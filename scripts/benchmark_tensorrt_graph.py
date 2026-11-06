"""Check arbitrary-order captured TensorRT replay against the same eager engine."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if importlib.util.find_spec('chess') is None:
    sys.path.append(str(next((ROOT / '.venv/lib').glob('python*/site-packages'))))
sys.path.insert(0, str(ROOT / 'output/puct-10x-20261031/trt-deps'))
import chess
import numpy as np
import torch
from neurochess.encoding import encode, legal_ids
from neurochess.multihead import MultiheadPlayer
from neurochess.tensorrt_inference import TensorRTModel
from neurochess.tensorrt_graph import TensorRTGraphMultiheadPlayer
from neurochess.train import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    root = ROOT / 'output/puct-10x-20261031'
    torch.set_num_threads(2)
    model, state = load_model(root / 'frozen-baseline/model.pt', 'cuda')
    config = model.config
    del model, state
    model = TensorRTModel(root / 'tensorrt-int8/model.plan', config)
    eager = MultiheadPlayer(model)
    tic = time.perf_counter()
    captured = TensorRTGraphMultiheadPlayer(model, batch_sizes=range(2, 9))
    setup = time.perf_counter() - tic
    fens = (ROOT / 'output/puct-2s-20261031/final-roots/roots.fen').read_text().splitlines()
    rows = []
    for n in [8, 1, 7, 2, 6, 3, 5, 4, 8, 1]:
        boards = [chess.Board(fens[(len(rows) + i) % len(fens)]) for i in range(n)]
        encoded = [encode(b) for b in boards]
        ids = np.full((n, 256), -1, np.int16)
        for i, b in enumerate(boards):
            li = legal_ids(b)[1]
            ids[i, :len(li)] = li
        args = (np.stack([e[0] for e in encoded]), np.stack([e[1] for e in encoded]), ids)
        ref = eager.evaluate_search_heads(*args)
        actual = captured.evaluate_search_heads(*args)
        for x, y in zip(actual, ref):
            np.testing.assert_allclose(x, y, atol=3e-6, rtol=3e-6, equal_nan=True)
        timings = {}
        for name, net in [('eager', eager), ('captured', captured)]:
            samples = []
            for _ in range(12):
                tic = time.perf_counter()
                net.evaluate_search_heads(*args)
                samples.append(time.perf_counter() - tic)
            timings[name] = dict(median=float(np.median(samples)), p95=float(np.quantile(samples, .95)))
        rows.append(dict(batch=n, timings=timings))
        print(json.dumps(rows[-1]), flush=True)
    (out / 'result.json').write_text(json.dumps(dict(setup_seconds=setup, rows=rows,
        note='Graph compared to the same quantized engine, not to original BF16 weights.'), indent=2))


if __name__ == '__main__':
    main()
