"""QA harness only: compare native runtime and Python scheduling on one engine."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'output/puct-10x-20261031/trt-deps'))
if importlib.util.find_spec('chess') is None:
    sys.path.append(str(next((ROOT / '.venv/lib').glob('python*/site-packages'))))
import chess
import numpy as np
import torch
from neurochess.encoding import encode, legal_ids
from neurochess.model import ModelConfig
from neurochess.multihead import MultiheadPlayer, fuse_heads
from neurochess.pipeline_mcts import PipelineMCTSPlayer
from neurochess.tensorrt_graph import TensorRTGraphMultiheadPlayer
from neurochess.tensorrt_inference import TensorRTModel


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class NativeNet:
    device = 'cuda'
    def __init__(self, library, plan):
        self.model = SimpleNamespace(config=ModelConfig(width=896, layers=16, heads=14, head_width=256, history_free=True))
        self.search_stats = dict(forward_positions=0)
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.nci_create.argtypes = [C.c_char_p, C.c_int, C.c_int]
        self.lib.nci_create.restype = C.c_void_p
        self.lib.nci_destroy.argtypes = [C.c_void_p]
        self.lib.nci_error.restype = C.c_char_p
        self.lib.nci_evaluate.argtypes = [C.c_void_p] * 4 + [C.c_int, C.c_void_p, C.c_void_p]
        self.lib.nci_fuse_device.argtypes = [C.c_void_p] * 6 + [C.c_int, C.c_size_t]
        self.handle = self.lib.nci_create(str(Path(plan).resolve()).encode(), 8, 1)
        if not self.handle:
            raise RuntimeError(self.lib.nci_error())

    def outputs(self, tokens, meta, moves, raw=False):
        out = np.empty((len(tokens), 515), np.float32)
        original = np.empty_like(out) if raw else None
        rc = self.lib.nci_evaluate(self.handle, tokens.ctypes.data, meta.ctypes.data, moves.ctypes.data,
            len(tokens), out.ctypes.data, original.ctypes.data if raw else None)
        if rc < 0:
            raise RuntimeError(self.lib.nci_error())
        return out, original

    def evaluate_search_heads(self, t, m, ids):
        out, _ = self.outputs(t, m, ids)
        self.search_stats['forward_positions'] += len(t)
        return out[:, :256], out[:, 256:512], out[:, 512]

    def close(self):
        if self.handle:
            self.lib.nci_destroy(self.handle)
            self.handle = None


class UCI:
    def __init__(self, binary, plan, cache=32768):
        self.process = subprocess.Popen([str(Path(binary).resolve()), '--engine', str(Path(plan).resolve()),
            '--uci', '--history-free', '--cache', str(cache)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.lines = queue.Queue()
        def read():
            for line in self.process.stdout:
                self.lines.put(line.rstrip())
            self.lines.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        self.send('uci')
        self.until('uciok')
        self.send('isready')
        self.until('readyok')

    def send(self, text):
        self.process.stdin.write(text + '\n')
        self.process.stdin.flush()

    def until(self, prefix):
        result = []
        while True:
            line = self.lines.get(timeout=60)
            if line is None:
                raise RuntimeError('Native executable exited')
            result.append(line)
            if line.startswith(prefix):
                return result

    def choose(self, board, fixed=None, seconds=None, reset=True):
        if reset:
            self.send('ucinewgame')
        history = ' '.join(move.uci() for move in board.move_stack)
        self.send('position fen ' + board.root().fen(en_passant='fen') + (' moves ' + history if history else ''))
        self.send('go nodes ' + str(fixed) if fixed else 'go movetime ' + str(round(seconds * 1000)))
        rows = self.until('bestmove ')
        telemetry = [line.split(' string telemetry ', 1)[1] for line in rows if ' string telemetry ' in line]
        if not telemetry:
            raise RuntimeError(rows)
        result = json.loads(telemetry[-1])
        assert result['move'] == rows[-1].split()[1]
        assert chess.Move.from_uci(result['move']) in board.legal_moves
        return result

    def close(self):
        if self.process.poll() is None:
            self.send('quit')
            self.process.wait(timeout=30)


def encoded(boards):
    data = [encode(b) for b in boards]
    ids = np.full((len(boards), 256), -1, np.int16)
    for i, board in enumerate(boards):
        li = legal_ids(board)[1]
        ids[i, :len(li)] = li
    return np.stack([e[0] for e in data]), np.stack([e[1] for e in data]), ids


def synthetic_heads(net):
    torch.manual_seed(11301)
    rows = []
    for n in [1, 7, 8]:
        p = torch.randn(n, 256, device='cuda')
        q = torch.randn_like(p).tanh()
        w = torch.randn(n, 3, device='cuda')
        ids = torch.full((n, 256), -1, device='cuda', dtype=torch.int32)
        for i in range(n):
            ids[i, :i + 1] = 0
        p[:, :8] = 1.0  # stable ties, fewer than five legal actions, tail mask
        out = torch.empty(n, 515, device='cuda')
        raw = torch.empty_like(out)
        rc = net.lib.nci_fuse_device(p.data_ptr(), q.data_ptr(), w.data_ptr(), ids.data_ptr(),
            out.data_ptr(), raw.data_ptr(), n, torch.cuda.current_stream().cuda_stream)
        assert rc == 0, net.lib.nci_error()
        actual = out.cpu().numpy()
        a, b, c, supported = fuse_heads(p.cpu().numpy(), q.cpu().numpy(), w.softmax(-1).cpu().numpy(), ids.cpu().numpy() >= 0)
        expected = np.concatenate([a, b, c[:, None]], 1)
        np.testing.assert_allclose(actual[:, :513], expected, atol=3e-6, rtol=3e-6, equal_nan=True)
        np.testing.assert_array_equal(np.isfinite(actual[:, 256:512]), supported)
        rows.append(dict(batch=n, passed=True))
    return rows


def main():
    parser = argparse.ArgumentParser()
    for name in ['binary', 'library', 'engine', 'positions', 'out']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--limit', type=int, default=16)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    boards = [chess.Board(fen) for fen in Path(args.positions).read_text().splitlines()[:args.limit]]
    native = NativeNet(args.library, args.engine)
    old_model = TensorRTModel(args.engine, native.model.config)
    old = TensorRTGraphMultiheadPlayer(old_model)
    raw_old = MultiheadPlayer(old_model)
    sources = ['neurochess/native/native_player.cpp', 'neurochess/native/trt_inference.cu',
        'neurochess/native/trt_inference.hpp', 'neurochess/native/CMakeLists.txt',
        'neurochess/native/mcts_pipeline.cpp', 'neurochess/native/chess_rules.hpp',
        'neurochess/native/vendor/chess.hpp', 'neurochess/pipeline_mcts.py',
        'neurochess/fused_heads.py', 'neurochess/tensorrt_graph.py', 'scripts/benchmark_native_runtime.py']
    load = lambda: subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], text=True).strip()
    report = dict(args=vars(args), engine_sha256=digest(args.engine), binary_sha256=digest(args.binary),
        library_sha256=digest(args.library), synthetic=synthetic_heads(native), output_parity=[], micro=[], fixed=[], timed=[],
        source_sha256={p:digest(p) for p in sources}, gpu_load_before=load(),
        executable_dependencies=subprocess.check_output(['ldd',str(Path(args.binary).resolve())],text=True),
        note='One identical TensorRT plan. Existing Python/Triton heads versus C++/CUDA heads. Fixed-node controls also compare identical native inference with different schedulers. Startup excluded; current background GPU load may affect timings.')
    for n in [8, 1, 7, 2, 6, 3, 5, 4]:
        inputs = encoded([boards[i % len(boards)] for i in range(n)])
        actual, raw = native.outputs(*inputs, raw=True)
        p, q, v = old.evaluate_search_heads(*inputs)
        expected = np.concatenate([p, q, v[:, None]], 1)
        reference_raw = raw_old.raw_outputs(*inputs)
        np.testing.assert_allclose(actual[:, :513], expected, atol=3e-6, rtol=3e-6, equal_nan=True)
        np.testing.assert_allclose(raw, reference_raw, atol=3e-6, rtol=3e-6)
        report['output_parity'].append(dict(batch=n, max_fused_error=float(np.nanmax(np.abs(actual[:, :513] - expected))),
            max_raw_error=float(np.max(np.abs(raw - reference_raw))), support_equal=bool(np.array_equal(np.isnan(actual[:, 256:512]), np.isnan(q)))))
    for n in [1, 8]:
        inputs = encoded([boards[i % len(boards)] for i in range(n)])
        samples = {'python': [], 'native_bridge': []}
        for i in range(40):
            for name in (['python', 'native_bridge'] if i % 2 == 0 else ['native_bridge', 'python']):
                tic = time.perf_counter()
                (old if name == 'python' else native).evaluate_search_heads(*inputs)
                samples[name].append(time.perf_counter() - tic)
        report['micro'].append(dict(batch=n, samples=samples,
            median={name: float(np.median(t)) for name, t in samples.items()}))
    controls = {}
    try:
        for capacity in [0, 32768]:
            controls[capacity] = UCI(args.binary, args.engine, capacity)
            for index, board in enumerate(boards):
                results = {}
                for name in (['native', 'python_native_nn', 'python_original'] if index % 2 == 0 else ['python_original', 'python_native_nn', 'native']):
                    if name == 'native':
                        results[name] = controls[capacity].choose(board, fixed=512)
                    else:
                        player = PipelineMCTSPlayer(native if name == 'python_native_nn' else old,
                            simulations=512, leaf_batch_size=8, inference_cache_size=capacity,
                            pending_loss_guard=True)
                        tic = time.perf_counter()
                        move = player.choose(board)
                        results[name] = dict(move=move.uci(), seconds=time.perf_counter() - tic,
                            root=player.last_result, stats=player.search_stats)
                        player.close()
                reference = results['python_native_nn']['root']
                root = results['native']['native']['roots'][0]
                assert results['native']['move'] == results['python_native_nn']['move'], (capacity, index, results)
                assert root['leaders'] == reference['leaders'], (capacity, index, root, reference)
                assert root['new_simulations'] == reference['new_simulations']
                report['fixed'].append(dict(index=index, fen=board.fen(en_passant='fen'), cache=capacity,
                    identical_native_nn_search=True, results=results))
                (out / 'progress.json').write_text(json.dumps(report, indent=2))
                print(json.dumps(dict(stage='fixed', index=index, cache=capacity,
                    moves={name: r['move'] for name, r in results.items()})), flush=True)
        for index, board in enumerate(boards):
            results = {}
            for name in (['native', 'python_original'] if index % 2 == 0 else ['python_original', 'native']):
                if name == 'native':
                    results[name] = controls[32768].choose(board, seconds=.5)
                else:
                    player = PipelineMCTSPlayer(old, simulations=16384, leaf_batch_size=8,
                        time_limit=.5, time_margin=.006, inference_cache_size=32768, pending_loss_guard=True)
                    tic = time.perf_counter()
                    move = player.choose(board)
                    results[name] = dict(move=move.uci(), seconds=time.perf_counter() - tic,
                        root=player.last_result, stats=player.search_stats)
                    player.close()
            report['timed'].append(dict(index=index, fen=board.fen(), results=results))
            (out / 'progress.json').write_text(json.dumps(report, indent=2))
    finally:
        for process in controls.values():
            process.close()
        native.close()
    report['summary'] = {}
    for capacity in [0, 32768]:
        selected = [r for r in report['fixed'] if r['cache'] == capacity]
        medians = {name: float(np.median([r['results'][name]['seconds'] for r in selected]))
                   for name in ['native', 'python_native_nn', 'python_original']}
        report['summary'][str(capacity)] = dict(median_seconds=medians,
            speedup_over_original=medians['python_original'] / medians['native'],
            original_move_agreement=sum(r['results']['native']['move'] == r['results']['python_original']['move'] for r in selected), samples=len(selected))
        ratios = np.array([r['results']['python_original']['seconds']/r['results']['native']['seconds'] for r in selected])
        rng = np.random.default_rng(11301)
        samples = np.exp(np.log(ratios)[rng.integers(len(ratios),size=(20000,len(ratios)))].mean(1))
        report['summary'][str(capacity)].update(paired_geomean_speedup=float(np.exp(np.log(ratios).mean())),
            paired_speedup_bootstrap95=np.quantile(samples,[.025,.975]).tolist())
    for name in ['native', 'python_original']:
        rows = [r['results'][name] for r in report['timed']]
        sims = [r['native']['roots'][0]['new_simulations'] if name == 'native' else r['root']['new_simulations'] for r in rows]
        report['summary'][name + '_500ms'] = dict(median_simulations=float(np.median(sims)),
            p95_seconds=float(np.quantile([r['seconds'] for r in rows], .95)), maximum_seconds=max(r['seconds'] for r in rows))
    report['gpu_load_after']=load()
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
