"""Export and calibrate a specified checkpoint without reusing another model's engine."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'output/puct-10x-20261031/trt-deps'))
if importlib.util.find_spec('chess') is None:
    sys.path.append(str(next((ROOT / '.venv/lib').glob('python*/site-packages'))))
import ml_dtypes
import numpy as np
import onnx
from onnx import helper, numpy_helper
import tensorrt as trt
import torch
from neurochess.fast_inference import resident_linears
from neurochess.train import load_model


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if digest(args.checkpoint) != args.expected_sha256:
        raise ValueError('Selected checkpoint fingerprint differs')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.set_num_threads(2)
    model, state = load_model(args.checkpoint, 'cuda')
    step = state.get('step')
    origin = state.get('origin_step')
    del state
    model = resident_linears(model).eval()
    amax, hooks = {}, []
    for name, layer in model.named_modules():
        if name.startswith('blocks.') and isinstance(layer, torch.nn.Linear):
            def hook(module, inputs, name=name):
                x = inputs[0].detach().to(torch.bfloat16).float().reshape(-1, inputs[0].shape[-1])
                value = x.abs().amax(0)
                amax[name] = torch.maximum(amax[name], value) if name in amax else value
            hooks.append(layer.register_forward_pre_hook(hook))
    shard = sorted((ROOT / 'data/evals-v2-5m').glob('train-*.npy'))[0]
    data = np.load(shard, mmap_mode='r')
    indices = np.random.default_rng(103110).choice(len(data), size=1024, replace=False)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for offset in range(0, len(indices), 64):
            rows = data[indices[offset:offset + 64]]
            model(*(torch.as_tensor(np.array(rows[field]), device='cuda')
                    for field in ['tokens', 'meta', 'moves']))
    for hook in hooks:
        hook.remove()
    amax = {name: value.cpu().numpy() for name, value in amax.items()}
    print('calibrated', time.perf_counter() - start, flush=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = (torch.zeros((8, 64), dtype=torch.int32, device='cuda'),
                  torch.zeros((8, 12), device='cuda'),
                  torch.zeros((8, 256), dtype=torch.int32, device='cuda'))
        names = ['tokens', 'meta', 'moves', 'policy', 'q', 'wdl_logits']
        torch.onnx.export(model, inputs, str(out / 'bf16.onnx'),
            input_names=names[:3], output_names=names[3:],
            dynamic_axes={name: {0: 'batch'} for name in names},
            opset_version=18, dynamo=False)
    del model, inputs
    gc.collect()
    torch.cuda.empty_cache()
    graph = onnx.load(out / 'bf16.onnx')
    graph.opset_import[0].version = 21
    weights = {w.name: w for w in graph.graph.initializer}
    nodes, changed = [], []
    for node in graph.graph.node:
        match = re.fullmatch(r'/blocks\.(\d+)/(qkv|out|ff/ff\.[02])/MatMul', node.name)
        if match is None or node.input[1] not in weights:
            nodes.append(node)
            continue
        layer = f'blocks.{match[1]}.' + match[2].replace('ff/', '')
        weight = weights[node.input[1]]
        values = numpy_helper.to_array(weight).astype(np.float32)
        activation_max = amax[layer]
        smooth = np.sqrt(np.maximum(activation_max, 1e-6) / np.maximum(np.abs(values).max(1), 1e-6))
        scaled = values * smooth[:, None]
        weight.CopyFrom(numpy_helper.from_array(scaled.astype(ml_dtypes.bfloat16), weight.name))
        base = node.name.replace('/', '_')
        names = {key: base + '_' + key for key in ['smooth', 'ascale', 'wscale', 'az', 'wz', 'div', 'aq', 'adq', 'wq', 'wdq']}
        arrays = dict(smooth=smooth.astype(ml_dtypes.bfloat16),
            ascale=np.array(max(float(np.max(activation_max / smooth)), 1e-6) / 127, dtype=ml_dtypes.bfloat16),
            wscale=(np.abs(scaled).max(0).clip(1e-6) / 127).astype(ml_dtypes.bfloat16),
            az=np.array(0, np.int8), wz=np.zeros(values.shape[1], np.int8))
        for key, value in arrays.items():
            graph.graph.initializer.append(numpy_helper.from_array(value, names[key]))
        nodes.extend([
            helper.make_node('Div', [node.input[0], names['smooth']], [names['div']]),
            helper.make_node('QuantizeLinear', [names['div'], names['ascale'], names['az']], [names['aq']]),
            helper.make_node('DequantizeLinear', [names['aq'], names['ascale'], names['az']], [names['adq']]),
            helper.make_node('QuantizeLinear', [node.input[1], names['wscale'], names['wz']], [names['wq']], axis=1),
            helper.make_node('DequantizeLinear', [names['wq'], names['wscale'], names['wz']], [names['wdq']], axis=1)])
        node.input[0], node.input[1] = names['adq'], names['wdq']
        nodes.append(node)
        changed.append(layer)
    if set(changed) != set(amax):
        raise RuntimeError('Exported quantized layers do not match calibrated layers')
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph)
    onnx.save(graph, out / 'model.onnx')
    del graph
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(out / 'model.onnx')):
        raise RuntimeError('\n'.join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    for name, tail in [('tokens', (64,)), ('meta', (12,)), ('moves', (256,))]:
        profile.set_shape(name, (1, *tail), (8, *tail), (16, *tail))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT build failed')
    (out / 'model.plan').write_bytes(bytes(plan))
    manifest = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=args.expected_sha256,
        checkpoint_step=step, origin_step=origin, tensorrt=trt.__version__, engine_sha256=digest(out / 'model.plan'),
        calibration_shard=str(shard.relative_to(ROOT)), calibration_indices=indices.tolist(),
        calibration='1024 old TRAIN inputs only; no evaluation positions/labels', alpha=.5,
        changed_layers=changed, total_seconds=time.perf_counter() - start, source_sha256=digest(__file__))
    (out / 'build.json').write_text(json.dumps(manifest, indent=2))
    print('built', manifest['total_seconds'], flush=True)


if __name__ == '__main__':
    main()
