"""Export a NeuroChess checkpoint to a TensorRT INT8 engine for the native UCI engine.

    python -m neurochess.export --model neurochess-13750.pt --out models/

The 64 Transformer matmuls are quantized to INT8 with SmoothQuant (alpha = 0.5,
per-channel activation maxima from 1024 calibration positions); embeddings,
LayerNorms and the output heads stay in BF16/FP32. TensorRT engines are tied to
the GPU architecture and TensorRT version, so build the engine on the machine
that will run it.

Requires the ``export`` extra: ``pip install -e '.[export]'`` plus a TensorRT
Python wheel matching the TensorRT SDK used to compile the engine.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn

from .encoding import encode_inputs
from .model import load_model

CALIBRATION = Path(__file__).with_name('calibration.fen')
INPUTS = ['tokens', 'meta', 'moves']
OUTPUTS = ['policy', 'q', 'wdl_logits']
MATMUL = re.compile(r'/blocks\.(\d+)/(qkv|out|ff/ff\.[02])/MatMul')


def calibration_batch(path=CALIBRATION, limit=1024):
    rows = [encode_inputs(chess.Board(line.strip())) for line in open(path) if line.strip()][:limit]
    return tuple(np.stack([row[i] for row in rows]) for i in range(3))


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16)


def activation_maxima(model, batch, device, chunk=64):
    """Per-input-channel |x| maxima of every Transformer linear layer."""
    amax, hooks = {}, []
    for name, layer in model.named_modules():
        if name.startswith('blocks.') and isinstance(layer, nn.Linear):
            def hook(module, inputs, name=name):
                x = inputs[0].detach().to(torch.bfloat16).float().reshape(-1, inputs[0].shape[-1])
                value = x.abs().amax(0)
                amax[name] = torch.maximum(amax[name], value) if name in amax else value
            hooks.append(layer.register_forward_pre_hook(hook))
    with torch.inference_mode(), autocast(device):
        for start in range(0, len(batch[0]), chunk):
            model(*(torch.as_tensor(x[start:start + chunk], device=device) for x in batch))
    for hook in hooks:
        hook.remove()
    return {name: value.cpu().numpy() for name, value in amax.items()}


def quantize_onnx(path, amax):
    """Insert SmoothQuant scaling and INT8 Q/DQ pairs around the Transformer matmuls."""
    import ml_dtypes
    import onnx
    from onnx import helper, numpy_helper

    graph = onnx.load(path)
    graph.opset_import[0].version = 21
    weights = {w.name: w for w in graph.graph.initializer}
    nodes, changed = [], []
    for node in graph.graph.node:
        match = MATMUL.fullmatch(node.name)
        if match is None or node.input[1] not in weights:
            nodes.append(node)
            continue
        layer = f'blocks.{match[1]}.' + match[2].replace('ff/', '')
        weight = weights[node.input[1]]
        values = numpy_helper.to_array(weight).astype(np.float32)
        activation = amax[layer]
        smooth = np.sqrt(np.maximum(activation, 1e-6) / np.maximum(np.abs(values).max(1), 1e-6))
        scaled = values * smooth[:, None]
        weight.CopyFrom(numpy_helper.from_array(scaled.astype(ml_dtypes.bfloat16), weight.name))
        base = node.name.replace('/', '_')
        n = {key: f'{base}_{key}' for key in ['smooth', 'ascale', 'wscale', 'az', 'wz', 'div', 'aq', 'adq', 'wq', 'wdq']}
        constants = dict(
            smooth=smooth.astype(ml_dtypes.bfloat16),
            ascale=np.array(max(float(np.max(activation / smooth)), 1e-6) / 127, dtype=ml_dtypes.bfloat16),
            wscale=(np.abs(scaled).max(0).clip(1e-6) / 127).astype(ml_dtypes.bfloat16),
            az=np.array(0, np.int8), wz=np.zeros(values.shape[1], np.int8))
        for key, value in constants.items():
            graph.graph.initializer.append(numpy_helper.from_array(value, n[key]))
        nodes += [
            helper.make_node('Div', [node.input[0], n['smooth']], [n['div']]),
            helper.make_node('QuantizeLinear', [n['div'], n['ascale'], n['az']], [n['aq']]),
            helper.make_node('DequantizeLinear', [n['aq'], n['ascale'], n['az']], [n['adq']]),
            helper.make_node('QuantizeLinear', [node.input[1], n['wscale'], n['wz']], [n['wq']], axis=1),
            helper.make_node('DequantizeLinear', [n['wq'], n['wscale'], n['wz']], [n['wdq']], axis=1)]
        node.input[0], node.input[1] = n['adq'], n['wdq']
        nodes.append(node)
        changed.append(layer)
    if set(changed) != set(amax):
        raise RuntimeError('Exported quantized layers do not match calibrated layers')
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph)
    onnx.save(graph, path)
    return changed


def build_engine(onnx_path, plan_path, max_batch):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    for name, width in [('tokens', 64), ('meta', 12), ('moves', 256)]:
        profile.set_shape(name, (1, width), (16, width), (max_batch, width))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT build failed')
    Path(plan_path).write_bytes(bytes(plan))
    return trt.__version__


def main():
    parser = argparse.ArgumentParser(description='Build a TensorRT INT8 engine from a NeuroChess checkpoint.')
    parser.add_argument('--model', required=True, help='path to the .pt checkpoint')
    parser.add_argument('--out', default='models', help='output directory')
    parser.add_argument('--name', default='neurochess-13750-int8')
    parser.add_argument('--max-batch', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device used for calibration and ONNX export')
    parser.add_argument('--keep-onnx', action='store_true')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path, plan_path = out / f'{args.name}.onnx', out / f'{args.name}.plan'
    device = torch.device(args.device)
    start = time.perf_counter()

    model = load_model(args.model, device)
    for layer in model.modules():  # same BF16 weights autocast would use
        if isinstance(layer, nn.Linear):
            layer.to(dtype=torch.bfloat16)
    amax = activation_maxima(model, calibration_batch(), device)
    print(f'calibrated {len(amax)} layers in {time.perf_counter() - start:.1f}s', flush=True)

    dummy = (torch.zeros((8, 64), dtype=torch.int32, device=device), torch.zeros((8, 12), device=device),
             torch.zeros((8, 256), dtype=torch.int32, device=device))
    with torch.inference_mode(), autocast(device):
        torch.onnx.export(model, dummy, str(onnx_path), input_names=INPUTS, output_names=OUTPUTS,
                          dynamic_axes={name: {0: 'batch'} for name in INPUTS + OUTPUTS},
                          opset_version=18, dynamo=False)
    del model, dummy
    changed = quantize_onnx(onnx_path, amax)
    print(f'quantized {len(changed)} matmuls; building TensorRT engine...', flush=True)
    version = build_engine(onnx_path, plan_path, args.max_batch)
    if not args.keep_onnx:
        for path in out.glob(f'{args.name}.onnx*'):
            path.unlink()
    print(f'wrote {plan_path} ({time.perf_counter() - start:.1f}s)')


if __name__ == '__main__':
    main()
