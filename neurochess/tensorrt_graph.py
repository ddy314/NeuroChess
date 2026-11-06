"""Per-shape TensorRT contexts captured with input casts and GPU head fusion.

Each graph owns its context and buffers. Sharing one dynamic TensorRT context
across captured shapes can invalidate workspace pointers; this class does not.
"""
import torch

from .fused_heads import FusedGraphMultiheadPlayer, fuse_cuda


class TensorRTGraphMultiheadPlayer(FusedGraphMultiheadPlayer):
    def __init__(self, model, device='cuda', batch_sizes=range(2, 9)):
        self.contexts = {}
        super().__init__(model, device, batch_sizes)

    def device_forward(self, tokens, meta, moves):
        n = len(tokens)
        if n not in self.contexts:
            import tensorrt as trt
            context = self.model.engine.create_execution_context()
            inputs = dict(
                tokens=torch.empty_like(tokens, dtype=torch.int32),
                meta=torch.empty_like(meta, dtype=torch.float32),
                moves=torch.empty_like(moves, dtype=torch.int32),
            )
            for name, tensor in inputs.items():
                if not context.set_input_shape(name, tuple(tensor.shape)):
                    raise ValueError('Batch outside TensorRT shape profile')
                context.set_tensor_address(name, tensor.data_ptr())
            outputs = {}
            for name in self.model.output_names:
                if self.model.engine.get_tensor_dtype(name) != trt.float32:
                    raise ValueError('Expected float32 TensorRT outputs')
                tensor = torch.empty(tuple(context.get_tensor_shape(name)),
                                     device=tokens.device, dtype=torch.float32)
                context.set_tensor_address(name, tensor.data_ptr())
                outputs[name] = tensor
            self.contexts[n] = context, inputs, outputs
        context, inputs, outputs = self.contexts[n]
        for name, tensor in [('tokens', tokens), ('meta', meta), ('moves', moves)]:
            inputs[name].copy_(tensor)
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('Captured TensorRT execution failed')
        return fuse_cuda(outputs['policy'], outputs['q'], outputs['wdl_logits'], moves)
