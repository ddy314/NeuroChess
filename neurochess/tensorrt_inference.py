"""Optional frozen TensorRT model wrapper; imported only in the isolated setup."""
from pathlib import Path
import torch
from torch import nn


class TensorRTModel(nn.Module):
    def __init__(self,plan,config,device='cuda'):
        super().__init__();import tensorrt as trt
        self.config=config;self.device=device;self.logger=trt.Logger(trt.Logger.WARNING);self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(plan).read_bytes())
        if self.engine is None:raise RuntimeError('Failed to deserialize TensorRT engine')
        self.context=self.engine.create_execution_context();self.inputs={};self.stream=torch.cuda.Stream(device=device)
        self.names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.output_names=[n for n in self.names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT]
        if set(self.output_names)!={'policy','q','wdl_logits'}:raise RuntimeError('Unexpected TensorRT output names')

    def forward(self,tokens,meta,moves):
        self.inputs=dict(tokens=tokens.to(torch.int32).contiguous(),meta=meta.float().contiguous(),moves=moves.to(torch.int32).contiguous())
        for name,value in self.inputs.items():
            if not self.context.set_input_shape(name,tuple(value.shape)):raise ValueError('Input outside TensorRT shape profile')
            self.context.set_tensor_address(name,value.data_ptr())
        outputs={name:torch.empty(tuple(self.context.get_tensor_shape(name)),dtype=torch.float32,device=self.device) for name in self.output_names}
        for name,value in outputs.items():self.context.set_tensor_address(name,value.data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream())
        if not self.context.execute_async_v3(self.stream.cuda_stream):raise RuntimeError('TensorRT execution failed')
        torch.cuda.current_stream().wait_stream(self.stream)
        return outputs['policy'],outputs['q'],outputs['wdl_logits']
