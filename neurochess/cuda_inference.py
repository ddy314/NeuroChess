"""Optional exact-shape CUDA graph inference for a frozen multihead model.

Prepare graphs before timing/playing. Uncaptured batch sizes use eager inference;
there is no padding, quantization, changed precision, or inference-output cache.
Use a separate instance after changing weights. Calls are sequential, not thread safe.
"""
import torch
from .multihead import MultiheadPlayer


class CudaGraphMultiheadPlayer(MultiheadPlayer):
    @torch.inference_mode()
    def __init__(self,model,device='cuda',batch_sizes=(64,)):
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise ValueError('CUDA graph inference requires CUDA')
        super().__init__(model,device)
        sizes=sorted({1,*map(int,batch_sizes)})
        if any(n<1 or n>256 for n in sizes):raise ValueError('Batch sizes must be in1..256')
        self.graphs={}
        # Private pools allow captured batch sizes to replay in arbitrary order.
        with torch.cuda.device(device):
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            for n in sizes:
                tokens=torch.zeros((n,64),dtype=torch.uint8,device=device)
                meta=torch.zeros((n,12),dtype=torch.float32,device=device)
                moves=torch.zeros((n,256),dtype=torch.int16,device=device)
                def forward():
                    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                        p,q,w=self.model(tokens,meta,moves)
                    return torch.cat([p,q,w.softmax(-1)],dim=1)
                with torch.cuda.stream(stream):
                    for _ in range(3):forward()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):output=forward()
                self.graphs[n]=(graph,tokens,meta,moves,output)
            torch.cuda.synchronize(device)

    @torch.inference_mode()
    def raw_outputs(self,tokens,meta,moves):
        n=len(tokens)
        if n not in self.graphs:return super().raw_outputs(tokens,meta,moves)
        if tokens.shape!=(n,64) or meta.shape!=(n,12) or moves.shape!=(n,256):
            raise ValueError('Invalid encoded input shapes')
        graph,t,m,ids,output=self.graphs[n]
        with torch.cuda.device(self.device):
            t.copy_(torch.as_tensor(tokens,dtype=torch.uint8))
            m.copy_(torch.as_tensor(meta,dtype=torch.float32))
            ids.copy_(torch.as_tensor(moves,dtype=torch.int16))
            graph.replay()
            # Return independent host storage: a later replay may overwrite
            # the captured output.
            return output.cpu().numpy()
