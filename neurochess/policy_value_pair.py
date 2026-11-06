"""Portable exact new-policy/Q + independently trainable old-value branch.

Both original trunks are preserved. This is not a value-layer weight transplant.
No original frozen baseline checkpoint or default model is overwritten.
"""
from dataclasses import asdict
import hashlib
import torch
from torch import nn
from .model import ChessModel,ModelConfig

ARCHITECTURE='policy_value_pair_v1'

class PolicyValuePair(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.policy_model=ChessModel(config);self.value_model=ChessModel(config)
        self.set_value_training('head')
    def set_value_training(self,mode):
        if mode not in ['head','backbone','joint']:raise ValueError('Training mode must be head, backbone or joint')
        self.requires_grad_(False)
        if mode=='head':self.value_model.value.requires_grad_(True)
        else:
            for name,p in self.value_model.named_parameters():
                if name.startswith(('piece.','rank.','file.','meta.','blocks.','norm.','value.')):p.requires_grad_(True)
        if mode=='joint':
            self.policy_model.requires_grad_(True)
            self.policy_model.value.requires_grad_(False) # This branch's WDL is unused.
        self.value_training_mode=mode
    def train(self,mode=True):
        super().train(mode)
        if self.value_training_mode!='joint':self.policy_model.eval()
        return self
    def value_only(self,tokens,meta):return self.value_model.value_only(tokens,meta)
    def forward(self,tokens,meta,moves):
        if self.value_training_mode=='joint':policy,q,_=self.policy_model(tokens,meta,moves)
        else:
            with torch.no_grad():policy,q,_=self.policy_model(tokens,meta,moves)
        return policy,q,self.value_only(tokens,meta)


def from_original_states(policy_checkpoint,value_checkpoint,device='cpu'):
    if policy_checkpoint['model_config']!=value_checkpoint['model_config']:raise ValueError('Original model configurations differ')
    model=PolicyValuePair(ModelConfig(**policy_checkpoint['model_config']))
    model.policy_model.load_state_dict(policy_checkpoint['model']);model.value_model.load_state_dict(value_checkpoint['model'])
    return model.to(device)


def policy_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.policy_model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def checkpoint(model,step,lineage,**extra):
    return dict(architecture=ARCHITECTURE,model_config=asdict(model.config),model=model.state_dict(),step=step,
                value_training_mode=model.value_training_mode,lineage=lineage,policy_q_frozen=model.value_training_mode!='joint',**extra)
