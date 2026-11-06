from dataclasses import asdict
import torch
from neurochess.model import ChessModel,ModelConfig
from neurochess.policy_value_pair import from_original_states,checkpoint,policy_digest
from neurochess.train import load_model


def test_exact_functional_pair_and_value_head_only_training_roundtrip(tmp_path):
 torch.manual_seed(31);config=ModelConfig(width=32,layers=1,heads=4,head_width=8,history_free=True)
 old=ChessModel(config).eval();new=ChessModel(config).eval();a=dict(model_config=asdict(config),model=new.state_dict());b=dict(model_config=asdict(config),model=old.state_dict())
 pair=from_original_states(a,b).eval();tokens=torch.randint(0,13,(3,64));meta=torch.randn(3,12);moves=torch.tensor([[1,20,-1],[100,200,-1],[300,400,-1]])
 with torch.no_grad():
  p,q,v=pair(tokens,meta,moves);np,nq,_=new(tokens,meta,moves);_,_,ov=old(tokens,meta,moves)
 assert torch.equal(p,np) and torch.equal(q,nq) and torch.equal(v,ov)
 before=policy_digest(pair);encoder={n:t.clone() for n,t in pair.value_model.state_dict().items() if not n.startswith('value.')}
 opt=torch.optim.AdamW([p for p in pair.parameters() if p.requires_grad],lr=.01)
 pair.train();loss=torch.nn.functional.cross_entropy(pair.value_only(tokens,meta),torch.tensor([0,1,2]));loss.backward();opt.step()
 assert policy_digest(pair)==before and all(torch.equal(t,pair.value_model.state_dict()[n]) for n,t in encoder.items())
 assert not torch.equal(pair.value_only(tokens,meta),ov)
 path=tmp_path/'pair.pt';torch.save(checkpoint(pair,1,dict(test=True)),path);restored,state=load_model(path,'cpu');restored.eval()
 with torch.no_grad():
  rp,rq,rv=restored(tokens,meta,moves);p,q,v=pair.eval()(tokens,meta,moves)
 assert torch.equal(rp,p) and torch.equal(rq,q) and torch.equal(rv,v)
 restored.set_value_training('backbone')
 assert restored.value_model.blocks[0].qkv.weight.requires_grad and not restored.policy_model.blocks[0].qkv.weight.requires_grad
 assert not restored.value_model.action[1].weight.requires_grad


def test_cached_encoder_features_and_kl_anchor_are_safe_for_autograd():
 config=ModelConfig(width=32,layers=1,heads=4,head_width=8);model=ChessModel(config)
 for p in model.parameters():p.requires_grad_(False)
 model.value.requires_grad_(True);tokens=torch.randint(0,13,(4,64));meta=torch.randn(4,12)
 with torch.inference_mode():parts=[model.encode_squares(tokens,meta).mean(1).float()]
 features=torch.cat(parts)
 with torch.inference_mode():anchor=model.value(features).softmax(-1).detach().clone()
 anchor=anchor.clone();logits=model.value(features);loss=torch.nn.functional.kl_div(logits.log_softmax(-1),anchor,reduction='batchmean')+torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,2,0]));loss.backward()
 assert model.value[0].weight.grad is not None and model.piece.weight.grad is None


def test_joint_training_updates_both_encoders_and_all_three_outputs(tmp_path):
 torch.manual_seed(48);c=ModelConfig(width=32,layers=1,heads=4,head_width=8)
 a=ChessModel(c);s=dict(model_config=asdict(c),model=a.state_dict())
 pair=from_original_states(s,s);pair.set_value_training('joint');pair.train()
 tokens=torch.randint(0,13,(3,64));meta=torch.randn(3,12);moves=torch.tensor([[1,20],[100,200],[300,400]])
 p,q,v=pair(tokens,meta,moves)
 loss=torch.nn.functional.cross_entropy(p,torch.tensor([0,1,0]))+q.square().mean()+torch.nn.functional.cross_entropy(v,torch.tensor([0,1,2]));loss.backward()
 for param in [pair.policy_model.action[1].weight,pair.policy_model.blocks[0].qkv.weight,pair.value_model.value[2].weight,pair.value_model.blocks[0].qkv.weight]:
  assert param.grad is not None and param.grad.abs().sum()>0
 assert pair.policy_model.value[0].weight.grad is None
 before=policy_digest(pair);torch.optim.SGD([p for p in pair.parameters() if p.requires_grad],lr=.01).step();assert policy_digest(pair)!=before
 file=tmp_path/'joint.pt';torch.save(checkpoint(pair,1,{}),file);loaded,state=load_model(file,'cpu')
 assert loaded.value_training_mode=='joint' and state['policy_q_frozen'] is False
 loaded.set_value_training('head');assert not loaded.policy_model.action[1].weight.requires_grad
