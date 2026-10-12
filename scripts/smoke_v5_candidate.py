"""Real-checkpoint 16-update smoke for isolated v5 candidate."""
import argparse, json, hashlib, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_expression_response import load_runtime, configure, state_digest, development_fold
from scripts.train_unified_v4 import build_base_cache, attach_base, motion_targets, make_refs, reference_ids, select_balanced
from kinetalk_v5.dual_timescale_model import DualTimescaleKineTalkV5, v5_losses

def sha(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main(a):
 configure(a.seed); device=torch.device(a.device)
 binding=json.loads(Path(a.binding).read_text())
 data,base=load_runtime(binding,device)
 q=data['splits']['train']; fit=development_fold(q,data['fit_sids'])['train']
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 model=DualTimescaleKineTalkV5.from_v4_checkpoint(ck,device=device)
 model.train(); opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01)
 ids=select_balanced(q,fit,a.batch_size,a.seed); cache=build_base_cache(q,base,ids,device)
 pool={}
 for i in fit: pool.setdefault(int(q['speaker_id'][i]),[]).append(i)
 import random; rng=random.Random(a.seed)
 refs_ids=[reference_ids(q,[i],pool,rng)[0] for i in ids]
 refs=make_refs(q,refs_ids,device)
 rows=[]
 for step in range(1,a.updates+1):
  batch=attach_base(q.batch(torch.tensor(ids),device),ids,cache)
  targets=motion_targets(batch)
  opt.zero_grad(set_to_none=True)
  out=model.training_forward(batch,refs,targets,seed=a.seed+step)
  losses=v5_losses(model,out,targets,batch)
  total=losses['total']
  if not torch.isfinite(total): raise RuntimeError('nonfinite loss')
  total.backward()
  grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
  heads=[model.decoder.affect_adapter,model.decoder.phonetic_adapter,model.envelope_head]
  checks=[]
  for h in heads:
   g=h.weight.grad
   checks.append(bool(g is not None and torch.isfinite(g).all() and g.abs().sum()>0))
  if not all(checks): raise RuntimeError(f'head gradients failed {checks}')
  if not torch.isfinite(grad): raise RuntimeError('nonfinite grad norm')
  opt.step()
  row={'step':step,'total':float(total),'grad_norm':float(grad),
       'envelope_distill':float(losses['envelope_distill']),
       'confidence':float(losses['confidence']),
       'factor_regularizer':float(losses['factor_regularizer']),
       'affect_gate':float(model.decoder.affect_gate.sigmoid()),
       'phonetic_gate':float(model.decoder.phonetic_gate.sigmoid()),'head_gradients':checks}
  rows.append(row); print(json.dumps(row),flush=True)
 Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 Path(a.output).write_text(json.dumps({'schema':'v5_candidate_smoke_v1','status':'complete','updates':a.updates,
  'checkpoint_sha256':sha(a.checkpoint),'binding_sha256':sha(a.binding),'test_loaded':False,
  'clip_indices':ids,'fit_clips':len(fit),'rows':rows},indent=2))
 print(json.dumps({'status':'complete','output':a.output,'updates':a.updates}))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--binding',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--updates',type=int,default=16); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--seed',type=int,default=47); p.add_argument('--device',default='cuda'); main(p.parse_args())
