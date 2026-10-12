"""Publish the verified model selection and check actual no-GT inference."""
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'docs/v6_factor_paths_20261012'))
from remote_transport import transport

REMOTE=r'''
import hashlib,json,sys,time
from pathlib import Path
import torch
selection=Path('/root/kinetalk_current_selection')
sys.path.insert(0,str(selection))
from current_model import load_current_model
device=torch.device('cuda')
model,record=load_current_model(selection/'current_model.json',device)
sys.path.insert(0,record['source_root'])
from scripts.train_expression_response import configure,load_runtime,state_digest
from scripts.train_unified_v4 import build_base_cache,attach_base
configure(47)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(record['binding'])==record['binding_sha256']
binding=json.loads(Path(record['binding']).read_text())
data,base=load_runtime(binding,device)
q=data['splits']['validation']
ids=[q['clip_id'].index(binding['display_clips'][emotion]) for emotion in ('angry','happy')]
cache=build_base_cache(q,base,ids,device)
before=state_digest(model)
checks=[]
for i in ids:
    batch=attach_base(q.batch(torch.tensor([i]),device),[i],cache)
    reference=data['refs'][int(q['speaker_id'][i])]
    assert q['clip_id'][i] not in reference['clip_id']
    refs={k:reference[k][0:1].to(device) for k in ('motion','valid','channel_mask')}
    with torch.inference_mode():
        output=model.predict(batch['audio_features'],batch['h0'],batch['b0'],batch['valid'],
                             refs,times=batch['times'],seed=95,mode='sample')
    assert torch.isfinite(output['motion']).all()
    assert output['motion'].shape==batch['b0'].shape
    checks.append(dict(clip=q['clip_id'][i],valid_frames=int(batch['valid'].sum()),
        motion_shape=list(output['motion'].shape),finite=True,reference_count=1,
        query_motion_passed_to_predict=False,teacher_called=False,mode='sample',seed=95))
assert before==state_digest(model)
receipt=dict(schema='joint_model_rollback_receipt_v1',status='applied',time=time.time(),
    selected_model=record['selected_model'],checkpoint_sha256=record['checkpoint_sha256'],
    selection_path=str(selection/'current_model.json'),selection_sha256=sha(selection/'current_model.json'),
    loader_sha256=sha(selection/'current_model.py'),strict_load=True,
    real_inference_checks=checks,model_unchanged=True,training_started=False,
    v6_status='stopped',test_loaded=False)
target=selection/'rollback_receipt.json'
target.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

if __name__=='__main__':
    transport({'command':'mkdir -p /root/kinetalk_current_selection'})
    transport({'upload':[
        [str(PROJECT/'configs/current_model.json'),'/root/kinetalk_current_selection/local_selection_copy.json'],
        [str(Path(__file__).with_name('remote_current_model.json')),'/root/kinetalk_current_selection/current_model.json'],
        [str(PROJECT/'scripts/current_model.py'),'/root/kinetalk_current_selection/current_model.py'],
        [str(PROJECT/'docs/CURRENT_MODEL_AND_TRAINING.md'),'/root/kinetalk_current_selection/CURRENT_MODEL_AND_TRAINING.md'],
        [str(PROJECT/'docs/CURRENT_OPTIMIZATION_STATE.md'),'/root/kinetalk_current_selection/CURRENT_OPTIMIZATION_STATE.md']]})
    result=json.loads(transport({'command':"/root/miniconda3/bin/python - <<'PY'\n"+REMOTE+"\nPY",'timeout':120}))
    Path(__file__).with_name('rollback_receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))
