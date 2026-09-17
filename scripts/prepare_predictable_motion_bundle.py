"""Stream locked metadata into compact residual bins and audio sidecars.

No full duplicated training cache is written. Test manifests are deliberately
not loaded: use validation for development and leave locked new_test unused.
Frozen B0/identity are evaluated without changing their state.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.neutral_data import native_clip
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.utils import freeze_module
from kinetalk_b0.emotion_ray import NUISANCE_CHANNELS_52
from kinetalk_b0.predictable_motion import bin_centered_frames
from scripts.extract_predictable_audio import extract
from scripts.extract_emotion2vec_pilot import load_extractor
from scripts.train_neutral_affect_pilot import sha, device_batch, observed
from scripts.train_neutral_affect_audio_ablation import state_hash


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def row_with_ids(row):
    return {**row, 'emotion_id':int(row.get('emotion_id',row.get('emotion',-1))),
        'intensity_id':int(row.get('intensity_id',row.get('intensity',-1)))}


@torch.no_grad()
def build_split(rows, root, identities, system, model, geometry, device, layers):
    motion_bins, weights, feature_bins, content_bins, records, clips = [],[],[],[],[],[]
    masks=[]
    stride=system.motion_teacher.stride
    for offset in range(0,len(rows),32):
        part=[native_clip(row_with_ids(r),root) for r in rows[offset:offset+32]]
        query=device_batch(part,device)
        base=system.base(query['content'],query['valid'])
        id_base=torch.stack([identities[int(s)]['baseline'][0] for s in query['speaker_id']])
        residual=torch.where(observed(query),query['motion']-base['b0']-id_base[:,None],0)
        residual[:,:,list(NUISANCE_CHANNELS_52)]=0
        y,w=bin_centered_frames(residual,query['valid'],stride)
        motion_bins.append(y.float()); weights.append(w.float())
        content,_=bin_centered_frames(query['content'],query['valid'],stride)
        content_bins.append(content.float())
        for clip in part:
            rec=extract(clip,model,geometry,layers,device)
            middle,_=bin_centered_frames(rec['middle'].float()[None],rec['valid'][None],stride)
            final,_=bin_centered_frames(rec['final'].float()[None],rec['valid'][None],stride)
            pro,_=bin_centered_frames(rec['prosody'][None],rec['valid'][None],stride)
            feature_bins.append((final[0].float(),middle[0].float(),pro[0].float()))
            records.append(rec['record'])
        masks.append(query['channel_mask'].cpu())
        clips.extend({k:c[k] for k in ('clip_id','sentence_id','emotion_id','speaker_id')} for c in part)
        print(json.dumps({'done':len(clips),'total':len(rows)}),flush=True)
    common=torch.cat(masks).all(0)
    common[list(NUISANCE_CHANNELS_52)]=False
    y=torch.cat(motion_bins)
    if (y[:,:,~common]!=0).any():
        raise ValueError('Missing-channel variation; select consistent observed layout')
    groups={'all_expression':common.nonzero(as_tuple=True)[0].tolist(),
        'upper_expression':[c for c in (5,6,12,13,41,42,43,44,45) if common[c]],
        'mouth':[c for c in range(14,41) if common[c]],'jaw17':[17] if common[17] else []}
    return {'motion_bins':y,'weight':torch.cat(weights),
        'features':{'acoustic':torch.stack([v[0] for v in feature_bins]),'content':torch.cat(content_bins),
            'middle':torch.stack([v[1] for v in feature_bins]),'prosody':torch.stack([v[2] for v in feature_bins])},
        'clip_id':[c['clip_id'] for c in clips],'sentence_id':[c['sentence_id'] for c in clips],
        'emotion_id':torch.stack([c['emotion_id'] for c in clips]),
        'speaker_id':torch.stack([c['speaker_id'] for c in clips]),
        'channel_mask':torch.cat(masks),'groups':groups,'audio_records':records}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest-dir','native-root','checkpoint','config','model-dir','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--layers',type=int,nargs='+',default=[2,4,6])
    args=p.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(args.config.read_text())
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if any(cfg.get(k)!=ck.get('config',{}).get(k) for k in ('data','model')):
        raise ValueError('Checkpoint/config mismatch')
    device='cuda'
    system=NeutralAffectSystem(cfg).to(device).eval()
    system.load_state_dict(ck['model'],strict=True); freeze_module(system)
    before=state_hash(system.state_dict())
    model,geometry,info=load_extractor(args.model_dir,device)
    if not args.layers or any(l < 1 or l >= len(model.blocks) for l in args.layers):
        raise ValueError('Require valid nonempty intermediate block indices')
    references=read_rows(args.manifest_dir/'enrollment.jsonl')
    identities={}
    with torch.no_grad():
        for sid in sorted({int(r['speaker_id']) for r in references}):
            clips=[native_clip(row_with_ids(r),args.native_root) for r in references if int(r['speaker_id'])==sid]
            q=device_batch(clips,device)
            residual=torch.where(observed(q),q['motion']-system.base(q['content'],q['valid'])['b0'],0)
            identities[sid]=system.encode_identity(residual[None],q['valid'][None])
    bundles={}
    for source,name in [('train','internal'),('validation','external_dev')]:
        rows=read_rows(args.manifest_dir/(source+'.jsonl'))
        if not rows:raise ValueError(f'Empty {source}')
        bundles[name]=build_split(rows,args.native_root,identities,system,model,geometry,device,args.layers)
    if bundles['internal']['groups'] != bundles['external_dev']['groups']:
        raise ValueError('Development observed channel layout differs from training')
    after=state_hash(system.state_dict())
    if before!=after:raise RuntimeError('Frozen system changed')
    provenance={'schema':'expanded_predictable_motion_v1','checkpoint_sha256':sha(args.checkpoint),
        'frozen_unchanged':True,'frozen_hash':before,'model':info,'geometry':geometry,'layers':args.layers,
        'manifest_hashes':{s:sha(args.manifest_dir/(s+'.jsonl')) for s in ('train','validation','enrollment')},
        'new_test_loaded':False,'script_sha256':sha(__file__),
        'extractor_sha256':sha(Path(__file__).with_name('extract_predictable_audio.py')),
        'feature_precision':'aligned middle/final float16 sidecar then float32 bins; prosody float32',
        'identity_caveat':'Static neutral bias cancels after clip centering; this probe cannot establish identity benefit',
        'scope':'Expanded training and reused/new validation only; locked new_test not read. Frozen pretrained modules may have seen these sentences.',
        'global_pretrained_on_internal_heldout':True,'external_dev_previously_inspected':True}
    # Training is all locked train; nested sentence CV is the only selector.
    out={'bundles':bundles,'train_ids':torch.arange(len(bundles['internal']['clip_id'])),
        'heldout_ids':torch.empty(0,dtype=torch.long),'provenance':provenance}
    torch.save(out,args.output/'diagnostic_bundle.pt')
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    print('COMPLETE',len(bundles['internal']['clip_id']),len(bundles['external_dev']['clip_id']),flush=True)


if __name__=='__main__':main()
