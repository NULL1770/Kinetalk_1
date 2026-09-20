"""Materialize complete approved train/validation clips, never test targets.

Separate content/motion reference roles, train-only statistics, resumable
per-clip acoustic shards and native clocks. No historical model is loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_emotion2vec_pilot import load_extractor, sha
from scripts.extract_predictable_audio import extract
from scripts.train_formal_predictable_projection import save_checkpoint, save_json, canonical_hash
from scripts.prepare_label_guided_audio_cache import fit_feature_statistics
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import stack_clips
from kinetalk_b0.label_guided_intensity import fit_intensity_scales

SCHEMA = 'paper_full_native_data_v1'


def validate_manifest(manifest):
    unsigned = dict(manifest); wanted = unsigned.pop('manifest_sha256')
    if canonical_hash(unsigned) != wanted:
        raise ValueError('Manifest canonical hash differs')
    if manifest.get('status') != 'approved_train_val_only' or manifest.get('sealed_test_targets_loaded') is not False:
        raise ValueError('Reviewed train/validation-only protocol required')
    roles = manifest['roles']
    if set(roles) != {'train', 'val', 'test'}:
        raise ValueError('Explicit three-way metadata roles required')
    seen = set(); speakers = set()
    reserved = set(manifest['reserved_sentences'])
    for role, value in roles.items():
        query, refs = value['query'], value['enrollment']
        ids = [r['clip_id'] for r in query + refs]
        people = {r['speaker'] for r in query + refs}
        if len(ids) != len(set(ids)) or seen & set(ids) or speakers & people:
            raise ValueError('Cross-role clip/identity overlap')
        if any(r['source_split'] != role or r['dataset'] != 'mead' for r in query + refs):
            raise ValueError('Source-role/dataset mismatch')
        if role == 'train' and reserved & {r['sentence'] for r in query + refs}:
            raise ValueError('Historical sealed sentence enters fitting')
        if {(r['speaker'],r['sentence']) for r in query} & {(r['speaker'],r['sentence']) for r in refs}:
            raise ValueError('Query/reference speaker-sentence overlap')
        for person in people:
            part = [r for r in refs if r['speaker'] == person]
            if len(part) < 2 or len({r['sentence'] for r in part}) != len(part) or any(r['emotion'] != 0 for r in part):
                raise ValueError('Independent neutral references required')
        seen.update(ids); speakers.update(people)


def read_native(row, native_root):
    # Fail before opening any file from a test role, even for enrollment.
    if row['source_split'] not in ('train','val'):
        raise ValueError('Test artifact reads are prohibited in preparation')
    root = Path(native_root).resolve(); path = (root / row['artifact']).resolve()
    if not path.is_relative_to(root) or sha(path) != row['artifact_sha256']:
        raise ValueError('Native path/hash differs: '+row['clip_id'])
    with np.load(path, allow_pickle=False) as z:
        values = {k: np.asarray(z[k]).copy() for k in ('motion','content','audio','times','mask','channel_mask')}
        provenance = json.loads(str(z['provenance'].item()))
    n = len(values['times'])
    if n != row['frames'] or provenance.get('clock_evidence') != 'embedded_video' or float(provenance['fps']) != 25:
        raise ValueError('Native frame/clock provenance differs')
    if not np.allclose(np.diff(values['times']), .04, rtol=0, atol=1e-5):
        raise ValueError('Native frame clock is not 25 fps')
    if values['mask'].shape != (n,) or values['channel_mask'].shape != (52,):
        raise ValueError('Mask dimensions differ')
    if any(not np.isin(values[k],[0,1]).all() for k in ('mask','channel_mask')):
        raise ValueError('Non-Boolean native masks')
    valid = values['mask'].astype(bool); channel = values['channel_mask'].astype(bool)
    if int(valid.sum()) != row['valid_frames'] or valid.sum() < 32:
        raise ValueError('Insufficient or inconsistent valid frames')
    for key,width in [('motion',52),('content',768),('audio',83)]:
        x = values[key]
        if x.shape != (n,width): raise ValueError('Native dimensions differ')
        mask = valid[:,None] & channel[None] if key == 'motion' else np.broadcast_to(valid[:,None],x.shape)
        if not np.isfinite(x[mask]).all(): raise ValueError('Observed nonfinite native values')
        values[key] = np.where(mask, x, 0.).astype(np.float32)
    return {**values,'valid':valid,'channel_mask':channel,'provenance':provenance}


def prepare(args):
    torch.set_num_threads(4)
    m = json.loads(args.manifest.read_text(encoding='utf8')); validate_manifest(m)
    args.output.mkdir(parents=True,exist_ok=True)
    recipe = {'schema':SCHEMA,'manifest_sha256':m['manifest_sha256'],
              'config_sha256':sha(args.config),'native_root':str(args.native_root.resolve()),
              'preparer_sha256':sha(__file__),'test_loaded':False,'frame_policy':'complete native sequence; padding never observed'}
    marker = args.output/'recipe.json'
    if marker.exists() and json.loads(marker.read_text()) != recipe:
        raise ValueError('Resume preparation recipe differs')
    save_json(marker,recipe)
    model,geometry,info = load_extractor(args.model_dir,args.device)
    recipe['extractor'] = info; recipe['geometry'] = geometry
    records=[];started=time.monotonic()
    for role in ('train','val'):
        for kind in ('enrollment','query'):
            for row in m['roles'][role][kind]:
                path=args.output/'clips'/(row['clip_id']+'.pt'); path.parent.mkdir(exist_ok=True)
                if path.exists():
                    saved=torch.load(path,map_location='cpu',weights_only=False)
                    if saved['row'] != row or saved['extractor'] != info: raise ValueError('Cached clip provenance differs')
                else:
                    native=read_native(row,args.native_root)
                    valid=torch.from_numpy(native['valid']);times=torch.from_numpy(native['times'].astype(np.float64))
                    clip={'clip_id':row['clip_id'],'sentence_id':row['sentence'],'valid':valid,'times':times,
                          'metadata':{'provenance':native['provenance']}}
                    record=extract(clip,model,geometry,[2,4,6],args.device) if kind=='query' else None
                    saved={'schema':SCHEMA,'row':row,'extractor':info,'valid':valid,'times':times,
                           'channel_mask':torch.from_numpy(native['channel_mask']),
                           'motion':torch.from_numpy(native['motion']),'content':torch.from_numpy(native['content']),
                           'audio':torch.from_numpy(native['audio']),'native_provenance':native['provenance']}
                    if record is not None:
                        saved['middle']=record['middle'];saved['prosody']=record['prosody'];saved['extraction_record']=record['record']
                    save_checkpoint(path,saved)
                records.append({'clip_id':row['clip_id'],'role':role,'kind':kind,'path':'clips/'+path.name,'sha256':sha(path),
                                'frames':len(saved['valid']),'valid_frames':int(saved['valid'].sum())})
                if len(records)%50==0:
                    state={'stage':'preparation','completed_clips':len(records),'elapsed_seconds':time.monotonic()-started}
                    save_json(args.output/'status.json',state);print(json.dumps(state),flush=True)
    save_json(args.output/'manifest.json',m)
    (args.output/'config.yaml').write_bytes(args.config.read_bytes())
    save_json(args.output/'index.json',{'schema':SCHEMA,'recipe':recipe,'records':records,'test_loaded':False})
    save_json(args.output/'status.json',{'stage':'prepared','clips':len(records),'elapsed_seconds':time.monotonic()-started,'test_loaded':False})
    print('PAPER_DATA_PREPARED',flush=True)


def pad_clip(saved, length, sid):
    row=saved['row']; n=len(saved['valid']); out={}
    for key in ('motion','content','audio','valid'):
        x=saved[key]; value=torch.zeros((length,*x.shape[1:]),dtype=x.dtype);value[:n]=x;out[key]=value
    out['times']=saved['times'][0]+torch.arange(length,dtype=torch.float64)/25;out['times'][:n]=saved['times']
    out.update(channel_mask=saved['channel_mask'],motion_valid=out['valid'].clone(),
        clip_id=row['clip_id'],sentence_id=row['sentence'],speaker=row['speaker'],
        speaker_id=torch.tensor(sid),emotion_id=torch.tensor(row['emotion']),
        intensity_id=torch.tensor(row['intensity']),intensity_valid=torch.tensor(row['intensity']>=0),
        dataset_id=torch.tensor(0),metadata={'source_split':row['source_split'],'artifact_sha256':row['artifact_sha256'],
                                         'observed_frames':n,'crop_start':0})
    if 'middle' in saved:
        feat=torch.cat((saved['content'],saved['middle'].float(),saved['prosody']),-1)
        x=torch.zeros(length,1540);x[:n]=feat;out['audio_features']=torch.where(out['valid'][:,None],x,0.)
    return out


def load_paper_data(directory, *, seed=47):
    root=Path(directory); index=json.loads((root/'index.json').read_text(encoding='utf8'))
    m=json.loads((root/'manifest.json').read_text(encoding='utf8'));validate_manifest(m)
    if m['manifest_sha256'] != index['recipe']['manifest_sha256'] or index.get('test_loaded') is not False:
        raise ValueError('Data index manifest mismatch')
    if sha(root/'config.yaml') != index['recipe']['config_sha256']:raise ValueError('Config hash differs')
    expected={(r['clip_id'],role,kind) for role in ('train','val') for kind in ('query','enrollment') for r in m['roles'][role][kind]}
    expected_rows={r['clip_id']:r for role in ('train','val') for kind in ('query','enrollment') for r in m['roles'][role][kind]}
    actual={(r['clip_id'],r['role'],r['kind']) for r in index['records']}
    if expected!=actual or len(actual)!=len(index['records']):raise ValueError('Data coverage differs')
    people=sorted({r['speaker'] for role in ('train','val') for r in m['roles'][role]['query']});sids={s:i for i,s in enumerate(people)}
    length=max(r['frames'] for r in index['records']);by_key={(r['role'],r['kind']):[] for r in index['records']}
    for rec in index['records']:
        path=(root/rec['path']).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=rec['sha256']:raise ValueError('Shard hash/path differs')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['row'] != expected_rows[rec['clip_id']]:raise ValueError('Shard metadata differs from approved manifest')
        if len(saved['valid']) != rec['frames'] or int(saved['valid'].sum()) != rec['valid_frames']:
            raise ValueError('Shard frame coverage differs')
        by_key[(rec['role'],rec['kind'])].append(pad_clip(saved,length,sids[saved['row']['speaker']]))
    refs={};anchors={};fit=[];dev=[]
    for role,arr in [('train',fit),('val',dev)]:
        part=by_key[(role,'enrollment')]
        for person in sorted({x['speaker'] for x in part}):
            sid=sids[person];arr.append(sid);q=stack_clips([x for x in part if x['speaker']==person]);refs[sid]=q
            obs=q['valid'][...,None]&q['channel_mask'][:,None]
            mean=torch.where(obs,q['motion'],0.).sum(1)/obs.sum(1).clamp_min(1)
            av=obs.sum(1).gt(0).all(0);anchor=mean.quantile(.5,dim=0)
            anchors[sid]=(torch.where(av,anchor,0.),av)
    splits={}
    for source,role in [('train','train'),('val','validation')]:
        q=stack_clips(by_key[(source,'query')]);q['anchors']=torch.stack([anchors[int(s)][0] for s in q['speaker_id']]);q['anchor_valid']=torch.stack([anchors[int(s)][1] for s in q['speaker_id']]);splits[role]=q
    train=splits['train'];stats=fit_feature_statistics(train['audio_features'],train['valid'],train['clip_id'])
    scales=fit_intensity_scales(train['motion'],train['valid'][...,None]&train['channel_mask'][:,None]&train['anchor_valid'][:,None],train['anchors'],floor=.02)
    cfg=yaml.safe_load((root/'config.yaml').read_text());torch.manual_seed(seed);system=NeutralAffectSystem(cfg).eval()
    provenance={'schema':SCHEMA,'index_sha256':sha(root/'index.json'),'manifest_sha256':m['manifest_sha256'],
                'config_sha256':sha(root/'config.yaml'),'fit_clips':len(train['clip_id']),'development_clips':len(splits['validation']['clip_id']),
                'fit_valid_frames':int(train['valid'].sum()),'initialization':'all KineTalk modules random; no historical checkpoint loaded',
                'frame_policy':'complete native sequences, dynamic batch trimming only','fit_sids':fit,'dev_sids':dev,
                'test_loaded':False,'statistics_source':'complete train query observed frames only'}
    return {'system':system,'config':cfg,'splits':splits,'refs':refs,'fit_sids':fit,'dev_sids':dev,
            'ref_groups':{},'feature_stats':stats,'target_scales':scales,'provenance':provenance}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','native-root','model-dir','config','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda');prepare(p.parse_args())
