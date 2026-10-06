"""Finalize a flat cache after an interrupted metadata/statistics write."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np, torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.packed_trainval_cache import SCHEMA, _pad_shard
from scripts.prepare_paper_full_data import validate_manifest, sha as file_sha
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    root=a.data.resolve(); out=a.output.resolve(); index=json.loads((root/'index.json').read_text()); manifest=json.loads((root/'manifest.json').read_text()); validate_manifest(manifest)
    people=sorted({r['speaker'] for role in ('train','val') for r in manifest['roles'][role]['query']}); sids={s:i for i,s in enumerate(people)}
    records={(r['clip_id'],r['role'],r['kind']):r for r in index['records']}; expected={r['clip_id']:r for role in ('train','val') for kind in ('query','enrollment') for r in manifest['roles'][role][kind]}; max_len=max(r['frames'] for r in index['records'])
    enrollment={sids[s]:[] for s in people}
    for role in ('train','val'):
      for row in manifest['roles'][role]['enrollment']:
        rec=records[(row['clip_id'],role,'enrollment')]; saved=torch.load(root/rec['path'],map_location='cpu',weights_only=False); enrollment[sids[row['speaker']]].append(_pad_shard(saved,max_len,sids[row['speaker']],require_audio=False))
    refs={}; anchors={}
    for sid,clips in enrollment.items():
      ref={key:torch.stack([c[key] for c in clips]) for key in ('motion','content','audio_features','valid','times','channel_mask','motion_valid')}
      for key in ('clip_id','sentence_id','speaker'): ref[key]=[c[key] for c in clips]
      ref['speaker_id']=torch.full((len(clips),),sid,dtype=torch.long); refs[sid]=ref; obs=ref['valid'][...,None]&ref['channel_mask'][:,None]; cnt=obs.sum(1); means=torch.where(obs,ref['motion'],0.).sum(1)/cnt.clamp_min(1); av=cnt.gt(0).all(0); med=means.quantile(.5,dim=0); anchors[sid]=(torch.where(av,med,torch.zeros_like(med)),av)
    feature_sum=torch.zeros(1540,dtype=torch.float64); feature_sq=torch.zeros(1540,dtype=torch.float64); feature_count=0; scale_sum=torch.zeros(52,dtype=torch.float64); scale_count=torch.zeros(52,dtype=torch.float64)
    split_meta={}
    for source_role,role in (('train','train'),('val','validation')):
      rd=out/role; lengths=np.load(rd/'lengths.npy',allow_pickle=False); offsets=np.load(rd/'offsets.npy',allow_pickle=False); ids=[r['clip_id'] for r in manifest['roles'][source_role]['query']]; split_meta[role]={'lengths':lengths.tolist(),'max_len':int(lengths.max()),'clip_id':ids,'sentence_id':[r['sentence'] for r in manifest['roles'][source_role]['query']],'speaker':[r['speaker'] for r in manifest['roles'][source_role]['query']],'metadata':[{'packed':True} for _ in ids]}
      audio=np.load(rd/'audio_features.npy',mmap_mode='r',allow_pickle=False); motion=np.load(rd/'motion.npy',mmap_mode='r',allow_pickle=False)
      valid=np.zeros(0,dtype=bool)
      for i,row in enumerate(manifest['roles'][source_role]['query']):
        lo,hi=int(offsets[i]),int(offsets[i+1]); saved_valid=np.load(root/records[(row['clip_id'],source_role,'query')]['path'],allow_pickle=False) if False else None
        # Native valid masks are already represented by the fixed padded file.
        fixed=np.load(rd/'valid.npy',mmap_mode='r',allow_pickle=False)[i,:hi-lo].astype(bool); x=np.asarray(audio[lo:hi],dtype=np.float32)[fixed];
        if source_role=='train':
          feature_sum+=torch.from_numpy(x).double().sum(0); feature_sq+=torch.from_numpy(x).double().square().sum(0); feature_count+=int(fixed.sum()); sid=sids[row['speaker']]; anchor,av=anchors[sid]; ch=np.load(rd/'channel_mask.npy',mmap_mode='r',allow_pickle=False)[i].astype(bool); ob=torch.from_numpy(fixed[:,None]&ch[None,:]&av.numpy()[None,:]); m=torch.from_numpy(np.asarray(motion[lo:hi])); d=torch.where(ob,m-anchor[None,:],torch.zeros_like(m)); scale_sum+=d.double().square().sum(0); scale_count+=ob.sum(0).double()
    mean=feature_sum/feature_count; var=(feature_sq/feature_count-mean.square()).clamp_min(0.); stats={'mean':mean.float(),'std':var.sqrt().clamp_min(1e-3).float(),'count':feature_count,'ddof':0,'std_floor':1e-3,'fit_clip_ids':split_meta['train']['clip_id'],'source':'train_valid_native_frames_only','application':'(raw_features - fit_mean)/fit_std then mask'}; scales=(scale_sum/scale_count.clamp_min(1)).sqrt().clamp_min(.02).float(); config=yaml.safe_load((root/'config.yaml').read_text()); NeutralAffectSystem(config).eval(); prov={'schema':'paper_full_native_data_v1','index_sha256':file_sha(root/'index.json'),'manifest_sha256':manifest['manifest_sha256'],'config_sha256':file_sha(root/'config.yaml'),'fit_clips':len(split_meta['train']['clip_id']),'development_clips':len(split_meta['validation']['clip_id']),'fit_valid_frames':feature_count,'fit_sids':sorted({sids[r['speaker']] for r in manifest['roles']['train']['query']}),'dev_sids':sorted({sids[r['speaker']] for r in manifest['roles']['val']['query']}),'test_loaded':False,'frame_policy':'complete native sequences, dynamic batch trimming only'}
    torch.save(refs,out/'refs.pt'); torch.save({'schema':SCHEMA,'source':str(root),'source_provenance':prov,'config':config,'fit_sids':prov['fit_sids'],'dev_sids':prov['dev_sids'],'feature_stats':stats,'target_scales':scales,'split_meta':split_meta,'test_loaded':False},out/'metadata.pt'); (out/'packed_cache.json').write_text(json.dumps({'schema':SCHEMA,'metadata':'metadata.pt','refs':'refs.pt','source':str(root),'test_loaded':False},indent=2))
if __name__=='__main__': main()
