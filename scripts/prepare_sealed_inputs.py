"""Extract frozen-test audio inputs only; query motion arrays are not read.

Neutral enrollment motion is an explicitly permitted deployment input. The
query targets stay inside their native NPZ files until predictions are sealed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_sealed_test_manifest import audit
from scripts.prepare_paper_full_data import validate_manifest
from scripts.extract_emotion2vec_pilot import load_extractor, sha
from scripts.extract_predictable_audio import extract
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, canonical_hash


def read_input(row, root, *, enrollment=False):
    if row['source_split']!='test': raise ValueError('Expected sealed test row')
    path=(root/row['artifact']).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path)!=row['artifact_sha256']:
        raise ValueError('Native test artifact path/hash differs')
    keys=['content','times','mask','channel_mask','provenance']
    if enrollment: keys.append('motion')
    # NPZ is lazy: indexing this allowlist never decompresses query motion.
    with np.load(path,allow_pickle=False) as z:
        values={key:np.asarray(z[key]).copy() for key in keys}
    provenance=json.loads(str(values.pop('provenance').item()))
    n=row['frames'];valid=values.pop('mask').astype(bool)
    if len(valid)!=n or int(valid.sum())!=row['valid_frames'] or valid.sum()<2:
        raise ValueError('Native test frame coverage differs')
    if values['content'].shape!=(n,768) or values['channel_mask'].shape!=(52,):
        raise ValueError('Native test input shape differs')
    if provenance.get('clock_evidence')!='embedded_video' or float(provenance['fps'])!=25:
        raise ValueError('Native test clock evidence differs')
    if not np.allclose(np.diff(values['times']),.04,rtol=0,atol=1e-5):
        raise ValueError('Native test frame rate differs')
    values['content']=np.where(valid[:,None],values['content'],0.).astype(np.float32)
    if not np.isfinite(values['content'][valid]).all():raise ValueError('Nonfinite test audio input')
    result={k:torch.from_numpy(v) for k,v in values.items()}
    result.update(valid=torch.from_numpy(valid), times=torch.from_numpy(values['times'].astype(np.float64)),
                  row=row, metadata={'provenance':provenance}, clip_id=row['clip_id'],sentence_id=row['sentence'])
    return result


def main():
    p=argparse.ArgumentParser()
    for key in ('manifest','train-cache','native-root','model-dir','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(2)
    report=audit(args.manifest)
    m=json.loads(args.manifest.read_text());validate_manifest(m)
    if set(r['emotion'] for r in m['roles']['test']['query'])!=set(range(8)):
        raise ValueError('Require all eight test emotions')
    train_meta=torch.load(args.train_cache/'metadata.pt',map_location='cpu',weights_only=False)
    source=json.loads((Path(train_meta['source'])/'manifest.json').read_text())
    for role in ('train','val'):
        if source['roles'][role]!=m['roles'][role]:raise ValueError('Train/development split changed')
    rows=m['roles']['test']['query'];refs=m['roles']['test']['enrollment']
    recipe=dict(manifest_sha256=m['manifest_sha256'],manifest_file_sha256=sha(args.manifest),
        test_role_sha256=canonical_hash(m['roles']['test']),train_manifest_sha256=source['manifest_sha256'],
        train_cache_metadata_sha256=sha(args.train_cache/'metadata.pt'),extractor_dir=str(args.model_dir),
        script_sha256=sha(__file__),query_motion_loaded=False,test_loaded=False,
        neutral_enrollment_motion_loaded=True,precision='FP16 cached audio, matching packed train inputs')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'recipe.json').exists() and json.loads((args.output/'recipe.json').read_text())!=recipe:
        raise ValueError('Sealed input recipe changed')
    save_json(args.output/'recipe.json',recipe);save_json(args.output/'manifest_audit.json',report)
    if (args.output/'complete.json').exists():return
    model,geometry,info=load_extractor(args.model_dir,args.device)
    lengths=np.asarray([r['frames'] for r in rows],dtype=np.int64)
    offsets=np.concatenate(([0],np.cumsum(lengths)));np.save(args.output/'offsets.npy',offsets)
    arrays={}
    for key,dtype,shape in [('audio_features',np.float16,(int(offsets[-1]),1540)),
                            ('times',np.float64,(int(offsets[-1]),)),('valid',np.bool_,(int(offsets[-1]),))]:
        path=args.output/(key+'.npy')
        arrays[key]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=dtype,shape=shape)
    done_path=args.output/'progress.json';done=json.loads(done_path.read_text())['completed'] if done_path.exists() else 0
    for i,row in enumerate(rows):
        if i<done:continue
        clip=read_input(row,args.native_root)
        feature=extract(clip,model,geometry,[2,4,6],args.device)
        x=torch.cat((clip['content'],feature['middle'].float(),feature['prosody']),-1)
        x=torch.where(clip['valid'][:,None],x,0.)
        if not torch.isfinite(x).all():raise ValueError('Nonfinite sealed input')
        lo,hi=int(offsets[i]),int(offsets[i+1])
        arrays['audio_features'][lo:hi]=x.half().numpy()
        arrays['times'][lo:hi]=clip['times'].numpy();arrays['valid'][lo:hi]=clip['valid'].numpy()
        if (i+1)%25==0 or i+1==len(rows):
            for value in arrays.values():value.flush()
            status=dict(completed=i+1,total=len(rows),query_motion_loaded=False)
            save_json(done_path,status);print(json.dumps(status),flush=True)
    enrollment={}
    for row in refs:
        clip=read_input(row,args.native_root,enrollment=True)
        observed=clip['valid'][:,None]&clip['channel_mask'][None]
        if not torch.isfinite(clip['motion'][observed]).all():raise ValueError('Nonfinite enrollment')
        clip['motion']=torch.where(observed,clip['motion'],0.)
        enrollment.setdefault(row['speaker'],[]).append({k:clip[k] for k in ('motion','content','valid','channel_mask','clip_id')})
    save_checkpoint(args.output/'enrollment.pt',enrollment)
    save_json(args.output/'rows.json',rows)
    save_json(args.output/'complete.json',{**recipe,'extractor':info,'clips':len(rows),
        'files':{p.name:sha(p) for p in args.output.glob('*.npy')},'enrollment_sha256':sha(args.output/'enrollment.pt')})


if __name__=='__main__':main()
