"""Bind frozen native acoustic baseline to independently extracted visual semantics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.diagnose_brow_state_nuisance import sha, write_json
from scripts.package_sparse_brow_teacher import checked

UPPER = [41,42,43,44,45,5,6,12,13]


def align_visual(sample_times, sample_valid, values, times, motion_valid, max_gap=.201):
    sample_times=np.asarray(sample_times,float);times=np.asarray(times,float)
    sample_valid=np.asarray(sample_valid,bool);values=np.asarray(values,float)
    if (values.shape[0]!=len(sample_times) or sample_valid.shape!=sample_times.shape
        or np.any(np.diff(sample_times)<=0) or np.any(np.diff(times)<=0)
        or not np.isfinite(values[sample_valid]).all()):
        raise ValueError('Invalid visual source clock or values')
    out=np.zeros((len(times),values.shape[1]),np.float32);valid=np.zeros(len(times),bool)
    for k in range(len(sample_times)):
        if not sample_valid[k]:continue
        exact=np.flatnonzero(np.abs(times-sample_times[k])<1e-7)
        out[exact]=values[k];valid[exact]=True
        if k+1==len(sample_times) or not sample_valid[k+1]:continue
        if sample_times[k+1]-sample_times[k]>max_gap:continue
        use=np.flatnonzero((times>=sample_times[k]-1e-7)&(times<=sample_times[k+1]+1e-7))
        if not len(use) or not motion_valid[use].all():continue
        fraction=(times[use]-sample_times[k])/(sample_times[k+1]-sample_times[k])
        out[use]=(1-fraction[:,None])*values[k]+fraction[:,None]*values[k+1]
        valid[use]=True
    valid &= motion_valid
    out[~valid]=0
    return out,valid


def build(selection, baseline, teacher, output):
    if output.exists():raise FileExistsError(output)
    read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
    selected=read(selection);bm=read(baseline/'manifest.json');tm=read(teacher/'manifest.json')
    bp=read(baseline/'provenance.json');tp=read(teacher/'protocol.json')
    if bp['selection_sha256']!=sha(selection) or tp['selection_sha256']!=sha(selection):
        raise ValueError('Selected source differs')
    for root,manifest,names in [(baseline,bm,['provenance.json']), (teacher,tm,['protocol.json'])]:
        for n in names:checked(root/n,manifest[n])
    if tp['stride']!=5 or tp['limited_smoke']:raise ValueError('Full5Hz teacher required')
    clips=[];reports=[]
    for row in selected['clips']:
        cid=row['clip_id'];bname='arrays/'+cid+'.npz';vname='video_npz/'+cid+'.npz'
        for root,manifest,name in [(baseline,bm,bname),(baseline,bm,vname),(teacher,tm,bname)]:checked(root/name,manifest[name])
        with np.load(baseline/bname,allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
        with np.load(baseline/vname,allow_pickle=False) as z:
            modes=z['mode_names'].tolist()
            target=z['motions'][modes.index('native coefficient reference (not inference input)')].copy()
        with np.load(teacher/bname,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
        if not np.allclose(b['times'],np.arange(len(b['valid']))/25,atol=1e-7,rtol=0):raise ValueError('Native origin differs')
        va,mask=align_visual(v['times'],v['valid'],v['va'],b['times'],b['valid'])
        posterior,pmask=align_visual(v['times'],v['valid'],v['q'],b['times'],b['valid'])
        if not np.array_equal(mask,pmask):raise ValueError('VA/posterior supports differ')
        if mask.sum()<25:raise ValueError('Insufficient visual support: '+cid)
        if not np.allclose(posterior[mask].sum(1),1,atol=1e-5):raise ValueError('Posterior not normalized')
        record={k:row[k] for k in ('clip_id','sentence','split','speaker','speaker_name','emotion')}
        record.update({k:torch.from_numpy(b[k]).float() for k in ('features','affect_global','affect_intensity','identity_code','identity_baseline','baseline52')})
        record.update(times=torch.from_numpy(b['times']),valid=torch.from_numpy(b['valid']),
                      motion=torch.from_numpy(target).float(),va=torch.from_numpy(va),posterior=torch.from_numpy(posterior),
                      semantic_valid=torch.from_numpy(mask))
        clips.append(record);reports.append({'clip_id':cid,'native_valid':int(b['valid'].sum()),'semantic_valid':int(mask.sum())})
    output.parent.mkdir(parents=True,exist_ok=True)
    provenance={'schema':'visual_semantic_dataset_v1','selection_sha256':sha(selection),
                'baseline_manifest_sha256':sha(baseline/'manifest.json'),'teacher_manifest_sha256':sha(teacher/'manifest.json'),
                'code_sha256':sha(__file__),'interpolation':'only adjacent valid visual samples<=.201s and no native gap; no extrapolation',
                'sealed_test_loaded':False,'dev405_indexed':False,'new_training_conditions_use_query_motion':False}
    torch.save({'schema':'visual_semantic_dataset_v1','clips':clips,'provenance':provenance},output)
    write_json(output.with_suffix('.json'),{**provenance,'dataset_sha256':sha(output),'clips':reports})
    print(json.dumps({'clips':len(clips),'dataset':str(output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('selection','baseline','teacher','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();build(a.selection,a.baseline,a.teacher,a.output)
