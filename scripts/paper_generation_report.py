"""Automatic coefficient and temporal scoring on declared development data."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.arkit_benchmark_report import score_fullface,build_report,write_report
from scripts.joint_motion_metrics import score_clip,summarize
from scripts.train_formal_predictable_projection import save_json
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES


def report_generation(curves,data,output,stage,args):
    q=data['splits']['validation'];lookup={cid:i for i,cid in enumerate(q['clip_id'])}
    modes=sorted({key.split('/',1)[1] for key in curves['predictions']})
    all_results={};scales=data['target_scales'][list(UPPER_INDICES)].numpy()
    for mode in modes:
        keys=[f'{seed}/{mode}' for seed in curves['noise_seeds'] if f'{seed}/{mode}' in curves['predictions']]
        samples=torch.stack([curves['predictions'][key] for key in keys]).numpy()
        rows=[];temporal=[]
        for i,cid in enumerate(curves['clip_id']):
            j=lookup[cid];y=curves['target'][i].numpy();valid=curves['valid'][i].numpy()
            mask=np.broadcast_to(curves['channel_mask'][i].numpy(),y.shape)
            meta={k:q[k][j] for k in ('speaker','sentence_id')}
            row=score_fullface(samples[:,i],{'clip_id':cid,'speaker':meta['speaker'],'sentence':meta['sentence_id'],
                'emotion':int(q['emotion_id'][j]),'valid':valid,'channel_mask':mask,'target52':y,'times':curves['times'][i].numpy()})
            rows.append(row)
            if len(keys)>=2:
                support=valid & mask[:,list(UPPER_INDICES)].all(1)
                record=score_clip(samples[:,i][:,:,list(UPPER_INDICES)],y[:,list(UPPER_INDICES)],support,scales)
                record.update(clip_id=cid,sentence=meta['sentence_id'],speaker=meta['speaker']);temporal.append(record)
        report=build_report(rows,scope='paper validation, full native sequences; not final test',sources={
            'manifest_sha256':data['provenance']['manifest_sha256'],'data_index_sha256':data['provenance']['index_sha256']})
        report['stage']=stage;report['condition_mode']=args.condition_mode;report['prediction_keys']=keys
        # Compact rows retain values/coverage and semantics; shared formula
        # definitions remain in the report, without repeating KBs per metric.
        report['per_clip']=[{k:r.get(k) for k in ('clip_id','sentence','speaker','emotion')} | {
            'metrics':{name:{k:v.get(k) for k in ('status','value','per_sample','coverage')} for name,v in r['metrics'].items()}}
            for r in rows]
        report['temporal']=summarize(temporal) if temporal else {'status':'pending','reason':'at least two draws required'}
        report['test_loaded']=False
        write_report(Path(output)/('arkit_'+mode+'.json'),report)
        if args.artifact_dir:
            (args.artifact_dir/stage).mkdir(parents=True,exist_ok=True)
            save_json(args.artifact_dir/stage/('temporal_'+mode+'.json'),temporal)
        all_results[mode]={'coefficient':report['summary'],'temporal':report['temporal'],'prediction_keys':keys}
    save_json(Path(output)/'benchmark_summary.json',all_results)
    return all_results
