"""Describe independent neutral-reference variability; no fitting or queries."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_expression_response import configure,load_runtime,cache_base,state_digest,sha,write
from scripts.refine_expression_prior import restore_model
from scripts.audit_reference_style_swap import summary,correlation,REGIONS
from scripts.evaluate_expression_response import finite

def distance(x,y):return float(np.sqrt(np.mean((np.asarray(x)-np.asarray(y))**2)))

def retrieval(rows,key):
    """A to independently recorded B, not generated target-identity accuracy."""
    if len(rows)<2:return None
    correct=0;same=[];different=[]
    for i,a in enumerate(rows):
        values=[distance(a[key][0],b[key][1]) for b in rows]
        correct+=int(np.argmin(values)==i);same.append(values[i])
        different.extend(v for j,v in enumerate(values) if j!=i)
    intra=float(np.mean(same));inter=float(np.mean(different))
    return dict(speakers=len(rows),A_to_B_retrieval=correct/len(rows),same_rms=intra,cross_rms=inter,
                same_cross_ratio=intra/inter if inter>1e-12 else None)

@torch.no_grad()
def run(a):
    configure(47);device=torch.device(a.device);out=Path(a.output)
    if out.exists():raise FileExistsError('Refuse existing diagnostic')
    out.mkdir(parents=True)
    b=json.loads(Path(a.binding).read_text(encoding='utf-8-sig'))
    for name,digest in b['source_files'].items():assert sha(Path(__file__).resolve().parents[1]/name)==digest,name
    model,_=restore_model(b,device);model.eval().requires_grad_(False)
    data,base=load_runtime(b,device)
    # Only enrollment B0 is evaluated. Empty query caches stay unused.
    cache_base(data,base,device,{'train':[],'validation':[]})
    before=(state_digest(model),state_digest(base.stage1))
    held=set(sorted(data['fit_sids'])[-2:]);fit=set(data['fit_sids'])-held;dev=set(data['dev_sids'])
    queries=set(data['splits']['train']['clip_id'])|set(data['splits']['validation']['clip_id'])
    old=json.loads(Path(a.style_report).read_text(encoding='utf-8-sig'));oldcodes={r['speaker_id']:r for r in old['style_code']}
    scales=model.scales.cpu().numpy();rows=[]
    for sid,ref in sorted(data['refs'].items()):
        assert len(ref['clip_id'])==2 and not queries.intersection(ref['clip_id'])
        assert all('_neutral_' in x for x in ref['clip_id'])
        role='train_fit' if sid in fit else 'internal_held_identity' if sid in held else 'external_development'
        assert sid in fit|held|dev
        codes=[];stats={'gt':[],'residual':[],'b0':[]};views=[]
        for j in range(2):
            refs={k:ref[k][j:j+1][None].to(device) for k in ('motion','b0','valid','channel_mask')}
            codes.append(model.encode_style(refs)['code'][0].cpu().numpy())
            x=ref['motion'][j].numpy();c=ref['b0'][j].numpy();v=ref['valid'][j].numpy();ch=ref['channel_mask'][j].numpy().copy();ch[51]=False;t=ref['times'][j].numpy()
            for key,value in [('gt',x),('b0',c),('residual',x-c)]:stats[key].append(summary(value,v,ch,t)/scales[None])
            closed=v&(c[:,17]<.05);closed_mean=summary(x,closed,ch,t)[0] if closed.any() else None
            views.append(dict(clip_id=ref['clip_id'][j],sentence=ref['sentence_id'][j],native_valid_frames=int(v.sum()),
                b0_closed_frames=int(closed.sum()),gt_mean=stats['gt'][-1][0]*scales,
                b0_mean=stats['b0'][-1][0]*scales,residual_mean=stats['residual'][-1][0]*scales,
                gt_at_b0_closed_mean=closed_mean))
        cosine=correlation(*codes);rms=distance(*codes)
        if sid in oldcodes:
            assert abs(cosine-oldcodes[sid]['A_B_cosine'])<1e-6 and abs(rms-oldcodes[sid]['A_B_rms'])<1e-6
        row=dict(speaker_id=sid,speaker=ref['speaker'][0],role=role,views=views,code=codes,A_B_code_cosine=cosine,A_B_code_rms=rms)
        for name,ix in REGIONS.items():
            for key in ('gt','residual','b0'):
                row[key+'/'+name+'/mean']=[s[0,ix] for s in stats[key]]
                row[key+'/'+name+'/dynamic']=[s[1:,ix].flatten() for s in stats[key]]
        rows.append(row)
    assert before==(state_digest(model),state_digest(base.stage1))
    keys=['code']+[k+'/'+name+'/'+stat for name in REGIONS for k in ('gt','residual','b0') for stat in ('mean','dynamic')]
    result={role:{key:retrieval([r for r in rows if r['role']==role],key) for key in keys}
            for role in ('train_fit','internal_held_identity','external_development')}
    report=dict(schema='phase49_neutral_reference_coordinates_v1',test_loaded=False,training_performed=False,
        query_motion_used=False,reference_query_clip_disjoint=True,frozen_state_exact=True,style_code_replayed=True,
        parent_checkpoint=b['parent_checkpoint'],neutral_checkpoint=b['neutral_checkpoint'],source_sha256=sha(__file__),
        data_manifest_sha256=b['data_manifest_sha256'],style_audit_report_sha256=sha(a.style_report),
        scope='Independent neutral reference A/B native GT/B0/residual statistics, not causal content leakage or target identity accuracy.',
        retrieval_by_role=result,per_speaker=rows,limits=['Two neutral performances per person; posture, content distribution and B0 error remain confounded.',
        'Normalization uses frozen TRAIN scales; no parameters fitted or development selection.',
        'B0-closed pose absent is reported absent, never extrapolated or filled.'])
    write(out/'report.json',finite(report));write(out/'state.json',dict(status='complete',speakers=len(rows),report_sha256=sha(out/'report.json'),test_loaded=False))
    print(json.dumps(dict(complete=True,speakers=len(rows),frozen_state_exact=True)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--style-report',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    run(p.parse_args())
