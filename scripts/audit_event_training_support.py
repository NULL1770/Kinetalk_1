"""Read-only training-support audit for source prior versus event receiver.

Only fit/inner motion tensors are indexed. No AE, GPU, training or generation.
This quantifies support changes and subset composition, not their causal effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.event_schedule_teacher import extract_schedule, fit_teacher, GROUPS, valid_runs
from scripts.probe_motion_condition_predictability import split_train_pool, sha
from scripts.train_prior_audio_adapter import diverse_diagnostics

SCHEMA = 'event_training_support_audit_v2'
PINNED = ('scripts/event_schedule_teacher.py', 'scripts/prepare_continuous_motion_dataset.py',
          'scripts/probe_motion_condition_predictability.py', 'scripts/train_event_schedule_pipeline.py',
          'scripts/train_prior_audio_adapter.py', 'scripts/train_continuous_motion_latent.py')


def arr(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def chunk_support(mask, minimum, maximum=200):
    mask = arr(mask)
    if mask.ndim != 1 or mask.dtype != bool or not 1 <= minimum <= maximum:
        raise ValueError('Boolean one-dimensional support and legal lengths required')
    kept = np.zeros_like(mask); chunks = []; dropped = []
    for a,b in valid_runs(mask):
        for start in range(a,b,maximum):
            end = min(start+maximum,b)
            if end-start < minimum: dropped.append((start,end))
            else: kept[start:end]=True; chunks.append((start,end))
    return kept, chunks, dropped


def _divide(x,y):
    return float(x/y) if y > 0 else None


def quantiles(values):
    values=np.asarray(values, dtype=float)
    if not len(values): return {'count': 0, 'min': None, 'p25': None, 'median': None, 'p75': None, 'p95': None, 'max': None}
    return dict(count=len(values), **dict(zip(('min','p25','median','p75','p95','max'),
                                            np.quantile(values,[0,.25,.5,.75,.95,1]).tolist())))


def _center_energy(motion, spans, channels):
    total = 0.
    for a,b in spans:
        x=motion[a:b][:,channels].astype(np.float64)
        x=x-x[:1]; x=x-x.mean(0,keepdims=True)
        total += float(np.square(x).sum())
    return total


def analyze_clip(clip, label):
    native=arr(clip['valid']); observed=arr(clip['motion_mask']); motion=arr(clip['motion9'])
    known=arr(label['known']); schedule=arr(label['schedule'])
    if native.ndim != 1 or native.dtype != bool or observed.shape != (len(native),9) or observed.dtype != bool:
        raise ValueError('Boolean native[T] and motion_mask[T,9] required')
    if motion.shape != (len(native),9) or known.shape != (len(native),4) or known.dtype != bool or schedule.shape != (len(native),12):
        raise ValueError('Invalid motion or schedule shapes')
    joint=native & observed.all(1)
    if not np.isfinite(motion[joint]).all(): raise ValueError('Nonfinite observed target motion')
    known_joint=joint & known.all(1)
    source, source_chunks, source_drop=chunk_support(joint,5)
    receiver, receiver_chunks, receiver_drop=chunk_support(known_joint,10)
    # Different 200-frame origins may admit source-dropped <5 tails. Preserve
    # this asymmetry explicitly rather than assuming receiver is source subset.
    active=schedule[:,:4]>.5
    result={'clip_id':clip['clip_id'], 'sentence':str(clip['sentence']), 'speaker':str(clip['speaker']),
            'emotion':str(clip['emotion']), 'native_clock_frames':len(native), 'native_valid_frames':int(native.sum()),
            'native_clock_le_200':len(native)<=200, 'native_valid_le_200':int(native.sum())<=200,
            'longest_native_run':max((b-a for a,b in valid_runs(native)),default=0),
            'frames':{'joint':int(joint.sum()), 'known4_joint':int(known_joint.sum()),
                      'source_kept':int(source.sum()), 'receiver_kept':int(receiver.sum()),
                      'source_lost_at_known4':int((source & ~known_joint).sum()),
                      'source_known_but_dropped_receiver_tail':int((source & known_joint & ~receiver).sum()),
                      'source_dropped_tail':int(sum(b-a for a,b in source_drop)),
                      'receiver_dropped_tail':int(sum(b-a for a,b in receiver_drop)),
                      'receiver_outside_source':int((receiver & ~source).sum())},
            'lengths':{'native_runs':[b-a for a,b in valid_runs(native)],
                       'joint_runs':[b-a for a,b in valid_runs(joint)],
                       'known4_runs':[b-a for a,b in valid_runs(known_joint)],
                       'source_segments':[b-a for a,b in source_chunks],
                       'receiver_segments':[b-a for a,b in receiver_chunks]}, 'groups':{}}
    for gi,(g,channels) in enumerate(GROUPS.items()):
        centered=np.zeros((len(native),len(channels)),np.float64)
        for a,b in valid_runs(joint):
            x=motion[a:b][:,channels].astype(np.float64);x=x-x[:1]
            centered[a:b]=x-x.mean(0,keepdims=True)
        power=np.square(centered).sum(1)
        events=[e for e in label['events'] if e['group_index']==gi]
        group={'joint_centered_energy':float(power[joint].sum()),
               'source_common_center_energy':float(power[source].sum()),
               'receiver_common_center_energy':float(power[receiver].sum()),
               'receiver_source_overlap_energy':float(power[receiver & source].sum()),
               'source_segment_center_energy':_center_energy(motion,source_chunks,channels),
               'receiver_segment_center_energy':_center_energy(motion,receiver_chunks,channels),
               'joint_frame_channels':int(joint.sum()*len(channels)),
               'source_frame_channels':int(source.sum()*len(channels)),
               'receiver_frame_channels':int(receiver.sum()*len(channels)),
               'native_known_frames':int((native & known[:,gi]).sum()),
               'joint_active_frames':int((joint & active[:,gi]).sum()),
               'source_active_frames':int((source & active[:,gi]).sum()),
               'receiver_active_frames':int((receiver & active[:,gi]).sum()),
               'source_active_lost_other_group_unknown':int((source & active[:,gi] & ~known.all(1)).sum()),
               'events':len(events),
               'source_complete_events':sum(bool(source[e['start']:e['end']].all()) for e in events),
               'receiver_complete_events':sum(bool(receiver[e['start']:e['end']].all()) for e in events)}
        result['groups'][g]=group
    return result


def summarize(rows):
    rows=list(rows); frames={k:sum(r['frames'][k] for r in rows) for k in (rows[0]['frames'] if rows else [])}
    out={'clips':len(rows),'sentences':len({r['sentence'] for r in rows}),'speakers':len({r['speaker'] for r in rows}),
         'source_clips_retained':sum(r['frames']['source_kept']>0 for r in rows),
         'receiver_clips_retained':sum(r['frames']['receiver_kept']>0 for r in rows),
         'receiver_excluded_ids':[r['clip_id'] for r in rows if not r['frames']['receiver_kept']],
         'native_clock_le_200':sum(r['native_clock_le_200'] for r in rows),
         'native_clock_gt_200':sum(not r['native_clock_le_200'] for r in rows),
         'native_valid_le_200':sum(r['native_valid_le_200'] for r in rows),
         'native_valid_gt_200':sum(not r['native_valid_le_200'] for r in rows),
         'native_valid_frames':sum(r['native_valid_frames'] for r in rows),'frames':frames,
         'length_quantiles':{},'groups':{}}
    out['receiver_source_frame_ratio']=_divide(frames.get('receiver_kept',0),frames.get('source_kept',0))
    for name in ('native_runs','joint_runs','known4_runs','source_segments','receiver_segments'):
        out['length_quantiles'][name]=quantiles([n for r in rows for n in r['lengths'][name]])
    for g in GROUPS:
        values={k:sum(r['groups'][g][k] for r in rows) for k in (rows[0]['groups'][g] if rows else [])}
        values['receiver_source_energy_ratio']=_divide(values.get('receiver_common_center_energy',0), values.get('source_common_center_energy',0))
        values['source_energy_retained_on_overlap']=_divide(values.get('receiver_source_overlap_energy',0),values.get('source_common_center_energy',0))
        values['source_active_coverage']=_divide(values.get('source_active_frames',0),values.get('joint_active_frames',0))
        values['receiver_active_coverage_vs_source']=_divide(values.get('receiver_active_frames',0),values.get('source_active_frames',0))
        values['joint_centered_rms']=np.sqrt(_divide(values['joint_centered_energy'],values['joint_frame_channels'])) if values.get('joint_frame_channels',0) else None
        values['source_segment_centered_rms']=np.sqrt(_divide(values['source_segment_center_energy'],values['source_frame_channels'])) if values.get('source_frame_channels',0) else None
        values['receiver_segment_centered_rms']=np.sqrt(_divide(values['receiver_segment_center_energy'],values['receiver_frame_channels'])) if values.get('receiver_frame_channels',0) else None
        out['groups'][g]=values
    return out


def _sampling_summary(rows):
    # common._sample_indices first samples one retained clip uniformly and
    # then a segment uniformly inside it. Loss token weights are separate.
    result={'sampling':'uniform retained clip then uniform segment within clip',
            'loss_weighting_caution':'selection probabilities only; flow loss pools valid latent-token coordinates, so segment length also affects gradient weighting'}
    for arm in ('source','receiver'):
        counts=np.array([len(r['lengths'][arm+'_segments']) for r in rows],dtype=float)
        total=float(counts.sum()); retained=int((counts>0).sum())
        weights=(counts>0).astype(float)/retained if retained else counts
        segment_weights=[float(w/n) for w,n in zip(weights,counts) if n>0 for _ in range(int(n))]
        result[arm]={'segments':int(total),
                     'expected_clip_probability_quantiles':quantiles(weights),
                     'expected_segment_probability_quantiles':quantiles(segment_weights),
                     'effective_clips':float(1/np.square(weights).sum()) if total else None,
                     'emotion_sampling_probability':{key:float(sum(w for r,w in zip(rows,weights) if r['emotion']==key)) for key in sorted({r['emotion'] for r in rows})},
                     'speaker_sampling_probability':{key:float(sum(w for r,w in zip(rows,weights) if r['speaker']==key)) for key in sorted({r['speaker'] for r in rows})}}
    result['clip_weight_total_variation']=float(.5*sum(abs(
        (1/result['source']['effective_clips'] if r['lengths']['source_segments'] else 0)-
        (1/result['receiver']['effective_clips'] if r['lengths']['receiver_segments'] else 0)) for r in rows))
    return result


def population(rows):
    out={'all':summarize(rows),'uniform_clip_then_segment_sampling':_sampling_summary(rows)}
    for axis in ('emotion','speaker'):
        out['by_'+axis]={key:summarize([r for r in rows if r[axis]==key]) for key in sorted({r[axis] for r in rows})}
    out['by_native_clock_length']={name:summarize([r for r in rows if r['native_clock_le_200']==flag])
                                 for name,flag in [('le_200',True),('gt_200',False)]}
    return out


def run(args):
    if args.output.exists(): raise FileExistsError('Fresh audit directory required')
    source=read(args.source_run/'protocol.json'); event=read(args.event_run/'protocol.json')
    if event.get('schema')!='event_schedule_pipeline_v1' or event.get('smoke'):
        raise ValueError('Formal event_schedule_pipeline_v1 run required')
    if event['source_protocol_sha256']!=sha(args.source_run/'protocol.json'):
        raise ValueError('Source protocol binding mismatch')
    if source['dataset_sha256']!=event['dataset_sha256'] or sha(args.dataset)!=source['dataset_sha256']:
        raise ValueError('Dataset binding mismatch')
    code=Path(__file__).resolve().parents[1]
    for rel in PINNED:
        if sha(code/rel)!=event['source_sha256'][rel]: raise ValueError('Executed event source hash mismatch: '+rel)
    if sha(code/'scripts/prepare_continuous_motion_dataset.py')!=source['source_sha256']['scripts/prepare_continuous_motion_dataset.py']:
        raise ValueError('Source segment implementation hash mismatch')
    for key,filename in [('fit_stats_sha256','fit_stats.pt'),('ae_sha256','ae_final.pt'),('prior_sha256','prior_final.pt')]:
        if sha(args.source_run/filename)!=event[key]: raise ValueError('Frozen source file binding mismatch: '+filename)
    teacher=read(args.event_run/'teacher.json')
    payload=torch.load(args.dataset,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('schema')!='continuous_motion_dataset_v1': raise ValueError('Dataset schema mismatch')
    fit,inner=split_train_pool(payload['clips'],source);del payload
    if len(fit)!=613 or len(inner)!=206: raise ValueError('Expected exact 613 fit / 206 inner population')
    if event['train_ids']!=[c['clip_id'] for c in fit] or event['validation_ids']!=[c['clip_id'] for c in inner]:
        raise ValueError('Event fit/inner split binding mismatch')
    if fit_teacher(fit)!=teacher: raise ValueError('Saved teacher differs from exact fit-only refit')
    diagnostics=diverse_diagnostics(inner,24);diagids=[c['clip_id'] for c in diagnostics]
    if read(args.event_run/'control/oracle/result.json')['selected_clip_ids']!=diagids:
        raise ValueError('24-clip receiver diagnostic membership differs')
    started=time.monotonic(); args.output.mkdir(parents=True)
    protocol={'schema':SCHEMA,'posthoc':True,'dataset_sha256':source['dataset_sha256'],
              'source_protocol_sha256':sha(args.source_run/'protocol.json'),'event_protocol_sha256':sha(args.event_run/'protocol.json'),
              'teacher_sha256':sha(args.event_run/'teacher.json'),'teacher_refit_exact':True,
              'script_sha256':sha(__file__),'pinned_code_sha256':{p:sha(code/p) for p in PINNED},
              'source_segmentation':'joint(native & all9 motion_mask), nonoverlap max200, min5',
              'receiver_segmentation':'joint & known4.all, nonoverlap max200, min10',
              'fit_ids':[c['clip_id'] for c in fit],'inner_ids':[c['clip_id'] for c in inner],'diagnostic24_ids':diagids,
              'outer_tensors_indexed':False,'outer_archive_mapped':True,'gpu_or_ae_used':False,
              'energy_definition':'Raw coefficient energy; each original joint run centered once; retain frame energies using each support. Also separately report actual training-segment-centered energies. No fit scale, query amplitude adjustment, or generation.',
              'sampling_definition':'Source and receiver use common._sample_indices: uniform retained clip then uniform segment within that clip. Report selection probabilities separately from frame coverage and length-dependent flow loss weights.',
              'scope':'Fit/inner diagnostic support only; differences are not proof of causal generation degradation'}
    write(args.output/'protocol.json',protocol)
    rows={}
    for name,clips in [('fit',fit),('inner',inner)]:
        rows[name]=[analyze_clip(c,extract_schedule(c['motion9'],c['valid'],teacher,motion_mask=c['motion_mask'])) for c in clips]
    selected=[r for r in rows['inner'] if r['clip_id'] in set(diagids)]
    remainder=[r for r in rows['inner'] if r['clip_id'] not in set(diagids)]
    summary={name:population(items) for name,items in {**rows,'inner_diagnostic24':selected,'inner_remaining182':remainder}.items()}
    comparisons={}
    for g in GROUPS:
        small=summary['inner_diagnostic24']['all']['groups'][g]; full=summary['inner']['all']['groups'][g]
        comparisons[g]={'diagnostic24_to_all206_joint_gt_rms':_divide(small['joint_centered_rms'],full['joint_centered_rms']),
                        'diagnostic24_source_energy_share_of206':_divide(small['source_common_center_energy'],full['source_common_center_energy']),
                        'diagnostic24_source_active_share_of206':_divide(small['source_active_frames'],full['source_active_frames']),
                        'receiver_energy_retention24':small['source_energy_retained_on_overlap'],
                        'receiver_energy_retention206':full['source_energy_retained_on_overlap']}
    result={'schema':SCHEMA,'populations':summary,'diagnostic24_vs_all206':comparisons,'per_clip':rows,
            'seconds':time.monotonic()-started,'causal_conclusion':False,'outer_tensors_indexed':False}
    write(args.output/'report.json',result)
    def fmt(value):return 'pending' if value is None else f'{value:.4f}'
    lines=['# Event training support audit','',
           'This is a post-hoc fit/inner support audit, without training, AE inference or outer target access. It does not establish causality.','',
           '| Population | Clips | Source retained | Receiver retained | Source frames | Receiver frames | Receiver/source | Native clock >200 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in summary:
        s=summary[name]['all'];lines.append(f"| {name} | {s['clips']} | {s['source_clips_retained']} | {s['receiver_clips_retained']} | {s['frames'].get('source_kept',0)} | {s['frames'].get('receiver_kept',0)} | {fmt(s['receiver_source_frame_ratio'])} | {s['native_clock_gt_200']} |")
    lines+=['','| Population | Group | Source energy retained on overlap | Active frames retained vs source | GT joint-run RMS | Source segment RMS | Receiver segment RMS |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name in summary:
        for g,v in summary[name]['all']['groups'].items():
            lines.append(f"| {name} | {g} | {fmt(v['source_energy_retained_on_overlap'])} | {fmt(v['receiver_active_coverage_vs_source'])} | {fmt(v['joint_centered_rms'])} | {fmt(v['source_segment_centered_rms'])} | {fmt(v['receiver_segment_centered_rms'])} |")
    lines+=['','Energy retention uses one original-joint-run center, so removed events cannot be hidden by recentering the shorter retained segment. Segment RMS separately reflects the actual crops presented to training.',
            'An event in one group can be removed because another group is unknown. Per-group source_active_lost_other_group_unknown and complete-event counts quantify that mechanism.',
            'The 24 clips are a metadata-selected subset, not the entire 206-clip inner population. Do not compare their model RMS ratios as if the populations were interchangeable.',
            'Full per-emotion/per-speaker/length strata, run and segment quantiles, dropped tails, and retained-frame asymmetries are in report.json.']
    lines+=['Both trainers first sample a retained clip uniformly, then a segment uniformly inside that clip. The JSON reports clip and segment selection probabilities; frame retention and flow-token loss weighting are separate quantities.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'output':str(args.output),'seconds':result['seconds'],'populations':{k:v['all']['clips'] for k,v in summary.items()}}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','source-run','event-run','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
