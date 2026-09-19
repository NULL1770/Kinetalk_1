"""Rescore fixed source and event checkpoints on every inner-validation clip.

Post-hoc development diagnostic. No fitting, checkpoint selection or promotion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from kinetalk_b0.models.continuous_upper_motion import ContinuousUpperAE, ContinuousLatentFlow
from scripts import train_continuous_motion_latent as common
from scripts import train_event_schedule_pipeline as event
from scripts import audit_event_receiver_bottleneck as audit
from scripts.probe_motion_condition_predictability import split_train_pool, sha


def run(args):
    if args.output.exists():
        raise FileExistsError('Fresh output required')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    ref = audit.read(args.source_run/'protocol.json')
    protocol = audit.read(args.event_run/'protocol.json')
    if protocol.get('smoke') or sha(args.dataset) != ref['dataset_sha256']:
        raise ValueError('Formal bound dataset required')
    if protocol['source_protocol_sha256'] != sha(args.source_run/'protocol.json'):
        raise ValueError('Source protocol mismatch')
    for key, filename in [('ae_sha256', 'ae_final.pt'), ('prior_sha256', 'prior_final.pt'), ('fit_stats_sha256', 'fit_stats.pt')]:
        if protocol[key] != sha(args.source_run/filename):
            raise ValueError('Source file mismatch: '+filename)
    payload = torch.load(args.dataset, map_location='cpu', weights_only=False, mmap=True)
    fit, clips = split_train_pool(payload['clips'], ref)
    del payload
    if protocol['train_ids'] != [c['clip_id'] for c in fit] or protocol['validation_ids'] != [c['clip_id'] for c in clips]:
        raise ValueError('Event split mismatch')
    stats = common._load(args.source_run/'fit_stats.pt')
    if set(stats['train_clip_ids']) != {c['clip_id'] for c in fit}:
        raise ValueError('Statistics fit mismatch')
    ae_ck = common._load(args.source_run/'ae_final.pt')
    ae = ContinuousUpperAE(**ae_ck['config']).to(args.device)
    ae.load_state_dict(ae_ck['state']); ae.eval().requires_grad_(False)
    teacher = audit.read(args.event_run/'teacher.json')
    teacher_hash = sha(args.event_run/'teacher.json')
    labels = {c['clip_id']: event.schedule_for(c, teacher) for c in clips}
    oracle = {cid: [r['schedule']] for cid, r in labels.items()}
    empty = {c['clip_id']: [np.zeros((len(c['valid']),12),np.float32)] for c in clips}
    source = {
        'ae_oracle': args.source_run/'ae_reconstruction/inner_validation/curves.pt',
        'source_prior': args.source_run/'prior_generation/inner_validation/curves.pt',
    }
    common._write(args.output/'protocol.json', {
        'schema':'event_validation_scope_audit_v1', 'posthoc':True,
        'source_protocol_sha256':sha(args.source_run/'protocol.json'),
        'event_protocol_sha256':sha(args.event_run/'protocol.json'),
        'dataset_sha256':ref['dataset_sha256'], 'selected_ids':[c['clip_id'] for c in clips],
        'seeds':list(audit.SEEDS), 'steps':24,
        'source_curves_sha256':{k:sha(v) for k,v in source.items()},
        'receiver_sha256':{k:sha(args.event_run/f'receiver_{k}_final.pt') for k in ('event','null')},
        'code_sha256':{p:sha(Path(__file__).resolve().parents[1]/p) for p in
                      ('scripts/audit_event_validation_scope.py',*event.SOURCES)},
        'scope':'all206 inner development; historical upstream exposure; not sealed test',
        'outer_targets_indexed':False,'default_replaced':False})
    table=[]
    for name, curves, deterministic in [('baseline',audit.baseline_curves(clips),True),
        ('ae_oracle',common._load(source['ae_oracle'])['clips'],True),
        ('source_prior',common._load(source['source_prior'])['clips'],False)]:
        result, report = audit.rescore_arm(curves, clips, stats, args.output/name,
                         deterministic=deterministic, scope=name+'; all206 inner development')
        table.append(audit.table_row(name,result,report))
    for name, checkpoint, schedules in [('event_oracle','event',oracle),('event_null','null',empty)]:
        common._write(args.output/'status.json',{'state':'evaluating','arm':name,'updated':time.time()})
        ck=common._load(args.event_run/f'receiver_{checkpoint}_final.pt')
        if ck['binding'] != {'protocol_sha256':sha(args.event_run/'protocol.json'), 'teacher_sha256':teacher_hash}:
            raise ValueError('Receiver checkpoint binding mismatch')
        model=ContinuousLatentFlow(**ck['config']).to(args.device)
        model.load_state_dict(ck['state']); model.eval().requires_grad_(False)
        before=common._value_sha(model.state_dict())
        event.evaluate_receiver(model,ae,clips,stats,schedules,args.output/name,args.device,name,teacher)
        if common._value_sha(model.state_dict()) != before:
            raise RuntimeError('Checkpoint changed')
        curves=common._load(args.output/name/'curves.pt')['clips']
        result,report=audit.rescore_arm(curves,clips,stats,args.output/name,
                           deterministic=False,scope=name+'; all206 inner development')
        table.append(audit.table_row(name,result,report))
    common._write(args.output/'summary.json',{'schema':'event_validation_scope_audit_v1',
        'clips':len(clips),'table':table,'default_replaced':False,'posthoc':True,
        'group_order':list(audit.GROUP_NAMES),'scope':'all206 inner development; no audio-predictability certification'})
    common._write(args.output/'status.json',{'state':'complete','seconds':time.monotonic()-started,
                                           'updated':time.time(),'default_replaced':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','source-run','event-run','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda')
    run(p.parse_args())
