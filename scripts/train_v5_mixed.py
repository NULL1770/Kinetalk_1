"""Resumable v5 MEAD/HDTF continuation with explicit domain/sample accounting.

HDTF provides content/motion/reference style only; emotion/intensity are unknown.
The neutral front-end is frozen. Motion weak events are parsed in source context
before cropping. Random rollouts receive sustained means, never GT event timing.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v4.unified_model_v4 import KineTalkV4Config, KineTalkV4Model
from kinetalk_v4.rollout_statistics import expression_mean_loss, pooled_state_loss
from kinetalk_v4.domain_routes import content_style_gradient_scope
from kinetalk_v4.state_posture import load_zero_posture_extension, posture_supervision
from kinetalk_v4.acoustic_geometry import load_zero_geometry_extension
from kinetalk_v4.geometry_support import geometry_support, clean_geometry_jacobian
from kinetalk_v4.unified_model_v4 import affine_geometry
from kinetalk_v2.unified_model import anchor_matrix
from kinetalk_v2.hdtf_dataset import HDTFSplit
from scripts.train_expression_response import configure, load_runtime, development_fold, state_digest, write, save, sha
from scripts.train_unified_v4 import (reference_ids, build_base_cache, attach_base, make_refs,
    motion_targets, objective, evaluate, select_balanced)
from scripts.train_full_staged import base_forward


def training_objective(model, batch, refs, targets, channels, weight,
                       posture_weight=0., teacher_posture_weight=0., geometry_rollout_weight=0.):
    loss, out, logs = objective(model, batch, refs, targets, warmup=False, channels=channels)
    if posture_weight or teacher_posture_weight:
        student_state = (out['conditioned']['state'] if model.cfg.state_supervision_space == 'decoder'
                         else out['student']['state'])
        posture = posture_supervision(student_state, out['teacher']['state'],
            targets['state'], batch['valid'], targets['quality'])
        loss = loss + posture_weight * posture['posture_direct'] + teacher_posture_weight * posture['posture_guidance']
        logs.update({k: float(v) for k, v in posture.items()})
    if weight or geometry_rollout_weight:
        rollout_out = model.decode(batch['h0'], batch['b0'], batch['valid'], out['conditioned'],
            out['style'], out['sampled_schedule']['events'].detach(),
            geometry_audio=out['geometry_audio_context'])
        rollout = rollout_out['motion']
    if weight:
        mean = expression_mean_loss(rollout, batch['motion'], batch['valid'], batch['channel_mask'], model.scales)
        state = (rollout - batch['b0']) @ anchor_matrix().to(rollout)
        state_mean = pooled_state_loss(state, targets['state'], batch['valid'], targets['quality'])
        loss = loss + weight * (mean + state_mean)
        logs.update(rollout_mean=float(mean), rollout_state_mean=float(state_mean))
    if geometry_rollout_weight:
        J=clean_geometry_jacobian(model.decoder.geometry_jacobian)
        truth=affine_geometry(torch.where(batch['channel_mask'][:,None],batch['motion'],0.),
                              J,model.decoder.geometry_offset)
        support=geometry_support(J,batch['channel_mask'])[:,None].expand_as(truth)
        projected=affine_geometry(rollout,J,model.decoder.geometry_offset)
        scale=model.decoder.geometry_scale
        final_mean=pooled_state_loss(projected/scale,truth/scale,batch['valid'],support)
        head_mean=pooled_state_loss(rollout_out['geometry_target']/scale,truth/scale,batch['valid'],support)
        loss=loss+geometry_rollout_weight*(final_mean+head_mean)
        logs.update(geometry_rollout_mean=float(final_mean),geometry_head_rollout_mean=float(head_mean))
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite objective')
    return loss, logs


@torch.no_grad()
def hdtf_batch(dataset, ids, base, device):
    context = dataset.context_batch(ids, device)
    context.update(base_forward(base, context['content'], context['valid']))
    targets = motion_targets(context)
    batch = dataset.batch(ids, device)
    batch.update({k: dataset.crop_context_tensor(context[k], context) for k in ('b0', 'h0')})
    return batch, {k: dataset.crop_context_tensor(v, context) for k, v in targets.items()}


def balanced_source_order(dataset, rng):
    groups = {}
    for i, row in enumerate(dataset.rows):
        groups.setdefault(row['speaker_group'], {}).setdefault(row['source_id'], []).append(i)
    for sources in groups.values():
        for values in sources.values(): rng.shuffle(values)
    order = []
    while any(values for sources in groups.values() for values in sources.values()):
        people = sorted(groups); rng.shuffle(people)
        for person in people:
            for source in sorted(groups[person]):
                if groups[person][source]: order.append(groups[person][source].pop())
    return order


def next_ids(order, position, count, rebuild):
    chosen = []
    while len(chosen) < count:
        if position >= len(order): order, position = rebuild(), 0
        end = min(len(order), position + count - len(chosen))
        chosen.extend(order[position:end]); position = end
    return chosen, order, position


def run(a):
    if a.epochs < 1 or a.steps < 1 or not 0 <= a.hdtf_weight < 1:
        raise ValueError('Invalid epoch/step/domain weights')
    if a.posture_weight < 0 or a.teacher_posture_weight < 0:
        raise ValueError('Posture weights must be nonnegative')
    if a.geometry_rollout_weight < 0:
        raise ValueError('Geometry rollout weight must be nonnegative')
    configure(a.seed); device = torch.device(a.device)
    root = Path(a.output)
    if root.exists() and not a.resume: raise FileExistsError('Fresh output required')
    root.mkdir(parents=True, exist_ok=True)
    binding = json.loads(Path(a.binding).read_text())
    data, base = load_runtime(binding, device); frozen = state_digest(base.stage1)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False); s = ck['model']
    cfg = KineTalkV4Config(**ck['config'])
    if a.posture_adapter:
        cfg = replace(cfg, use_posture_adapter=True)
    if a.teacher_guidance_mode:
        cfg = replace(cfg, teacher_guidance_mode=a.teacher_guidance_mode)
    if a.state_supervision_space:
        cfg = replace(cfg, state_supervision_space=a.state_supervision_space)
    if a.geometry_audio_mode:
        cfg=replace(cfg,geometry_audio_mode=a.geometry_audio_mode)
    if a.geometry_audio_mode and a.posture_adapter:
        raise ValueError('New geometry and posture extensions require separate matched pilots')
    if a.teacher_posture_weight and cfg.teacher_guidance_mode != 'posture':
        raise ValueError('Privileged posture guidance must replace framewise guidance explicitly')
    if cfg.teacher_guidance_mode == 'posture' and not a.teacher_posture_weight:
        raise ValueError('Posture guidance mode requires a nonzero explicit guidance weight')
    model = KineTalkV4Model(cfg, s['feature_mean'], s['feature_std'], s['scales'],
        s['decoder.geometry_jacobian'], s['decoder.geometry_offset'], s['decoder.geometry_scale']).to(device)
    extended = (load_zero_geometry_extension(model,s) if cfg.geometry_audio_mode != 'none'
                else load_zero_posture_extension(model, s))
    # Model construction can consume different amounts of RNG under an optional
    # head; reset before matched training and preserve the independent data RNG.
    configure(a.seed)
    if (model.decoder.geometry_scale[[0, 3]] < .1).any():
        raise ValueError('Start mixed continuation from the repaired geometry checkpoint')
    q = data['splits']['train']; fold = development_fold(q, data['fit_sids'])
    fit = fold['train']; held = select_balanced(q, fold['speaker_dev'] + fold['sentence_dev'], a.eval_limit, a.seed)
    pool = {}
    for i in fit: pool.setdefault(int(q['speaker_id'][i]), []).append(i)
    rng = random.Random(a.seed)
    refs = {i: reference_ids(q, [i], pool, rng)[0] for i in fit}
    cache = build_base_cache(q, base, sorted(set(fit + held + list(refs.values()))), device)
    hdtf = None
    if a.hdtf_weight:
        if not a.hdtf_manifest or not a.hdtf_features or not a.hdtf_source_root:
            raise ValueError('Quality-bound HDTF artifacts required')
        hdtf = HDTFSplit(a.hdtf_manifest, a.hdtf_features, split='train',
            source_root=a.hdtf_source_root, verify_hashes=True, cache_sources=a.hdtf_cache_sources,
            motion_policy='raw')
        if len(hdtf.eligible_reference_indices()) != len(hdtf):
            raise ValueError('HDTF binding contains ineligible independent references')
    channels = ck['protocol']['geometry_channels']
    protocol = dict(schema='v5_mixed_continuation_v1', seed=a.seed, epochs=a.epochs,
        steps_per_epoch=a.steps, mead_fit_clips=len(fit), internal_dev_clips=len(held),
        source_checkpoint_sha256=sha(a.checkpoint), binding_sha256=sha(a.binding),
        source_sha256=sha(__file__), neutral_digest=frozen, config=asdict(cfg),
        geometry_channels=channels, geometry_scale_train_fitted=model.decoder.geometry_scale.cpu().tolist(),
        domain_weight=dict(mead=1-a.hdtf_weight, hdtf=a.hdtf_weight),
        hdtf_provenance=hdtf.provenance if hdtf is not None else None,
        hdtf_windows=len(hdtf) if hdtf is not None else 0,
        hdtf_people=len(set(hdtf['speaker_group'])) if hdtf is not None else 0,
        hdtf_sources=len(set(hdtf['source_id'])) if hdtf is not None else 0,
        hdtf_emotion_targets=0, hdtf_intensity_targets=0, hdtf_neutral_targets=0,
        text_enabled=False, test_loaded=False, flags={k:v for k,v in vars(a).items() if k!='resume'},
        statistical_rollout='population coefficient/state means; no sampled-GT pointwise timing loss',
        temporal_targets='observed motion events parsed on source context; cropped boundaries censored')
    protocol.update(posture_extension_added=extended,
        teacher_guidance_mode=cfg.teacher_guidance_mode,
        state_supervision_space=cfg.state_supervision_space,
        posture_guidance=('TRAIN-only detached teacher advantage on supported '+cfg.state_supervision_space+' student clip means and raw motion-teacher coordinates'
            if cfg.teacher_guidance_mode == 'posture' else 'disabled' if cfg.teacher_guidance_mode == 'none'
            else 'legacy framewise state distillation'),
        posture_adapter_input='pooled audio hidden only; no text/query motion/label lookup',
        model_source_sha256=sha(sys.modules[KineTalkV4Model.__module__].__file__),
        posture_source_sha256=sha(sys.modules[posture_supervision.__module__].__file__),
        training_references_sha256=hashlib.sha256(json.dumps(sorted(refs.items())).encode()).hexdigest(),
        fit_indices_sha256=hashlib.sha256(json.dumps(fit).encode()).hexdigest(),
        extra_posture_supervision_domain='MEAD only')
    protocol.update(geometry_receiver_mode=cfg.geometry_audio_mode,
        geometry_rollout_weight=a.geometry_rollout_weight,
        geometry_extension_added=extended if cfg.geometry_audio_mode != 'none' else False,
        geometry_source_sha256=sha(sys.modules[load_zero_geometry_extension.__module__].__file__),
        geometry_receiver_input='native HuBERT768 or fixed coarse-h0 tiling + same TRAIN-scaled prosody4',
        geometry_mean_supervision='supported native clip means, sampled final motion and head; no event timing oracle')
    protocol['posture_extension_added']=extended if cfg.geometry_audio_mode=='none' else False
    common_initial={k:v for k,v in model.state_dict().items() if not k.startswith('decoder.geometry_audio.')}
    digest=hashlib.sha256()
    for name,value in sorted(common_initial.items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    protocol['common_initial_parameters_sha256']=digest.hexdigest()
    if model.decoder.geometry_audio is not None:
        protocol['geometry_receiver_initial_sha256']=state_digest(model.decoder.geometry_audio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    def mead_order():
        order = list(fit); rng.shuffle(order); return order
    mo = mead_order(); mp = 0
    ho = balanced_source_order(hdtf, rng) if hdtf is not None else []; hp = 0
    used_m, used_h = set(), set(); history = []; step = 0; start_epoch = 0
    used_mead_sequence = []
    routing_verified = False
    posture_gradient_verified = False
    geometry_gradient_verified = False
    if a.resume:
        saved = torch.load(a.resume, map_location=device, weights_only=False)
        if saved['protocol'] != protocol: raise ValueError('Resume protocol differs')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start_epoch, step, history = saved['epoch'], saved['step'], saved['history']
        mo, mp, ho, hp = saved['orders']; used_m, used_h = set(saved['used_mead']), set(saved['used_hdtf'])
        used_mead_sequence = saved.get('used_mead_sequence', [])
        rng.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda': torch.cuda.set_rng_state_all(saved['cuda_rng'])
    write(root / 'protocol.json', protocol)
    print(json.dumps({'event': 'start', 'mead': len(fit), 'hdtf': len(hdtf) if hdtf is not None else 0}), flush=True)
    begin = time.monotonic()
    for epoch in range(start_epoch, a.epochs):
        model.train(); sums = {}
        for update in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            mids, mo, mp = next_ids(mo, mp, a.mead_batch_size, mead_order)
            used_m.update(mids)
            used_mead_sequence.extend(mids)
            batch = attach_base(q.batch(torch.tensor(mids), device), mids, cache)
            references = make_refs(q, [refs[i] for i in mids], device)
            targets = motion_targets(batch)
            mloss, mlogs = training_objective(model, batch, references, targets, channels,
                a.rollout_mean_weight, a.posture_weight, a.teacher_posture_weight,a.geometry_rollout_weight)
            ((1-a.hdtf_weight) * mloss).backward()
            hloss = torch.zeros((), device=device)
            if hdtf is not None:
                hids, ho, hp = next_ids(ho, hp, a.hdtf_batch_size, lambda: balanced_source_order(hdtf, rng))
                used_h.update(hids)
                hb, ht = hdtf_batch(hdtf, hids, base, device)
                hr = hdtf.single_references(hids, device, seed=a.seed + step)
                if not hr['reference_valid'].all(): raise ValueError('Independent HDTF reference absent')
                if (hb['emotion_id'].ge(0).any() or hb['emotion_valid'].any()
                        or hb['intensity_id'].ge(0).any() or hb['intensity_valid'].any()
                        or hb['neutral_target_valid'].any()):
                    raise ValueError('Unknown HDTF emotion/intensity/neutral targets not masked')
                if a.hdtf_scope == 'content_style':
                    # Existing MEAD gradients are retained. The temporary
                    # scope suppresses only HDTF gradients in affect-specific
                    # parameters; shared decoder effects remain measurable.
                    guarded = [p for module in (model.teacher, model.student, model.events,
                        model.emotion_to_state, model.decoder.affect) for p in module.parameters()]
                    previous = [None if p.grad is None else p.grad.detach().clone() for p in guarded] if not routing_verified else []
                    with content_style_gradient_scope(model):
                        hloss, hlogs = training_objective(model, hb, hr, ht, channels, a.rollout_mean_weight)
                        (a.hdtf_weight * hloss).backward()
                    if not routing_verified:
                        for parameter, expected in zip(guarded, previous):
                            if expected is None:
                                if parameter.grad is not None:
                                    raise RuntimeError('HDTF introduced affect-specific gradients')
                            elif parameter.grad is None or not torch.equal(parameter.grad, expected):
                                raise RuntimeError('HDTF modified existing MEAD affect gradients')
                        routing_verified = True
                else:
                    hloss, hlogs = training_objective(model, hb, hr, ht, channels, a.rollout_mean_weight)
                    (a.hdtf_weight * hloss).backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient')
            if cfg.use_posture_adapter and not posture_gradient_verified:
                head = model.student.posture_adapter.output
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise RuntimeError('Audio posture head has missing/nonfinite gradients')
                if head.weight.grad.abs().sum() == 0:
                    raise RuntimeError('Audio posture head received zero weight gradient')
                posture_gradient_verified = True
            if cfg.geometry_audio_mode != 'none' and not geometry_gradient_verified:
                head=model.decoder.geometry_audio.output
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise RuntimeError('Geometry receiver head has missing/nonfinite gradients')
                if head.weight.grad.abs().sum()==0:
                    raise RuntimeError('Geometry receiver head received zero gradients')
                geometry_gradient_verified=True
            optimizer.step(); step += 1
            sums['mead_loss'] = sums.get('mead_loss', 0.) + float(mloss)
            sums['hdtf_loss'] = sums.get('hdtf_loss', 0.) + float(hloss)
            for name in ('posture_direct', 'posture_guidance', 'teacher_advantage_fraction',
                         'geometry_rollout_mean','geometry_head_rollout_mean'):
                sums[name] = sums.get(name, 0.) + mlogs.get(name, 0.)
            if step % 20 == 1:
                state = dict(status='training', epoch=epoch+1, step=step, total_updates=a.epochs*a.steps,
                    mead_loss=float(mloss), hdtf_loss=float(hloss), gradient_norm=float(grad),
                    unique_mead_clips=len(used_m), unique_hdtf_windows=len(used_h),
                    elapsed_seconds=time.monotonic()-begin)
                state.update({k:mlogs[k] for k in ('posture_direct','posture_guidance',
                    'teacher_advantage_fraction','geometry_rollout_mean','geometry_head_rollout_mean') if k in mlogs})
                write(root / 'state.json', state); print(json.dumps(state), flush=True)
        dev = evaluate(model, q, held, cache, data, device, a.seed, channels)
        if state_digest(base.stage1) != frozen: raise RuntimeError('Frozen neutral B0 changed')
        history.append(dict(epoch=epoch+1, step=step, internal_dev=dev,
            train={k:v/a.steps for k,v in sums.items()}, unique_mead_clips=len(used_m), unique_hdtf_windows=len(used_h)))
        saved = dict(model=model.state_dict(), config=asdict(cfg), protocol=protocol, optimizer=optimizer.state_dict(),
            epoch=epoch+1, step=step, history=history, orders=(mo,mp,ho,hp),
            used_mead=sorted(used_m), used_hdtf=sorted(used_h), python_rng=rng.getstate(),
            used_mead_sequence=used_mead_sequence,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        save(root / 'resume.pt', saved); write(root / 'history.json', history)
    save(root / 'final.pt', {k:v for k,v in saved.items() if k!='optimizer'})
    write(root / 'complete.json', dict(status='complete', updates=step, protocol=protocol,
        elapsed_seconds=time.monotonic()-begin, frozen_b0_sha256=frozen,
        unique_mead_clips=len(used_m), unique_hdtf_windows=len(used_h),
        unique_hdtf_sources=len({hdtf['source_id'][i] for i in used_h}) if hdtf is not None else 0,
        unique_hdtf_people=len({hdtf['speaker_group'][i] for i in used_h}) if hdtf is not None else 0,
        hdtf_affect_gradient_preservation_verified=routing_verified if a.hdtf_scope=='content_style' else None,
        posture_head_gradient_verified=posture_gradient_verified if cfg.use_posture_adapter else None,
        geometry_head_gradient_verified=geometry_gradient_verified if cfg.geometry_audio_mode!='none' else None,
        mead_training_sequence_sha256=hashlib.sha256(json.dumps(used_mead_sequence).encode()).hexdigest(),
        paper_result=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--binding',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--hdtf-manifest');p.add_argument('--hdtf-features');p.add_argument('--hdtf-source-root')
    p.add_argument('--hdtf-weight',type=float,default=.25);p.add_argument('--hdtf-cache-sources',type=int,default=236)
    p.add_argument('--hdtf-scope',choices=['all_factors','content_style'],default='all_factors')
    p.add_argument('--posture-adapter',action='store_true')
    p.add_argument('--posture-weight',type=float,default=0.)
    p.add_argument('--teacher-posture-weight',type=float,default=0.)
    p.add_argument('--teacher-guidance-mode',choices=['frame','posture','none'])
    p.add_argument('--state-supervision-space',choices=['raw','decoder'])
    p.add_argument('--geometry-audio-mode',choices=['none','coarse','hubert'])
    p.add_argument('--geometry-rollout-weight',type=float,default=0.)
    p.add_argument('--epochs',type=int,default=2);p.add_argument('--steps',type=int,default=1363)
    p.add_argument('--mead-batch-size',type=int,default=8);p.add_argument('--hdtf-batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=47);p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--rollout-mean-weight',type=float,default=.5);p.add_argument('--eval-limit',type=int,default=128)
    p.add_argument('--device',default='cuda');p.add_argument('--resume')
    run(p.parse_args())
