"""Short v5-initialized staged pilot and independent factor interventions.

Fit only prepared TRAIN examples. No independent emotion probe is used in
the objective. Query motion is confined to teacher/reconstruction/evaluation.
Random student trajectories have flow likelihood supervision, NOT a pointwise
loss against one observed performance. HDTF semantics are explicitly masked.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.factor_rebuild import V5FactorRebuild, RebuildConfig
from kinetalk_b0.models.expression_response import masked_pool
from kinetalk_v2.unified_model import teacher_targets, adjacency
from kinetalk_v4.unified_model_v4 import semantic, masked_mean, affine_geometry
from scripts.train_expression_response import configure, load_runtime, state_digest, write, save
from scripts.train_v7_pilot import load, sha, batch, FixedProbe, meanabs, motion_stats
from scripts.diagnose_v7_factor_routes import cross_person_donors


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -g


def reference(refs, slot):
    out = {k: v[:, slot] for k, v in refs.items()}
    # HDTF may have only one legal independent support. Do not make an empty
    # reference an artificial zero-code training example.
    missing = ~out['valid'].any(-1)
    if missing.any():
        for k in out: out[k] = torch.where(missing.reshape(-1, *([1] * (out[k].ndim - 1))),
                                           refs[k][:, 0], out[k])
    return out


def supervision(logits, intensity_logits, b):
    """Only genuine MEAD labels; unknown HDTF never becomes neutral."""
    known = b['dataset_id'].eq(0)
    emo = torch.where(known, b['emotion_id'], -1)
    level = (b['intensity'] * 3).round().long()
    use = known & b['intensity_valid'] & b['intensity'].ge(0)
    level = torch.where(use, level, -1)
    return semantic(logits, emo) + .2 * semantic(intensity_logits, level)


def reconstruction(model, y, b):
    scale = model.core.scales
    obs = b['valid'][..., None] & b['channel_mask'][:, None]
    position = masked_mean((y - b['motion']).abs() / scale, obs)
    edge = adjacency(b['valid'], b['times'])
    velocity = masked_mean(((y[:, 1:] - y[:, :-1]) -
                            (b['motion'][:, 1:] - b['motion'][:, :-1])).abs() / scale,
                           edge[..., None] & b['channel_mask'][:, None])
    j = model.core.decoder.geometry_jacobian
    dependencies = j.abs().gt(1e-7)
    supported = (b['channel_mask'][:, None] | ~dependencies[None]).all(-1)
    target = affine_geometry(torch.where(obs, b['motion'], 0.), j,
                              model.core.decoder.geometry_offset)
    predicted = affine_geometry(y, j, model.core.decoder.geometry_offset)
    geometry = masked_mean((target - predicted).abs() / model.core.decoder.geometry_scale,
                            b['valid'][..., None] & supported[:, None])
    return position + .35 * velocity + .1 * geometry


def event_loss(schedule, targets):
    hazard = masked_mean(F.binary_cross_entropy_with_logits(schedule['hazard_logits'],
                                  targets['onset'].float(), reduction='none'), targets['onset_valid'])
    logits = schedule['duration_logits']
    dur = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets['duration'].flatten(),
                           reduction='none').reshape_as(targets['duration'])
    params = schedule['amplitude_params']; logstd = params[..., 1].clamp(-4, 0)
    nll = .5 * ((targets['amplitude'].clamp_min(.005).log() - params[..., 0]) /
                 logstd.exp()).square() + logstd
    return hazard + .2 * masked_mean(dur, targets['duration_valid']) + \
        .02 * masked_mean(nll, targets['duration_valid'])


def contrastive_style(a, b, identities, temperature=.2):
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    scores = a @ b.T / temperature
    positive = identities[:, None].eq(identities[None])
    return (scores.logsumexp(-1) - scores.masked_fill(~positive, -torch.inf).logsumexp(-1)).mean()


def raw_audio(data, mean, std):
    """v7 arrays are normalized 1540D; v5 accepts RAW affect 772D."""
    out = dict(data)
    for key in ('audio_features', 'ref_audio_features'):
        out[key] = data[key][..., 768:] * std[768:] + mean[768:]
        mask = data['valid' if key == 'audio_features' else 'ref_valid']
        out[key] = torch.where(mask[..., None], out[key], 0.)
    return out


def attach_ref_labels(data, rows, lookup):
    labels = torch.full(data['ref_valid'].shape[:2], -1, dtype=torch.long)
    for row in rows:
        for slot, ref in enumerate(row['references']):
            info = lookup.get(ref['clip_id'])
            if info is not None: labels[row['row'], slot] = info
    data['ref_emotion_id'] = labels


@torch.no_grad()
def evaluate(model, data, device, binding, seed, output, role, contextual=True):
    model.eval(); n = len(data['motion'])
    # Evaluation probes are loaded only AFTER fitting. Never optimize them.
    probes = [FixedProbe(binding, i).to(device) for i in range(len(binding['probes']))]
    donor = cross_person_donors(data['speaker_id'], data['emotion_id'])
    enrolled = []
    for start in range(0, n, 8):
        _, r = batch(data, list(range(start, min(n, start + 8))), device)
        enrolled.append(model.encode_style(reference(r, 0), contextual=contextual))
    enrolled = torch.cat(enrolled)
    per = []; examples = {}; all_y = {k: [] for k in ('mean', 'sample', 'sample2', 'teacher', 'other_ref', 'wrong', 'changed')}
    all_gt = []; all_valid = []; all_labels = []
    for start in range(0, n, 8):
        ids = list(range(start, min(n, start + 8)))
        b, r = batch(data, ids, device); valid = b['valid'].bool()
        r0, r1 = reference(r, 0), reference(r, 1)
        s0, s1, wrong = enrolled[ids], model.encode_style(r1, contextual=contextual), enrolled[donor[ids]]
        p = model.affect_audio(b['audio_features'], valid)
        t = model.visual_teacher(b)
        targets = teacher_targets(b['motion'], b['b0'], valid, b['channel_mask'], b['times'])
        ys = {'mean': model.decode(b, p, s0, mode='modal')['motion'],
              'teacher': model.decode(b, t, s0, events=targets['events'])['motion'],
              'other_ref': model.decode(b, p, s1, mode='modal')['motion'],
              'wrong': model.decode(b, p, wrong, mode='modal')['motion']}
        samples = []
        for offset in (0, 91):
            samples.append(model.predict(b['audio_features'], b['h0'], b['b0'], valid, r0,
                          times=b['times'], seed=seed + start + offset, contextual_style=contextual))
        ys.update(sample=samples[0]['motion'], sample2=samples[1]['motion'])
        labels = b['emotion_id']; changed = (labels.clamp_min(0) + 1) % model.cfg.classes
        newemo = F.one_hot(changed, model.cfg.classes).float()
        ys['changed'] = model.decode(b, p, s0, mode='modal', emotion=newemo)['motion']
        for k in all_y: all_y[k].append(ys[k].cpu())
        all_gt.append(b['motion'].cpu()); all_valid.append(valid.cpu()); all_labels.append(labels.cpu())
        for j, i in enumerate(ids):
            v = valid[j:j+1]; truth = b['motion'][j:j+1]
            obs = v[..., None] & b['channel_mask'][j:j+1, None]
            row = {'row': i, 'emotion': int(labels[j])}
            for name, y in ys.items():
                row[name + '_mae'] = float(masked_mean((y[j:j+1] - truth).abs(), obs))
                row[name + '_mouth_mae'] = float(masked_mean((y[j:j+1, :, 14:40] - truth[:, :, 14:40]).abs(), obs[:, :, 14:40]))
            row['same_style_code_distance'] = float((s0[j] - s1[j]).square().mean().sqrt())
            row['cross_style_code_distance'] = float((s0[j] - wrong[j]).square().mean().sqrt())
            row['reference_motion_delta'] = meanabs(ys['mean'][j:j+1], ys['other_ref'][j:j+1], v)
            row['cross_style_motion_delta'] = meanabs(ys['mean'][j:j+1], ys['wrong'][j:j+1], v)
            row['seed_diversity'] = meanabs(ys['sample'][j:j+1], ys['sample2'][j:j+1], v)
            for name in ('mean', 'sample'):
                row.update({name + '_' + k: val for k, val in motion_stats(ys[name][j:j+1], v).items()})
            per.append(row)
            emo = int(labels[j])
            if emo >= 0 and emo not in examples:
                examples[emo] = i
                np.savez_compressed(output / f'{role}_emotion_{emo}.npz',
                    motions=np.stack([truth[0].cpu().numpy(), b['b0'][j].cpu().numpy(),
                                     *[ys[k][j].cpu().numpy() for k in ('mean', 'sample', 'teacher', 'other_ref', 'wrong')]]),
                    mode_names=np.array(['GT', 'B0', 'audio_modal', 'audio_flow_sample', 'privileged_teacher',
                                         'same_person_other_emotion_ref', 'different_person_ref']),
                    valid=v[0].cpu().numpy(), channel_mask=b['channel_mask'][j].cpu().numpy(),
                    times=b['times'][j].cpu().numpy(), row=np.array(i), emotion=np.array(emo))
    output.mkdir(parents=True, exist_ok=True)
    truth = torch.cat(all_gt); valid = torch.cat(all_valid); labels = torch.cat(all_labels)
    known = labels.ge(0)
    pred = {k: torch.cat(v) for k, v in all_y.items()}
    summary = {k: float(np.mean([x[k] for x in per])) for k in per[0] if k not in ('row', 'emotion')}
    summary['style_distance_ratio'] = summary['same_style_code_distance'] / max(summary['cross_style_code_distance'], 1e-8)
    summary['independent_probes'] = {}
    for ix, probe in enumerate(probes):
        classes = {k: probe.classes(v, valid).cpu() for k, v in {'gt': truth, **pred}.items()}
        summary['independent_probes'][str(ix)] = {k: float(v[known].eq(labels[known]).float().mean()) for k, v in classes.items()}
        summary['independent_probes'][str(ix)].update(
            style_class_preserved=float(classes['wrong'][known].eq(classes['mean'][known]).float().mean()),
            reference_class_preserved=float(classes['other_ref'][known].eq(classes['mean'][known]).float().mean()),
            changed_class_correct=float(classes['changed'][known].eq((labels[known] + 1) % model.cfg.classes).float().mean()))
        for row in per:
            i = row['row']; row[f'probe{ix}_mean_correct'] = int(classes['mean'][i] == labels[i])
            row[f'probe{ix}_sample_correct'] = int(classes['sample'][i] == labels[i])
    # Timing proxies; these are diagnostic, not a certified phonetic decoder.
    from kinetalk_v4.unified_model_v4 import affine_geometry
    j = model.core.decoder.geometry_jacobian.cpu(); off = model.core.decoder.geometry_offset.cpu()
    geometry = {k: affine_geometry(v, j, off) for k, v in pred.items()}
    summary['style_geometry_delta'] = meanabs(geometry['mean'], geometry['wrong'], valid)
    summary['seed_geometry_delta'] = meanabs(geometry['sample'], geometry['sample2'], valid)
    summary['gt_dynamics'] = motion_stats(truth, valid)
    summary['per_emotion'] = {str(e): {'n': len([r for r in per if r['emotion'] == e]),
        'mean_mae': float(np.mean([r['mean_mae'] for r in per if r['emotion'] == e])),
        'sample_mae': float(np.mean([r['sample_mae'] for r in per if r['emotion'] == e]))}
        for e in range(model.cfg.classes) if any(r['emotion'] == e for r in per)}
    write(output / f'{role}_per_clip.json', per)
    return summary


def train(a):
    configure(a.seed); device = torch.device(a.device); out = Path(a.output)
    if out.exists(): raise FileExistsError('fresh output required')
    out.mkdir(parents=True); start = time.time()
    binding = json.loads(Path(a.binding).read_text())
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if sha(a.checkpoint) != a.sha256: raise ValueError('retained v5 hash mismatch')
    runtime, base = load_runtime(binding, torch.device('cpu'))
    mean, std = runtime['feature_stats']['mean'].cpu(), runtime['feature_stats']['std'].cpu()
    data = {role: raw_audio(load(Path(a.data) / (role + '.npz')), mean, std)
            for role in ('fit', 'held_sentence', 'unseen_speaker')}
    provenance = json.loads((Path(a.data) / 'provenance.json').read_text())
    lookup = {}
    for split in ('train', 'validation'):
        q = runtime['splits'][split]
        lookup.update({cid: int(q['emotion_id'][i]) for i, cid in enumerate(q['clip_id'])})
    for role, arrays in data.items():
        attach_ref_labels(arrays, [r for r in provenance['records'] if r['role'] == role], lookup)
        if role == 'fit':
            # Unique person keys across data domains; source grouping is audited
            # by the prepared provenance, not inferred from local numeric IDs.
            arrays['person_key'] = arrays['speaker_id'] + arrays['dataset_id'] * 100000
    del runtime, base
    original = V5FactorRebuild.from_v5(checkpoint, RebuildConfig(flow_steps=a.flow_steps,
                                    flow_noise_kernel=a.flow_noise_kernel)).to(device)
    baseline = copy.deepcopy(original)
    baseline.requires_grad_(False)
    fit = data['fit']; rng = np.random.default_rng(a.seed)
    order = [rng.choice(len(fit['motion']), min(a.batch, len(fit['motion'])), replace=False).tolist()
             for _ in range(max(a.visual_steps, a.student_steps, a.control_steps))]
    history = []
    def next_batch(step): return batch(fit, order[step], device)
    def log(stage, step, loss, **extra):
        row = dict(stage=stage, step=step+1, loss=float(loss), elapsed_seconds=time.time()-start, **extra)
        history.append(row)
        if (step+1) % 32 == 0:
            write(out / 'state.json', row); print(json.dumps(row), flush=True)
    # Matched continuation: same original decoder/style/student, same TRAIN
    # batches; supervision protocol and budget are recorded explicitly.
    control = copy.deepcopy(original).requires_grad_(False)
    control.core.decoder.requires_grad_(True); control.core.style.requires_grad_(True)
    control.core.student.requires_grad_(True); control.core.events.requires_grad_(True)
    control.core.emotion_to_state.requires_grad_(True)
    params = [p for p in control.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    for step in range(a.control_steps):
        control.train(); b, refs = next_batch(step); opt.zero_grad(set_to_none=True)
        p = control.affect_audio(b['audio_features'], b['valid'])
        s = control.encode_style(reference(refs, step % 2), contextual=False)
        targets = teacher_targets(b['motion'], b['b0'], b['valid'], b['channel_mask'], b['times'])
        y = control.decode(b, p, s, events=targets['events'])
        loss = reconstruction(control, y['motion'], b) + supervision(p['emotion_logits'], p['intensity_logits'], b) + event_loss(y['schedule'], targets)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.); opt.step(); log('matched_control', step, loss)
    save(out / 'control.pt', control.checkpoint())
    # Visual teacher and conditional reference training. Native audio anchor is
    # fixed; all genuine references predict the SAME observed query Y.
    model = original; model.requires_grad_(False)
    # Preserve the retained v5 carrier while learning the new factors. The
    # decoder/style base is reopened only in a later ablation after this
    # route proves it does not erase v5 emotion. Otherwise a short pilot can
    # confuse decoder drift with a teacher/student failure.
    for module in (model.teacher, model.reference_context,
                   model.style_emotion, model.style_content, model.affect_style,
                   model.style_response):
        module.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    for step in range(a.visual_steps):
        model.train(); model.anchor_student.eval(); b, refs = next_batch(step); opt.zero_grad(set_to_none=True)
        r0, r1 = reference(refs, 0), reference(refs, 1)
        s0, s1 = model.encode_style(r0), model.encode_style(r1)
        t = model.visual_teacher(b)
        targets = teacher_targets(b['motion'], b['b0'], b['valid'], b['channel_mask'], b['times'])
        y0, y1 = [model.decode(b, t, s, events=targets['events']) for s in (s0, s1)]
        loss = .5 * (reconstruction(model, y0['motion'], b) + reconstruction(model, y1['motion'], b))
        loss += supervision(t['emotion_logits'], t['intensity_logits'], b)
        scale = torch.cat((s0.detach(), s1.detach())).std(0).clamp_min(.25)
        consistency = ((s0 - s1) / scale).square().mean()
        loss += a.style_consistency * consistency + .02 * contrastive_style(s0, s1, b['person_key'])
        # Auxiliary nuisance adversaries; heads learn normally, encoders
        # receive reversed gradients. No claim that chance accuracy proves purity.
        nuisance = semantic(model.style_emotion(Reverse.apply(s0)), r0['emotion_id'])
        nuisance += .02 * F.mse_loss(model.style_content(Reverse.apply(s0)),
                                     masked_pool(r0['h0'], r0['valid']).detach())
        nuisance += .02 * F.mse_loss(model.affect_style(Reverse.apply(masked_pool(t['state'], b['valid']))),
                                     .5 * (s0 + s1).detach())
        loss += a.nuisance_weight * nuisance
        loss += .002 * t['correction'].square()[b['valid']].mean()
        # Keep audio deployment expressive while visual state coordinates are
        # learned: a small, REAL paired anchor, not a random-prior MSE.
        with torch.no_grad(): p = model.affect_audio(b['audio_features'], b['valid'], anchor=True)
        loss += .25 * reconstruction(model, model.decode(b, p, s0, events=targets['events'])['motion'], b)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite visual objective')
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.); opt.step()
        log('visual', step, loss, style_consistency=float(consistency))
    save(out / 'visual.pt', model.checkpoint())
    visual_digest = state_digest(model.teacher); decoder_digest = state_digest(model.core.decoder)
    model.requires_grad_(False)
    model.flow.requires_grad_(True)
    model.core.student.emotion.requires_grad_(True)
    model.core.student.intensity.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    # TRAJECTORY scales use fit only, in native raw-state coordinates.
    diffs = []
    model.eval()
    with torch.no_grad():
        for step in range(min(16, a.student_steps)):
            b, _ = next_batch(step); t = model.visual_teacher(b)
            p = model.affect_audio(b['audio_features'], b['valid'])
            diffs.append((t['state'] - p['state'])[b['valid']])
        model.flow.scale.copy_(torch.cat(diffs).square().mean(0).sqrt().clamp_min(.025))
    for step in range(a.student_steps):
        model.eval(); model.flow.train()
        b, refs = next_batch(step); opt.zero_grad(set_to_none=True)
        with torch.no_grad(): t = model.visual_teacher(b); s = model.encode_style(reference(refs, step % 2))
        p = model.affect_audio(b['audio_features'], b['valid']); c = model.core.condition(p, b['valid'])
        loss = model.flow.loss(t['state'], p['state'], p['hidden'], c['emotion'], c['intensity'], b['valid'],
                               times=b['times'])
        # Only the global semantic readouts are fitted here. Keep the retained
        # encoder/state carrier fixed; flow handles the trajectory correction.
        loss += supervision(p['emotion_logits'], p['intensity_logits'], b)
        loss += .2 * F.kl_div(p['emotion_logits'].log_softmax(-1), t['emotion_logits'].softmax(-1), reduction='batchmean')
        targets = teacher_targets(b['motion'], b['b0'], b['valid'], b['channel_mask'], b['times'])
        y = model.decode(b, p, s, events=targets['events'])
        loss += .25 * reconstruction(model, y['motion'], b)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite student objective')
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.); opt.step(); log('student_flow', step, loss)
    # Enable stochastic endpoint only after distribution fitting. This gate is
    # explicit in the receipt and keeps the v5 continuation exactly stable
    # before the new dynamic module earns its place.
    model.flow.sample_gain.fill_(1.0)
    if state_digest(model.teacher) != visual_digest or state_digest(model.core.decoder) != decoder_digest:
        raise RuntimeError('teacher/decoder changed during distribution distillation')
    save(out / 'repair.pt', model.checkpoint()); write(out / 'history.json', history)
    results = {}
    for name, net, contextual in [('retained_v5_cross_ref', baseline, False), ('matched_control', control, False), ('repair', model, True)]:
        dest = out / name; dest.mkdir()
        results[name] = {role: evaluate(net, data[role], device, binding, a.seed, dest, role, contextual)
                         for role in ('held_sentence', 'unseen_speaker')}
    report = {'schema': 'v5_factor_rebuild_pilot_v1', 'elapsed_seconds': time.time()-start,
              'base_checkpoint': a.checkpoint, 'base_sha256': a.sha256, 'results': results,
              'budgets': {'control': a.control_steps, 'visual': a.visual_steps, 'student': a.student_steps},
              'fit_dataset_counts': {str(i): int(fit['dataset_id'].eq(i).sum()) for i in (0, 1)},
              'sealed_test_loaded': False, 'independent_probes_used_in_training': False,
              'paper_ready': False, 'limits': ['short pilot, not paper evidence',
                 'v5 historical benchmark used neutral enrollment; here cross-emotion refs are matched',
                 'weak event parser reused from v5; event truth not independent psychological annotation',
                 'state flow is a distribution-learning prototype; finite seeds need visual/distribution audit',
                 'geometric deltas are not certified phonetic invariance',
                 'HDTF same-source style pairs cannot prove cross-source or cross-emotion style']}
    write(out / 'report.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--binding', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--sha256', required=True); p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, default=47)
    p.add_argument('--batch', type=int, default=8); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--control-steps', type=int, default=384); p.add_argument('--visual-steps', type=int, default=256)
    p.add_argument('--student-steps', type=int, default=128); p.add_argument('--flow-steps', type=int, default=12)
    p.add_argument('--flow-noise-kernel', type=int, default=1)
    p.add_argument('--style-consistency', type=float, default=.1); p.add_argument('--nuisance-weight', type=float, default=.02)
    train(p.parse_args())
