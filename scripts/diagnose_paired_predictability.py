"""TRAIN-only linear probes of safe native expression and neutral-base dynamics.

Probe coefficients never modify the parent. Association is not causal leakage.
All targets share each channel's safe event/observation mask and native clock.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import ExpressionResponse, ResponseConfig
from scripts.diagnose_expression_targets import frame_features
from scripts.train_expression_response import (
    configure, load_runtime, cache_base, development_fold, state_digest, sha, write)
from scripts.train_full_staged import load_safe_native_targets, safe_target_batch
from scripts.audit_reference_style_swap import REGIONS

KINDS = ('centered', 'displacement')
FEATURES = ('u', 'hidden', 'audio', 'base', 'conditional')
COMPONENTS = ('expression_difference', 'neutral_base_error', 'total_residual')


def conditional_features(base, emotion, intensity):
    """Explicit receiving-side features; requires no query label or motion."""
    if base.ndim != 3 or base.shape[-1] != 52 or emotion.shape != (len(base), 8) or intensity.shape != (len(base), 4):
        raise ValueError('Conditional feature dimensions differ')
    for value in (emotion, intensity):
        if not torch.isfinite(value).all() or (value < 0).any() or not torch.allclose(value.sum(-1), torch.ones_like(value[:, 0]), atol=1e-6):
            raise ValueError('Finite normalized probabilities required')
    return torch.cat((base,
        (base[..., None, :]*emotion[:, None, :, None]).flatten(-2),
        (base[..., None, :]*intensity[:, None, :, None]).flatten(-2)), -1)


def coordinate_targets(motion, neutral, base, mask):
    if motion.shape != neutral.shape or motion.shape != base.shape or motion.shape != mask.shape:
        raise ValueError('Coordinate/mask shape mismatch')
    if mask.dtype != torch.bool:
        raise ValueError('Boolean channel mask required')
    for x in (motion, neutral, base):
        if not torch.isfinite(x[mask]).all():
            raise ValueError('Nonfinite observed coordinate')
    # Promote before subtraction so sum/center identities are not rounded twice.
    x, n, b = [torch.where(mask, v, 0.).double() for v in (motion, neutral, base)]
    return torch.stack((x-n, n-b, x-b), -1)


def views(x, y, mask, times, kind):
    """Yield groups with identical channel support; preserve equal clip weights."""
    if x.ndim != 3 or y.ndim != 4 or x.shape[:2] != mask.shape[:2] or y.shape[:3] != mask.shape:
        raise ValueError('Feature/target/channel clocks differ')
    if times.shape != mask.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('Native time/mask shape mismatch')
    observed = mask.any(-1)
    if not torch.isfinite(times[observed]).all():
        raise ValueError('Nonfinite observed time')
    if not torch.isfinite(x[observed]).all() or not torch.isfinite(y[mask]).all():
        raise ValueError('Nonfinite observed probe value')
    dt = times[:, 1:]-times[:, :-1]
    if ((dt <= 0) & observed[:, 1:] & observed[:, :-1]).any():
        raise ValueError('Nonmonotonic observed times')
    x, y = x.double(), y.double()
    if kind == 'displacement':
        mask = mask[:, 1:] & mask[:, :-1] & torch.isclose(
            dt, torch.full_like(dt, .04), rtol=1e-4, atol=1e-7)[..., None]
        x, y = x[:, 1:]-x[:, :-1], y[:, 1:]-y[:, :-1]
    elif kind != 'centered':
        raise ValueError('Unknown dynamics kind')
    # Typically mouth and upper-face masks form only a few groups. Grouping is
    # an algebraic optimization; individual channels retain their own fit count.
    unique, inverse = torch.unique(mask.flatten(0, 1).T, dim=0, return_inverse=True)
    for group, flat_mask in enumerate(unique):
        ids = torch.where(inverse == group)[0]
        valid = flat_mask.reshape(mask.shape[:2])
        count = valid.sum(1)
        if not count.any():
            continue
        a = torch.where(valid[..., None], x, 0.)
        b = torch.where(valid[..., None, None], y[:, :, ids], 0.)
        if kind == 'centered':
            a = torch.where(valid[..., None], a-a.sum(1)[:, None]/count.clamp_min(1)[:, None, None], 0.)
            b = torch.where(valid[..., None, None], b-b.sum(1)[:, None]/count.clamp_min(1)[:, None, None, None], 0.)
        yield ids, a, b, valid, count


class ChannelRidge:
    """Per-channel clip-equal ridge, TRAIN RMS scaling, no intercept/sweep."""
    def __init__(self, dim, channels=52, targets=3, device='cpu'):
        self.dim, self.channels, self.targets = dim, channels, targets
        self.xx = torch.zeros(channels, dim, dim, dtype=torch.float64, device=device)
        self.xy = torch.zeros(channels, dim, targets, dtype=torch.float64, device=device)
        self.yy = torch.zeros(channels, targets, dtype=torch.float64, device=device)
        self.count = torch.zeros(channels, dtype=torch.float64, device=device)

    def add(self, x, y, mask, times, kind):
        for ids, a, b, valid, count in views(x, y, mask, times, kind):
            w = (valid.to(a.dtype)/count.clamp_min(1)[:, None]).sqrt()
            aw = (a*w[..., None]).flatten(0, 1)
            bw = (b*w[..., None, None]).flatten(0, 1)
            self.xx[ids] += aw.T @ aw
            self.xy[ids] += torch.einsum('td,tco->cdo', aw, bw)
            self.yy[ids] += bw.square().sum(0)
            self.count[ids] += (count > 0).sum()

    def solve(self, ridge=1e-3):
        if ridge <= 0 or not self.count.any():
            raise ValueError('Positive fixed ridge and nonempty TRAIN fit required')
        xx = self.xx/self.count.clamp_min(1)[:, None, None]
        xy = self.xy/self.count.clamp_min(1)[:, None, None]
        rms = xx.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).sqrt()
        z = xx/(rms[:, :, None]*rms[:, None, :])
        weight = torch.linalg.solve(z+ridge*torch.eye(self.dim, device=z.device, dtype=z.dtype), xy/rms[:, :, None])/rms[:, :, None]
        energy = self.yy/self.count.clamp_min(1)[:, None]
        return weight, energy


def score(x, y, mask, times, kind, weight):
    error = torch.zeros(len(x), mask.shape[-1], y.shape[-1], dtype=torch.float64, device=x.device)
    zero, support = error.clone(), torch.zeros(mask.shape[::2], dtype=torch.bool, device=x.device)
    for ids, a, b, valid, count in views(x, y, mask, times, kind):
        pred = torch.einsum('btd,cdo->btco', a, weight[ids])
        error[:, ids] = (torch.where(valid[..., None, None], (pred-b).square(), 0.).sum(1)
                          /count.clamp_min(1)[:, None, None])
        zero[:, ids] = b.square().sum(1)/count.clamp_min(1)[:, None, None]
        support[:, ids] = count[:, None] > 0
    return error, zero, support


def summarize(error, zero, support, energy):
    result = {}
    for region, ix in REGIONS.items():
        result[region] = {}
        for j, name in enumerate(COMPONENTS):
            mask = support[:, ix]
            def average(values, valid):
                count = valid.sum(1)
                keep = count > 0
                if not keep.any():
                    return None
                return float((np.where(valid, values, 0.).sum(1)[keep]/count[keep]).mean())
            mse, baseline = average(error[:, ix, j], mask), average(zero[:, ix, j], mask)
            normal = energy[ix, j]
            active = mask & (normal[None] > 1e-8)
            result[region][name] = dict(
                clips=int(mask.any(1).sum()), mse=mse, zero_mse=baseline,
                energy_reduction=None if baseline is None or baseline <= 1e-12 else 1-mse/baseline,
                train_energy_normalized_mse=average(error[:, ix, j]/np.maximum(normal[None], 1e-8), active),
                normalized_active_channels=int((normal > 1e-8).sum()),
                normalized_clips=int(active.any(1).sum()))
    return result


@torch.no_grad()
def run(a):
    configure(47)
    device, out = torch.device(a.device), Path(a.output)
    if out.exists():
        raise FileExistsError('Fresh diagnostic output required')
    out.mkdir(parents=True)
    binding = json.loads(Path(a.binding).read_text())
    source = Path(__file__).resolve().parents[1]
    for name, digest in binding['source_files'].items():
        assert sha(source/name) == digest, name
    assert sha(a.checkpoint) == binding['parent_checkpoint']['sha256']
    manifest = Path(a.safe_root)/'teacher_manifest_gated.jsonl'
    assert sha(manifest) == binding['safe_manifest_sha256']
    data, base = load_runtime(binding, device)
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    cfg = ResponseConfig(**ck['config'])
    assert cfg.center_local
    model = ExpressionResponse(cfg, ck['model']['feature_mean'], ck['model']['feature_std'], ck['model']['scales']).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval().requires_grad_(False)
    before, neutral_digest = state_digest(model), state_digest(base.stage1)
    q = data['splits']['train']
    fold = development_fold(q, data['fit_sids'])
    approved = {v['source_clip_id'] for v in map(json.loads, manifest.read_text().splitlines()) if v.get('teacher_eligible') is True}
    neutral_id = data['config']['data']['emotion_classes'].index('neutral')
    chosen = {role:[i for i in fold[role] if q['clip_id'][i] in approved and int(q['emotion_id'][i]) != neutral_id]
              for role in ('train', 'speaker_dev', 'sentence_dev')}
    if a.smoke:
        chosen = {role:ids[:8] for role, ids in chosen.items()}
    else:
        assert {k:len(v) for k,v in chosen.items()} == dict(train=2024, speaker_dev=281, sentence_dev=278)
    ids = sorted(i for values in chosen.values() for i in values)
    targets = load_safe_native_targets(a.safe_root, [q['clip_id'][i] for i in ids])
    assert set(targets) == {q['clip_id'][i] for i in ids}
    canonical = sorted({j for i in ids for j in range(i//32*32, min((i//32+1)*32, len(q['clip_id'])))})
    cache_base(data, base, device, {'train':canonical, 'validation':[]})
    capture = {}
    hook = model.prior.encoder.register_forward_hook(lambda m, inp, h: capture.update(h=h))
    feature_dims = [('base',52), ('conditional',676)] if a.feature_set == 'conditional' else [
        ('u',cfg.local_dim), ('hidden',cfg.hidden), ('audio',772), ('base',52)]
    probes = {(kind, feat):ChannelRidge(dim, device=device)
              for kind in KINDS for feat, dim in feature_dims}
    weights, energies, summaries, arrays, metadata = {}, {}, {}, {}, []
    checked = False
    for role, indices in chosen.items():
        passes = ('fit', 'score') if role == 'train' else ('score',)
        for phase in passes:
            batches = {}
            for step, sub in enumerate(torch.tensor(indices, dtype=torch.long).split(32)):
                b = q.batch(sub, device)
                n, valid, channel = safe_target_batch(targets, b['clip_id'], b['valid'].shape[1], device, native_times=b['times'])
                mask = valid[..., None] & channel & b['valid'][..., None] & b['channel_mask'][:, None] & base.motion_support
                mask[:, :, 51] = False
                y = coordinate_targets(b['motion'], n, b['b0'], mask)
                p = model.audio_prior(b['audio_features'], b['valid'])
                h = capture['h']
                _, u = model.conditions(p, b['valid'])
                features = {'u':u, **frame_features(model, b, h)}
                # Diagnostic only: a receiver-side interaction between the
                # frozen B0 content base and audio-predicted clip emotion /
                # intensity probabilities. No query labels, GT motion or
                # content feature enters the audio prior/student.
                if a.feature_set == 'conditional':
                    emotion_prob = torch.softmax(model.emotion_head(p['g_mean']), -1)
                    intensity_prob = torch.softmax(model.intensity_head(p['g_mean']), -1)
                    features['conditional'] = conditional_features(features['base'], emotion_prob, intensity_prob)
                torch.testing.assert_close(features['hidden'] @ model.prior.local_head.weight[:cfg.local_dim].T,
                                           u, rtol=1e-4, atol=1e-5)
                if not checked:
                    corrupted = b['audio_features'].clone(); corrupted[..., :768] = float('nan')
                    other_p = model.audio_prior(corrupted, b['valid'])
                    _, other_u = model.conditions(other_p, b['valid'])
                    torch.testing.assert_close(other_u, u, rtol=0, atol=0)
                    torch.testing.assert_close(other_p['g_mean'], p['g_mean'], rtol=0, atol=0)
                    checked = True
                for key, probe in probes.items():
                    kind, feat = key
                    if phase == 'fit':
                        probe.add(features[feat], y, mask, b['times'], kind)
                    else:
                        vals = score(features[feat], y, mask, b['times'], kind, weights[key])
                        batches.setdefault(key, []).append(tuple(v.cpu().numpy() for v in vals))
                if phase == 'score':
                    metadata.extend(dict(role=role, index=i, clip_id=q['clip_id'][i], emotion=int(q['emotion_id'][i])) for i in sub.tolist())
                if step % 8 == 0:
                    write(out/'state.json', dict(status='running', role=role, phase=phase, clips=min((step+1)*32,len(indices)), total=len(indices), test_loaded=False))
            if phase == 'fit':
                for key, probe in probes.items():
                    weights[key], energies[key] = probe.solve()
                write(out/'state.json', dict(status='fit_complete', fit_clips=len(indices), test_loaded=False))
            else:
                summaries[role] = {}
                for key, vals in batches.items():
                    error, zero, support = [np.concatenate([v[j] for v in vals]) for j in range(3)]
                    label = '/'.join(key)
                    energy = energies[key].cpu().numpy()
                    summaries[role][label] = summarize(error, zero, support, energy)
                    emotion = np.asarray([int(q['emotion_id'][i]) for i in indices])
                    summaries[role][label]['by_emotion'] = {str(e):summarize(error[emotion==e],zero[emotion==e],support[emotion==e],energy) for e in np.unique(emotion)}
                    for name, val in [('mse',error), ('zero',zero), ('support',support)]:
                        arrays[role+'/'+label+'/'+name] = val.astype(np.float32) if name != 'support' else val
            print(json.dumps(dict(role=role, phase=phase, clips=len(indices))), flush=True)
    hook.remove()
    assert state_digest(model) == before and state_digest(base.stage1) == neutral_digest
    assert sha(manifest) == binding['safe_manifest_sha256']
    np.savez_compressed(out/'per_clip_errors.npz', **arrays)
    fit = {}
    for key in probes:
        fit['/'.join(key)+'/weight'] = weights[key].cpu().numpy()
        fit['/'.join(key)+'/energy'] = energies[key].cpu().numpy()
        fit['/'.join(key)+'/fit_count'] = probes[key].count.cpu().numpy()
    np.savez_compressed(out/'linear_fit.npz', **fit)
    write(out/'clips.json', metadata)
    write(out/'pair_artifacts.json', {cid:targets[cid][4] for cid in sorted(targets)})
    report = dict(schema='paired_dynamics_predictability_v2', feature_set=a.feature_set, selected_counts={k:len(v) for k,v in chosen.items()},
        smoke=a.smoke, ridge=.001, summaries=summaries, checkpoint_sha256=sha(a.checkpoint), binding_sha256=sha(a.binding),
        safe_manifest_sha256=sha(manifest), data_manifest_sha256=binding['data_manifest_sha256'],
        source_sha256=sha(__file__), frozen_parent_exact=True, frozen_neutral_exact=True, content_nan_isolation=True,
        hidden_u_mapping_verified=True, test_loaded=False, validation_queries_used=False, neural_training_performed=False,
        diagnostic_linear_fit=True, components=COMPONENTS, features=[f for f,d in feature_dims], kinds=KINDS,
        limits=['Imperfect aligned neutral includes alignment differences; targets are not pure emotion GT.',
                'Input capacities differ. Linear association does not prove leakage, independence or neural learnability.',
                'Per-clip channel centering uses observed target support for diagnosis, not a new inference path.',
                'Components are nonorthogonal; scores cannot be interpreted as causal shares. No development/test fitting.',
                'This audit does not change generated motion or any full-development model metric.'])
    write(out/'report.json', report)
    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    assert size < 48*2**20, size
    write(out/'state.json', dict(status='complete', report_sha256=sha(out/'report.json'), bytes=size, test_loaded=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('binding', 'checkpoint', 'safe-root', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--feature-set', choices=('original','conditional'), default='original')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        # Do not touch a completed/preexisting output on a mistaken relaunch.
        if not isinstance(exc, FileExistsError) and Path(args.output).is_dir():
            write(Path(args.output)/'failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
