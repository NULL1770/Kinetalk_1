"""TRAIN-only linear recoverability of supported native expression descriptors.

No generator change. Pair descriptors are not pure emotion ground truth.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import (
    ExpressionResponse, ResponseConfig, clean, stride_pool, native_interpolate)
from scripts.train_expression_response import (
    configure, load_runtime, development_fold, state_digest, sha, write)
from scripts.train_full_staged import load_safe_native_targets, safe_target_batch
from scripts.audit_paired_target_reliability import window_moments
from scripts.audit_reference_style_swap import REGIONS
from scripts.diagnose_paired_predictability import ChannelRidge, score

TARGETS = ('local_mean', 'local_std')
WIDTH = 5


def descriptors(query, neutral, mask, times):
    if query.shape != neutral.shape or query.shape != mask.shape:
        raise ValueError('Pair shape mismatch')
    # Promotion before subtraction; unobserved values are never used.
    expression = query.astype(np.float64)-neutral.astype(np.float64)
    mean, std, supported = window_moments(expression, mask, times, WIDTH)
    supported &= (supported.sum(0) >= 2)[None]
    return np.stack((mean, std), -1), supported


def feature_windows(x, valid, times):
    if valid.shape != (len(x),) or valid.dtype != bool:
        raise ValueError('Audio validity must retain native slots')
    mask = np.broadcast_to(valid[:, None], x.shape)
    mean, _, support = window_moments(x, mask, times, WIDTH)
    return mean, support[:, 0]


def reverse_control(x, valid, target_mask, lengths):
    """Mirror complete native indices, including gaps; score a common support."""
    if x.shape[:2] != valid.shape or target_mask.shape[:2] != valid.shape:
        raise ValueError('Control clocks differ')
    if valid.dtype != torch.bool or target_mask.dtype != torch.bool:
        raise ValueError('Boolean control masks required')
    reverse = torch.zeros_like(x)
    reverse_valid = torch.zeros_like(valid)
    for j, length in enumerate(lengths):
        n = int(length)
        if n < 1 or n > x.shape[1]:
            raise ValueError('Invalid native length')
        reverse[j, :n] = x[j, :n].flip(0)
        reverse_valid[j, :n] = valid[j, :n].flip(0)
    common = target_mask & valid[..., None] & reverse_valid[..., None]
    common &= (common.sum(1) >= 2)[:, None]
    return reverse, common


def summarize(error, zero, support, energy):
    result = {}
    for region, ix in REGIONS.items():
        result[region] = {}
        for j, name in enumerate(TARGETS):
            mask = support[:, ix]
            def average(values, valid):
                n = valid.sum(1)
                keep = n > 0
                return float((np.where(valid, values, 0.).sum(1)[keep]/n[keep]).mean()) if keep.any() else None
            mse, baseline = average(error[:, ix, j], mask), average(zero[:, ix, j], mask)
            normal = energy[ix, j]
            active = mask & (normal[None] > 1e-8)
            result[region][name] = dict(
                clips=int(mask.any(1).sum()), mse=mse, zero_mse=baseline,
                energy_reduction=None if baseline is None or baseline <= 1e-12 else 1-mse/baseline,
                train_energy_normalized_mse=average(error[:, ix, j]/np.maximum(normal[None], 1e-8), active),
                normalized_active_channels=int((normal > 1e-8).sum()), normalized_clips=int(active.any(1).sum()))
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
    assert sha(binding['prior_artifacts']) == binding['prior_artifacts_sha256']
    manifest = Path(a.safe_root)/'teacher_manifest_gated.jsonl'
    assert sha(manifest) == binding['safe_manifest_sha256']
    data, base = load_runtime(binding, device)
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    cfg = ResponseConfig(**ck['config'])
    model = ExpressionResponse(cfg, ck['model']['feature_mean'], ck['model']['feature_std'], ck['model']['scales']).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval().requires_grad_(False)
    before, neutral_digest = state_digest(model), state_digest(base.stage1)
    q = data['splits']['train']
    fold = development_fold(q, data['fit_sids'])
    approved = {v['source_clip_id'] for v in map(json.loads, manifest.read_text().splitlines()) if v.get('teacher_eligible') is True}
    neutral_id = data['config']['data']['emotion_classes'].index('neutral')
    chosen = {role: [i for i in fold[role] if q['clip_id'][i] in approved and int(q['emotion_id'][i]) != neutral_id]
              for role in ('train', 'speaker_dev', 'sentence_dev')}
    if a.smoke:
        chosen = {role: ids[:8] for role, ids in chosen.items()}
    else:
        assert {k: len(v) for k, v in chosen.items()} == dict(train=2024, speaker_dev=281, sentence_dev=278)
    ids = sorted(i for values in chosen.values() for i in values)
    targets = load_safe_native_targets(a.safe_root, [q['clip_id'][i] for i in ids])
    expected = json.loads(Path(binding['prior_artifacts']).read_text())
    assert set(targets) == {q['clip_id'][i] for i in ids}
    assert all(targets[cid][4] == expected[cid] for cid in targets)
    capture = {}
    hook = model.prior.encoder.register_forward_hook(lambda m, inp, h: capture.update(h=h))
    dims = {'u': cfg.local_dim, 'hidden': cfg.hidden, 'audio': 772, 'prosody': 4}
    probes = {name: ChannelRidge(dim, targets=2, device=device) for name, dim in dims.items()}
    weights, energies, summaries, arrays, metadata = {}, {}, {}, {}, []
    checked = False
    for role, indices in chosen.items():
        for phase in (('fit', 'score') if role == 'train' else ('score',)):
            batches = {}
            for step, sub in enumerate(torch.tensor(indices, dtype=torch.long).split(32)):
                b = q.batch(sub, device, keys=('audio_features', 'valid', 'times', 'motion', 'channel_mask', 'clip_id'))
                n, valid, channel = safe_target_batch(targets, b['clip_id'], b['valid'].shape[1], device, native_times=b['times'])
                mask = valid[..., None] & channel & b['valid'][..., None] & b['channel_mask'][:, None] & base.motion_support
                mask[:, :, 51] = False
                p = model.audio_prior(b['audio_features'], b['valid'])
                h = capture['h']
                _, u = model.conditions(p, b['valid'])
                tokens, token_valid = stride_pool(h, b['valid'], cfg.stride)
                hidden = native_interpolate(tokens, token_valid, b['valid'], cfg.stride)
                audio = (clean(b['audio_features'][..., 768:], b['valid'])-model.feature_mean)/model.feature_std
                source_features = dict(u=u, hidden=hidden, audio=audio)
                if not checked:
                    corrupted = b['audio_features'].clone()
                    corrupted[..., :768] = float('nan')
                    other = model.audio_prior(corrupted, b['valid'])
                    _, other_u = model.conditions(other, b['valid'])
                    torch.testing.assert_close(other_u, u, rtol=0, atol=0)
                    torch.testing.assert_close(other['g_mean'], p['g_mean'], rtol=0, atol=0)
                    checked = True
                # CPU native window arithmetic matches Phase56 exactly.
                motion_np, neutral_np, mask_np, times_np = [v.cpu().numpy() for v in (b['motion'], n, mask, b['times'])]
                feature_np = {k: v.cpu().numpy() for k, v in source_features.items()}
                valid_np = b['valid'].cpu().numpy()
                shape = mask_np.shape
                y_np = np.zeros((*shape, 2), dtype=np.float64)
                supported = np.zeros(shape, dtype=bool)
                x_np = {k: np.zeros((*shape[:2], dim), dtype=np.float64) for k, dim in dims.items() if k != 'prosody'}
                audio_valid = np.zeros(shape[:2], dtype=bool)
                lengths = [int(q['_lengths'][i]) for i in sub.tolist()]
                for j, (i, length) in enumerate(zip(sub.tolist(), lengths)):
                    y_np[j, :length], supported[j, :length] = descriptors(
                        motion_np[j, :length], neutral_np[j, :length], mask_np[j, :length], times_np[j, :length])
                    for name in x_np:
                        x_np[name][j, :length], v = feature_windows(feature_np[name][j, :length], valid_np[j, :length], times_np[j, :length])
                        if name == 'u':
                            audio_valid[j, :length] = v
                        else:
                            assert np.array_equal(audio_valid[j, :length], v)
                    supported[j, :length] &= audio_valid[j, :length, None]
                    supported[j] &= (supported[j].sum(0) >= 2)[None]
                    if phase == 'score':
                        metadata.append(dict(role=role, index=i, clip_id=q['clip_id'][i], emotion=int(q['emotion_id'][i]),
                            coverage={reg: dict(native_observations=int(mask_np[j, :length, ix].sum()),
                                descriptor_observations=int(supported[j, :length, ix].sum()),
                                native_slots=int(length*len(ix))) for reg, ix in REGIONS.items()}))
                x_np['prosody'] = x_np['audio'][..., -4:]
                features = {k: torch.as_tensor(v, device=device) for k, v in x_np.items()}
                y = torch.as_tensor(y_np, device=device)
                mask = torch.as_tensor(supported, device=device)
                feature_valid = torch.as_tensor(audio_valid, device=device)
                for name, probe in probes.items():
                    x = features[name]
                    if phase == 'fit':
                        probe.add(x, y, mask, b['times'], 'centered')
                        continue
                    rev, common = reverse_control(x, feature_valid, mask, lengths)
                    for control, fx, fm in [('forward', x, mask), ('control_forward', x, common), ('control_reverse', rev, common)]:
                        values = score(fx, y, fm, b['times'], 'centered', weights[name])
                        batches.setdefault(name+'/'+control, []).append(tuple(v.cpu().numpy() for v in values))
                if step % 8 == 0:
                    write(out/'state.json', dict(status='running', role=role, phase=phase, clips=min((step+1)*32, len(indices)), total=len(indices), test_loaded=False))
            if phase == 'fit':
                for name, probe in probes.items():
                    weights[name], energies[name] = probe.solve(.001)
            else:
                summaries[role] = {}
                emotion = np.asarray([int(q['emotion_id'][i]) for i in indices])
                for label, values in batches.items():
                    error, zero, support = [np.concatenate([v[j] for v in values]) for j in range(3)]
                    energy = energies[label.split('/')[0]].cpu().numpy()
                    summaries[role][label] = summarize(error, zero, support, energy)
                    summaries[role][label]['by_emotion'] = {str(e): summarize(error[emotion == e], zero[emotion == e], support[emotion == e], energy) for e in np.unique(emotion)}
                    for name, val in [('mse', error), ('zero', zero), ('support', support)]:
                        arrays[role+'/'+label+'/'+name] = val.astype(np.float32) if name != 'support' else val
            print(json.dumps(dict(role=role, phase=phase, clips=len(indices))), flush=True)
    hook.remove()
    assert state_digest(model) == before and state_digest(base.stage1) == neutral_digest
    assert sha(manifest) == binding['safe_manifest_sha256']
    np.savez_compressed(out/'per_clip_errors.npz', **arrays)
    fit = {}
    for name, probe in probes.items():
        fit[name+'/weight'] = weights[name].cpu().numpy()
        fit[name+'/energy'] = energies[name].cpu().numpy()
        fit[name+'/fit_count'] = probe.count.cpu().numpy()
    np.savez_compressed(out/'linear_fit.npz', **fit)
    write(out/'clips.json', metadata)
    write(out/'pair_artifacts.json', {cid: targets[cid][4] for cid in sorted(targets)})
    report = dict(schema='phase57_expression_descriptors_v1', selected_counts={k: len(v) for k, v in chosen.items()},
        width=WIDTH, ridge=.001, targets=TARGETS, features=dims, smoke=a.smoke, summaries=summaries,
        checkpoint_sha256=sha(a.checkpoint), binding_sha256=sha(a.binding), source_sha256=sha(__file__),
        safe_manifest_sha256=sha(manifest), data_manifest_sha256=binding['data_manifest_sha256'],
        frozen_parent_exact=True, frozen_neutral_exact=True, original_pair_hashes_exact=True, content_nan_isolation=True,
        test_loaded=False, validation_queries_used=False, neural_training_performed=False, b0_forward_used=False,
        diagnostic_linear_fit=True, limits=[
            'Paired descriptors include independent-performance/alignment differences; not pure emotion GT.',
            'Targets and features centered on observed per-channel support; diagnostic, not a deployed prediction path.',
            'Linear recoverability and reverse sensitivity do not establish nonlinear limits or causal disentanglement.',
            'Five-frame descriptor support is sparse in the mouth. No missing-frame filling or mask expansion.',
            'No generated motion, new deployment metric, model promotion, or test/development fitting.'])
    write(out/'report.json', report)
    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    assert size < 32*2**20, size
    write(out/'state.json', dict(status='complete', bytes=size, report_sha256=sha(out/'report.json'), test_loaded=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('binding', 'checkpoint', 'safe-root', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        if not isinstance(exc, FileExistsError) and Path(args.output).is_dir():
            write(Path(args.output)/'failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
