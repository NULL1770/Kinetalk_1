"""Matched center96 evaluation of center and full-native acoustic rollouts.

Only acoustic conditions and independent identity codes reach generation.
Full rollouts use generated history from native frame zero, then the exact
historical center crop is scored. The compact archive contains upper9 only.
"""
from __future__ import annotations

import time

import torch

from scripts import native_context_runtime as runtime
from scripts import train_audio_prefix_adaptation as adaptation
from scripts import train_context_mechanism as context
from scripts import train_prefix_upper as p
from scripts.evaluate_audio_prefix_adaptation import _composition_invariants
from scripts.evaluate_prefix_formal import _same_bits
from scripts.evaluate_temporal_adapter_transfer import (
    CONDITION_KEYS, GROUPS, SEEDS, _score_predictions, compose_dc_upper,
)


SCHEMA = 'full_native_context_common_center_evaluation_v1'


def extract_center(value, starts):
    """Extract fixed native positions, including unchanged invalid padding."""
    if value.ndim < 2 or len(value) != len(starts):
        raise ValueError('Center extraction requires one start per row')
    if any(int(start) < 0 or int(start)+96 > value.shape[1] for start in starts):
        raise ValueError('Pad the native clock before extracting center96')
    return torch.stack([value[row, int(start):int(start)+96] for row, start in enumerate(starts)])


def _pad(value, frames):
    if value.shape[1] > frames:
        raise ValueError('Batch exceeds the population native noise clock')
    result = value.new_zeros(len(value), frames, *value.shape[2:])
    result[:, :value.shape[1]] = value
    return result


def pad_encoded_full(encoded_full, ids, lengths, valid, device):
    """Allowlisted full-audio encodings; no old center-local cache is copied."""
    rows = [encoded_full[int(i)] for i in ids]
    if any(any(key not in row for key in ('h0', 'audio_global', 'audio_intensity', 'static_upper')) for row in rows):
        raise ValueError('Complete full-native acoustic encodings required')
    width = rows[0]['h0'].shape[-1]
    h0 = torch.zeros(len(rows), valid.shape[1], width, device=device, dtype=torch.float32)
    for row_index, (row, length) in enumerate(zip(rows, lengths)):
        value = row['h0']
        if (not torch.is_tensor(value) or value.shape != (int(length), width)
                or not value.is_floating_point()):
            raise ValueError('Full h0 must use its declared native length')
        h0[row_index, :int(length)] = value.to(device=device, dtype=h0.dtype)
    result = {'valid': valid, 'h0': h0}
    for key in ('audio_global', 'audio_intensity', 'static_upper'):
        values = []
        for row in rows:
            value = row[key]
            if not torch.is_tensor(value) or not value.is_floating_point():
                raise ValueError('Acoustic encoding fields must be floating tensors')
            value = value.reshape(-1) if key == 'audio_intensity' else value
            if value.ndim != 1:
                raise ValueError('Full clip encodings must be one-dimensional')
            values.append(value.to(device=device, dtype=torch.float32))
        result[key] = torch.stack(values)
    return result


def _validate(upper, local, system, store, scales, args, steps, mode, encoded_full):
    if mode not in ('center', 'full'):
        raise ValueError('Evaluation mode must be center or full')
    if any(getattr(module, 'training', False) for module in (upper, local, system)):
        raise ValueError('Evaluation modules must already be in eval mode')
    if type(args.batch_size) is not int or args.batch_size < 1 or steps != 12:
        raise ValueError('Positive batch size and fixed twelve solver steps required')
    q = store.q
    required = ('motion', 'valid', 'channel_mask', 'times', 'clip_id', 'sentence_id',
                'speaker_id', 'emotion_id', 'h0', 'audio_features', 'audio_global',
                'audio_intensity', 'static_upper')
    if any(key not in q for key in required):
        raise ValueError('Complete historical center reference and acoustic conditions required')
    n = len(q['valid'])
    if (not n or q['valid'].shape != (n, 96) or q['valid'].dtype != torch.bool
            or not q['valid'].any(1).all() or q['motion'].shape != (n, 96, 52)
            or q['channel_mask'].shape != (n, 52) or q['channel_mask'].dtype != torch.bool
            or not q['channel_mask'][:, p.CC].all()
            or len(q['clip_id']) != n or len(set(q['clip_id'])) != n
            or list(store.clip_ids) != list(q['clip_id'])
            or len(q['sentence_id']) != n or any(not isinstance(s, str) or not s for s in q['sentence_id'])
            or q['speaker_id'].shape != (n,) or q['emotion_id'].shape != (n,)):
        raise ValueError('Invalid common center96 reference, membership or masks')
    if (q['times'].shape != (n, 96) or not torch.isfinite(q['times']).all()
            or not torch.allclose(q['times'][:, 1:]-q['times'][:, :-1],
                torch.full_like(q['times'][:, 1:], .04), atol=1e-7, rtol=1e-5)
            or not torch.isfinite(q['motion'][..., p.CC][q['valid']]).all()):
        raise ValueError('Finite observed center upper motion and native25Hz clock required')
    lengths = runtime._integer_metadata(store.native_lengths, 'native_lengths')
    starts = runtime._integer_metadata(store.center_starts, 'center_starts')
    if len(lengths) != n or len(starts) != n:
        raise ValueError('Native lengths and crop starts must match population')
    if (not torch.is_tensor(scales) or scales.shape != (9,) or not scales.is_floating_point()
            or not torch.isfinite(scales).all() or not (scales > 0).all()):
        raise ValueError('Nine finite positive frozen residual scales required')
    if mode == 'full' and (not isinstance(encoded_full, (list, tuple)) or len(encoded_full) != n):
        raise ValueError('One full-native encoded record per population clip required')
    return lengths, starts


def _validate_conditions(b):
    valid = b['valid']
    if valid.dtype != torch.bool or valid.ndim != 2 or not valid.any(1).all():
        raise ValueError('Boolean native mask with observed entries required')
    for key in ('h0', 'audio_features'):
        value = b[key]
        if (value.ndim != 3 or value.shape[:2] != valid.shape
                or not value.is_floating_point() or not torch.isfinite(value[valid]).all()):
            raise ValueError('Finite observed acoustic '+key+' required')
    if b['audio_features'].shape[-1] != 1540:
        raise ValueError('Expected1540 raw acoustic features')
    for key, width in (('static_upper', 9), ('audio_intensity', 1), ('audio_global', None)):
        value = b[key]
        if (value.ndim != 2 or len(value) != len(valid) or (width is not None and value.shape[-1] != width)
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError('Finite per-clip acoustic '+key+' required')
    # Sanitizing invalid encodings here protects even a replacement decoder.
    b['h0'] = torch.where(valid[..., None], b['h0'], 0.)
    b['audio_features'] = torch.where(valid[..., None], b['audio_features'], 0.)


def actual_decoder_seams(pred, target, valid, starts, *, mode):
    """Native right-index multiples16, restricted to common-center pairs."""
    offsets = starts if mode == 'full' else torch.zeros_like(starts)
    right = offsets[:, None]+torch.arange(1, valid.shape[1])[None]
    adjacent = valid[:, 1:] & valid[:, :-1]
    seams = adjacent & right.remainder(p.CHUNK).eq(0)
    result = {'observed_seam_pairs': int(seams.sum()),
              'per_clip_observed_seam_pairs': seams.sum(1).tolist(), 'groups': {}}
    for name, cc in GROUPS.items():
        pd = pred[:, 1:, cc].double()-pred[:, :-1, cc].double()
        td = target[:, 1:, cc].double()-target[:, :-1, cc].double()
        groups = {}
        for label, mask in (('seam', seams), ('nonseam', adjacent & ~seams)):
            if not mask.any():
                groups[label] = {'pairs': 0, 'prediction_displacement_rms': None,
                    'target_displacement_rms': None, 'displacement_mse': None}
            else:
                groups[label] = {'pairs': int(mask.sum()),
                    'prediction_displacement_rms': float(pd[mask].square().mean().sqrt()),
                    'target_displacement_rms': float(td[mask].square().mean().sqrt()),
                    'displacement_mse': float((pd-td)[mask].square().mean())}
        result['groups'][name] = groups
    return result


def _validate_bases(q, bases):
    if bases is None:
        return
    if bases.get('clip_id') != q['clip_id'] or bases.get('noise_seeds') != list(SEEDS):
        raise ValueError('Baseline clip order or noise declarations differ')
    for key, source in (('target', 'motion'), ('valid', 'valid'), ('channel_mask', 'channel_mask'),
                        ('times', 'times'), ('b0', 'b0'), ('emotion_id', 'emotion_id'), ('speaker_id', 'speaker_id')):
        if key not in bases or source not in q:
            raise ValueError('Baseline native metadata differs: '+key)
        left, right = q[source].cpu(), bases[key].cpu()
        # Historical caches may retain FP16 motion while the baseline archive
        # stores its exact FP32 promotion. Compare values without rounding down
        # or introducing a tolerance; actual generated-output protection below
        # remains bit-exact against the original FP32 baseline tensor.
        same = _same_bits(left, right)
        if not same and left.is_floating_point() and right.is_floating_point():
            same = left.shape == right.shape and torch.equal(left.double(), right.double())
        if not same:
            raise ValueError('Baseline native metadata differs: '+key)
    observed = q['valid'].cpu()[..., None] & q['channel_mask'].cpu()[:, None]
    for seed in SEEDS:
        value = bases.get('predictions', {}).get(f'{seed}/base')
        if (not torch.is_tensor(value) or value.shape != q['motion'].shape or value.dtype != torch.float32
                or not torch.isfinite(value.cpu()[observed]).all()):
            raise ValueError('Complete finite FP32 baseline samples required')


def _check_protection(predictions, dc, bases, valid):
    if bases is None:
        return {}
    checked = {}
    for key, raw in predictions.items():
        baseline = bases['predictions'][key.split('/')[0]+'/base'].cpu()
        for kind, upper in (('raw', raw), ('dc', dc[key])):
            output = p.compose_upper_face(baseline, upper.to(baseline), valid)
            if (not _same_bits(output[..., list(p.r.NOT_UPPER)], baseline[..., list(p.r.NOT_UPPER)])
                    or not _same_bits(output[~valid], baseline[~valid])):
                raise RuntimeError('Composition changed protected43 channels or invalid baseline')
            checked[key+'/'+kind] = {'nonupper43_bit_exact': True, 'invalid_baseline_bit_exact': True}
    return checked


@torch.no_grad()
def evaluate_native(upper, local, system, store, identities, scales, args, steps,
                    *, mode, encoded_full, bases=None):
    """Raw primary, common-center DC secondary; no oracle or teacher readout."""
    began = time.monotonic()
    lengths, starts = _validate(upper, local, system, store, scales, args, steps, mode, encoded_full)
    q = store.q
    _validate_bases(q, bases)
    n = len(lengths)
    reference = p.r.subset({key: q[key] for key in ('valid', 'times', 'clip_id',
        'sentence_id', 'emotion_id', 'speaker_id')}, torch.arange(n), 'cpu')
    valid = reference['valid']
    reference['target_upper9'] = torch.where(valid[..., None], q['motion'][..., p.CC].cpu().float(), 0.)
    reference['channel_mask_upper'] = q['channel_mask'][:, p.CC].cpu().clone()
    reference.update(native_lengths=lengths.clone(), center_starts=starts.clone())
    # Draw over the entire cohort before batching. Dedicated generators do not
    # touch process RNG; center/full consume exactly the same native positions.
    noises = {}
    for seed in SEEDS:
        noise, _, draw = runtime.paired_noise(lengths, starts, torch.Generator().manual_seed(seed), mode)
        noises[seed] = noise
    frames = draw['draw_frames'] if mode == 'full' else 96
    keys = [(seed, intervention) for seed in SEEDS
            for intervention in (('full', 'local_static', 'local_reverse') if seed == 42 else ('full',))]
    parts = {f'{seed}/{intervention}': [] for seed, intervention in keys}
    static_parts = []
    for ids in torch.arange(n).split(args.batch_size):
        source = store.batch(ids, mode=mode, device=args.device)
        selected_starts = starts[ids]
        if mode == 'full':
            mask = _pad(source['valid'], frames)
            if (mask & (torch.arange(frames, device=mask.device)[None] >= lengths[ids].to(mask.device)[:, None])).any():
                raise ValueError('Frames beyond native length must remain invalid')
            if not torch.equal(extract_center(mask, selected_starts).cpu(), valid[ids]):
                raise ValueError('Full/common center observation masks differ')
            for row, (start, length) in enumerate(zip(selected_starts, lengths[ids])):
                count = min(96, int(length)-int(start))
                if not torch.allclose(source['times'][row, int(start):int(start)+count].cpu().double(),
                        reference['times'][ids[row], :count].double(), atol=1e-7, rtol=0):
                    raise ValueError('Full/common center native timestamps differ')
            b = pad_encoded_full(encoded_full, ids, lengths[ids], mask, args.device)
            b['audio_features'] = _pad(source['audio_features'], frames)
        else:
            b = {key: source[key].float() if source[key].is_floating_point() else source[key]
                 for key in (*CONDITION_KEYS, 'static_upper', 'audio_features')}
            if (not torch.equal(b['valid'].cpu(), valid[ids])
                    or not torch.equal(source['times'].cpu(), reference['times'][ids])):
                raise ValueError('Center source reference changed')
        _validate_conditions(b)
        identity = p.r.batch_identity(identities, {'speaker_id': q['speaker_id'][ids].to(args.device)})
        native = adaptation.local_features(local, b['audio_features'], b['valid'])
        if (native.ndim != 3 or native.shape[:2] != b['valid'].shape
                or not torch.isfinite(native[b['valid']]).all()):
            raise ValueError('Local encoder must return finite observed native conditions')
        native = torch.where(b['valid'][..., None], native, 0.)
        static_parts.append(b['static_upper'].cpu().clone())
        for seed, intervention in keys:
            current = native
            if intervention != 'full':
                current, _ = p.time_intervention(native, b['h0'], b['valid'], intervention.removeprefix('local_'))
            conditions = {key: b[key] for key in CONDITION_KEYS}
            dynamic = context.decode_context(upper, conditions, {'code': identity['code']}, current,
                noises[seed][ids].to(current), steps=steps, mode='full', arm='chunk_teacher')
            if (not torch.is_tensor(dynamic) or dynamic.shape != (len(ids), frames, 9)
                    or not torch.isfinite(dynamic[b['valid']]).all()):
                raise ValueError('Decoder must return finite observed native upper motion')
            raw = torch.where(b['valid'][..., None], b['static_upper'][:, None]+dynamic*scales.to(dynamic), 0.)
            common = extract_center(raw, selected_starts) if mode == 'full' else raw
            parts[f'{seed}/{intervention}'].append(common.cpu())
    predictions = {key: torch.cat(values) for key, values in parts.items()}
    static = torch.cat(static_parts)
    dc = {key: compose_dc_upper(value, static, valid) for key, value in predictions.items()}
    protection = _check_protection(predictions, dc, bases, valid)
    score_args = (reference['target_upper9'], valid, reference['channel_mask_upper'], reference['emotion_id'])
    compositions = {}
    for kind, samples in (('raw', predictions), ('dc', dc)):
        score = _score_predictions(samples, *score_args)
        for key, value in samples.items():
            score['modes'][key]['actual_decoder_seams'] = actual_decoder_seams(
                value, reference['target_upper9'], valid, starts, mode=mode)
        compositions[kind] = score
    report = {'schema': SCHEMA, 'mode': mode, 'evaluation_role': 'common_historical_center96',
        'clips': n, 'sentence_count': len(set(reference['sentence_id'])), 'decode_steps': steps,
        'noise_seeds': list(SEEDS), 'population_noise_draw_frames': draw['draw_frames'],
        'noise_pairing': 'One population [N,rounded_max_native_T,9] draw per seed, then native center slices; independent of batch size.',
        'compositions': compositions, 'primary_composition': 'raw', 'secondary_composition': 'dc',
        'dc_invariants': {key: _composition_invariants(value, dc[key], static, valid) for key, value in predictions.items()},
        'static_origin': 'full_native_audio_mean' if mode == 'full' else 'inherited_center96_audio_mean',
        'dc_mean_scope': 'COMMON observed center96 generated output; never full-native mean or GT mean',
        'boundary_definition': 'boundaries is the fixed center-index16 grid diagnostic, not full-native decoder seams.',
        'actual_seam_definition': 'Adjacent observed common-center pair whose native right index is divisible by16; center mode uses right center index.',
        'nonupper_protection_checked': bases is not None, 'protection_checks': protection,
        'baseline_binding': 'Caller must bind the external baseline archive SHA256; no baseline hash is invented here.' if bases is not None else None,
        'fullface_deployment_evaluation': False, 'teacher_readout_scored': False, 'nonupper_scored': False,
        'GT_was_input': False, 'oracle_modes': [], 'test_loaded': False, 'default_replaced': False,
        'per_clip_order': list(reference['clip_id']), 'seconds': time.monotonic()-began,
        'limitations': [
            'Only the exact historical center96 is scored; full-native suffix/prefix quality is not certified.',
            'DC is offline predicted-center mean composition and never feeds autoregressive history.',
            'Baseline43/invalid exactness, when checked, is an engineering invariant and does not certify lipsync, identity or global emotion quality.',
            'Teacher emotion readout and perceptual full-face evaluation are not evaluated.',
            'Training/source exposure and held-out status must be supplied by the caller; this report does not establish unseen-source generalization.',
        ]}
    curves = {**reference, 'schema': SCHEMA, 'mode': mode, 'static_used': static.clone(),
        'static_upper': static.clone(), 'upper_predictions9': predictions, 'upper_channel_indices': list(p.CC),
        'noise_seeds': list(SEEDS), 'population_noise_draw_frames': draw['draw_frames'], 'decode_steps': steps,
        'GT_was_input': False, 'oracle_modes': [], 'dc_saved': False,
        'dc_reconstruction': 'static_used + center_valid(raw COMMON center96 upper9); zero invalid',
        'invalid_upper_policy': 'zero diagnostic placeholders; restore bound baseline for full-face composition',
        'fullface_deployment_evaluation': False, 'nonupper_protection_checked': bases is not None,
        'baseline_binding': report['baseline_binding'], 'static_origin': report['static_origin']}
    return report, curves


__all__ = ['evaluate_native', 'extract_center', 'pad_encoded_full', 'actual_decoder_seams']
