"""Compact, target-free generation diagnostics on the incremental holdout.

This is an upper-nine diagnostic, not a full-face deployment evaluator. The
shared pretrained source has seen the original fitting pool, including these
sentences; only the new adaptation stage holds them out. No target trajectory,
label, or target mean enters decoding. All five saved samples remain raw; DC
is a separately scored, prediction-only offline composition.
"""
from __future__ import annotations

import time

import torch

from scripts import train_context_mechanism as context
from scripts import train_prefix_upper as p
from scripts.audit_temporal_repair import center, compact_score, paired_metrics
from scripts.audio_flow_metrics import trajectory_energy_score, adjacent_variogram_score
from scripts.evaluate_audio_prefix_adaptation import _composition_invariants


SCHEMA = 'temporal_adapter_incremental_transfer_v1'
GROUPS = {'brows': list(range(5)), 'eyes_expression': list(range(5, 9))}
SEEDS = tuple(p.r.SEEDS)
CONDITION_KEYS = ('valid', 'h0', 'audio_global', 'audio_intensity')


def _validate(upper, q, scales, args, steps):
    required = ('motion', 'valid', 'channel_mask', 'times', 'clip_id',
                'sentence_id', 'emotion_id', 'speaker_id', 'static_upper',
                'prefix_local', 'h0', 'audio_global', 'audio_intensity')
    if any(key not in q for key in required):
        raise ValueError('Complete upper transfer metadata, including sentence_id, required')
    if getattr(upper, 'training', False):
        raise ValueError('Transfer evaluation requires upper in eval mode')
    if type(args.batch_size) is not int or args.batch_size < 1 or steps != 12:
        raise ValueError('Positive batch size and fixed twelve solver steps required')
    valid, motion = q['valid'], q['motion']
    if (valid.ndim != 2 or valid.shape[1] != 96 or len(valid) < 1
            or valid.dtype != torch.bool or not valid.any(1).all()
            or motion.shape != (*valid.shape, 52)
            or q['channel_mask'].shape != (len(valid), 52)
            or q['channel_mask'].dtype != torch.bool or not q['channel_mask'][:, p.CC].all()
            or q['static_upper'].shape != (len(valid), 9)
            or q['prefix_local'].shape != (*valid.shape, 64)
            or q['h0'].ndim != 3 or q['h0'].shape[:2] != valid.shape):
        raise ValueError('Invalid native96 upper transfer shapes or observation masks')
    if (len(q['clip_id']) != len(valid) or len(set(q['clip_id'])) != len(valid)
            or len(q['sentence_id']) != len(valid)
            or any(not isinstance(s, str) or not s for s in q['sentence_id'])
            or q['emotion_id'].shape != (len(valid),) or q['speaker_id'].shape != (len(valid),)):
        raise ValueError('Unique clips and explicit per-clip sentence/label metadata required')
    times = q['times']
    if (times.shape != valid.shape or not torch.isfinite(times).all()
            or not torch.allclose(times[:, 1:]-times[:, :-1], torch.full_like(times[:, 1:], .04),
                                  atol=1e-7, rtol=1e-5)):
        raise ValueError('Native 25Hz clock required')
    for key in ('motion', 'prefix_local', 'h0'):
        value = q[key][..., p.CC] if key == 'motion' else q[key]
        if not torch.isfinite(value[valid]).all():
            raise ValueError('Finite observed upper motion and conditions required')
    for key in ('static_upper', 'audio_global', 'audio_intensity'):
        if len(q[key]) != len(valid) or not torch.isfinite(q[key]).all():
            raise ValueError('Finite per-clip audio conditions required')
    if scales.shape != (9,) or not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError('Nine finite positive frozen source scales required')


def compose_dc_upper(raw_upper, static_upper, valid):
    """No reference input; invalid entries are diagnostic zero placeholders."""
    observed = valid[..., None].expand_as(raw_upper)
    return torch.where(observed, static_upper[:, None]+center(raw_upper, observed), 0.)


def _full_placeholder(upper):
    value = upper.new_zeros(*upper.shape[:2], 52)
    value[..., p.CC] = upper
    return value


def _per_clip(pred, target, observed):
    rows = [paired_metrics(pred[i:i+1], target[i:i+1], observed[i:i+1]) for i in range(len(target))]
    return {key: [row[key] for row in rows] for key in rows[0]}


def _score_predictions(predictions, target, valid, channel_mask, labels):
    """Only brows and eye-expression regions; no placeholder channel is scored."""
    observed = valid[..., None] & channel_mask[:, None]
    full_target = _full_placeholder(target)
    modes = {}
    for key, value in predictions.items():
        full_prediction = _full_placeholder(value)
        modes[key] = {
            'paired': {name: paired_metrics(value[..., cc], target[..., cc], observed[..., cc])
                       for name, cc in GROUPS.items()},
            'per_clip': {name: _per_clip(value[..., cc], target[..., cc], observed[..., cc])
                         for name, cc in GROUPS.items()},
            'temporal': (p.temporal_stats(full_prediction, full_target, valid)
                         if (valid[:, 1:] & valid[:, :-1]).any() else {name: None for name in GROUPS}),
            'boundaries': p.h.boundary_report(full_prediction, full_target, valid),
        }
    stack = torch.stack([predictions[f'{seed}/full'] for seed in SEEDS])
    populations = {}
    for population, rows in (('all', torch.ones(len(valid), dtype=torch.bool)),
                             ('neutral', labels == 0), ('nonneutral', labels != 0)):
        if not rows.any():
            continue
        groups = {}
        for name, cc in GROUPS.items():
            truth, mask = target[rows][..., cc], observed[rows][..., cc]
            samples = stack[:, rows][..., cc]
            per_seed = [paired_metrics(x, truth, mask) for x in samples]
            means = {key: sum(row[key] for row in per_seed)/len(SEEDS)
                     if per_seed[0][key] is not None else None for key in per_seed[0]}
            distribution = {}
            for kind, x, y in (('raw', samples, truth),
                                ('centered_coefficients', center(samples.double(), mask[None]), center(truth.double(), mask))):
                distribution[kind] = {
                    'trajectory_energy_score': compact_score(trajectory_energy_score(x, y, mask)),
                    'adjacent_variogram_score': compact_score(adjacent_variogram_score(x, y, mask, power=.5)),
                }
            groups[name] = {'mean_over_three_seeds': means,
                            'per_seed': dict(zip(map(str, SEEDS), per_seed)), 'distribution': distribution}
        populations[population] = {'clips': int(rows.sum()), 'groups': groups}
    interventions = {}
    for key in ('42/local_static', '42/local_reverse'):
        groups = {}
        for name, cc in GROUPS.items():
            mask = observed[..., cc]
            response = (predictions[key][..., cc]-predictions['42/full'][..., cc])[mask].double()
            paired = modes[key]['paired'][name]
            baseline = modes['42/full']['paired'][name]
            groups[name] = {**paired, 'matched_noise_response_rms': float(response.square().mean().sqrt()),
                            'delta_raw_mse_vs_full': paired['raw_mse']-baseline['raw_mse'],
                            'delta_centered_mse_vs_full': paired['centered_mse']-baseline['centered_mse']}
        interventions[key] = groups
    return {'modes': modes, 'distribution': {'noise_seeds': list(SEEDS), 'populations': populations},
            'deployable_interventions_seed42': interventions}


@torch.no_grad()
def evaluate_transfer(upper, q, identities, scales, args, steps):
    """Return an upper-only report and compact raw upper-nine sample archive.

    Complete supplied membership is evaluated with three fixed full samples
    and seed42 local-only static/reverse interventions. Target/labels never
    reach the decoder. All raw and DC scores use valid observed entries only.
    """
    began = time.monotonic()
    _validate(upper, q, scales, args, steps)
    count = len(q['valid'])
    indices = torch.arange(count)
    reference = p.r.subset({key: q[key] for key in ('valid', 'times', 'clip_id', 'sentence_id',
        'emotion_id', 'speaker_id', 'static_upper')}, indices, 'cpu')
    reference['target_upper9'] = q['motion'][..., p.CC].detach().cpu().float().clone()
    reference['channel_mask_upper'] = q['channel_mask'][:, p.CC].detach().cpu().clone()
    valid = reference['valid']
    reference['target_upper9'] = torch.where(valid[..., None], reference['target_upper9'], 0.)
    # Select the permitted interface BEFORE batching; q['motion'], labels and
    # metadata cannot accidentally enter the rollout by unpacking the query.
    inference = {key: q[key] for key in (*CONDITION_KEYS, 'prefix_local', 'static_upper', 'speaker_id')}
    noises = {seed: torch.randn(count, 96, 9, generator=torch.Generator().manual_seed(seed)) for seed in SEEDS}
    predictions = {}
    for seed in SEEDS:
        for mode in (('full', 'local_static', 'local_reverse') if seed == 42 else ('full',)):
            outputs = []
            for ix in indices.split(args.batch_size):
                b = p.r.subset(inference, ix, args.device)
                identity = p.r.batch_identity(identities, b)
                native = b['prefix_local']
                if mode != 'full':
                    native, _ = p.time_intervention(native, b['h0'], b['valid'], mode.removeprefix('local_'))
                conditions = {key: b[key] for key in CONDITION_KEYS}
                dynamic = context.decode_context(upper, conditions, {'code': identity['code']}, native,
                    noises[seed][ix].to(native), steps=steps, mode='full', arm='chunk_teacher')
                if (not torch.is_tensor(dynamic) or dynamic.shape != (len(ix), 96, 9)
                        or not torch.isfinite(dynamic[b['valid']]).all()):
                    raise ValueError('Decoder must return finite observed native96 upper motion')
                raw = b['static_upper'][:, None]+dynamic*scales.to(dynamic)
                outputs.append(torch.where(b['valid'][..., None], raw, 0.).cpu())
            predictions[f'{seed}/{mode}'] = torch.cat(outputs)
    dc = {key: compose_dc_upper(value, reference['static_upper'], valid) for key, value in predictions.items()}
    invariants = {key: _composition_invariants(value, dc[key], reference['static_upper'], valid)
                  for key, value in predictions.items()}
    score_args = (reference['target_upper9'], valid, reference['channel_mask_upper'], reference['emotion_id'])
    report = {
        'schema': SCHEMA, 'evaluation_role': 'incremental_adaptation_sentence_transfer',
        'clips': count, 'sentence_count': len(set(reference['sentence_id'])),
        'noise_seeds': list(SEEDS), 'decode_steps': steps, 'smoke': bool(getattr(args, 'smoke', False)),
        'compositions': {'raw': _score_predictions(predictions, *score_args), 'dc': _score_predictions(dc, *score_args)},
        'dc_invariants': invariants, 'GT_was_input': False, 'oracle_modes': [],
        'fullface_deployment_evaluation': False, 'nonupper_scored': False,
        'teacher_readout_scored': False, 'nonupper_protection_checked': False,
        'boundary_definition': 'Generated output displacement at adjacent valid native indices15->16,31->32,...; no GT prefix endpoint.',
        'per_clip_order': list(reference['clip_id']),
        'test_loaded': False, 'default_replaced': False,
        'limitations': [
            'Shared source components were motion-supervised on the complete original fit pool; this is incremental adaptation transfer, not unseen-encoder or sealed-test generalization.',
            'Upper-nine diagnostic uses zero invalid placeholders and contains no full-face baseline; it cannot certify43-channel preservation, lipsync, identity, or emotion quality.',
            'DC subtracts the predicted valid whole-clip mean and adds frozen audio static_upper; it is offline and never feeds history.',
            'Three fixed inference seeds and per-clip paired scores do not establish naturalness or population calibration.',
        ],
    }
    report['seconds'] = time.monotonic()-began
    curves = {**reference, 'schema': SCHEMA, 'upper_predictions9': predictions,
              'upper_channel_indices': list(p.CC), 'noise_seeds': list(SEEDS), 'decode_steps': steps,
              'GT_was_input': False, 'invalid_upper_policy': 'zero diagnostic placeholders, never scored',
              'dc_saved': False, 'dc_reconstruction': 'static_upper + center_valid(raw_upper9); zero invalid',
              'fullface_deployment_evaluation': False}
    return report, curves


__all__ = ['evaluate_transfer', 'compose_dc_upper']
