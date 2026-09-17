"""Evaluate offline audio-mean composition of saved deployable prefix outputs.

No model sampling, fitting, filesystem writes, or oracle composition occurs.
The caller retains pre-composition GT-prefix diagnostics in its raw report.
"""
from __future__ import annotations

import time

import torch

from scripts import train_prefix_upper as p
from scripts.audit_temporal_repair import metadata_equal, summarize
from scripts.evaluate_prefix_formal import _same_bits, chunk_diagnostics
from scripts.train_centered_temporal_prior import temporal_stats


SCHEMA = 'audio_prefix_dc_composition_v1'
TOLERANCE = 2e-6
REQUIRED = {f'{seed}/full' for seed in p.r.SEEDS} | {
    '42/empty', '42/reverse_history', '42/static', '42/reverse',
    '42/local_static', '42/local_reverse',
}


def _validate(q, bases, raw_curves, args, compose_dc_fn):
    if not callable(compose_dc_fn):
        raise ValueError('Explicit DC composition callable required')
    if type(args.batch_size) is not int or args.batch_size < 1:
        raise ValueError('Positive evaluation batch size required')
    required = ('motion', 'valid', 'static_upper', 'clip_id', 'times', 'channel_mask',
                'b0', 'emotion_id', 'speaker_id')
    if any(key not in q for key in required):
        raise ValueError('Full reference metadata and audio static_upper are required')
    valid, motion = q['valid'], q['motion']
    if (motion.ndim != 3 or motion.shape[-1] != 52 or valid.shape != motion.shape[:2]
            or valid.dtype != torch.bool or min(valid.shape) < 1 or not valid.any(1).all()
            or len(q['clip_id']) != len(valid) or len(set(q['clip_id'])) != len(valid)
            or q['channel_mask'].shape != (len(valid), 52) or q['channel_mask'].dtype != torch.bool
            or not q['channel_mask'][:, p.CC].all()
            or q['static_upper'].shape != (len(valid), 9) or not torch.isfinite(q['static_upper']).all()):
        raise ValueError('Invalid full reference shape, masks, unique IDs, or audio mean')
    reference = p.r.subset(q, torch.arange(len(valid)), 'cpu')
    if (reference['times'].shape != valid.shape or not torch.isfinite(reference['times']).all()
            or not torch.allclose(reference['times'][:, 1:] - reference['times'][:, :-1],
                                  torch.full_like(reference['times'][:, 1:], .04), atol=1e-7, rtol=1e-5)):
        raise ValueError('Native 25Hz clock required')
    metadata = {**reference, 'target': reference['motion']}
    metadata_equal(metadata, bases)
    metadata_equal(metadata, raw_curves)
    if raw_curves.get('noise_seeds') != list(p.r.SEEDS):
        raise ValueError('Raw curves must contain the three fixed seed declarations')
    candidates = raw_curves.get('predictions')
    if not isinstance(candidates, dict):
        raise ValueError('Raw prediction dictionary required')
    # Exclude any oracle-named entry before inspecting or evaluating its value.
    # Even malformed oracle payloads cannot enter composition or scores.
    predictions = {key: value for key, value in candidates.items()
                   if isinstance(key, str) and 'oracle' not in key.lower()}
    if not REQUIRED <= predictions.keys():
        raise ValueError('Missing required deployable full/local/history interventions')
    observed = p.r.obs(reference)
    if not torch.isfinite(reference['motion'][observed]).all():
        raise ValueError('Observed target values must be finite')
    for key, value in predictions.items():
        seed, separator, mode = key.partition('/')
        if not separator or seed not in {str(seed) for seed in p.r.SEEDS} or not mode:
            raise ValueError('Invalid deployable prediction name')
        if (not torch.is_tensor(value) or value.shape != motion.shape or value.dtype != torch.float32
                or not torch.isfinite(value.cpu()[observed]).all()):
            raise ValueError('Invalid saved deployable float32 trajectory')
    for seed in p.r.SEEDS:
        baseline = bases.get('predictions', {}).get(f'{seed}/base')
        if (not torch.is_tensor(baseline) or baseline.shape != motion.shape or baseline.dtype != torch.float32
                or not torch.isfinite(baseline.cpu()[observed]).all()):
            raise ValueError('Missing or invalid seed baseline')
    return reference, predictions, len(candidates)-len(predictions)


def _composition_invariants(raw_upper, composed_upper, static_upper, valid):
    """Tensor-level DC invariants, before nonlinear metric normalizations."""
    maxima = {'valid_mean_max_abs_error': 0., 'centered_motion_max_abs_difference': 0.,
              'adjacent_displacement_max_abs_difference': 0.}
    adjacent_count = 0
    for row in range(len(valid)):
        mask = valid[row]
        before, after = raw_upper[row].double(), composed_upper[row].double()
        before_mean, after_mean = before[mask].mean(0), after[mask].mean(0)
        mean_error = float((after_mean-static_upper[row].double()).abs().max())
        centered_error = float(((after[mask]-after_mean)-(before[mask]-before_mean)).abs().max())
        maxima['valid_mean_max_abs_error'] = max(maxima['valid_mean_max_abs_error'], mean_error)
        maxima['centered_motion_max_abs_difference'] = max(maxima['centered_motion_max_abs_difference'], centered_error)
        pairs = mask[1:] & mask[:-1]
        adjacent_count += int(pairs.sum())
        if pairs.any():
            displacement_error = float(((after[1:]-after[:-1])-(before[1:]-before[:-1]))[pairs].abs().max())
            maxima['adjacent_displacement_max_abs_difference'] = max(
                maxima['adjacent_displacement_max_abs_difference'], displacement_error)
    if any(value > TOLERANCE for value in maxima.values()):
        raise RuntimeError('DC mean, centered trajectory, or adjacent-displacement invariant failed: '+str(maxima))
    return {**maxima, 'observed_adjacent_frame_pairs': adjacent_count, 'absolute_tolerance': TOLERANCE}


@torch.no_grad()
def evaluate_compositions(system, q, identities, bases, raw_curves, args, compose_dc_fn):
    """Return DC-only deployable report/curves using fixed audio upper means.

    ``compose_dc_fn(baseline, raw_upper9, static_upper9, valid)`` must return
    a full [B,T,52] trajectory preserving 43 other channels and invalid frames.
    Target motion is used only for scoring; the compose callback never sees it.
    All oracle predictions are ignored, and no actual-prefix metric is called.
    """
    started = time.monotonic()
    reference, raw_predictions, excluded = _validate(q, bases, raw_curves, args, compose_dc_fn)
    if getattr(system, 'training', False):
        raise ValueError('DC evaluator requires the motion teacher already in eval mode')
    count = len(reference['valid'])
    indices = torch.arange(count)
    predictions, records, mode_seconds = {}, {}, {}
    for key, raw_prediction in raw_predictions.items():
        mode_started = time.monotonic()
        seed = key.split('/', 1)[0]
        outputs, correct = [], 0
        for ix in indices.split(args.batch_size):
            b = p.r.subset(q, ix, args.device)
            ident = p.r.batch_identity(identities, b)
            baseline = bases['predictions'][seed+'/base'][ix].to(args.device)
            raw = raw_prediction[ix].to(args.device)
            if (not _same_bits(raw[..., list(p.r.NOT_UPPER)], baseline[..., list(p.r.NOT_UPPER)])
                    or not _same_bits(raw[~b['valid']], baseline[~b['valid']])):
                raise RuntimeError('Raw saved output does not preserve the bound baseline')
            # Clones prevent an impure composition callback from modifying the
            # bound inputs used for independent checks or later mode scoring.
            composed = compose_dc_fn(baseline.clone(), raw[..., p.CC].clone(),
                                     b['static_upper'].clone(), b['valid'].clone())
            if (not torch.is_tensor(composed) or composed.shape != baseline.shape
                    or composed.dtype != baseline.dtype or composed.device != baseline.device
                    or not torch.isfinite(composed[p.r.obs(b)]).all()):
                raise ValueError('DC composition must return finite observed matching full trajectories')
            if (not _same_bits(composed[..., list(p.r.NOT_UPPER)], baseline[..., list(p.r.NOT_UPPER)])
                    or not _same_bits(composed[~b['valid']], baseline[~b['valid']])):
                raise RuntimeError('DC composition changed protected 43 channels or invalid frames')
            teacher = system.encode_motion(
                torch.where(p.r.obs(b), composed-b['b0']-ident['baseline'][:, None], 0.), b['valid'])
            logits = teacher['emotion_logits']
            if logits.ndim != 2 or len(logits) != len(ix) or not torch.isfinite(logits).all():
                raise ValueError('Invalid composed-motion teacher logits')
            correct += int((logits.argmax(-1) == b['emotion_id']).sum())
            outputs.append(composed.cpu())
        prediction = torch.cat(outputs)
        invariants = _composition_invariants(raw_prediction.cpu()[..., p.CC], prediction[..., p.CC],
                                              reference['static_upper'], reference['valid'])
        predictions[key] = prediction
        records[key] = {
            'populations': p.r.populations(prediction, reference),
            'temporal': temporal_stats(prediction, reference['motion'], reference['valid']),
            'boundaries': p.h.boundary_report(prediction, reference['motion'], reference['valid']),
            'boundary_definition': 'Adjacent final-output displacement at 15->16,31->32,...; coefficient/frame. No GT-prefix endpoint is scored.',
            'chunk_diagnostics': chunk_diagnostics(prediction, reference['motion'], reference['valid'], reference['channel_mask']),
            'generated_emotion_accuracy_nonindependent': correct/count,
            'dc_invariants': invariants,
        }
        mode_seconds[key] = time.monotonic()-mode_started
    curves = {key: reference[key] for key in ('clip_id', 'valid', 'times', 'channel_mask', 'b0', 'emotion_id', 'speaker_id')}
    curves.update(schema=SCHEMA, target=reference['motion'], static_upper=reference['static_upper'].clone(),
                  predictions=predictions, noise_seeds=list(p.r.SEEDS), decode_steps=raw_curves.get('decode_steps'),
                  composition='fixed_audio_mean_plus_raw_generated_motion_minus_its_valid_clip_mean',
                  source_raw_schema=raw_curves.get('schema'), oracle_prediction_keys=[],
                  prediction_scope='Deployable offline DC composition only; no oracle trajectories or GT-prefix endpoint diagnostics.')
    for key in ('arm', 'adaptation_arm', 'recipe_sha256'):
        if key in raw_curves:
            curves[key] = raw_curves[key]
    distribution = summarize(curves, reference['emotion_id'])
    interventions = distribution.pop('single_seed_interventions')
    report = {
        'schema': SCHEMA, 'clips': count, 'noise_seeds': list(p.r.SEEDS), 'decode_steps': raw_curves.get('decode_steps'),
        'evaluation_role': 'internal_development', 'smoke': bool(getattr(args, 'smoke', False)),
        'modes': records, 'distribution': distribution, 'deployable_interventions_seed42': interventions,
        'oracle_predictions_excluded': excluded, 'oracle_composition_performed': False,
        'actual_prefix_scoring_performed': False, 'mean_source': 'Supplied frozen audio-derived static_upper; saved in dc_curves.',
        'nonupper_exact': True, 'invalid_baseline_exact': True, 'dc_invariants_passed': True,
        'invariant_absolute_tolerance': TOLERANCE, 'evaluation_seconds': time.monotonic()-started,
        'mode_seconds_including_scoring': mode_seconds, 'test_loaded': False, 'default_replaced': False,
        'limitations': [
            'Whole-clip mean removal is an offline model composition and is not causal streaming generation.',
            'A constant shift cannot improve temporal correlation, centered shape, or adjacent displacement except numerical rounding.',
            'Pre-composition actual GT-prefix endpoint diagnostics remain exclusively in the caller raw report.',
            'Motion teacher emotion readout is nonindependent; inherited mouth coefficients do not certify lip sync or identity.',
            'Repeated internal development with three fixed seeds does not establish sealed-test generalization or perceptual quality.',
        ],
    }
    for key in ('arm', 'adaptation_arm', 'recipe_sha256'):
        if key in raw_curves:
            report[key] = raw_curves[key]
    return report, curves


__all__ = ['evaluate_compositions']
