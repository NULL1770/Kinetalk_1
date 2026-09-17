"""Batched fixed-seed evaluation of deployable prefix flow and separate oracle.

No fitting or checkpoint selection is performed. The caller binds the supplied
model, cached audio/identity inputs and baseline files to its training recipe.
"""
from __future__ import annotations

import torch

from scripts import train_prefix_upper as p
from scripts.audit_temporal_repair import metadata_equal, summarize
from scripts.audit_history_upper import chunk_profile
from scripts.train_centered_temporal_prior import temporal_stats
from kinetalk_b0.models.slow_state_affect import compose_upper_face


MODES = ('full', 'empty', 'reverse_history', 'static', 'reverse', 'oracle_history')


def _same_bits(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def actual_prefix_continuation(pred, target, valid, channel_mask, supplied_past):
    """Current first frame minus the endpoint actually supplied as known past.

    Call only when a prefix was supplied, using GT for oracle or the generated
    trajectory for ordinary rollout. Reverse-history endpoints differ and are
    intentionally unscored here. Missing immediate predecessor means no pair.
    """
    pairs = valid[:, 1:] & valid[:, :-1]
    seam = torch.arange(1, valid.shape[1], device=valid.device).remainder(p.CHUNK) == 0
    pairs = pairs & seam[None]
    result = {}
    for name in ('brows', 'eyes_expression'):
        channels = list(p.GROUPS[name])
        observed = pairs[..., None] & channel_mask[:, None, channels]
        count = int(observed.sum())
        if not count:
            result[name] = {'observed_channel_pairs': 0, 'rms': None,
                            'reference_rms': None, 'displacement_mse': None}
            continue
        step = (pred[:, 1:, channels].double() - supplied_past[:, :-1, channels].double())[observed]
        truth = (target[:, 1:, channels].double() - target[:, :-1, channels].double())[observed]
        if not torch.isfinite(step).all() or not torch.isfinite(truth).all():
            raise ValueError('Nonfinite observed actual-prefix continuation')
        result[name] = {'observed_channel_pairs': count, 'rms': float(step.square().mean().sqrt()),
                        'reference_rms': float(truth.square().mean().sqrt()),
                        'displacement_mse': float((step-truth).square().mean())}
    return result


def chunk_diagnostics(pred, target, valid, channel_mask):
    """Descriptive raw profiles, early-frame errors, and within-block speed.

    The common-clip profile uses one coverage rule determined only by masks;
    it is independent of mode and generated results. All displacements use
    coefficient/frame units, not coefficient/second.
    """
    def profiles(values, truth, good, channels):
        output = chunk_profile(values, truth, good, channels)
        observed = good[..., None] & channels[:, None]
        for name, cc in p.GROUPS.items():
            x, y, mask = (values[..., list(cc)].double(), truth[..., list(cc)].double(),
                          observed[..., list(cc)])
            for row in output[name]:
                start, stop = row['start_frame'], row['stop_frame_exclusive']
                m = mask[:, start:stop]
                row['prediction_raw_mean'] = float(x[:, start:stop][m].mean()) if m.any() else None
                row['reference_raw_mean'] = float(y[:, start:stop][m].mean()) if m.any() else None
                row['early_frame_errors'] = {}
                for width in (2, 4):
                    end = min(stop, start+width)
                    early = mask[:, start:end]
                    error = (x[:, start:end]-y[:, start:end])[early]
                    row['early_frame_errors'][f'first_{width}_native_frames'] = {
                        'observed_values': int(early.sum()),
                        'raw_mse': float(error.square().mean()) if early.any() else None,
                        'signed_error_mean': float(error.mean()) if early.any() else None}
                pairs = m[:, 1:] & m[:, :-1]
                dx = (x[:, start+1:stop]-x[:, start:stop-1])[pairs]
                dy = (y[:, start+1:stop]-y[:, start:stop-1])[pairs]
                row['within_chunk_displacement'] = {
                    'observed_channel_pairs': int(pairs.sum()),
                    'prediction_rms': float(dx.square().mean().sqrt()) if pairs.any() else None,
                    'reference_rms': float(dy.square().mean().sqrt()) if pairs.any() else None,
                    'displacement_mse': float((dx-dy).square().mean()) if pairs.any() else None}
        return output

    eligible = torch.ones(len(valid), dtype=torch.bool, device=valid.device)
    for start in range(0, valid.shape[1], p.CHUNK):
        length = min(p.CHUNK, valid.shape[1]-start)
        eligible &= valid[:, start:start+length].sum(1) >= (length+1)//2
    return {'all_observed': profiles(pred, target, valid, channel_mask),
            'common_clip_coverage': {
                'rule': 'At least half the native frames observed in every chunk; identical clips for all modes.',
                'clip_count': int(eligible.sum()), 'clip_indices': eligible.nonzero(as_tuple=True)[0].tolist(),
                'profiles': profiles(pred[eligible], target[eligible], valid[eligible], channel_mask[eligible])},
            'units': 'Uncentered coefficient and coefficient/frame displacement.',
            'scope': 'Descriptive rollout profiles; changing late error alone does not prove recursive drift.'}


def _validate(q, bases, scales, args, steps, use_prefix):
    if type(use_prefix) is not bool or type(steps) is not int or steps < 1:
        raise ValueError('Explicit Boolean prefix setting and positive integer solver steps required')
    if type(args.batch_size) is not int or args.batch_size < 1:
        raise ValueError('Positive evaluation batch size required')
    required = ('motion', 'valid', 'h0', 'prefix_local', 'audio_global', 'audio_intensity', 'static_upper',
                'clip_id', 'times', 'channel_mask', 'b0', 'emotion_id', 'speaker_id')
    if any(key not in q for key in required):
        raise ValueError('Formal evaluation requires full cached conditions and reference metadata')
    motion, valid = q['motion'], q['valid']
    if (motion.ndim != 3 or motion.shape[-1] != 52 or valid.shape != motion.shape[:2]
            or valid.dtype != torch.bool or min(valid.shape) < 1 or not valid.any(1).all()
            or len(q['clip_id']) != len(valid) or len(set(q['clip_id'])) != len(valid)
            or q['channel_mask'].shape != (len(valid), 52) or q['channel_mask'].dtype != torch.bool
            or not q['channel_mask'][:, p.CC].all()):
        raise ValueError('Invalid formal curve shape, unique IDs, or observed upper channels')
    if (q['times'].shape != valid.shape or not torch.isfinite(q['times']).all()
            or not torch.allclose(q['times'][:, 1:] - q['times'][:, :-1],
                                  torch.full_like(q['times'][:, 1:], .04), atol=1e-7, rtol=1e-5)):
        raise ValueError('Native 25Hz evaluation clock required')
    if (not torch.is_tensor(scales) or scales.shape != (9,) or not scales.is_floating_point()
            or not torch.isfinite(scales).all() or (scales <= 0).any()):
        raise ValueError('Nine finite positive fixed training scales required')
    reference = p.r.subset(q, torch.arange(len(valid)), 'cpu')
    metadata_equal({**reference, 'target': reference['motion']}, bases)
    observed = reference['valid'][..., None] & reference['channel_mask'][:, None]
    if not torch.isfinite(reference['motion'][observed]).all():
        raise ValueError('Nonfinite observed reference')
    for seed in p.r.SEEDS:
        value = bases.get('predictions', {}).get(f'{seed}/base')
        if (not torch.is_tensor(value) or value.shape != motion.shape or value.dtype != torch.float32
                or not torch.isfinite(value.cpu()[observed]).all()):
            raise ValueError('Missing or invalid fixed-seed float32 baseline')
    return reference


@torch.no_grad()
def evaluate_formal(upper, system, q, identities, scales, bases, args, steps, use_prefix):
    """Return report/curves for all supplied clips, with globally indexed noise.

    No smoke truncation or per-batch RNG restart is applied. All three full
    [N,T,9] Gaussian arrays are drawn before inference with independent CPU
    generators seeded 42/123/2026; every intervention reuses that seed's array.
    Prefix/global/local audio conditions are cached and fixed by the caller.
    """
    reference = _validate(q, bases, scales, args, steps, use_prefix)
    if getattr(upper, 'training', False) or getattr(system, 'training', False):
        raise ValueError('Formal evaluator requires both models already in eval mode')
    count, frames = reference['valid'].shape
    indices = torch.arange(count)
    noises = {seed: torch.randn(count, frames, 9, generator=torch.Generator().manual_seed(seed))
              for seed in p.r.SEEDS}
    predictions, modes, oracle_records = {}, {}, {}
    for seed in p.r.SEEDS:
        for mode in MODES if seed == 42 else ('full',):
            outputs, correct = [], 0
            for ix in indices.split(args.batch_size):
                b = p.r.subset(q, ix, args.device)
                identity = p.r.batch_identity(identities, b)
                native = b['prefix_local']
                fixed_scales = scales.to(native)
                kwargs = {'oracle_target': p.h.normalized_target(b, fixed_scales)} if mode == 'oracle_history' else {}
                dynamic = p.rollout(upper, b, identity, native, noises[seed][ix].to(native),
                                    steps=steps, mode=mode, use_prefix=use_prefix, **kwargs)
                baseline = bases['predictions'][f'{seed}/base'][ix].to(device=args.device)
                pred = compose_upper_face(baseline, b['static_upper'][:, None] + dynamic*fixed_scales, b['valid'])
                if (not _same_bits(pred[..., list(p.r.NOT_UPPER)], baseline[..., list(p.r.NOT_UPPER)])
                        or not _same_bits(pred[~b['valid']], baseline[~b['valid']])):
                    raise RuntimeError('Frozen 43-channel or invalid-frame baseline protection failed')
                readout = system.encode_motion(
                    torch.where(p.r.obs(b), pred-b['b0']-identity['baseline'][:, None], 0.), b['valid'])
                logits = readout['emotion_logits']
                if logits.ndim != 2 or len(logits) != len(ix) or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite or invalid generated-motion teacher logits')
                correct += int((logits.argmax(-1) == b['emotion_id']).sum())
                outputs.append(pred.cpu())
            key = f'{seed}/{mode}'
            prediction = torch.cat(outputs)
            predictions[key] = prediction
            scored_prefix = use_prefix and mode in ('full', 'oracle_history')
            prefix_source = ('tracked_reference_GT' if mode == 'oracle_history' else 'generated_past') if scored_prefix else None
            if not use_prefix or mode == 'empty':
                reason = 'No prefix was supplied; stitched boundary is descriptive only.'
            elif mode == 'reverse_history':
                reason = 'Reversed supplied endpoint differs from chronological previous frame; actual-prefix score omitted.'
            elif mode in ('static', 'reverse'):
                reason = 'Only unmodified full and explicit oracle modes receive actual-prefix scoring in this evaluator.'
            else:
                reason = 'Current first frame minus its actually supplied immediately preceding prefix endpoint.'
            actual = actual_prefix_continuation(prediction, reference['motion'], reference['valid'], reference['channel_mask'],
                        reference['motion'] if mode == 'oracle_history' else prediction) if scored_prefix else None
            record = {'populations': p.r.populations(prediction, reference),
                      'temporal': temporal_stats(prediction, reference['motion'], reference['valid']),
                      'chunk_diagnostics': chunk_diagnostics(prediction, reference['motion'], reference['valid'], reference['channel_mask']),
                      'boundaries': p.h.boundary_report(prediction, reference['motion'], reference['valid']),
                      'boundary_definition': 'Stitched adjacent output displacement at native indices 15->16,31->32,...; coefficient/frame.',
                      'actual_prefix_continuation': {'scored': scored_prefix, 'source': prefix_source, 'metrics': actual, 'note': reason},
                      'same_gt_endpoint_output_diagnostic': {
                          'metrics': actual_prefix_continuation(prediction, reference['motion'], reference['valid'],
                                                               reference['channel_mask'], reference['motion']),
                          'GT_was_supplied_as_history': bool(use_prefix and mode == 'oracle_history'),
                          'note': 'Same reference endpoint for comparisons across arms. This output measurement is not evidence that GT was supplied to the model; only the prefix-enabled oracle receives GT history.'},
                      'generated_emotion_accuracy_nonindependent': correct/count,
                      'oracle_target_history': mode == 'oracle_history', 'prefix_enabled': use_prefix}
            (oracle_records if mode == 'oracle_history' else modes)[key] = record
    full42 = predictions['42/full']
    for mode in ('empty', 'reverse_history', 'oracle_history'):
        value = predictions['42/'+mode]
        if not _same_bits(value[:, :p.CHUNK], full42[:, :p.CHUNK]):
            raise RuntimeError('A history intervention changed the history-free first chunk')
        if not use_prefix and not _same_bits(value, full42):
            raise RuntimeError('No-prefix rollout unexpectedly depended on history')
    curves = {key: reference[key] for key in ('clip_id', 'valid', 'times', 'channel_mask', 'b0', 'emotion_id', 'speaker_id')}
    curves.update(schema=p.SCHEMA, target=reference['motion'], predictions=predictions,
                  noise_seeds=list(p.r.SEEDS), decode_steps=steps, prefix_enabled=use_prefix,
                  oracle_prediction_keys=['42/oracle_history'],
                  prediction_scope='Only */full are three-seed deployment samples; 42/oracle_history is GT-conditioned diagnosis.')
    distribution = summarize(curves, reference['emotion_id'])
    interventions = distribution.pop('single_seed_interventions')
    oracle_scores = interventions.pop('42/oracle_history')
    report = {'schema': p.SCHEMA, 'clips': count, 'noise_seeds': list(p.r.SEEDS), 'decode_steps': steps,
              'evaluation_role': 'internal_development', 'modes': modes, 'distribution': distribution,
              'deployable_interventions_seed42': interventions,
              'ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY': {'modes': oracle_records, 'paired_metrics_vs_full': oracle_scores},
              'nonupper_exact': True, 'invalid_baseline_exact': True, 'first_chunk_history_interventions_equal': True,
              'no_prefix_history_interventions_equal': True if not use_prefix else None,
              'prefix_enabled': use_prefix, 'test_loaded': False, 'default_replaced': False,
              'deploy_history': 'Generated strictly preceding chunks; fixed audio-derived static reference; no query GT mean.',
              'limitations': ['Repeated internal development, not sealed test or evidence of unseen-identity generalization.',
                             'Motion teacher emotion readout is nonindependent; protected mouth coefficients do not certify lip sync.',
                             'Oracle actual-prefix continuity uses the supplied GT endpoint; its stitched trajectory is a different diagnostic.',
                             'No-prefix removes previous audio tokens together with motion tokens; paired effects concern prior token context.',
                             'Offline audio features may include future context; this is not causal audio generation.']}
    return report, curves


__all__ = ['evaluate_formal', 'actual_prefix_continuation', 'chunk_diagnostics']
