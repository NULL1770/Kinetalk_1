"""Fixed-seed evaluation for empty chunks, teacher-trained chunks, and whole windows.

The caller supplies the complete evaluation subset and matching baseline metadata.
This evaluator neither truncates clips nor selects checkpoints or samples.
"""
from __future__ import annotations

import time

import torch

from scripts import train_prefix_upper as p
from scripts.evaluate_prefix_formal import (
    _same_bits, _validate, actual_prefix_continuation, chunk_diagnostics,
)
from scripts.audit_temporal_repair import summarize
from scripts.train_centered_temporal_prior import temporal_stats
from kinetalk_b0.models.slow_state_affect import compose_upper_face


SCHEMA = 'context_mechanism_v1'
ARMS = ('chunk_empty', 'chunk_teacher', 'whole')
ORACLE_MODES = ('oracle_history', 'oracle_reverse_history')
MODES = ('full', 'empty', 'reverse_history', 'static', 'reverse') + ORACLE_MODES


def supplied_history_endpoints(pred, target, valid, mode):
    """Return raw-coordinate endpoints actually presented as known history.

    Only immediately adjacent valid boundary pairs can be scored downstream.
    Reverse-history decoding reverses the observed history slots in place;
    its last known slot therefore contains the first valid value in the prior
    eight-frame window. No missing frame is filled and no pair crosses a gap.
    """
    if mode not in MODES:
        raise ValueError('Unknown history endpoint mode')
    source = target if mode in ORACLE_MODES else pred
    if (source.shape != pred.shape or source.shape != target.shape or source.ndim != 3
            or source.shape[-1] != 52 or valid.shape != source.shape[:2] or valid.dtype != torch.bool):
        raise ValueError('Endpoint curves must be matching [B,T,52] with Boolean validity')
    supplied = source.clone()
    if mode not in ('reverse_history', 'oracle_reverse_history'):
        return supplied
    for start in range(p.CHUNK, valid.shape[1], p.CHUNK):
        eligible = valid[:, start-1] & valid[:, start]
        ids = eligible.nonzero(as_tuple=True)[0]
        if not len(ids):
            continue
        first = max(0, start-p.HISTORY)
        history = valid[ids, first:start]
        # Adjacent predecessor validity ensures at least one observed slot.
        earliest = history.to(torch.int64).argmax(1) + first
        supplied[ids, start-1] = source[ids, earliest]
    return supplied


@torch.no_grad()
def evaluate(upper, system, q, identities, scales, bases, args, steps, arm, rollout_fn):
    """Evaluate all supplied clips with one globally indexed noise array per seed.

    Only ``chunk_teacher`` uses past motion during decoding. Its deployable
    modes receive generated history; GT is passed explicitly only to its
    separate oracle diagnostic. For other arms the history interventions must
    be bit-identical and even the nominal oracle receives no target tensor.
    """
    started = time.monotonic()
    if arm not in ARMS:
        raise ValueError('Unknown context-mechanism arm')
    if not callable(rollout_fn):
        raise ValueError('Explicit rollout callable required')
    use_prefix = arm == 'chunk_teacher'
    reference = _validate(q, bases, scales, args, steps, use_prefix)
    if getattr(upper, 'training', False) or getattr(system, 'training', False):
        raise ValueError('Context evaluator requires both models already in eval mode')
    count, frames = reference['valid'].shape
    indices = torch.arange(count)
    noises = {seed: torch.randn(count, frames, 9, generator=torch.Generator().manual_seed(seed))
              for seed in p.r.SEEDS}
    predictions, records, oracle_records, durations = {}, {}, {}, {}
    for seed in p.r.SEEDS:
        for mode in MODES if seed == 42 else ('full',):
            mode_started = time.monotonic()
            outputs, correct = [], 0
            oracle_active = use_prefix and mode in ORACLE_MODES
            for ix in indices.split(args.batch_size):
                b = p.r.subset(q, ix, args.device)
                identity = p.r.batch_identity(identities, b)
                native = b['prefix_local']
                fixed_scales = scales.to(native)
                kwargs = {'oracle_target': p.h.normalized_target(b, fixed_scales)} if oracle_active else {}
                # Keep target motion and labels out of the rollout interface;
                # teacher readout/scoring below retains the complete reference.
                conditions = {key: b[key] for key in (
                    'valid', 'h0', 'audio_global', 'audio_intensity', 'static_upper',
                    'clip_id', 'speaker_id', 'times')}
                dynamic = rollout_fn(upper, conditions, identity, native, noises[seed][ix].to(native),
                                     steps=steps, mode=mode, arm=arm, **kwargs)
                if not torch.is_tensor(dynamic) or dynamic.shape != (len(ix), frames, 9):
                    raise ValueError('Rollout must return the complete [B,T,9] dynamic trajectory')
                baseline = bases['predictions'][f'{seed}/base'][ix].to(device=args.device)
                pred = compose_upper_face(baseline, b['static_upper'][:, None] + dynamic*fixed_scales, b['valid'])
                if (not _same_bits(pred[..., list(p.r.NOT_UPPER)], baseline[..., list(p.r.NOT_UPPER)])
                        or not _same_bits(pred[~b['valid']], baseline[~b['valid']])):
                    raise RuntimeError('Frozen 43-channel or invalid-frame baseline protection failed')
                teacher = system.encode_motion(
                    torch.where(p.r.obs(b), pred-b['b0']-identity['baseline'][:, None], 0.), b['valid'])
                logits = teacher['emotion_logits']
                if logits.ndim != 2 or len(logits) != len(ix) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid generated-motion teacher logits')
                correct += int((logits.argmax(-1) == b['emotion_id']).sum())
                outputs.append(pred.cpu())
            key = f'{seed}/{mode}'
            prediction = torch.cat(outputs)
            predictions[key] = prediction
            scored_prefix = use_prefix and mode != 'empty'
            reversed_history = mode in ('reverse_history', 'oracle_reverse_history')
            supplied_past = supplied_history_endpoints(prediction, reference['motion'], reference['valid'], mode) if scored_prefix else None
            if arm == 'whole':
                reason = 'Single whole-window generation has no supplied motion prefix and no decoder stitch at the 16-frame diagnostic grid.'
            elif not use_prefix or mode == 'empty':
                reason = 'No motion prefix was supplied; adjacent boundary displacement is descriptive output only.'
            elif reversed_history:
                reason = 'Current first frame minus the actual last known value after reversing valid prior-eight-frame motion slots; compare only adjacent valid native boundary pairs.'
            else:
                reason = 'Current first frame minus the immediately preceding endpoint actually supplied as prefix.'
            record = {
                'populations': p.r.populations(prediction, reference),
                'temporal': temporal_stats(prediction, reference['motion'], reference['valid']),
                'chunk_diagnostics': chunk_diagnostics(prediction, reference['motion'], reference['valid'], reference['channel_mask']),
                'boundaries': p.h.boundary_report(prediction, reference['motion'], reference['valid']),
                'boundary_definition': ('Fixed 16-frame diagnostic grid within one whole-window decode; these are not decoder seams.'
                                        if arm == 'whole' else 'Decoder stitch at native indices 15->16,31->32,...; coefficient/frame.'),
                'boundaries_are_decoder_stitches': arm != 'whole',
                'actual_prefix_continuation': {
                    'scored': scored_prefix,
                    'source': (('tracked_reference_GT' if oracle_active else 'generated_past') + ('_reversed' if reversed_history else '')) if scored_prefix else None,
                    'metrics': actual_prefix_continuation(prediction, reference['motion'], reference['valid'],
                                                         reference['channel_mask'], supplied_past) if scored_prefix else None,
                    'note': reason},
                'same_gt_endpoint_output_diagnostic': {
                    'metrics': actual_prefix_continuation(prediction, reference['motion'], reference['valid'],
                                                         reference['channel_mask'], reference['motion']),
                    'GT_was_supplied_as_history': oracle_active,
                    'note': 'Common chronological reference endpoint for output comparisons only. GT is an input exclusively for chunk_teacher oracle modes; reversed GT history does not end at the chronological preceding frame.'},
                'generated_emotion_accuracy_nonindependent': correct/count,
                'oracle_mode_requested': mode in ORACLE_MODES,
                'oracle_target_history': oracle_active,
                'history_intervention_effective_by_design': use_prefix,
                'prefix_enabled': use_prefix,
            }
            (oracle_records if mode in ORACLE_MODES else records)[key] = record
            durations[key] = time.monotonic()-mode_started
    full42 = predictions['42/full']
    for mode in ('empty', 'reverse_history') + ORACLE_MODES:
        value = predictions['42/'+mode]
        if not _same_bits(value[:, :p.CHUNK], full42[:, :p.CHUNK]):
            raise RuntimeError('History intervention changed the history-free first chunk')
        if not use_prefix and not _same_bits(value, full42):
            raise RuntimeError('History-free arm unexpectedly depended on history')
    curves = {key: reference[key] for key in ('clip_id', 'valid', 'times', 'channel_mask', 'b0', 'emotion_id', 'speaker_id')}
    curves.update(schema=SCHEMA, arm=arm, target=reference['motion'], predictions=predictions,
                  noise_seeds=list(p.r.SEEDS), decode_steps=steps, prefix_enabled=use_prefix,
                  oracle_prediction_keys=['42/'+mode for mode in ORACLE_MODES], oracle_has_target_input=use_prefix,
                  prediction_scope='Only */full are three-seed deployment samples. Nominal oracle is separate; only chunk_teacher receives GT history.')
    distribution = summarize(curves, reference['emotion_id'])
    interventions = distribution.pop('single_seed_interventions')
    oracle_scores = {'42/'+mode: interventions.pop('42/'+mode) for mode in ORACLE_MODES}
    report = {
        'schema': SCHEMA, 'arm': arm, 'clips': count, 'noise_seeds': list(p.r.SEEDS), 'decode_steps': steps,
        'smoke': bool(getattr(args, 'smoke', False)), 'evaluation_role': 'internal_development',
        'modes': records, 'distribution': distribution, 'deployable_interventions_seed42': interventions,
        'ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY': {
            'active_target_conditioning': use_prefix, 'modes': oracle_records, 'paired_metrics_vs_full': oracle_scores,
            'note': 'Oracle is a GT-conditioned diagnostic for chunk_teacher; inactive history control for chunk_empty/whole.'},
        'nonupper_exact': True, 'invalid_baseline_exact': True, 'first_chunk_history_interventions_equal': True,
        'history_free_arm_interventions_equal': True if not use_prefix else None,
        'prefix_enabled': use_prefix, 'test_loaded': False, 'default_replaced': False,
        'noise_indexing': 'Three full supplied-subset [N,T,9] arrays drawn before batching; all modes share each seed array.',
        'evaluation_seconds': time.monotonic()-started, 'mode_seconds_including_scoring': durations,
        'limitations': [
            'Repeated internal development, not sealed test or evidence of unseen-identity generalization.',
            'Motion teacher emotion readout is nonindependent; protected mouth coefficients do not certify perceptual lip sync or identity.',
            'Whole-window diagnostics at multiples of sixteen are comparable frame positions, not actual decoder stitch boundaries.',
            'Chunk_empty drops previous motion and acoustic tokens; chunk_teacher vs chunk_empty does not isolate motion values alone.',
            'Oracle actual-prefix endpoint and stitched output displacement are distinct; oracle results never enter deployment aggregation.',
            'Offline audio features and whole-window attention use context; this is not causal audio generation.',
        ],
    }
    return report, curves


__all__ = ['evaluate', 'supplied_history_endpoints', 'SCHEMA', 'ARMS']
