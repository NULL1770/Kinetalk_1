"""Train-only static neutral-mouth calibration from independent enrollment.

The target is each neutral query's mean (motion - current B0); the input is
the independent enrollment's mean residual for the same speaker. This fits
one gain/intercept per jaw/mouth coefficient, not a temporal motion model.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import torch

MOUTH = tuple(range(14, 41))
SCHEMA = 'reference_mouth_calibration_v1'


def _fit_provenance(provenance: dict[str, Any] | None, speakers: list[str]) -> dict[str, Any]:
    if not isinstance(provenance, dict) or provenance.get('split') != 'train':
        raise ValueError('Calibration requires explicit train-only independent-reference provenance')
    names = ('query_clip_ids', 'reference_clip_ids', 'query_sentence_ids', 'reference_sentence_ids')
    n = len(speakers)
    if any(not isinstance(provenance.get(key), list) or len(provenance[key]) != n for key in names):
        raise ValueError('Provenance must bind every query/reference clip and sentence')
    def strings(values):
        return isinstance(values, list) and bool(values) and all(isinstance(v, str) and v for v in values)
    if not strings(provenance['query_clip_ids']) or not strings(provenance['query_sentence_ids']):
        raise ValueError('Query clip/sentence IDs must be nonempty strings')
    if len(set(provenance['query_clip_ids'])) != n:
        raise ValueError('Repeated query clip would change calibration weighting')
    if any(not strings(part) for key in ('reference_clip_ids', 'reference_sentence_ids') for part in provenance[key]):
        raise ValueError('Every query requires explicit independent enrollment IDs')
    groups = defaultdict(list)
    for row, speaker in enumerate(speakers):
        groups[speaker].append(row)
    query_clips = set(provenance['query_clip_ids'])
    reference_clips = {clip for part in provenance['reference_clip_ids'] for clip in part}
    if query_clips & reference_clips:
        raise ValueError('Query and enrollment clips must be independent')
    for rows in groups.values():
        query_sentences = {provenance['query_sentence_ids'][i] for i in rows}
        ref_sentences = {s for i in rows for s in provenance['reference_sentence_ids'][i]}
        if query_sentences & ref_sentences:
            raise ValueError('Query and enrollment sentences must be independent within each speaker')
        first = rows[0]
        for i in rows:
            if (set(provenance['reference_clip_ids'][i]) != set(provenance['reference_clip_ids'][first])
                    or set(provenance['reference_sentence_ids'][i]) != set(provenance['reference_sentence_ids'][first])):
                raise ValueError('Each speaker requires one fixed independent enrollment pool')
            if len(provenance['reference_clip_ids'][i]) != len(set(provenance['reference_clip_ids'][i])):
                raise ValueError('Repeated enrollment clips would change reference pooling')
    return {key: provenance[key] for key in names} | {'split': 'train'}


def _observed_mask(value: torch.Tensor, mask: torch.Tensor | None, name: str) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(value, dtype=torch.bool)
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != value.shape or mask.device != value.device:
        raise ValueError(f'{name} must be Boolean and match the residual means')
    if not torch.isfinite(value[mask]).all():
        raise ValueError('Observed calibration residual means must be finite')
    return mask


@torch.no_grad()
def fit_reference_mouth_calibration(query_means: torch.Tensor, reference_means: torch.Tensor,
                                    speaker_ids, *, query_mask: torch.Tensor | None = None,
                                    reference_mask: torch.Tensor | None = None,
                                    ridge: float = 1e-3, bias_ridge: float = 1e-4,
                                    provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit 27 independent constrained ridge maps, with equal speaker weight.

    Minimize E_s E_clip[(y - gain*x - bias)^2] + ridge*(gain-1)^2
    + bias_ridge*bias^2. Each channel uses only observed x/y. The constrained
    gain is in [0,1]; its intercept is re-solved after applying that bound.
    Inputs must be mean residuals computed with the *same current B0*.
    Provenance is checked for cross-clip and within-speaker cross-sentence
    independence. Tensor values alone cannot establish that data contract.
    """
    if (not torch.is_tensor(query_means) or query_means.ndim != 2 or query_means.shape[-1] != 27
            or not query_means.is_floating_point() or not torch.is_tensor(reference_means)
            or reference_means.shape != query_means.shape or not reference_means.is_floating_point()
            or reference_means.device != query_means.device or len(query_means) < 2):
        raise ValueError('Calibration requires paired floating query/reference means [N,27]')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (ridge, bias_ridge)):
        raise ValueError('Predetermined gain and bias ridge strengths must be positive finite values')
    ids = speaker_ids.detach().cpu().tolist() if torch.is_tensor(speaker_ids) else list(speaker_ids)
    if len(ids) != len(query_means) or any(not isinstance(v, (str, int)) or isinstance(v, bool) or str(v) == '' for v in ids):
        raise ValueError('One valid speaker ID is required per neutral query')
    speakers = [str(v) for v in ids]
    binding = _fit_provenance(provenance, speakers)
    query_mask = _observed_mask(query_means, query_mask, 'query_mask')
    reference_mask = _observed_mask(reference_means, reference_mask, 'reference_mask')
    observed = (query_mask & reference_mask).detach().cpu()
    x = torch.where(reference_mask, reference_means, 0.).detach().cpu().double()
    y = torch.where(query_mask, query_means, 0.).detach().cpu().double()
    groups = defaultdict(list)
    for row, speaker in enumerate(speakers):
        groups[speaker].append(row)
    if len(groups) < 2:
        raise ValueError('Speaker-balanced calibration requires at least two training speakers')
    ref_mask = reference_mask.detach().cpu()
    for rows in groups.values():
        first = rows[0]
        for i in rows[1:]:
            if (not torch.equal(ref_mask[i], ref_mask[first])
                    or not torch.equal(x[i][ref_mask[i]], x[first][ref_mask[first]])):
                raise ValueError('Each speaker must use the same pooled independent reference means')
    weights = torch.zeros_like(x)
    speaker_counts = torch.zeros(27, dtype=torch.long)
    for rows in groups.values():
        count = observed[rows].sum(0)
        weights[rows] = observed[rows].double() / count.clamp_min(1)
        speaker_counts += count > 0
    if (speaker_counts < 2).any():
        raise ValueError('Every calibrated mouth channel needs observations from at least two training speakers')
    weights /= speaker_counts
    mx, my = (weights*x).sum(0), (weights*y).sum(0)
    xx, xy = (weights*x*x).sum(0), (weights*x*y).sum(0)
    gain = ((xy + ridge - mx*my/(1+bias_ridge)) /
            (xx + ridge - mx.square()/(1+bias_ridge))).clamp(0., 1.)
    bias = (my-gain*mx)/(1+bias_ridge)
    prediction = gain*x+bias
    return {'schema': SCHEMA, 'channels': list(MOUTH), 'gain': gain.tolist(), 'bias': bias.tolist(),
            'ridge': float(ridge), 'bias_ridge': float(bias_ridge), 'gain_prior': 1.,
            'observed_channels': [True]*27, 'fit_rows': len(y), 'fit_speakers': len(groups),
            'channel_observed_rows': observed.sum(0).tolist(), 'channel_observed_speakers': speaker_counts.tolist(),
            'speaker_balanced_fit_mse': float((weights*(prediction-y).square()).sum(0).mean()),
            'fit_scope': 'train_neutral_queries_and_independent_neutral_enrollment_only',
            'development_used_for_fit': False, 'test_used_for_fit': False,
            'input': 'independent_enrollment_mean_motion_minus_current_B0',
            'target': 'neutral_query_mean_motion_minus_current_B0',
            'weights': 'equal speaker, then equal observed neutral query per channel',
            'provenance': binding}


def calibration_parameters(calibration: dict[str, Any], *, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate fixed calibration; missing channels cannot become zero labels."""
    if (not isinstance(calibration, dict) or calibration.get('schema') != SCHEMA
            or calibration.get('channels') != list(MOUTH)
            or calibration.get('fit_scope') != 'train_neutral_queries_and_independent_neutral_enrollment_only'
            or calibration.get('development_used_for_fit') is not False
            or calibration.get('test_used_for_fit') is not False
            or calibration.get('observed_channels') != [True]*27):
        raise ValueError('Require bound train-only calibration for all 27 observed mouth channels')
    gain = torch.as_tensor(calibration.get('gain'), device=device, dtype=dtype)
    bias = torch.as_tensor(calibration.get('bias'), device=device, dtype=dtype)
    if (gain.shape != (27,) or bias.shape != (27,) or not torch.isfinite(gain).all()
            or not torch.isfinite(bias).all() or ((gain < 0)|(gain > 1)).any()):
        raise ValueError('Mouth gain/bias must be finite [27], with gain in [0,1]')
    return gain, bias


def apply_reference_mouth_calibration(reference_means: torch.Tensor, calibration: dict[str, Any], *,
                                      reference_mask: torch.Tensor) -> torch.Tensor:
    """Static deployment mapping; accepts independent reference data only."""
    if not torch.is_tensor(reference_means) or reference_means.shape[-1] != 27 or not reference_means.is_floating_point():
        raise ValueError('Reference residual means must have 27 mouth channels')
    mask = _observed_mask(reference_means, reference_mask, 'reference_mask')
    if not mask.all():
        raise ValueError('Cannot apply mouth calibration to absent reference mouth channels')
    gain, bias = calibration_parameters(calibration, device=reference_means.device, dtype=reference_means.dtype)
    return reference_means*gain+bias
