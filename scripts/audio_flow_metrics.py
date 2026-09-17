"""Finite-ensemble diagnostics for audio-conditioned motion generation.

All samples for a clip share audio/identity and use a predeclared noise-seed list.
Scores compare distributions with one observed target per clip; they do not need
multiple GT recordings of identical audio. Descriptive finite-K scores are not
proof of calibration, natural motion, or perceptual quality. No best-of-K.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_multiseed_stochasticity import center, decompose
from scripts.audit_teacher_schedule_probe import GROUPS


def _validate(predictions, target, observed):
    if target.ndim != 3 or predictions.ndim != 4 or predictions.shape[1:] != target.shape:
        raise ValueError("Require predictions [K,N,T,D] and target [N,T,D]")
    if predictions.shape[0] < 3:
        raise ValueError("At least three predeclared samples are required")
    if observed.shape != target.shape or observed.dtype != torch.bool:
        raise ValueError("Observed must be a Boolean mask matching target")
    if not observed.flatten(1).any(1).all():
        raise ValueError("Every clip needs observed entries")
    if not torch.isfinite(target[observed]).all() or not torch.isfinite(predictions[:, observed]).all():
        raise ValueError("Observed target/sample entries must be finite")


def _summary(rows):
    if not rows:
        return {"clips": 0, "mean": None, "per_clip": []}
    values = torch.stack(rows).double()
    return {"clips": len(rows), "mean": float(values.mean()), "per_clip": values.tolist()}


def trajectory_energy_score(predictions, target, observed):
    """Whole observed-trajectory Euclidean ES, divided by sqrt(n_observed).

    The empirical ES scores the finite empirical forecast distribution. Fair ES
    uses all off-diagonal seed pairs and is unbiased for a population ES if
    samples are iid from that forecast. The divisor is target-independent and
    fixed by each clip's observation layout. Lower is better.
    """
    _validate(predictions, target, observed)
    k = len(predictions)
    first, spread, empirical, fair = [], [], [], []
    for clip in range(len(target)):
        mask = observed[clip]
        samples = predictions[:, clip, :, :][:, mask].double()
        truth = target[clip][mask].double()
        n = samples.shape[1]
        distance = (samples - truth).square().mean(-1).sqrt().mean()
        # One scalar distance per complete trajectory; not averaged frame ES.
        pair_sum = torch.pdist(samples, p=2).sum() / n ** .5
        empirical_spread = pair_sum / (k * k)
        fair_spread = pair_sum / (k * (k - 1))
        first.append(distance); spread.append(fair_spread)
        empirical.append(distance - empirical_spread)
        fair.append(distance - fair_spread)
    return {"fair": _summary(fair), "empirical": _summary(empirical),
            "target_distance": _summary(first), "fair_pair_spread_term": _summary(spread),
            "normalization": "Per-clip complete observed trajectory L2 / sqrt(observed elements); equal clip mean",
            "fair_pair_denominator": k * (k - 1), "sample_count": k}


def adjacent_variogram_score(predictions, target, observed, *, power=.5):
    """Adjacent-time variogram score per channel, excluding invalid gaps.

    VS = mean_pairs (abs(delta_GT)**p - E_s abs(delta_Xs)**p)**2.
    Fair correction subtracts sample variance/K, removing Monte Carlo bias of
    squared mean error under iid forecast samples. A finite fair estimate may
    be negative. This selected-pair score is proper but not strictly proper:
    it cannot assess all aspects of the joint trajectory distribution.
    """
    _validate(predictions, target, observed)
    if not 0 < power < 2:
        raise ValueError("Variogram power must lie in (0, 2)")
    k = len(predictions)
    pair_mask = observed[:, 1:] & observed[:, :-1]
    empirical, fair, corrections, clip_ids, pair_counts = [], [], [], [], []
    for clip in range(len(target)):
        mask = pair_mask[clip]
        if not mask.any():
            continue
        # Index after differencing; invalid inputs are never included in score.
        sample_delta = torch.diff(predictions[:, clip].double(), dim=1)[:, mask].abs().pow(power)
        target_delta = torch.diff(target[clip].double(), dim=0)[mask].abs().pow(power)
        error = (sample_delta.mean(0) - target_delta).square().mean()
        correction = sample_delta.var(0, unbiased=True).mean() / k
        empirical.append(error); corrections.append(correction); fair.append(error - correction)
        clip_ids.append(clip); pair_counts.append(int(mask.sum()))
    return {"power": power, "fair": _summary(fair), "empirical": _summary(empirical),
            "finite_k_correction": _summary(corrections), "included_clip_indices": clip_ids,
            "per_clip_observed_pairs": pair_counts, "sample_count": k,
            "pair_rule": "Same channel at adjacent valid frame indices; invalid gaps excluded; equal clip means"}


def velocity_statistics(predictions, target, observed, times):
    """Physical per-second velocities on adjacent observed frames only."""
    _validate(predictions, target, observed)
    if times.shape != target.shape[:2]:
        raise ValueError("Times must match [clip,time]")
    pair_mask = observed[:, 1:] & observed[:, :-1]
    dt = torch.diff(times.double(), dim=1)
    valid_dt = pair_mask.any(-1)
    if not torch.isfinite(dt[valid_dt]).all() or (dt[valid_dt] <= 0).any():
        raise ValueError("Observed adjacent times must be finite and increasing")
    if not pair_mask.any():
        return {"observed_pairs": 0, "decomposition": None}
    # Clamp only unobserved dt; positivity of every scored pair checked above.
    safe_dt = torch.where(valid_dt, dt, torch.ones_like(dt))
    pvel = torch.diff(predictions.double(), dim=2) / safe_dt[None, :, :, None]
    tvel = torch.diff(target.double(), dim=1) / safe_dt[:, :, None]
    return {"observed_pairs": int(pair_mask.sum()), "decomposition": decompose(pvel, tvel, pair_mask)}


def paired_condition_response(left, right, observed):
    """Compare conditions at matched seed; do not confuse with seed variance."""
    if left.shape != right.shape or left.ndim != 4 or left.shape[1:] != observed.shape:
        raise ValueError("Paired predictions/mask shapes differ")
    if not observed.any():
        raise ValueError("Paired response needs observations")
    delta = (left - right)[:, observed].double()
    if not torch.isfinite(delta).all():
        raise ValueError("Observed paired predictions must be finite")
    mean = delta.mean(0)
    return {"paired_response_mse": float(delta.square().mean()),
            "paired_response_rms": float(delta.square().mean().sqrt()),
            "ensemble_mean_response_rms": float(mean.square().mean().sqrt()),
            "response_seed_std": float((delta - mean).square().mean().sqrt())}


def audit_audio_flow_samples(curves, reference, *, expected_seeds: Sequence[int], variogram_power=.5):
    """Aggregate fixed-seed curves using the existing motion audit contract.

    ``curves`` contains ``noise_seeds``, ``decode_steps``, and
    ``motion[str(seed)][mode]`` tensors of shape [N,T,52]. Modes must include
    full and have the same names for every seed. ``reference`` has q.motion,
    q.valid, q.channel_mask, q.emotion_id, q.times, base.b0, identity.baseline.
    All scoring is CPU float64. Source/protocol hashing belongs to the runner.
    """
    seeds = list(expected_seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("Predeclare at least three distinct integer seeds")
    if curves.get("noise_seeds") != seeds or set(curves.get("motion", {})) != set(map(str, seeds)):
        raise ValueError("Saved seeds/order differ from predeclared seeds")
    if type(curves.get("decode_steps")) is not int or curves["decode_steps"] < 1:
        raise ValueError("Positive decode_steps required")
    modes = tuple(curves["motion"][str(seeds[0])])
    if "full" not in modes or any(set(curves["motion"][str(seed)]) != set(modes) for seed in seeds):
        raise ValueError("Need full mode and matching mode names across seeds")
    q = reference["q"]
    target = q["motion"].detach().cpu().double()
    valid, channels = q["valid"].detach().cpu(), q["channel_mask"].detach().cpu()
    if valid.dtype != torch.bool or channels.dtype != torch.bool:
        raise ValueError("Reference observation masks must be Boolean")
    if target.ndim != 3 or valid.shape != target.shape[:2] or channels.shape != (len(target), target.shape[-1]):
        raise ValueError("Reference shapes differ")
    observed = valid[..., None] & channels[:, None, :]
    baseline = reference["base"]["b0"].detach().cpu().double() + reference["identity"]["baseline"].detach().cpu().double()[:, None]
    if baseline.shape != target.shape or not torch.isfinite(baseline[observed]).all():
        raise ValueError("Reference baseline differs or is nonfinite")
    times = q["times"].detach().cpu().double()
    labels = q["emotion_id"].detach().cpu()
    if labels.shape != (len(target),):
        raise ValueError("Emotion labels must match clips")
    populations = {"all": torch.arange(len(target)), "neutral": (labels == 0).nonzero(as_tuple=True)[0],
                   "nonneutral": (labels != 0).nonzero(as_tuple=True)[0]}
    if "speaker_id" in q:
        speakers = q["speaker_id"].detach().cpu()
        if speakers.shape != labels.shape:
            raise ValueError("Speaker IDs must match clips")
        for speaker in speakers.unique().tolist():
            populations[f"speaker_{speaker}/nonneutral"] = ((speakers == speaker) & (labels != 0)).nonzero(as_tuple=True)[0]
    populations = {name: ids for name, ids in populations.items() if len(ids)}
    result = {"schema": "audio_flow_distribution_metrics_v1", "noise_seeds": seeds, "sample_count": len(seeds),
              "decode_steps": curves["decode_steps"], "modes": list(modes),
              "population_clip_indices": {name: ids.tolist() for name, ids in populations.items()},
              "scope": "Descriptive finite-K samples, one GT trajectory per clip. No best-of-K or calibration claim. Lower ES/VS is better. Fair estimators assume iid model samples; VS is not strictly proper.",
              "scores": {}, "paired_condition_response": {}}
    centered_target = center(target - baseline, observed)
    for kind, truth in (("raw_motion", target), ("centered_residual", centered_target)):
        result["scores"][kind] = {}
        mode_fields = {}
        for mode in modes:
            stack = torch.stack([curves["motion"][str(seed)][mode].detach().cpu().double() for seed in seeds])
            _validate(stack, target, observed)
            if kind == "centered_residual":
                stack = center(stack - baseline, observed.expand_as(stack))
            mode_fields[mode] = stack
            result["scores"][kind][mode] = {}
            for population, ids in populations.items():
                rows = {}
                for group, cc in GROUPS.items():
                    if max(cc) >= target.shape[-1]:
                        raise ValueError("Reference lacks required motion group channels")
                    mask = observed[ids][..., cc]
                    # A group can be entirely unavailable for some datasets.
                    usable = mask.flatten(1).any(1)
                    if not usable.all():
                        raise ValueError(f"Every scored clip must observe {group}")
                    pred, actual = stack[:, ids][..., cc], truth[ids][..., cc]
                    rows[group] = {"error_decomposition": decompose(pred, actual, mask),
                                   "trajectory_energy_score": trajectory_energy_score(pred, actual, mask),
                                   "adjacent_variogram_score": adjacent_variogram_score(pred, actual, mask, power=variogram_power),
                                   "velocity": velocity_statistics(pred, actual, mask, times[ids])}
                result["scores"][kind][mode][population] = rows
        result["paired_condition_response"][kind] = {}
        for control in ("zero", "reverse"):
            if control not in modes:
                continue
            rows = {}
            for population, ids in populations.items():
                rows[population] = {group: paired_condition_response(mode_fields["full"][:, ids][..., cc],
                    mode_fields[control][:, ids][..., cc], observed[ids][..., cc]) for group, cc in GROUPS.items()}
            result["paired_condition_response"][kind]["full_minus_" + control] = rows
    return result
