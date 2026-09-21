"""Train and gate an oracle regional-intensity receiver.

This is deliberately an *oracle-only* diagnostic.  The target brow/eye
activity field is extracted from observed validation/train motion and is never
used as a deployable audio condition.  A frozen Stage4 source prediction
provides temporal shape; :class:`RegionalIntensityGainAdapter` can only scale
that shape while preserving the source clip means.  The script therefore
answers a narrow question before an audio student is attempted:

    Can a protected receiver use a regional activity field to improve the
    frozen source prior on the same upper-face target?

If this gate fails, callers must not start audio-intensity training.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinetalk_b0.models.mean_preserving_upper import UPPER_INDICES, compose_mean_preserving_upper
from kinetalk_b0.models.regional_intensity_gain import (
    RegionalIntensityGainAdapter,
    regional_envelope,
)
from scripts.extract_emotion2vec_pilot import sha
from scripts.train_full_staged import NOT_UPPER, subset
from scripts.train_isolated_audio_state import load_context, old_prediction
from scripts.train_formal_predictable_projection import canonical_hash, save_checkpoint


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf8")


def _center(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid[..., None]
    clean = torch.where(mask, x, 0.)
    count = valid.sum(1, keepdim=True).to(x.dtype).clamp_min(1.)
    mean = clean.sum(1, keepdim=True) / count[..., None]
    return torch.where(mask, x - mean, 0.)


def _upper_valid(q: dict) -> torch.Tensor:
    valid = q["valid"]
    channels = q.get("channel_mask")
    if channels is None:
        return valid
    # ``channel_mask`` is clip-level [B,52] (see ``train_full_staged.obs``),
    # while ``valid`` is frame-level [B,T].  Expand the all-upper-channel
    # eligibility over the native frame clock explicitly; relying on
    # broadcasting would align [B] with T and fail for the usual T != B.
    upper = channels[:, list(UPPER_INDICES)].all(-1)
    result = valid & upper[:, None]
    if not result.any(1).all():
        raise ValueError("Every clip needs at least one frame with all nine upper channels")
    return result


@torch.no_grad()
def _cache_source(data: dict, system: nn.Module, audio: nn.Module,
                  identities: dict, split: str, device: str, seed: int) -> dict:
    """Build a frozen Stage4 source upper trajectory and oracle target field."""
    q = data["splits"][split]
    rows = []
    generator = torch.Generator().manual_seed(seed)
    for ids in torch.arange(len(q["valid"])).split(16):
        b = subset(q, ids, device)
        b["frozen_local"] = audio(b["audio_features"], b["valid"])["local"]
        # Keep one paired noise stream per clip.  It is re-created from the
        # clip index below so train/validation caches are deterministic.
        # Draw on CPU because the portable generator is CPU-backed, then move
        # the exact paired tensor to the selected execution device.
        noise = torch.randn((*b["valid"].shape, 52), generator=generator).to(device)
        base = old_prediction(system, b, identities, noise)
        upper = base[..., list(UPPER_INDICES)]
        target = b["motion"][..., list(UPPER_INDICES)].float()
        valid = _upper_valid({"valid": b["valid"], "channel_mask": b.get("channel_mask")})
        target = torch.where(valid[..., None], target, 0.)
        target_centered = _center(target, valid)
        target_env = regional_envelope(target_centered, valid)
        rows.append({
            "ids": ids.cpu(),
            "prior": upper.detach().cpu(),
            "target": target_centered.detach().cpu(),
            "target_envelope": target_env.detach().cpu(),
            "valid": valid.detach().cpu(),
            "base_full": base.detach().cpu(),
        })
    return {
        "prior": torch.cat([r["prior"] for r in rows]),
        "target": torch.cat([r["target"] for r in rows]),
        "target_envelope": torch.cat([r["target_envelope"] for r in rows]),
        "valid": torch.cat([r["valid"] for r in rows]),
        "base_full": torch.cat([r["base_full"] for r in rows]),
    }


def _masked_mse(x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid[..., None].expand_as(x)
    return (x[mask] - y[mask]).square().mean()


_REGION_NAMES = ("brow", "eye")


def _contiguous_runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return half-open true runs without joining native-clock mask gaps."""
    positions = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not positions:
        return []
    runs: list[tuple[int, int]] = []
    left = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            runs.append((left, previous + 1))
            left = position
        previous = position
    runs.append((left, previous + 1))
    return runs


def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> float | None:
    """Pearson correlation, returning None for a constant/short sequence."""
    if x.numel() < 2 or y.numel() < 2:
        return None
    x = x - x.mean(); y = y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    if not torch.isfinite(denominator) or float(denominator) <= 1e-12:
        return None
    value = (x * y).sum() / denominator
    return float(value.clamp(-1., 1.)) if torch.isfinite(value) else None


def _peak_lag(x: torch.Tensor, y: torch.Tensor, max_lag: int) -> int | None:
    """Return lag (in frames) of the best demeaned cross-correlation.

    Positive lag means ``x`` is delayed relative to ``y``.  This is computed
    separately on every contiguous valid run, so a padding gap cannot create
    an artificial peak.  ``None`` is returned for constant or too-short runs.
    """
    n = int(x.numel())
    if n < 3:
        return None
    best_lag: int | None = None
    best_score = -float("inf")
    for lag in range(-min(max_lag, n - 2), min(max_lag, n - 2) + 1):
        if lag >= 0:
            xx, yy = x[lag:], y[:n - lag]
        else:
            xx, yy = x[:n + lag], y[-lag:]
        score = _safe_corr(xx, yy)
        if score is not None and score > best_score:
            best_score, best_lag = score, lag
    return best_lag


def _temporal_diagnostics(pred: torch.Tensor, target: torch.Tensor,
                          valid: torch.Tensor) -> dict:
    """Compare regional envelopes by correlation, latency, and missing events."""
    correlations: list[list[float]] = [[], []]
    lags: list[list[int]] = [[], []]
    active_frames = [0, 0]
    missing_frames = [0, 0]
    missing_denominator = [0, 0]
    for batch_index in range(pred.shape[0]):
        for left, right in _contiguous_runs(valid[batch_index]):
            for region in range(2):
                x = pred[batch_index, left:right, region]
                y = target[batch_index, left:right, region]
                corr = _safe_corr(x, y)
                if corr is not None:
                    correlations[region].append(corr)
                lag = _peak_lag(x, y, max_lag=15)
                if lag is not None:
                    lags[region].append(lag)
                # ``active`` marks target events; a source value below ten
                # percent of its own observed range is treated as a missing
                # event.  The threshold is per clip/region and is recorded in
                # the output schema so this remains a diagnostic, not a tuned
                # training objective.
                target_threshold = max(1e-6, float(torch.quantile(y, .75)))
                source_threshold = max(1e-6, .10 * float(x.max()))
                active = y >= target_threshold
                denominator = int(active.sum())
                missing = int((active & (x <= source_threshold)).sum())
                active_frames[region] += denominator
                missing_frames[region] += missing
                missing_denominator[region] += denominator

    def _mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def _region(values: list[list[float | int]]) -> dict:
        return {name: _mean(values[index]) for index, name in enumerate(_REGION_NAMES)}

    missing = {
        name: (missing_frames[index] / missing_denominator[index]
               if missing_denominator[index] else None)
        for index, name in enumerate(_REGION_NAMES)
    }
    return {
        "correlation": _region(correlations),
        "correlation_macro": _mean([v for values in correlations for v in values]),
        "peak_lag_frames": _region(lags),
        "peak_abs_lag_frames_macro": _mean([abs(v) for values in lags for v in values]),
        "prior_missing_event_coverage": missing,
        "prior_missing_event_coverage_macro": (
            sum(missing_frames) / sum(missing_denominator)
            if sum(missing_denominator) else None),
        "active_event_frames": dict(zip(_REGION_NAMES, active_frames)),
        "peak_lag_definition": "positive means prediction is delayed; max_lag=15 native frames",
        "missing_event_definition": "target >= per-clip q75 and source <= max(1e-6, 0.1*source_max)",
    }


@torch.no_grad()
def _score(adapter: nn.Module, cache: dict, device: str) -> tuple[dict, torch.Tensor]:
    prior = cache["prior"].to(device)
    target = cache["target"].to(device)
    target_env = cache["target_envelope"].to(device)
    valid = cache["valid"].to(device)
    out = adapter(prior, target_env, valid)
    static = _center(prior, valid)
    dynamic = _center(out["dynamic_upper"], valid)
    source_env = regional_envelope(static, valid)
    dynamic_env = regional_envelope(dynamic, valid)
    static_mse = _masked_mse(static, target, valid)
    oracle_mse = _masked_mse(dynamic, target, valid)
    static_env = _masked_mse(source_env, target_env, valid)
    oracle_env = _masked_mse(dynamic_env, target_env, valid)
    static_temporal = _temporal_diagnostics(source_env, target_env, valid)
    oracle_temporal = _temporal_diagnostics(dynamic_env, target_env, valid)
    static_corr = static_temporal["correlation_macro"]
    oracle_corr = oracle_temporal["correlation_macro"]
    static_lag = static_temporal["peak_abs_lag_frames_macro"]
    oracle_lag = oracle_temporal["peak_abs_lag_frames_macro"]
    static_missing = static_temporal["prior_missing_event_coverage_macro"]
    oracle_missing = oracle_temporal["prior_missing_event_coverage_macro"]
    # The receiver gate is deliberately stricter than a lower MSE: an oracle
    # that only amplifies a mistimed prior must not unlock audio training.
    # ``None`` means the clip set was constant/too short and therefore fails
    # the temporal gate rather than being treated as a pass.
    temporal_gate = {
        "envelope_correlation_improved": bool(
            oracle_corr is not None and static_corr is not None
            and oracle_corr >= static_corr + .02),
        "peak_lag_not_worse": bool(
            oracle_lag is not None and static_lag is not None
            and oracle_lag <= static_lag + 1.0),
        "missing_event_coverage_not_worse": bool(
            oracle_missing is not None and static_missing is not None
            and oracle_missing <= static_missing),
    }
    temporal_passed = all(temporal_gate.values())
    count = valid.sum(1, keepdim=True).to(prior.dtype)
    prior_mean = (torch.where(valid[..., None], prior, 0.).sum(1) / count).cpu()
    output_mean = (torch.where(valid[..., None], out["dynamic_upper"], 0.).sum(1) / count).cpu()
    result = {
        "clips": int(len(valid)),
        "centered_upper_mse": {"static": float(static_mse), "oracle": float(oracle_mse)},
        "regional_envelope_mse": {"static": float(static_env), "oracle": float(oracle_env)},
        "temporal_diagnostics": {"static": static_temporal, "oracle": oracle_temporal},
        "temporal_gate": temporal_gate,
        "temporal_receiver_passed": temporal_passed,
        "relative_improvement": {
            "centered_upper": float((static_mse - oracle_mse) / static_mse.clamp_min(1e-12)),
            "envelope": float((static_env - oracle_env) / static_env.clamp_min(1e-12)),
        },
        "max_mean_drift": float((output_mean - prior_mean).abs().max()),
        "gain": {
            "min": float(out["gain"][valid].min()),
            "max": float(out["gain"][valid].max()),
            "mean": float(out["gain"][valid].mean()),
        },
        "receiver_passed": bool(oracle_mse < static_mse and oracle_env < static_env
                                 and temporal_passed),
    }
    return result, out["dynamic_upper"].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.output.exists():
        raise FileExistsError("Fresh oracle output directory required")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data, system, audio, identities, _ = load_context(args.data, args.source, args.device, args.seed)
    if args.smoke:
        for role, q in data["splits"].items():
            data["splits"][role] = subset(q, torch.arange(min(32, len(q["valid"]))), "cpu")
    frozen = {"system": sha(args.source), "script": sha(Path(__file__))}
    train_cache = _cache_source(data, system, audio, identities, "train", args.device, args.seed)
    val_cache = _cache_source(data, system, audio, identities, "validation", args.device, args.seed + 1)
    # The source trajectory and oracle field are cached on CPU; adapter itself
    # is tiny and receives no identity/global/audio gradients.
    adapter = RegionalIntensityGainAdapter(hidden=32, max_gain=3.).to(args.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-3, weight_decay=1e-4)
    protocol = {
        "schema": "regional_intensity_oracle_v1",
        "oracle_only": True, "audio_student_started": False,
        "data_manifest_sha256": data["provenance"]["manifest_sha256"],
        "source_sha256": sha(args.source), "script_sha256": sha(Path(__file__)),
        "epochs": 1 if args.smoke else args.epochs, "batch_size": args.batch_size,
        "seed": args.seed, "target": "centered upper9 regional RMS envelope from observed motion",
        "receiver": "mean-preserving brow5/eye4 gain over frozen Stage4 source trajectory",
        "test_loaded": False, "default_replaced": False,
        "temporal_gate": {
            "required": ["envelope_correlation", "peak_lag", "prior_missing_event_coverage"],
            "description": "Oracle must improve envelope correlation, avoid worsening peak lag, and not miss more target events than the frozen prior before audio OOF training.",
        },
    }
    _save_json(args.output / "protocol.json", protocol)
    train_n = len(train_cache["valid"])
    budget = 1 if args.smoke else args.epochs
    for epoch in range(budget):
        adapter.train(); losses = []
        order = torch.randperm(train_n, generator=torch.Generator().manual_seed(args.seed + epoch))
        for ids in order.split(args.batch_size):
            prior = train_cache["prior"][ids].to(args.device)
            target = train_cache["target"][ids].to(args.device)
            envelope = train_cache["target_envelope"][ids].to(args.device)
            valid = train_cache["valid"][ids].to(args.device)
            out = adapter(prior, envelope, valid)
            pred = _center(out["dynamic_upper"], valid)
            loss = _masked_mse(pred, target, valid)
            # A small envelope term keeps the stated receiver objective
            # visible while the coefficient fit remains the main criterion.
            loss = loss + .25 * _masked_mse(regional_envelope(pred, valid), envelope, valid)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        _save_json(args.output / f"epoch{epoch + 1:03d}.json", {
            "status": "training", "epoch": epoch + 1, "epochs": budget,
            "loss": float(np.mean(losses)), "test_loaded": False,
        })

    adapter.eval(); train_score, train_dynamic = _score(adapter, train_cache, args.device)
    val_score, val_dynamic = _score(adapter, val_cache, args.device)
    # Protection checks use the full frozen Stage4 source prediction.  The
    # receiver only replaces upper9, so the other 43 channels must be exact.
    val_base = val_cache["base_full"]
    val_valid = val_cache["valid"]
    # Restore the frozen Stage4 upper mean around the oracle's centered
    # trajectory.  Assigning the centered field directly would create an
    # artificial mean drift and fail the very protection gate this diagnostic
    # is meant to verify.
    composed = compose_mean_preserving_upper(
        val_base, val_dynamic, val_valid
    )
    base_subset = torch.where(val_valid[..., None], val_base, 0.)
    composed_subset = torch.where(val_valid[..., None], composed, 0.)
    nonupper_exact = bool(torch.equal(composed[..., list(NOT_UPPER)].view(torch.int32),
                                      val_base[..., list(NOT_UPPER)].view(torch.int32)))
    mean_drift = float((composed_subset[..., list(UPPER_INDICES)].mean(1)
                        - base_subset[..., list(UPPER_INDICES)].mean(1)).abs().max())
    result = {
        "oracle_only": True, "audio_student_started": False,
        "train": train_score, "validation": val_score,
        "protection": {"nonupper43_exact": nonupper_exact,
                        "upper_mean_preserved": mean_drift < 1e-5,
                        "max_upper_mean_drift": mean_drift},
        "receiver_passed": bool(val_score["receiver_passed"] and nonupper_exact and mean_drift < 1e-5),
        "test_loaded": False, "default_replaced": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _save_json(args.output / "evaluation.json", result)
    save_checkpoint(args.output / "final.pt", {
        "model": adapter.state_dict(), "config": {"hidden": 32, "max_gain": 3.},
        "protocol": protocol, "protocol_sha256": canonical_hash(protocol),
        "oracle_only": True, "test_loaded": False,
    })
    _save_json(args.output / "status.json", {
        "status": "complete", "receiver_passed": result["receiver_passed"],
        "oracle_only": True, "audio_student_started": False,
        "final_sha256": sha(args.output / "final.pt"),
        "elapsed_seconds": result["elapsed_seconds"], "test_loaded": False,
    })
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
