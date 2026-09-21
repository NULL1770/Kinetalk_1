"""Probe an explicit signed upper-face receiver before audio prediction.

This is a diagnostic ceiling, not a deployable audio model.  It keeps the
frozen Stage4 full-face output and its upper-face mean, then replaces only the
centered upper trajectory with a four-state signed regional basis
(``raise/down/squint/wide``).  The observed motion supplies the state for the
oracle arm; zero, constant, reverse and permutation controls use the same
clips, noise and masks.  Audio training is intentionally absent from this
script.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinetalk_b0.models.mean_preserving_upper import UPPER_INDICES, compose_mean_preserving_upper
from kinetalk_b0.models.regional_intensity_gain import regional_envelope
from kinetalk_b0.models.slow_state_affect import UPPER_STATE_GROUPS
from scripts.train_full_staged import NOT_UPPER, subset
from scripts.train_isolated_audio_state import load_context, old_prediction
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_regional_intensity_oracle import (
    _center, _masked_mse, _temporal_diagnostics, _save_json,
)
from scripts.extract_emotion2vec_pilot import sha


def _state_from_centered(upper: torch.Tensor, valid: torch.Tensor,
                         scales: torch.Tensor) -> torch.Tensor:
    """Project centered raw UPPER9 into four signed, scale-normalized states."""
    groups = UPPER_STATE_GROUPS
    states = []
    for group in groups:
        idx = torch.as_tensor(group, device=upper.device)
        value = upper[..., idx] / scales[idx]
        states.append(value.mean(-1))
    state = torch.stack(states, -1)
    return torch.where(valid[..., None], state, 0.)


def _lift_state(state: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Lift four signed states into UPPER9 raw coefficient order."""
    mapping = (1, 1, 0, 0, 0, 2, 3, 2, 3)
    idx = torch.as_tensor(mapping, device=state.device)
    return state[..., idx] * scales


def _reverse(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    out = torch.where(valid[..., None], value, 0.).clone()
    for row in range(len(out)):
        ids = valid[row].nonzero(as_tuple=True)[0]
        out[row, ids] = value[row, ids.flip(0)]
    return out


def _constant(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """A deterministic nonzero, zero-mean control with the same support."""
    clean = torch.where(valid[..., None], value, 0.)
    count = valid.sum(1, keepdim=True).to(value.dtype).clamp_min(1.)
    level = clean.square().sum(1, keepdim=True).div(count[..., None]).sqrt()
    clock = torch.arange(value.shape[1], device=value.device, dtype=value.dtype)[None, :, None]
    centered_clock = torch.where(valid[..., None], clock, 0.)
    centered_clock = centered_clock - centered_clock.sum(1, keepdim=True) / count[..., None]
    return torch.where(valid[..., None], centered_clock / centered_clock.abs().amax(1, keepdim=True).clamp_min(1.) * level, 0.)


@torch.no_grad()
def _cache(data: dict, system, audio, identities, device: str, seed: int) -> dict:
    q = data["splits"]["validation"]
    rows = []
    generator = torch.Generator().manual_seed(seed)
    scales = data["target_scales"][list(UPPER_INDICES)].to(device)
    for ids in torch.arange(len(q["valid"])).split(16):
        b = subset(q, ids, device)
        b["frozen_local"] = audio(b["audio_features"], b["valid"])["local"]
        noise = torch.randn((*b["valid"].shape, 52), generator=generator).to(device)
        base = old_prediction(system, b, identities, noise)
        valid = b["valid"] & b["channel_mask"][:, list(UPPER_INDICES)].all(-1)[:, None]
        target = torch.where(valid[..., None], b["motion"][..., list(UPPER_INDICES)], 0.)
        target = _center(target, valid)
        states = _state_from_centered(target, valid, scales)
        rows.append({"base": base.cpu(), "target": target.cpu(),
                     "state": states.cpu(), "valid": valid.cpu()})
    return {key: torch.cat([row[key] for row in rows]) for key in rows[0]}


def _score(cache: dict, device: str) -> dict:
    base = cache["base"].to(device)
    target = cache["target"].to(device)
    state = cache["state"].to(device)
    valid = cache["valid"].to(device)
    scales = torch.ones(9, device=device, dtype=base.dtype)
    # State is already normalized by the fitted channel scales; recover raw
    # units by reading the same scales from the caller into the cache below.
    scales = cache["upper_scales"].to(device)
    modes = {
        "zero": torch.zeros_like(state),
        "constant": _constant(state, valid),
        "real": state,
        "reverse": _reverse(state, valid),
        "permutation": _reverse(state.roll(1, 0), valid),
    }
    source = _center(base[..., list(UPPER_INDICES)], valid)
    target_env = regional_envelope(target, valid)
    source_env = regional_envelope(source, valid)
    scores = {}
    for name, condition in modes.items():
        dynamic = _lift_state(condition, scales)
        composed = compose_mean_preserving_upper(base, dynamic, valid)
        centered = _center(composed[..., list(UPPER_INDICES)], valid)
        env = regional_envelope(centered, valid)
        temporal = _temporal_diagnostics(env, target_env, valid)
        scores[name] = {
            "centered_upper_mse": float(_masked_mse(centered, target, valid)),
            "regional_envelope_mse": float(_masked_mse(env, target_env, valid)),
            "temporal_diagnostics": temporal,
            "upper_response_rms": float(_masked_mse(centered, source, valid).sqrt()),
            "mouth_jaw_and_nonupper_exact": bool(torch.equal(
                composed[..., list(NOT_UPPER)].view(torch.int32),
                base[..., list(NOT_UPPER)].view(torch.int32))),
            "upper_mean_drift": float((
                torch.where(valid[..., None], composed[..., list(UPPER_INDICES)], 0.).sum(1)
                / valid.sum(1, keepdim=True).to(base.dtype)
                - torch.where(valid[..., None], base[..., list(UPPER_INDICES)], 0.).sum(1)
                / valid.sum(1, keepdim=True).to(base.dtype)
            ).abs().max()),
        }
    static = scores["zero"]
    real = scores["real"]
    static_corr = static["temporal_diagnostics"]["correlation_macro"]
    real_corr = real["temporal_diagnostics"]["correlation_macro"]
    static_lag = static["temporal_diagnostics"]["peak_abs_lag_frames_macro"]
    real_lag = real["temporal_diagnostics"]["peak_abs_lag_frames_macro"]
    gate = {
        "real_beats_zero_upper_mse": real["centered_upper_mse"] < static["centered_upper_mse"],
        "real_beats_zero_envelope_mse": real["regional_envelope_mse"] < static["regional_envelope_mse"],
        # Zero has no temporal correlation by construction; requiring an
        # improvement over it would make the diagnostic fail vacuously.
        "real_correlation_absolute": real_corr is not None and real_corr >= .50,
        "real_peak_lag_absolute": real_lag is not None and real_lag <= 2.0,
        "controls_preserved": all(v["mouth_jaw_and_nonupper_exact"] and v["upper_mean_drift"] < 1e-5 for v in scores.values()),
        "real_beats_reverse": real["regional_envelope_mse"] < scores["reverse"]["regional_envelope_mse"],
        "real_beats_permutation": real["regional_envelope_mse"] < scores["permutation"]["regional_envelope_mse"],
    }
    return {"conditions": scores, "gate": gate, "receiver_passed": bool(all(gate.values()))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Fresh signed receiver output required")
    a.output.mkdir(parents=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    data, system, audio, identities, _ = load_context(a.data, a.source, a.device, a.seed)
    if a.smoke:
        data["splits"]["validation"] = subset(data["splits"]["validation"], torch.arange(min(32, len(data["splits"]["validation"]["valid"]))), "cpu")
    scales = data["target_scales"][list(UPPER_INDICES)].float()
    cache = _cache(data, system, audio, identities, a.device, a.seed + 1)
    cache["upper_scales"] = scales.cpu()
    result = _score(cache, a.device)
    result.update({"oracle_only": True, "audio_student_started": False,
                   "signed_basis": ["raise", "down", "squint", "wide"],
                   "data_manifest_sha256": data["provenance"]["manifest_sha256"],
                   "source_sha256": sha(a.source), "test_loaded": False,
                   "default_replaced": False})
    _save_json(a.output / "evaluation.json", result)
    _save_json(a.output / "protocol.json", {
        "schema": "signed_upper_receiver_probe_v1", "oracle_only": True,
        "audio_student_started": False, "smoke": a.smoke, "seed": a.seed,
        "condition_modes": ["real", "zero", "constant", "reverse", "permutation"],
        "mean": "frozen Stage4 upper mean", "basis": "signed raise/down/squint/wide group lift",
        "test_loaded": False, "default_replaced": False,
    })
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
