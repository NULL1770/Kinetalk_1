"""Read-only noise-seed audit of already saved internal-development curves.

For the same audio/identity, separates average single-sample squared error into
ensemble-mean error and between-seed variance. This is a distribution diagnostic,
not a test of whether diverse samples are all plausible or correctly conditioned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_teacher_schedule_probe import GROUPS as _AUDIT_GROUPS
from scripts.train_formal_predictable_projection import canonical_hash

# Use the exact channel groups already used by the schedule/audit protocol.
GROUPS = {name: list(channels) for name, channels in _AUDIT_GROUPS.items()}
MODES = ("full", "zero", "reverse", "oracle")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def center(values, observed):
    """Center only observed time points, independently for each clip/channel."""
    clean = torch.where(observed, values, torch.zeros_like(values))
    mean = clean.sum(-2, keepdim=True) / observed.sum(-2, keepdim=True).clamp_min(1)
    return torch.where(observed, clean - mean, torch.zeros_like(clean))


def correlation(x, y):
    x, y = x - x.mean(), y - y.mean()
    denominator = (x.square().sum() * y.square().sum()).sqrt()
    return None if denominator <= 0 else float((x * y).sum() / denominator)


def decompose(predictions, target, observed):
    """Exact finite-seed decomposition; variance uses denominator K, not K-1."""
    if predictions.ndim != target.ndim + 1 or predictions.shape[1:] != target.shape:
        raise ValueError("Predictions require [seed, *target.shape]")
    if observed.shape != target.shape or observed.dtype != torch.bool:
        raise ValueError("Observed mask must be Boolean and match target")
    if len(predictions) < 2 or not observed.any():
        raise ValueError("At least two seeds and observed entries required")
    p, t = predictions[:, observed].double(), target[observed].double()
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise ValueError("Observed predictions/targets must be finite")
    average = p.mean(0)
    sample_mse = (p - t).square().mean()
    mean_mse = (average - t).square().mean()
    variance = (p - average).square().mean()
    torch.testing.assert_close(sample_mse, mean_mse + variance, rtol=1e-11, atol=1e-14)
    target_energy = t.square().mean()
    sample_energy = p.square().mean()
    mean_energy = average.square().mean()
    torch.testing.assert_close(sample_energy, mean_energy + variance, rtol=1e-11, atol=1e-14)
    return {"observations_per_seed": int(observed.sum()),
            "single_sample_mse": float(sample_mse), "ensemble_mean_mse": float(mean_mse),
            "seed_variance": float(variance), "seed_std": float(variance.sqrt()),
            "seed_variance_fraction_of_mse": float(variance / sample_mse) if sample_mse > 0 else None,
            "sample_rms": float(sample_energy.sqrt()), "ensemble_mean_rms": float(mean_energy.sqrt()),
            "target_rms": float(target_energy.sqrt()),
            "seed_variance_to_target_energy": float(variance / target_energy) if target_energy > 0 else None,
            "ensemble_mean_correlation": correlation(average, t),
            "per_seed_correlation": [correlation(row, t) for row in p],
            "per_seed_mse": [float((row - t).square().mean()) for row in p],
            "decomposition_absolute_error": float((sample_mse - mean_mse - variance).abs())}


def audit(curves, reference):
    seeds = curves.get("noise_seeds")
    if seeds != [42, 123, 2026] or curves.get("decode_steps") != 12:
        raise ValueError("Require existing three-noise twelve-step curve protocol")
    if set(curves.get("motion", {})) != set(map(str, seeds)):
        raise ValueError("Curve seed keys differ")
    if any(set(curves["motion"][str(seed)]) != set(MODES) for seed in seeds):
        raise ValueError("Require full/zero/reverse/oracle at every seed")
    q = reference["q"]
    target, valid, channels = q["motion"].double(), q["valid"].bool(), q["channel_mask"].bool()
    if target.ndim != 3 or valid.shape != target.shape[:2] or channels.shape != (len(target), target.shape[-1]):
        raise ValueError("Reference observation shapes differ")
    observed = valid[..., None] & channels[:, None, :]
    baseline = reference["base"]["b0"].double() + reference["identity"]["baseline"].double()[:, None]
    if baseline.shape != target.shape:
        raise ValueError("Reference baseline shape mismatch")
    populations = {"all": torch.arange(len(target)),
                   "neutral": (q["emotion_id"] == 0).nonzero(as_tuple=True)[0],
                   "nonneutral": (q["emotion_id"] != 0).nonzero(as_tuple=True)[0]}
    if "speaker_id" in q:
        for speaker in q["speaker_id"].unique().tolist():
            populations[f"speaker_{speaker}/nonneutral"] = ((q["speaker_id"] == speaker) & (q["emotion_id"] != 0)).nonzero(as_tuple=True)[0]
    populations = {name: ids for name, ids in populations.items() if len(ids)}
    fields = {"raw_motion": {}, "centered_residual": {}}
    targets = {"raw_motion": target, "centered_residual": center(target - baseline, observed)}
    for mode in MODES:
        stack = torch.stack([curves["motion"][str(seed)][mode].double() for seed in seeds])
        if stack.shape[1:] != target.shape:
            raise ValueError("Curve/reference shape mismatch")
        fields["raw_motion"][mode] = stack
        fields["centered_residual"][mode] = center(stack - baseline, observed.expand_as(stack))
    output = {"noise_seeds": seeds, "decode_steps": 12, "population_counts": {name: len(ids) for name, ids in populations.items()},
              "error_decomposition": {}, "paired_condition_response": {}, "decomposition_verified": True}
    for kind, by_mode in fields.items():
        output["error_decomposition"][kind] = {}
        for mode, stack in by_mode.items():
            rows = {}
            for pop, ids in populations.items():
                rows[pop] = {}
                for group, cc in GROUPS.items():
                    if max(cc) >= target.shape[-1]:
                        raise ValueError("Reference lacks required group channels")
                    rows[pop][group] = decompose(stack[:, ids][..., cc], targets[kind][ids][..., cc], observed[ids][..., cc])
            output["error_decomposition"][kind][mode] = rows
        output["paired_condition_response"][kind] = {}
        for contrast in ("zero", "reverse"):
            rows = {}
            delta = by_mode["full"] - by_mode[contrast]
            for pop, ids in populations.items():
                rows[pop] = {}
                for group, cc in GROUPS.items():
                    # Same seed in both modes isolates condition response from noise.
                    selected = delta[:, ids][..., cc]
                    masks = observed[ids][..., cc]
                    values = selected[:, masks]
                    mean = values.mean(0)
                    rows[pop][group] = {"paired_response_mse": float(values.square().mean()),
                        "paired_response_rms": float(values.square().mean().sqrt()),
                        "ensemble_mean_response_rms": float(mean.square().mean().sqrt()),
                        "response_seed_std": float((values - mean).square().mean().sqrt())}
            output["paired_condition_response"][kind]["full_minus_" + contrast] = rows
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True, help="Training run provenance.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh output required")
    torch.set_num_threads(4)
    provenance = json.loads(args.provenance.read_text(encoding="utf8"))
    recipe = provenance["recipe"]
    if provenance.get("recipe_sha256") != canonical_hash(recipe):
        raise ValueError("Recipe canonical hash mismatch")
    sidecar_path = args.curves.with_name(args.curves.stem + ".provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf8"))
    if sidecar.get("schema") != "projection_schedule_curves_provenance_v1" or sidecar.get("curve_sha256") != sha(args.curves):
        raise ValueError("Curve provenance mismatch")
    checkpoint_path = args.curves.with_name(args.curves.name.replace("_curves.pt", ".pt"))
    if not checkpoint_path.exists() or sidecar.get("checkpoint_sha256") != sha(checkpoint_path):
        raise ValueError("Curve/checkpoint provenance mismatch")
    cache_path = Path(recipe["args"]["cache"])
    if sidecar["recipe_sha256"] != provenance["recipe_sha256"] or sidecar["cache_sha256"] != recipe["input_sha256"]["cache"] or sha(cache_path) != sidecar["cache_sha256"]:
        raise ValueError("Curve/recipe/cache binding mismatch")
    for key in ("outer280_loaded", "new_identity439_loaded", "test_loaded"):
        if recipe.get(key) is not False:
            raise ValueError("Only previously audited internal development supported")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    if cache.get("schema") != "predictable_renderer_cache_v1":
        raise ValueError("Wrong renderer cache schema")
    reference = cache["splits"]["validation"]
    curves = torch.load(args.curves, map_location="cpu", weights_only=False, mmap=True)
    result = {"schema": "multiseed_stochasticity_audit_v1", "source_sha256": sha(__file__),
              "curve_sha256": sidecar["curve_sha256"], "cache_sha256": sidecar["cache_sha256"],
              "recipe_sha256": sidecar["recipe_sha256"], "data_scope": recipe.get("data_scope"),
              "interpretation": "Finite-seed error decomposition only. Sample diversity does not establish plausible emotion, calibrated uncertainty, or audio synchrony. Ensemble mean is a diagnostic, not a new deployed output. No best-of-K selection.",
              "variance_denominator": "K (population across saved seeds)",
              "test_loaded": False, "outer280_loaded": False, "new_identity439_loaded": False,
              "training_performed": False, "default_replaced": False, **audit(curves, reference)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({"complete": True, "decomposition_verified": True,
          "full_nonneutral_centered": result["error_decomposition"]["centered_residual"]["full"]["nonneutral"]}), flush=True)


if __name__ == "__main__":
    main()
