"""Package existing V2 causal controls without fitting or reopening data.

This is a read-only evidence packer.  It consumes result.json files already
produced by the frozen continuous-latent evaluations and writes a compact
comparison plus explicit claim boundaries.  It never loads motion/audio
arrays, model checkpoints, or sealed-test targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def row(path: Path, label: str) -> dict:
    data = load(path)
    summary = data.get("summary") or {}
    es = summary.get("joint_fair_es") or {}
    return {
        "label": label,
        "source": str(path),
        "clips": data.get("clips"),
        "scored_clips": data.get("scored_clips"),
        "split": (data.get("split") or "inner_validation_206") if data.get("clips") == 206 else data.get("split"),
        "intervention": data.get("intervention"),
        "use_audio": data.get("use_audio"),
        "centered_es": es.get("centered"),
        "raw_es": es.get("raw"),
        "variogram": (summary.get("variogram") or {}).get("aggregate"),
        "raw_oob": summary.get("raw_oob"),
        "naturalness_certified": data.get("naturalness_certified", False),
    }


def delta(a: dict, b: dict, key: str = "centered_es") -> float | None:
    if a.get(key) is None or b.get(key) is None:
        return None
    return float(a[key]) - float(b[key])


def subset_metric(path: Path, clip_ids: set[str], metric: str) -> float:
    values = []
    for item in load(path).get("per_clip_scores", []):
        if item.get("clip_id") not in clip_ids:
            continue
        if metric == "centered_es":
            value = (item.get("joint_fair_es") or {}).get("centered")
        elif metric == "variogram":
            value = (item.get("variogram") or {}).get("aggregate")
        else:
            raise ValueError(metric)
        if value is not None:
            values.append(float(value))
    if len(values) != len(clip_ids):
        raise RuntimeError(f"missing {metric} values for {path}")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/research_takeover_20260920/v2_causal_diagnostic_pack_20260920.json"))
    args = parser.parse_args()
    root = Path("artifacts")
    files = {
        # Existing 64-clip outer diagnostic from the formal V2 run.
        "outer_206_audio": root / "multiscale_audio_v2_20260919/point_1000/audio/result.json",
        "outer_206_matched_static": root / "multiscale_audio_v2_20260919/point_1000/matched_static/result.json",
        "outer_206_prior": root / "multiscale_audio_v2_20260919/point_1000/prior/result.json",
        "outer_206_reverse": root / "multiscale_audio_v2_20260919/point_1000/reverse/result.json",
        "outer_206_mismatch": root / "multiscale_audio_v2_20260919/point_1000/mismatch/result.json",
        # The older 64-clip evaluation is retained only as a historical
        # diagnostic; its 64 clips were not the formal paper split.
        "historical_64_audio": root / "continuous_latent_20260919/audio_generation/holdout/result.json",
        "historical_64_static": root / "continuous_latent_20260919/audio_static/holdout/result.json",
        "historical_64_reverse": root / "continuous_latent_20260919/audio_reverse/holdout/result.json",
        "historical_64_mismatch": root / "continuous_latent_20260919/audio_mismatch/holdout/result.json",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing existing diagnostic artifacts: " + ", ".join(missing))
    rows = {name: row(path, name) for name, path in files.items()}
    mismatch_ids = {item["clip_id"] for item in load(files["outer_206_mismatch"]).get("per_clip_scores", [])}
    audio_ids = {item["clip_id"] for item in load(files["outer_206_audio"]).get("per_clip_scores", [])}
    static_ids = {item["clip_id"] for item in load(files["outer_206_matched_static"]).get("per_clip_scores", [])}
    common_ids = mismatch_ids & audio_ids & static_ids
    if len(common_ids) != len(mismatch_ids):
        raise RuntimeError("mismatch is not a strict subset of the shared 206-clip pool")
    rows["outer_206_mismatch"]["paired_subset_clips"] = len(common_ids)
    paired = {
        "audio_centered_es": subset_metric(files["outer_206_audio"], common_ids, "centered_es"),
        "static_centered_es": subset_metric(files["outer_206_matched_static"], common_ids, "centered_es"),
        "mismatch_centered_es": subset_metric(files["outer_206_mismatch"], common_ids, "centered_es"),
        "audio_variogram": subset_metric(files["outer_206_audio"], common_ids, "variogram"),
        "static_variogram": subset_metric(files["outer_206_matched_static"], common_ids, "variogram"),
        "mismatch_variogram": subset_metric(files["outer_206_mismatch"], common_ids, "variogram"),
        "clip_count": len(common_ids),
    }
    comparisons = {
        "outer_206_audio_minus_matched_static_centered_es": delta(rows["outer_206_audio"], rows["outer_206_matched_static"]),
        "outer_206_audio_minus_matched_static_variogram": delta(rows["outer_206_audio"], rows["outer_206_matched_static"], "variogram"),
        "outer_206_audio_minus_prior_centered_es": delta(rows["outer_206_audio"], rows["outer_206_prior"]),
        "outer_206_audio_minus_reverse_centered_es": delta(rows["outer_206_audio"], rows["outer_206_reverse"]),
        "outer_206_paired_audio_minus_mismatch_centered_es": paired["audio_centered_es"] - paired["mismatch_centered_es"],
        "outer_206_paired_audio_minus_static_centered_es": paired["audio_centered_es"] - paired["static_centered_es"],
        "outer_206_paired_audio_minus_mismatch_variogram": paired["audio_variogram"] - paired["mismatch_variogram"],
        "outer_206_paired_audio_minus_static_variogram": paired["audio_variogram"] - paired["static_variogram"],
        "historical_64_audio_minus_static_centered_es": delta(rows["historical_64_audio"], rows["historical_64_static"]),
    }
    pack = {
        "schema": "kinetalk_v2_causal_diagnostic_pack_v1",
        "status": "descriptive_read_only",
        "sources": rows,
        "comparisons": comparisons,
        "paired_subset": paired,
        "interpretation": {
            "primary": "On the 206-clip inner validation, real audio is not better than matched static audio on centered ES or variogram; this does not establish a deployable local-audio dynamic gain.",
            "training_signal": "The separate frozen 32-fit/32-inner-dev diagnostic remains evidence that audio can reduce error on seen fit clips; it is not a held-out paper result.",
            "reverse": "Reverse is a timing intervention, but a worse score alone is not proof of causal audio timing because the intervention shifts the feature distribution and solver response.",
            "mismatch": "Same-emotion, different-sentence mismatch is a stress control with resampling and distribution shift; it is not a clean counterfactual baseline.",
            "oracle": "No motion-oracle receiver control is included: AE reconstruction would read target motion and cannot certify audio conditioning.",
            "metrics": "FaceDiffuser ARKit FDD/MBE/LBE and the 206-clip values are preserved in source artifacts, but FDD excludes brows and is not a local timing metric.",
            "sealed_test": "No sealed-test target was loaded by this packer.",
        },
        "claim_boundaries": [
            "Do not claim audio dynamic success from lower FM error, nonzero motion, or a video that moves.",
            "Do not use the historical 64-clip or 206-clip diagnostics as final paper test results.",
            "Do not call the current 613/206 protocol full-data training.",
        ],
    }
    pack["source_artifact_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
    }
    canonical = json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")
    pack["pack_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"output": str(args.output), "pack_sha256": pack["pack_sha256"],
                      "comparisons": comparisons}, ensure_ascii=False))


if __name__ == "__main__":
    main()
