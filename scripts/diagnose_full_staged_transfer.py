"""Read-only same-checkpoint global/local condition swaps on locked data.

Both renderer checkpoints receive the *same* completed Stage4 audio encoder
and frozen motion teacher. Teacher conditions use target motion and are oracle
diagnostics, never deployable audio results. No fitting or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from scripts.full_staged_data import load_training_inputs, sha
from scripts.train_formal_predictable_projection import canonical_hash, save_checkpoint, save_json
from scripts.train_full_staged import (
    batch_identity, cache_current_base, identity_cache, obs, region_report, subset,
    targets, teacher_affect,
)
from scripts.train_predictable_renderer import state_hash

SCHEMA = "full_staged_transfer_diagnostic_v1"
MODES = ("audio_all", "audio_global_teacher_local", "teacher_global_audio_local", "teacher_all", "audio_global_zero_local")
MODE_SCOPE = {
    "audio_all": "Deployable Stage4 audio conditions; no target motion in generation input.",
    "audio_global_teacher_local": "ORACLE: target-motion teacher local, audio global and intensity.",
    "teacher_global_audio_local": "ORACLE: target-motion teacher global and intensity, audio local.",
    "teacher_all": "ORACLE: target-motion teacher global, intensity and local.",
    "audio_global_zero_local": "Audio global/intensity with local explicitly zero; no target motion in generation input.",
}


def swap_condition(audio, teacher, mode):
    """Global and intensity always move together; labels are never conditions."""
    if mode not in MODES:
        raise ValueError("Unknown condition swap")
    use_teacher_global = mode in ("teacher_global_audio_local", "teacher_all")
    global_source = teacher if use_teacher_global else audio
    local = teacher["local"] if mode in ("audio_global_teacher_local", "teacher_all") else audio["local"]
    if mode == "audio_global_zero_local":
        local = torch.zeros_like(local)
    return {"global": global_source["global"], "intensity_value": global_source["intensity_value"], "local": local}


def sequence_metrics(prediction, target, valid):
    """Raw and within-clip temporal diagnostics over observed values only."""
    if prediction.shape != target.shape or prediction.ndim != 3 or valid.shape != prediction.shape[:2]:
        raise ValueError("Expected matching [B,T,D] and whole-frame masks")
    mask = valid[..., None].expand_as(prediction)
    x, y = prediction.double(), target.double()
    if not mask.any() or not torch.isfinite(x[mask]).all() or not torch.isfinite(y[mask]).all():
        raise ValueError("Observed latent diagnostics must be finite and nonempty")
    n = mask.sum(1, keepdim=True).clamp_min(1)
    mean_x, mean_y = torch.where(mask, x, 0.).sum(1, keepdim=True) / n, torch.where(mask, y, 0.).sum(1, keepdim=True) / n
    xc, yc = torch.where(mask, x - mean_x, 0.), torch.where(mask, y - mean_y, 0.)
    sse, energy = (xc - yc).square().sum(), yc.square().sum()
    return {"raw_mse": float((x[mask] - y[mask]).square().mean()),
            "clip_mean_mse": float((mean_x - mean_y).square().mean()),
            "centered_mse": float(sse / mask.sum()),
            "centered_correlation": float((xc * yc).sum() / (xc.square().sum() * energy).sqrt().clamp_min(1e-12)),
            "centered_r2": float(1. - sse / energy.clamp_min(1e-12)),
            "prediction_temporal_rms": float((xc.square().sum() / mask.sum()).sqrt()),
            "target_temporal_rms": float((energy / mask.sum()).sqrt()),
            "prediction_rms": float(x[mask].square().mean().sqrt()),
            "target_rms": float(y[mask].square().mean().sqrt())}


def global_metrics(prediction, target):
    """Across-clip coordinate diagnostics, not temporal/global-label accuracy."""
    x, y = prediction.double(), target.double()
    if x.shape != y.shape or x.ndim != 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Global diagnostics require finite matching [B,D]")
    xc, yc = x - x.mean(0), y - y.mean(0)
    denom = (xc.square().sum(0) * yc.square().sum(0)).sqrt()
    good = denom > 1e-12
    correlations = (xc * yc).sum(0)[good] / denom[good]
    return {"raw_mse": float((x - y).square().mean()),
            "across_clip_pooled_centered_correlation": float((xc * yc).sum() / (xc.square().sum() * yc.square().sum()).sqrt().clamp_min(1e-12)),
            "across_clip_mean_nonconstant_dimension_correlation": float(correlations.mean()) if good.any() else None,
            "nonconstant_dimensions": int(good.sum()),
            "mean_cosine_similarity": float(torch.nn.functional.cosine_similarity(x, y, dim=-1).mean()),
            "prediction_rms": float(x.square().mean().sqrt()), "target_rms": float(y.square().mean().sqrt())}


def _make_audio(state, stride, device):
    model = SlowStateAffect(state["feature_mean"], state["feature_std"],
                            global_dim=state["global_head.weight"].shape[0],
                            hidden=state["input.weight"].shape[0],
                            local_dim=state["local_head.weight"].shape[0],
                            num_emotions=state["emotion_classifier.weight"].shape[0],
                            num_levels=state["intensity_classifier.weight"].shape[0], stride=stride)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def condition_diagnostics(system, audio, data, identities, role, ids, *, batch_size, stride, device):
    q = data["splits"][role]
    values = {k: [] for k in ("audio_global", "teacher_global", "audio_local", "teacher_local", "audio_state", "target_state", "audio_intensity", "teacher_intensity")}
    audio_classes, teacher_classes, emotion = [], [], []
    for ix in ids.split(batch_size):
        b = subset(q, ix, device); ident = batch_identity(identities, b)
        target = teacher_affect(system, b, ident)
        prediction = audio(b["audio_features"], b["valid"])
        slow, _ = targets(b, data["target_scales"].to(device), stride)
        for key, value in (("audio_global", prediction["global"]), ("teacher_global", target["global"]),
                           ("audio_local", prediction["local"]), ("teacher_local", target["local"]),
                           ("audio_state", prediction["state"]), ("target_state", slow["state"]),
                           ("audio_intensity", prediction["intensity_value"]), ("teacher_intensity", target["intensity_value"])):
            values[key].append(value.cpu())
        audio_classes.append(prediction["emotion_logits"].argmax(-1).cpu())
        teacher_classes.append(target["emotion_logits"].argmax(-1).cpu())
        emotion.append(b["emotion_id"].cpu())
    values = {k: torch.cat(v) for k, v in values.items()}
    reference = subset(q, ids, "cpu")
    labels = torch.cat(emotion)
    result = {"role": role, "clips": len(ids), "clip_ids": reference["clip_id"],
              "global_audio_vs_teacher": global_metrics(values["audio_global"], values["teacher_global"]),
              "intensity_audio_vs_teacher": global_metrics(values["audio_intensity"], values["teacher_intensity"]),
              "local_audio_vs_teacher": sequence_metrics(values["audio_local"], values["teacher_local"], reference["valid"]),
              "audio_state_vs_gt_slow_state": sequence_metrics(values["audio_state"], values["target_state"], reference["valid"]),
              "audio_emotion_accuracy": float((torch.cat(audio_classes) == labels).float().mean()),
              "teacher_emotion_accuracy": float((torch.cat(teacher_classes) == labels).float().mean()),
              "local_comparison_limit": "Audio local and teacher local are not explicitly aligned; coordinate MSE measures interface discrepancy, not necessarily equivalent semantic error after renderer coadaptation.",
              "scope": "First 64 fit rows; training-fit diagnostic, not independent generalization" if role == "train" else "All locked internal development; inherited model exposure and shared scripts remain"}
    return result, {**values, "clip_id": reference["clip_id"], "valid": reference["valid"], "times": reference["times"]}


@torch.no_grad()
def evaluate_swaps(system, audio, data, identities, *, seed, steps, batch_size, device):
    q = data["splits"]["validation"]
    ids = torch.arange(len(q["valid"]))
    noise = torch.randn(*q["motion"].shape, generator=torch.Generator().manual_seed(seed))
    outputs = {mode: [] for mode in MODES}
    classes = {mode: [] for mode in MODES}
    for ix in ids.split(batch_size):
        b = subset(q, ix, device); ident = batch_identity(identities, b)
        teacher = teacher_affect(system, b, ident)
        predicted = audio(b["audio_features"], b["valid"])
        for mode in MODES:
            affect = swap_condition(predicted, teacher, mode)
            motion = system.generate(b["content"], b["valid"], ident, affect, initial_noise=noise[ix].to(device),
                                     steps=steps, base={k: b[k] for k in ("b0", "h0")})["motion"]
            readout = system.encode_motion(torch.where(obs(b), motion - b["b0"] - ident["baseline"][:, None], 0.), b["valid"])
            classes[mode].append((readout["emotion_logits"].argmax(-1) == b["emotion_id"]).cpu())
            outputs[mode].append(motion.cpu())
        print(json.dumps({"event": "swap_batch", "done": int(ix[-1]) + 1, "total": len(ids)}), flush=True)
    outputs = {mode: torch.cat(parts) for mode, parts in outputs.items()}
    report = {mode: {"scope": MODE_SCOPE[mode], "metrics": region_report(value, q),
                     "generated_teacher_emotion_accuracy_nonindependent": float(torch.cat(classes[mode]).float().mean())}
              for mode, value in outputs.items()}
    return report, {"clip_id": q["clip_id"], "target": q["motion"], "valid": q["valid"], "times": q["times"],
                    "channel_mask": q["channel_mask"], "b0": q["b0"], "initial_noise": noise, "predictions": outputs}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "audio", "targets", "enrollment", "native-root", "trained-run", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    return p


def main():
    args = parser().parse_args()
    if args.output.exists(): raise FileExistsError("A fresh diagnostic output directory is required")
    if args.batch_size < 1: raise ValueError("batch-size must be positive")
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    run_provenance = json.loads((args.trained_run / "provenance.json").read_text(encoding="utf8"))
    recipe = run_provenance["recipe"]
    if run_provenance.get("recipe_sha256") != canonical_hash(recipe): raise ValueError("Trained-run recipe hash differs")
    data = load_training_inputs(args.source_run, args.audio, args.targets, args.enrollment, args.native_root)
    if data["provenance"]["input_sha256"] != recipe["data_provenance"]["input_sha256"]:
        raise ValueError("Diagnostic inputs differ from completed training")
    if len(data["splits"]["validation"]["valid"]) != 405:
        raise ValueError("This diagnostic requires the complete locked 405-clip development split")
    if len(data["splits"]["train"]["valid"]) < 64: raise ValueError("At least 64 fit clips required")
    paths = {stage: args.trained_run / stage / "final.pt" for stage in ("teacher", "audio")}
    hashes = {stage: sha(path) for stage, path in paths.items()}
    checkpoints = {}
    for stage, path in paths.items():
        complete = json.loads(path.with_name("complete.json").read_text(encoding="utf8"))
        if complete.get("final_sha256") != hashes[stage]: raise ValueError("Final checkpoint does not match completion hash")
        value = torch.load(path, map_location="cpu", weights_only=False)
        if (value.get("stage") != stage or value.get("recipe_sha256") != run_provenance["recipe_sha256"]
                or value.get("completed_epochs") != recipe["epochs_per_stage"]):
            raise ValueError("Checkpoint stage, completion or recipe differs")
        if value["config"] != data["config"]: raise ValueError("Checkpoint model configuration differs")
        if not torch.equal(value["scales"], data["target_scales"]): raise ValueError("Checkpoint target scales differ")
        checkpoints[stage] = value
    # A common identity/global teacher/B0 is essential for a renderer-only test.
    non_renderer = {s: {k: v for k, v in ck["system"].items() if not k.startswith("renderer.")}
                    for s, ck in checkpoints.items()}
    non_renderer_hashes = {s: state_hash(v) for s, v in non_renderer.items()}
    if len(set(non_renderer_hashes.values())) != 1:
        raise ValueError("Stage3/4 differ outside renderer; cannot isolate adaptation with shared conditions")
    system = data["system"].to(args.device).eval().requires_grad_(False)
    system.load_state_dict(checkpoints["audio"]["system"], strict=True)
    stride = int(recipe["stride_frames"]); steps = int(recipe["args"]["decode_steps"]); seed = 42
    audio = _make_audio(checkpoints["audio"]["audio"], stride, args.device)
    initial_audio_hash = state_hash(audio.state_dict())
    cache_current_base(system, data, args.device, args.batch_size)
    identities = identity_cache(system, data, args.device)
    args.output.mkdir(parents=True)
    provenance = {"schema": SCHEMA, "trained_run": str(args.trained_run.resolve()),
                  "trained_recipe_sha256": run_provenance["recipe_sha256"], "checkpoint_sha256": hashes,
                  "data_provenance": data["provenance"], "noise_seed": seed, "decode_steps": steps,
                  "batch_size": args.batch_size, "common_non_renderer_state_sha256": non_renderer_hashes,
                  "audio_encoder_source": "Completed Stage4 audio checkpoint, identical for both renderers",
                  "teacher_source": "Frozen motion teacher identical in Stage3 and Stage4; uses query GT motion",
                  "mode_scopes": MODE_SCOPE, "test_loaded": False, "training_performed": False,
                  "checkpoint_selection_performed": False, "default_replaced": False,
                  "source_sha256": {name: sha(Path(__file__).resolve().parents[1] / name) for name in
                                    ("scripts/diagnose_full_staged_transfer.py", "scripts/train_full_staged.py",
                                     "scripts/full_staged_data.py", "kinetalk_b0/models/neutral_affect.py",
                                     "kinetalk_b0/models/slow_state_affect.py", "kinetalk_b0/models/dit.py")}}
    save_json(args.output / "provenance.json", provenance)
    diagnostics, diagnostic_tensors = {}, {}
    for role, ids in (("train", torch.arange(64)), ("validation", torch.arange(405))):
        report, values = condition_diagnostics(system, audio, data, identities, role, ids, batch_size=args.batch_size,
                                              stride=stride, device=args.device)
        diagnostics[role] = report; diagnostic_tensors[role] = values
    save_json(args.output / "condition_diagnostics.json", diagnostics)
    save_checkpoint(args.output / "condition_diagnostics.pt", {"schema": SCHEMA, "provenance": provenance, "splits": diagnostic_tensors})
    results = {}
    for stage in ("teacher", "audio"):
        system.renderer.load_state_dict({k[len("renderer."):]: v for k, v in checkpoints[stage]["system"].items() if k.startswith("renderer.")}, strict=True)
        before = state_hash(system.state_dict())
        print(json.dumps({"event": "renderer", "stage": stage}), flush=True)
        report, curves = evaluate_swaps(system, audio, data, identities, seed=seed, steps=steps, batch_size=args.batch_size, device=args.device)
        if state_hash(system.state_dict()) != before or state_hash(audio.state_dict()) != initial_audio_hash:
            raise RuntimeError("Read-only diagnostic mutated model state")
        if any(p.grad is not None for m in (system, audio) for p in m.parameters()): raise RuntimeError("Unexpected diagnostic gradients")
        results[stage] = report
        save_json(args.output / (stage + "_renderer.json"), {"schema": SCHEMA, "renderer_stage": stage, "modes": report, "provenance": provenance})
        curve_path = args.output / (stage + "_renderer_curves.pt")
        save_checkpoint(curve_path, {"schema": SCHEMA, "renderer_stage": stage, "provenance": provenance, **curves})
        save_json(curve_path.with_suffix(".provenance.json"), {"curve_sha256": sha(curve_path), "checkpoint_sha256": hashes[stage],
                                                               "diagnostic_provenance_sha256": sha(args.output / "provenance.json")})
    output_hashes = {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}
    save_json(args.output / "summary.json", {"schema": SCHEMA, "status": "complete", "clips": 405, "fit_diagnostic_clips": 64,
                                            "noise_seeds": [seed], "results": results, "condition_diagnostics": diagnostics,
                                            "output_sha256": output_hashes, "test_loaded": False, "training_performed": False,
                                            "interpretation": "Single-seed same-input intervention; target-motion teacher swaps are oracle and may be distribution-shifted. No significance or unique-cause claim."})
    print("FULL_STAGED_TRANSFER_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
