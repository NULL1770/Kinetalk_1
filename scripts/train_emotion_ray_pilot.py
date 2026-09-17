"""Fixed-budget, frozen-backbone test of emotion direction x audio scalar.

Label-conditioned arms are diagnostics. Actual inference uses cached audio,
its frozen global probabilities, and train-fitted rays only. No renderer or
default checkpoint is updated. Every arm is scored against identical targets.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

import torch
from torch.nn import functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_conditioned_dynamic import EmotionConditionedDynamicPredictor
from kinetalk_b0.emotion_ray import fit_emotion_rays, project_ray_field, direction_from_probs
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset, EMOTIONS
from kinetalk_b0.utils import freeze_module
from scripts.emotion_ray_metrics import field_metrics
from scripts.probe_dynamic_predictability import bin_frames, center_controls, sentence_split, weighted_mse
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_feature_probe import fit_feature_stats, prepare_feature_batch
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, device_batch, observed, optimize, prepare,
    select, sha, write_json,
)


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    return value


def as_json(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    return value


def scalar_prediction(model, frames, probability, valid, stride, scale):
    raw = model(frames, probability, valid)
    binned, weight = bin_frames(raw, valid, stride)
    return center_controls(binned, weight) * scale


def temporal_intervention(values, weight, mode):
    """Alter only time, keeping the destination emotion direction fixed."""
    if mode == "full":
        return values
    result = torch.zeros_like(values)
    if mode == "zero":
        return result
    for i in range(len(values)):
        positions = (weight[i] > 0).nonzero(as_tuple=True)[0]
        if mode == "reverse":
            result[i, positions] = values[i, positions.flip(0)]
        elif mode == "shuffle":
            source = (i + 1) % len(values)
            active = values[source, weight[source] > 0]
            grid = torch.linspace(0, len(active) - 1, len(positions), device=values.device).round().long()
            result[i, positions] = active[grid]
        else:
            raise ValueError(mode)
    return center_controls(result, weight)


def metrics(pred, target, bundle, ids, bootstrap=1000):
    if not len(ids):
        return None
    return field_metrics(pred[ids], target[ids], bundle["weight"][ids],
                         [bundle["sentence_id"][int(i)] for i in ids],
                         bootstrap_samples=bootstrap)


def evaluate_field(scalar, direction, bundle, ids, bootstrap=1000):
    """Use the same GT-ray proxy AND real motion for every candidate."""
    selected = ids[bundle["nonneutral"][ids]]
    neutral = ids[bundle["emotion_id"][ids] == 0]
    vector = scalar * direction[:, None]
    result = {
        "nonneutral_n": len(selected), "neutral_n": len(neutral),
        "inactive_nonneutral_n": int(((bundle["emotion_id"][ids] != 0) & ~bundle["nonneutral"][ids]).sum()),
        "scalar_nonneutral": metrics(scalar, bundle["scalar"], bundle, selected, bootstrap),
        "proxy_nonneutral": metrics(vector, bundle["proxy"], bundle, selected, bootstrap),
        "motion_nonneutral": {},
        "neutral_proxy": metrics(vector, bundle["proxy"], bundle, neutral, 0),
    }
    for name, channels in bundle["groups"].items():
        if channels:
            result["motion_nonneutral"][name] = metrics(vector[:, :, channels],
                bundle["motion_bins"][:, :, channels], bundle, selected, bootstrap)
    return result


@torch.no_grad()
def prepare_bundle(system, dataset, identities, checkpoint, rays, device):
    query, _ = prepare(system, dataset, device)
    residual = affect_residual(query, select(identities, query["speaker_id"]))
    # Keep the original trained global transform separate from new head stats.
    global_query = prepare_feature_batch(query, checkpoint["audio_source"], checkpoint["feature_stats"])
    probability = system.encode_audio(global_query["audio"], query["valid"])["emotion_logits"].softmax(-1)
    labels = F.one_hot(query["emotion_id"], len(EMOTIONS)).float()
    true_direction = rays["rays"][query["emotion_id"]]
    _, info = direction_from_probs(probability, rays, return_info=True)
    # Preserve confidence/neutral suppression. Do not normalize tiny emotional
    # mass to a unit direction and do not gate inference with GT labels.
    actual_direction = probability @ rays["rays"]
    common = rays["observed_channels"]
    if not query["channel_mask"][:, common].all():
        raise ValueError("Evaluation lacks an observed training-ray channel")
    residual = torch.where(common[None, None], residual, 0)
    scalar, weight, motion_bins = project_ray_field(residual, query["valid"], true_direction, rays["stride"])
    groups = {"all_expression": common.nonzero(as_tuple=True)[0].tolist(),
        "upper_expression": [i for i in (5, 6, 12, 13, 41, 42, 43, 44, 45) if common[i]],
        "mouth": [i for i in range(14, 41) if common[i]],
        "jaw17": [17] if common[17] else []}
    clip_means = residual.sum(1) / query["valid"].sum(1, keepdim=True)
    return {"query": query, "scalar": scalar, "weight": weight, "motion_bins": motion_bins,
        "residual_clip_means": clip_means, "speaker_id": query["speaker_id"], "channel_mask": query["channel_mask"],
        "proxy": scalar * true_direction[:, None], "true_direction": true_direction,
        "actual_direction": actual_direction, "probability": probability, "labels": labels,
        "emotion_id": query["emotion_id"], "sentence_id": query["sentence_id"],
        "clip_id": query["clip_id"], "nonneutral": rays["active"][query["emotion_id"]],
        "direction_info": info, "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "data", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--reuse-heads", type=Path, help="Evaluate previously trained identical-config heads without retraining")
    args = parser.parse_args()
    if min(args.steps, args.hidden, args.batch_size) < 1:
        raise ValueError("Positive steps/width/batch size required")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "audio" or checkpoint.get("audio_source") != "acoustic":
        raise ValueError("Use frozen run09 audio checkpoint")
    if any(cfg.get(k) != checkpoint.get("config", {}).get(k) for k in ("data", "model")):
        raise ValueError("Use checkpoint effective model/data config")
    data_hash = sha(args.data / "train.pt")
    if data_hash != checkpoint.get("provenance", {}).get("train_sha256"):
        raise ValueError("Training cache differs from frozen checkpoint provenance")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(cfg.get("device", "cuda"))
    train_data = NeutralAffectDataset(args.data / "train.pt")
    dev_data = NeutralAffectDataset(args.data / "heldout.pt")
    train_cpu, hold_cpu = sentence_split([q["sentence_id"] for q in train_data.queries], args.seed)
    train_sentences = {train_data.queries[int(i)]["sentence_id"] for i in train_cpu}
    hold_sentences = {train_data.queries[int(i)]["sentence_id"] for i in hold_cpu}
    dev_sentences = {q["sentence_id"] for q in dev_data.queries}
    if train_sentences & hold_sentences or (train_sentences | hold_sentences) & dev_sentences:
        raise ValueError("Sentence split overlap")
    if {q["clip_id"] for q in train_data.queries} & {q["clip_id"] for q in dev_data.queries}:
        raise ValueError("External development query overlap")
    for query in dev_data.queries:
        for ref in train_data.identity_references[int(query["speaker_id"])]:
            if ref["clip_id"] == query["clip_id"] or ref["sentence_id"] == query["sentence_id"]:
                raise ValueError("Training enrollment overlaps development query")
    train_ids, hold_ids = train_cpu.to(device), hold_cpu.to(device)
    system = NeutralAffectSystem(cfg).to(device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    freeze_module(system)
    original_hash = state_hash(system.state_dict())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source").mkdir()
    sources = [Path(__file__), Path("kinetalk_b0/emotion_ray.py"),
        Path("kinetalk_b0/emotion_conditioned_dynamic.py"), Path("scripts/emotion_ray_metrics.py"),
        Path("scripts/probe_dynamic_predictability.py"), Path("scripts/train_neutral_affect_pilot.py"),
        Path("scripts/train_neutral_affect_feature_probe.py"), Path("kinetalk_b0/models/neutral_affect.py")]
    for path in sources:
        shutil.copyfile(path, args.output / "source" / path.name)
    provenance = {"schema": "emotion_ray_fixed_budget_v1", "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "checkpoint_sha256": sha(args.checkpoint), "train_sha256": data_hash,
        "external_dev_sha256": sha(args.data / "heldout.pt"), "frozen_before": original_hash,
        "source": {str(p): sha(p) for p in sources},
        "global_pretrained_on_internal_heldout": True,
        "external_dev_previously_inspected": True,
        "scope": "New dynamic head heldout sentences of enrolled speakers. External development is not untouched testing; earlier B0 pretraining overlap is not excluded.",
        "selection": "Fixed steps/lr/width/decay, no heldout checkpoint selection or amplitude calibration",
        "loss": "One train-RMS-normalized scalar MSE; no renderer training",
        "train_indices": train_cpu.tolist(), "internal_heldout_indices": hold_cpu.tolist()}
    write_json(args.output / "provenance.json", provenance)
    (args.output / "effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")
    with torch.no_grad():
        query, _ = prepare(system, train_data, device)
        refs = train_data.identity_references
        if sorted(refs) != list(range(len(refs))) or len({len(v) for v in refs.values()}) != 1:
            raise ValueError("Contiguous speaker IDs and equal enrollment reference counts required")
        reference = device_batch([r for s in sorted(refs) for r in refs[s]], device)
        ref_residual = torch.where(observed(reference), reference["motion"] -
            system.base(reference["content"], reference["valid"])["b0"], 0)
        identities = cached_identity(system, {"residual": ref_residual.reshape(len(refs),len(refs[0]),*ref_residual.shape[1:]),
            "valid": reference["valid"].reshape(len(refs),len(refs[0]),-1)})
        residual = affect_residual(query, select(identities, query["speaker_id"]))
        rays = fit_emotion_rays(residual, query["valid"], query["channel_mask"], query["speaker_id"],
            query["emotion_id"], train_ids, len(EMOTIONS), system.motion_teacher.stride)
        bundles = {"internal": prepare_bundle(system, train_data, identities, checkpoint, rays, device),
                   "external_dev": prepare_bundle(system, dev_data, identities, checkpoint, rays, device)}
        tr = bundles["internal"]
        fit_non = train_ids[tr["nonneutral"][train_ids]]
        scale = ((tr["scalar"][fit_non].square() * tr["weight"][fit_non, :, None]).sum() /
                 tr["weight"][fit_non].sum()).sqrt().clamp_min(1e-4)
        stats, features = {}, {}
        for source in ("acoustic", "content"):
            stats[source] = fit_feature_stats(select(tr["query"], train_ids), source)
            features[source] = {name: prepare_feature_batch(b["query"], source, stats[source])["audio"]
                                for name,b in bundles.items()}
    splits = {"train": ("internal", train_ids), "internal_heldout": ("internal", hold_ids),
              "external_dev": ("external_dev", torch.arange(len(bundles["external_dev"]["scalar"]), device=device))}
    report = {"target_scale": float(scale), "rays": as_json(rays), "splits": {}, "arms": {}}
    for name, (key, ids) in splits.items():
        b = bundles[key]
        diagnostics = {k: {"mean": float(v[ids].mean()), "max": float(v[ids].max())}
                       for k,v in b["direction_info"].items()}
        info = {"n": len(ids), "sentences": sorted({b["sentence_id"][int(i)] for i in ids}),
            "global_accuracy": float((b["probability"][ids].argmax(-1) == b["emotion_id"][ids]).float().mean()),
            "direction": diagnostics,
            "oracle_ray": evaluate_field(b["scalar"], b["true_direction"], b, ids),
            "gt_curve_audio_direction": evaluate_field(b["scalar"], b["actual_direction"], b, ids)}
        report["splits"][name] = info
    # Preserve inputs binned for a later strictly train-selected ridge audit.
    diagnostic_bundle = {}
    for key,b in bundles.items():
        saved = {k: v for k,v in b.items() if k != "query"}
        saved["features"] = {s: center_controls(*bin_frames(features[s][key], b["query"]["valid"], rays["stride"]))
                             for s in features}
        diagnostic_bundle[key] = cpu(saved)
    torch.save({"bundles": diagnostic_bundle, "rays": cpu(rays), "train_ids": train_cpu,
        "heldout_ids": hold_cpu, "provenance": provenance}, args.output / "diagnostic_bundle.pt")
    write_json(args.output / "summary.json", report)
    arms = [("e2v_unconditioned", "acoustic", "uniform"), ("e2v_oracle", "acoustic", "labels"),
            ("e2v_actual", "acoustic", "probability"), ("content_actual", "content", "probability")]
    for name, source, condition in arms:
        torch.manual_seed(args.seed)
        head = EmotionConditionedDynamicPredictor(features[source]["internal"].shape[-1], len(EMOTIONS), args.hidden).to(device)
        if args.reuse_heads:
            previous = torch.load(args.reuse_heads / (name + ".pt"), map_location="cpu", weights_only=False)
            for key in ("seed", "steps", "hidden", "batch_size", "learning_rate", "weight_decay"):
                if previous["args"][key] != getattr(args, key):
                    raise ValueError(f"Reused head training config differs: {key}")
            if previous["frozen_checkpoint_sha256"] != provenance["checkpoint_sha256"]:
                raise ValueError("Reused head backbone differs")
            for key in ("mean", "std"):
                torch.testing.assert_close(previous["feature_stats"][key], stats[source][key], rtol=0, atol=0)
            torch.testing.assert_close(previous["rays"]["rays"], rays["rays"].cpu(), rtol=0, atol=0)
            torch.testing.assert_close(previous["scale"], scale.cpu(), rtol=0, atol=0)
            head.load_state_dict(previous["model"], strict=True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        rng = torch.Generator(device=device).manual_seed(args.seed + 100)
        probabilities = {k: torch.full_like(b["labels"], 1 / len(EMOTIONS)) if condition == "uniform" else b[condition]
                         for k,b in bundles.items()}
        started = time.time()
        for step in range(1, 1 if args.reuse_heads else args.steps + 1):
            head.train()
            sample = train_ids[torch.randint(len(train_ids), (args.batch_size,), generator=rng, device=device)]
            pred = scalar_prediction(head, features[source]["internal"][sample], probabilities["internal"][sample],
                                     tr["query"]["valid"][sample], rays["stride"], scale)
            loss = weighted_mse(pred / scale, tr["scalar"][sample] / scale, tr["weight"][sample])
            norm = optimize(loss, optimizer, list(head.parameters()))
            if step == 1 or step % 100 == 0 or step == args.steps:
                row = {"arm": name, "step": step, "loss": float(loss), "gradient_norm": norm,
                       "elapsed_seconds": round(time.time() - started, 2)}
                with (args.output / "training.jsonl").open("a", encoding="utf8") as handle:
                    import json
                    handle.write(json.dumps(row) + "\n")
                print(row, flush=True)
        head.eval()
        arm_report, curves = {}, {}
        with torch.no_grad():
            preds = {k: scalar_prediction(head, features[source][k], probabilities[k], b["query"]["valid"],
                                         rays["stride"], scale) for k,b in bundles.items()}
            for split, (key, ids) in splits.items():
                b = bundles[key]
                prediction = preds[key]
                direction = b["true_direction"] if condition in ("labels", "uniform") else b["actual_direction"]
                score = {"primary_direction_source": "oracle_label" if condition in ("labels", "uniform") else "frozen_audio",
                         "primary": evaluate_field(prediction, direction, b, ids),
                         "same_curve_gt_direction": evaluate_field(prediction, b["true_direction"], b, ids),
                         "same_curve_audio_direction": evaluate_field(prediction, b["actual_direction"], b, ids),
                         "per_emotion": {}}
                if split != "train":
                    local = {k: v[ids] if torch.is_tensor(v) else v for k,v in b.items()}
                    local["sentence_id"] = [b["sentence_id"][int(i)] for i in ids]
                    local_ids = torch.arange(len(ids), device=device)
                    score["interventions"] = {mode: evaluate_field(temporal_intervention(prediction[ids], b["weight"][ids], mode),
                        direction[ids], local, local_ids) for mode in ("zero", "reverse", "shuffle")}
                for eid in b["emotion_id"][ids].unique().tolist():
                    own = ids[b["emotion_id"][ids] == eid]
                    score["per_emotion"][EMOTIONS[eid]] = evaluate_field(prediction, direction, b, own, 0)
                arm_report[split] = score
                curves[split] = {"prediction": cpu(prediction[ids]), "target": cpu(b["scalar"][ids]),
                    "weight": cpu(b["weight"][ids]), "clips": [b["clip_id"][int(i)] for i in ids]}
                print({"arm": name, "split": split,
                    "scalar_r2": score["primary"]["scalar_nonneutral"]["r2_against_zero"],
                    "proxy_r2": score["primary"]["proxy_nonneutral"]["r2_against_zero"],
                    "upper_motion_r2": score["primary"]["motion_nonneutral"]["upper_expression"]["r2_against_zero"]}, flush=True)
            if condition == "labels":
                arm_report["oracle_head_with_actual_condition"] = {}
                for split in ("internal_heldout", "external_dev"):
                    key, ids = splits[split]
                    b = bundles[key]
                    p = scalar_prediction(head, features[source][key], b["probability"], b["query"]["valid"], rays["stride"], scale)
                    arm_report["oracle_head_with_actual_condition"][split] = evaluate_field(p, b["actual_direction"], b, ids)
        report["arms"][name] = arm_report
        torch.save({"model": cpu(head.state_dict()), "source": source, "condition": condition,
            "feature_stats": stats[source], "scale": cpu(scale), "rays": cpu(rays),
            "args": provenance["args"], "frozen_checkpoint_sha256": provenance["checkpoint_sha256"],
            "curves": curves}, args.output / (name + ".pt"))
        write_json(args.output / "summary.json", report)
    report["frozen_after"] = state_hash(system.state_dict())
    report["frozen_unchanged"] = report["frozen_after"] == original_hash
    if not report["frozen_unchanged"]:
        raise RuntimeError("Frozen model changed")
    report["generation_evaluated"] = False
    write_json(args.output / "summary.json", report)
    print("COMPLETE: frozen model unchanged; no default model replacement", flush=True)


if __name__ == "__main__":
    main()
