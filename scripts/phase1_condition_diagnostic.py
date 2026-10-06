"""Phase-1 read-only oracle/zero/shuffle affect condition diagnostic.

This script does not train or modify checkpoints. It keeps content, identity,
validation order and initial noise fixed while changing only renderer affect
conditions. It is intended to run against the frozen remote audio checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.packed_trainval_cache import load_packed
from scripts.train_full_staged import (
    audio_affect, base_forward, batch_identity, merge_teacher_audio, teacher_affect,
)
from scripts.arkit_benchmark_report import score_fullface
from kinetalk_b0.emotion_probe import MotionEmotionProbe, classification_metrics, motion_features
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect


MOUTH = tuple(range(14, 41))
JAW_OPEN = 17


def _identity_cache(system, data, device):
    result = {}
    for sid, ref in data["refs"].items():
        content = ref["content"].to(device)
        valid = ref["valid"].to(device)
        motion = ref["motion"].to(device)
        channel = ref["channel_mask"].to(device)
        base = base_forward(system, content, valid)
        residual = torch.where(valid[..., None] & channel[:, None], motion - base["b0"], 0.0)
        result[int(sid)] = system.encode_identity(
            residual[None], valid[None], reference_channel_mask=channel[None]
        )
    return result


def _region_energy(pred, target, valid, channel_mask, channels):
    m = valid[..., None] & channel_mask[:, None]
    m = m[..., list(channels)]
    x = pred[..., list(channels)]
    y = target[..., list(channels)]
    denom = m.sum().clamp_min(1)
    return {
        "pred_rms": float(torch.where(m, x.square(), torch.zeros_like(x)).sum().sqrt() / denom.sqrt()),
        "target_rms": float(torch.where(m, y.square(), torch.zeros_like(y)).sum().sqrt() / denom.sqrt()),
        "pred_mean_abs": float(torch.where(m, x.abs(), torch.zeros_like(x)).sum() / denom),
        "target_mean_abs": float(torch.where(m, y.abs(), torch.zeros_like(y)).sum() / denom),
    }


def _merge_energy(rows):
    keys = rows[0]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def _paired_delta(first, second, valid, channel, channels):
    """Paired same-noise condition delta, excluding invalid/padded frames."""
    indices = list(channels)
    if channel.ndim == 1:
        channel = channel.unsqueeze(0)
    if valid.ndim == 1:
        valid = valid.unsqueeze(0)
    m = valid[..., None] & channel[:, None, :]
    m = m[..., indices]
    delta = first[..., indices] - second[..., indices]
    denom = m.sum().clamp_min(1)
    return {
        "delta_rms": float(torch.where(m, delta.square(), torch.zeros_like(delta)).sum().sqrt() / denom.sqrt()),
        "delta_mean_abs": float(torch.where(m, delta.abs(), torch.zeros_like(delta)).sum() / denom),
    }


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    provenance = json.loads((args.run_root / "provenance.json").read_text(encoding="utf8"))
    recipe = provenance["recipe"]
    if checkpoint.get("stage") != "audio":
        raise ValueError("Phase-1 diagnostic requires the final audio-stage checkpoint")
    if checkpoint.get("recipe_sha256") != provenance.get("recipe_sha256"):
        raise ValueError("Checkpoint and provenance recipe hashes differ")
    data = load_packed(args.data, materialize=False, with_refs=True)
    if data["provenance"].get("manifest_sha256") != checkpoint.get("data_manifest_sha256"):
        raise ValueError("Packed data manifest differs from checkpoint")
    system = NeutralAffectSystem(checkpoint["config"]).to(device).eval()
    system.load_state_dict(checkpoint["system"], strict=True)
    stats = checkpoint["feature_stats"]
    audio = SlowStateAffect(stats["mean"], stats["std"], stride=recipe["args"]["stride"]).to(device).eval()
    audio.load_state_dict(checkpoint["audio"], strict=True)
    identities = _identity_cache(system, data, device)
    probe_ck = torch.load(args.probe, map_location="cpu", weights_only=False)
    if probe_ck.get("train_manifest_sha256") != checkpoint.get("data_manifest_sha256") or probe_ck.get("test_used_for_selection"):
        raise ValueError("Independent probe is not bound to the checkpoint train manifest")
    probe = MotionEmotionProbe(probe_ck["feature_dim"], probe_ck["hidden"], len(probe_ck["classes"])).eval()
    probe.load_state_dict(probe_ck["model"])
    support = probe_ck["channel_support"].bool()

    q = data["splits"]["validation"]
    total = len(q["_lengths"])
    ids = torch.arange(total)
    if args.limit:
        ids = ids[: min(args.limit, total)]
    # One deterministic permutation is reused for all batches and all conditions.
    perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.shuffle_seed))
    global_all = []
    intensity_all = []
    for ix in ids.split(args.batch_size):
        b = q.batch(ix, device)
        out = audio_affect(audio, b["audio_features"], b["valid"])
        global_all.append(out["global"].detach().cpu())
        intensity_all.append(out["intensity_value"].detach().cpu())
    global_all = torch.cat(global_all)
    intensity_all = torch.cat(intensity_all)

    names = ["audio_pred", "teacher_oracle", "zero_all", "zero_global", "shuffle_global", "intensity_low", "intensity_high"]
    collected = {name: {"pred": [], "target": [], "valid": [], "channel": [], "labels": []} for name in names}
    noise_gen = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    for local_start, ix in enumerate(ids.split(args.batch_size)):
        b = q.batch(ix, device)
        ident = batch_identity(identities, b)
        base = {"b0": b["b0"], "h0": b["h0"]} if "b0" in b else base_forward(system, b["content"], b["valid"])
        # Packed validation batches intentionally omit frame caches. Teacher
        # extraction follows the same current B0 used by generation.
        b["b0"] = base["b0"]
        b["h0"] = base["h0"]
        audio_out = audio_affect(audio, b["audio_features"], b["valid"])
        teacher_out = teacher_affect(system, b, ident)
        oracle = merge_teacher_audio(teacher_out, audio_out)
        n = torch.randn((len(ix), b["valid"].shape[1], 52), generator=noise_gen)
        n = n.to(device)
        shuffled_global = global_all[perm[local_start * args.batch_size: local_start * args.batch_size + len(ix)]].to(device)
        shuffled_intensity = intensity_all[perm[local_start * args.batch_size: local_start * args.batch_size + len(ix)]].to(device)

        def clone_affect(source):
            return {k: v for k, v in source.items()}

        conditions = {
            "audio_pred": audio_out,
            "teacher_oracle": oracle,
        }
        zero = clone_affect(audio_out)
        zero["global"] = torch.zeros_like(zero["global"])
        zero["intensity_value"] = torch.zeros_like(zero["intensity_value"])
        zero["u_a"] = torch.zeros_like(zero["u_a"])
        conditions["zero_all"] = zero
        zero_global = clone_affect(audio_out)
        zero_global["global"] = torch.zeros_like(zero_global["global"])
        zero_global["intensity_value"] = torch.zeros_like(zero_global["intensity_value"])
        conditions["zero_global"] = zero_global
        shuffled = clone_affect(audio_out)
        shuffled["global"] = shuffled_global
        shuffled["intensity_value"] = shuffled_intensity
        conditions["shuffle_global"] = shuffled
        low = clone_affect(audio_out); low["intensity_value"] = torch.zeros_like(low["intensity_value"])
        high = clone_affect(audio_out); high["intensity_value"] = torch.full_like(high["intensity_value"], 3.0)
        conditions["intensity_low"] = low
        conditions["intensity_high"] = high

        for name, affect in conditions.items():
            pred = system.generate(
                b["content"], b["valid"], ident, affect,
                initial_noise=n,
                steps=(args.decode_steps or recipe["args"]["decode_steps"]),
                base=base,
            )["motion"].detach().cpu()
            cpu_b = {k: v.detach().cpu() for k, v in b.items() if torch.is_tensor(v)}
            out_store = collected[name]
            out_store["pred"].extend([pred[j, :int(q["_lengths"][int(i)])] for j, i in enumerate(ix.tolist())])
            out_store["target"].extend([cpu_b["motion"][j, :int(q["_lengths"][int(i)])] for j, i in enumerate(ix.tolist())])
            out_store["valid"].extend([cpu_b["valid"][j, :int(q["_lengths"][int(i)])] for j, i in enumerate(ix.tolist())])
            out_store["channel"].extend([cpu_b["channel_mask"][j] for j in range(len(ix))])
            out_store["labels"].extend(cpu_b["emotion_id"].tolist())

    result = {
        "schema": "phase1_condition_diagnostic_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "data": str(args.data),
        "data_manifest_sha256": checkpoint["data_manifest_sha256"],
        "probe": str(args.probe),
        "clips": len(ids),
        "conditions": {},
        "fixed": {"noise_seed": args.noise_seed, "shuffle_seed": args.shuffle_seed,
                  "decode_steps": args.decode_steps or recipe["args"]["decode_steps"],
                  "recipe_decode_steps": recipe["args"]["decode_steps"],
                  "batch_size": args.batch_size},
        "support": {"articulatory_mouth": [14,15,16,17,18,19,20,21,22,31,32,37,38,39,40], "mouth": list(MOUTH), "probe_channels": int(support.sum())},
    }
    for name, store in collected.items():
        probe_pred, labels = [], []
        clip_metrics, energies, jaw = [], [], []
        for pred, target, valid, channel, label in zip(store["pred"], store["target"], store["valid"], store["channel"], store["labels"]):
            sample = pred.unsqueeze(0).numpy()
            scored = score_fullface(sample, {"target52": target.numpy(), "valid": valid.numpy(), "times": np.arange(len(valid), dtype=np.float64) / 25.0, "channel_mask": np.broadcast_to(channel.numpy(), target.shape)})
            clip_metrics.append({k: scored["metrics"][k]["value"] for k in ("arkit_mbe", "arkit_lbe")})
            energies.append({"mouth": _region_energy(pred.unsqueeze(0), target.unsqueeze(0), valid.unsqueeze(0), channel.unsqueeze(0), MOUTH), "jaw_open_pred_mean": float(pred[valid, JAW_OPEN].mean()), "jaw_open_target_mean": float(target[valid, JAW_OPEN].mean()), "jaw_open_pred_rms": float(pred[valid, JAW_OPEN].square().mean().sqrt()), "jaw_open_target_rms": float(target[valid, JAW_OPEN].square().mean().sqrt())})
            feat = motion_features(pred[:, support], valid)
            with torch.no_grad(): probe_pred.append(int(probe(feat[None]).argmax(-1)))
            labels.append(label)
        mouth_rows = [x["mouth"] for x in energies]
        result["conditions"][name] = {
            "mbe": float(np.mean([x["arkit_mbe"] for x in clip_metrics])),
            "lbe": float(np.mean([x["arkit_lbe"] for x in clip_metrics])),
            "mouth": _merge_energy(mouth_rows),
            "jaw_open_pred_mean": float(np.mean([x["jaw_open_pred_mean"] for x in energies])),
            "jaw_open_target_mean": float(np.mean([x["jaw_open_target_mean"] for x in energies])),
            "jaw_open_pred_rms": float(np.mean([x["jaw_open_pred_rms"] for x in energies])),
            "jaw_open_target_rms": float(np.mean([x["jaw_open_target_rms"] for x in energies])),
            "independent_probe": classification_metrics(torch.tensor(labels), torch.tensor(probe_pred), probe_ck["classes"]),
        }
    # Every condition was generated with the same per-clip noise tensor.  Save
    # paired deltas explicitly so aggregate RMS cannot be mistaken for causal
    # condition sensitivity.
    paired = {}
    reference_name = "audio_pred"
    for name in names:
        if name == reference_name:
            continue
        rows = []
        for first, second, valid, channel in zip(
            collected[name]["pred"], collected[reference_name]["pred"],
            collected[name]["valid"], collected[name]["channel"]):
            row = {
                "mouth": _paired_delta(first, second, valid, channel, MOUTH),
                "articulatory_mouth": _paired_delta(first, second, valid, channel,
                                                     (14,15,16,17,18,19,20,21,22,31,32,37,38,39,40)),
                "jaw_open": _paired_delta(first, second, valid, channel, (JAW_OPEN,)),
            }
            rows.append(row)
        paired[f"{name}_minus_{reference_name}"] = {
            region: {metric: float(np.mean([row[region][metric] for row in rows]))
                     for metric in rows[0][region]}
            for region in rows[0]
        }
    result["paired_deltas"] = paired
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"event": "complete", "output": str(args.output), "clips": len(ids)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=42)
    p.add_argument("--shuffle-seed", type=int, default=20261003)
    p.add_argument("--decode-steps", type=int, default=0,
                   help="Optional solver-step override; 0 uses the checkpoint recipe.")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.decode_steps < 0:
        raise ValueError("decode steps must be positive, or 0 to use the checkpoint recipe")
    run(args)


if __name__ == "__main__":
    main()
