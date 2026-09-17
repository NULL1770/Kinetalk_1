"""Audio-only head-initialization ablation from one frozen motion teacher.

The random arm preserves the audio weights already in teacher.pt. The other
arm copies only its emotion/intensity classifiers. No decoding is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, device_batch, distill, observed, optimize,
    prepare, select, semantics, sha, write_json,
)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def summarize(system, query, teacher, scales):
    system.eval()
    audio = system.encode_audio(query["audio"], query["valid"])
    valid = teacher["control_mask"]
    level = query["intensity_valid"] & (query["intensity_id"] >= 0)
    return {"n": len(query["motion"]), "clips": query["clip_id"],
            "distill": float(distill(audio, teacher, scales)), "semantic": float(semantics(audio, query)),
            "audio_teacher_global_mse": float((audio["global"] - teacher["global"]).square().mean()),
            "audio_teacher_control_mse": float((audio["controls"] - teacher["controls"])[valid].square().mean()),
            "zero_teacher_control_mse": float(teacher["controls"][valid].square().mean()),
            "audio_teacher_intensity_value_mae": float((audio["intensity_value"] - teacher["intensity_value"]).abs().mean()),
            "emotion_accuracy": float((audio["emotion_logits"].argmax(-1) == query["emotion_id"]).float().mean()),
            "intensity_accuracy": float((audio["intensity_logits"].argmax(-1)[level] == query["intensity_id"][level]).float().mean()) if level.any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "teacher-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heads", choices=("random", "teacher_init"), required=True)
    parser.add_argument("--scales-data", type=Path, help="Fix distillation scaling to this training cache for a data-size comparison")
    args = parser.parse_args()
    if args.steps < 0:
        raise ValueError("steps must be nonnegative")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(cfg.get("device", "cuda"))
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "motion_teacher":
        raise ValueError("A motion_teacher checkpoint is required")
    original_config = checkpoint.get("provenance", {}).get("config", {})
    if any(original_config.get(k) != cfg.get(k) for k in ("data", "model")):
        raise ValueError("Model/data configuration must match the teacher checkpoint")
    system = NeutralAffectSystem(cfg).to(device)
    system.load_state_dict(checkpoint["model"], strict=True)
    checkpoint_audio_hash = state_hash(system.audio_encoder.state_dict())
    freeze_module(system)
    if args.heads == "teacher_init":
        for name in ("emotion_classifier", "intensity_classifier"):
            getattr(system.audio_encoder, name).load_state_dict(getattr(system.motion_teacher, name).state_dict(), strict=True)
    system.audio_encoder.requires_grad_(True)
    system.eval()
    frozen = lambda: {k: v for k, v in system.state_dict().items() if not k.startswith("audio_encoder.")}
    initial_frozen_hash = state_hash(frozen())
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root / "scripts/train_neutral_affect_pilot.py",
               root / "kinetalk_b0/models/neutral_affect.py", root / "kinetalk_b0/neutral_data.py",
               root / "kinetalk_b0/models/dit.py", root / "kinetalk_b0/models/model.py",
               root / "kinetalk_b0/models/encoders.py", root / "kinetalk_b0/utils.py"]
    provenance = {"schema": "neutral_affect_audio_heads_ablation_v1", "heads": args.heads,
                  "seed": args.seed, "steps": args.steps, "config": cfg, "config_sha256": sha(args.config),
                  "teacher_checkpoint": str(args.teacher_checkpoint.resolve()), "teacher_sha256": sha(args.teacher_checkpoint),
                  "train_sha256": sha(args.data / "train.pt"), "heldout_sha256": sha(args.data / "heldout.pt"),
                  "scales_cache_sha256": sha((args.scales_data or args.data) / "train.pt"),
                  "source_sha256": {str(p.relative_to(root)): sha(p) for p in sources},
                  "checkpoint_audio_state_sha256": checkpoint_audio_hash,
                  "initial_state_sha256": state_hash(system.state_dict()),
                  "initial_audio_state_sha256": state_hash(system.audio_encoder.state_dict()),
                  "initial_frozen_state_sha256": initial_frozen_hash, "torch": torch.__version__,
                  "scope": "Heldout sentences of enrolled people; latent/semantic summary only, no generation-quality conclusion."}
    write_json(args.output / "provenance.json", provenance)
    data = {name: NeutralAffectDataset(args.data / (name + ".pt")) for name in ("train", "heldout")}
    queries = {name: prepare(system, dataset, device)[0] for name, dataset in data.items()}
    refs = data["train"].identity_references
    if sorted(refs) != list(range(len(refs))) or len({len(v) for v in refs.values()}) != 1:
        raise ValueError("Enrollment requires contiguous speaker IDs and equal reference counts")
    reference = device_batch([r for speaker in sorted(refs) for r in refs[speaker]], device)
    with torch.no_grad():
        residual = torch.where(observed(reference), reference["motion"] - system.base(reference["content"], reference["valid"])["b0"], 0)
        identities = cached_identity(system, {"residual": residual.reshape(len(refs), len(refs[0]), *residual.shape[1:]),
                                            "valid": reference["valid"].reshape(len(refs), len(refs[0]), -1)})
        targets = {name: system.encode_motion(affect_residual(q, select(identities, q["speaker_id"])), q["valid"]) for name, q in queries.items()}
        teacher = targets["train"]
        scaling_teacher = teacher
        if args.scales_data:
            scaling_data = NeutralAffectDataset(args.scales_data / "train.pt")
            index = {clip: i for i, clip in enumerate(queries["train"]["clip_id"])}
            scaling_ids = torch.tensor([index[q["clip_id"]] for q in scaling_data.queries], device=device)
            scaling_teacher = select(teacher, scaling_ids)
        scales = {"global": scaling_teacher["global"].std(0).clamp_min(0.05),
                  "controls": scaling_teacher["controls"][scaling_teacher["control_mask"]].std(0).clamp_min(0.05)}
    report = {"before": {name: summarize(system, q, targets[name], scales) for name, q in queries.items()}}
    params = list(system.audio_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(cfg["training"].get("audio_lr", 0.0005)), weight_decay=1e-5)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    minibatch_hash = hashlib.sha256()
    system.audio_encoder.train()
    for step in range(args.steps):
        cpu_ids = torch.randint(len(data["train"]), (int(cfg["training"].get("batch_size", 8)),), generator=rng)
        minibatch_hash.update(cpu_ids.numpy().tobytes())
        ids = cpu_ids.to(device)
        query, target = select(queries["train"], ids), select(targets["train"], ids)
        audio = system.encode_audio(query["audio"], query["valid"])
        latent, semantic = distill(audio, target, scales), semantics(audio, query)
        loss = latent + 0.1 * semantic
        norm = optimize(loss, optimizer, params)
        record = {"time": time.time(), "step": step + 1, "loss": float(loss), "distill": float(latent), "semantic": float(semantic), "grad_norm": norm}
        with (args.output / "training.jsonl").open("a", encoding="utf8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if step % 100 == 0 or step + 1 == args.steps:
            print(json.dumps(record, allow_nan=False), flush=True)
    if state_hash(frozen()) != initial_frozen_hash:
        raise RuntimeError("Frozen teacher/generator state changed")
    report.update(after={name: summarize(system, q, targets[name], scales) for name, q in queries.items()},
                  minibatch_sha256=minibatch_hash.hexdigest(), frozen_state_unchanged=True)
    write_json(args.output / "summary.json", report)
    torch.save({"model": system.state_dict(), "provenance": provenance, "stage": "audio", "steps": args.steps,
                "audio_stats": data["train"].cache["audio_stats"], "scales": scales,
                "optimizer": optimizer.state_dict(), "minibatch_sha256": minibatch_hash.hexdigest()}, args.output / "audio.pt")


if __name__ == "__main__":
    main()
