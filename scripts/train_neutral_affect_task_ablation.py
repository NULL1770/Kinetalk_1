"""Matched audio continuation: latent controls versus frozen-renderer flow loss.

Both arms start from exactly the same audio checkpoint and preserve its scales
and audio statistics. Summary metrics concern latents/semantics, not decoding.
"""
from __future__ import annotations

import argparse
import copy
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
from scripts.train_neutral_affect_audio_ablation import state_hash, summarize
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, derivative_loss, device_batch, masked_mse,
    observed, optimize, prepare, select, semantics, sha, write_json,
)


def task_objective(system, query, base, identity, teacher, scales, objective, flow_rng):
    """Frozen renderer parameters still transmit the flow gradient to audio."""
    audio = system.encode_audio(query["audio"], query["valid"])
    global_loss = ((audio["global"] - teacher["global"]) / scales["global"]).square().mean()
    semantic = semantics(audio, query)
    flow = velocity = global_loss.new_zeros(())
    if objective == "latent":
        error = ((audio["controls"] - teacher["controls"]) / scales["controls"]).square()
        weight = teacher["control_weight"].unsqueeze(-1)
        task = (error * weight).sum() / (weight.sum() * error.shape[-1]).clamp_min(1)
    elif objective == "flow":
        batch = query["motion"].shape[0]
        device = query["motion"].device
        times = torch.rand(batch, device=device, generator=flow_rng)
        times[torch.rand(batch, device=device, generator=flow_rng) < 0.2] = 0
        noise = torch.randn(query["motion"].shape, device=device, generator=flow_rng, dtype=query["motion"].dtype)
        output = system.flow(query["motion"], query["content"], query["valid"], identity, audio,
                             noise=noise, time=times, base=base)
        flow = masked_mse(output["prediction"], output["velocity_target"], observed(query))
        velocity = derivative_loss(output["motion"], query["motion"], query) / system.residual_scale ** 2
        task = flow + 0.1 * velocity
    else:
        raise ValueError("objective must be latent or flow")
    return global_loss + task + 0.1 * semantic, {"global": global_loss, "task": task,
                                               "semantic": semantic, "flow": flow, "velocity": velocity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--objective", choices=("latent", "flow"), required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--audio-dilations", nargs=3, type=int, help="Controlled context-only change; keeps three TCN layers and all checkpoint tensors")
    args = parser.parse_args()
    if args.steps < 0:
        raise ValueError("steps must be nonnegative")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "audio" or not all(k in checkpoint for k in ("scales", "audio_stats")):
        raise ValueError("An audio checkpoint with fixed scales and audio_stats is required")
    original = checkpoint.get("provenance", {}).get("config", {})
    if any(original.get(k) != cfg.get(k) for k in ("data", "model")):
        raise ValueError("Model/data configuration must match the checkpoint")
    cfg = copy.deepcopy(cfg)
    if args.audio_dilations is not None:
        if min(args.audio_dilations) < 1:
            raise ValueError("Audio dilations must be positive")
        cfg["model"]["audio_dilations"] = args.audio_dilations
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(cfg.get("device", "cuda"))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")
    system = NeutralAffectSystem(cfg).to(device)
    system.load_state_dict(checkpoint["model"], strict=True)
    freeze_module(system)
    system.audio_encoder.requires_grad_(True)
    system.eval()
    frozen = lambda: {k: v for k, v in system.state_dict().items() if not k.startswith("audio_encoder.")}
    frozen_hash = state_hash(frozen())
    scales = {k: v.detach().to(device) for k, v in checkpoint["scales"].items()}
    if any(not torch.isfinite(scales[k]).all() or not (scales[k] > 0).all() for k in ("global", "controls")):
        raise ValueError("Checkpoint scales must be finite and positive")
    root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), root / "scripts/train_neutral_affect_audio_ablation.py",
                    root / "scripts/train_neutral_affect_pilot.py", root / "kinetalk_b0/neutral_data.py",
                    root / "kinetalk_b0/models/neutral_affect.py", root / "kinetalk_b0/models/dit.py",
                    root / "kinetalk_b0/models/model.py", root / "kinetalk_b0/models/encoders.py", root / "kinetalk_b0/utils.py"]
    provenance = {"schema": "neutral_affect_audio_task_ablation_v1", "objective": args.objective,
                  "steps": args.steps, "seed": args.seed, "config": cfg, "config_sha256": sha(args.config),
                  "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha(args.checkpoint),
                  "train_sha256": sha(args.data / "train.pt"), "heldout_sha256": sha(args.data / "heldout.pt"),
                  "source_sha256": {str(p.relative_to(root)): sha(p) for p in source_paths},
                  "initial_state_sha256": state_hash(system.state_dict()),
                  "initial_audio_state_sha256": state_hash(system.audio_encoder.state_dict()),
                  "initial_frozen_state_sha256": frozen_hash, "scales_sha256": state_hash(checkpoint["scales"]),
                  "torch": torch.__version__, "optimizer": "New matched AdamW for each arm; checkpoint optimizer not resumed",
                  "scope": "Heldout sentences of enrolled people. No decoding or unseen-person claim; nonzero response is not success."}
    write_json(args.output / "provenance.json", provenance)
    data = {name: NeutralAffectDataset(args.data / (name + ".pt")) for name in ("train", "heldout")}
    for name, dataset in data.items():
        if any(not torch.equal(dataset.cache["audio_stats"][k].cpu(), checkpoint["audio_stats"][k].cpu()) for k in ("mean", "std")):
            raise ValueError(f"{name} audio normalization differs from the checkpoint")
    prepared = {name: prepare(system, dataset, device) for name, dataset in data.items()}
    queries, bases = {k: v[0] for k, v in prepared.items()}, {k: v[1] for k, v in prepared.items()}
    refs = data["train"].identity_references
    if sorted(refs) != list(range(len(refs))) or len({len(v) for v in refs.values()}) != 1:
        raise ValueError("Enrollment requires contiguous IDs and equal reference counts")
    reference = device_batch([r for speaker in sorted(refs) for r in refs[speaker]], device)
    with torch.no_grad():
        residual = torch.where(observed(reference), reference["motion"] - system.base(reference["content"], reference["valid"])["b0"], 0)
        identities = cached_identity(system, {"residual": residual.reshape(len(refs), len(refs[0]), *residual.shape[1:]),
                                            "valid": reference["valid"].reshape(len(refs), len(refs[0]), -1)})
        targets = {name: system.encode_motion(affect_residual(q, select(identities, q["speaker_id"])), q["valid"]) for name, q in queries.items()}
    report = {"before": {name: summarize(system, q, targets[name], scales) for name, q in queries.items()}}
    params = list(system.audio_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(cfg["training"].get("audio_lr", 0.0005)), weight_decay=1e-5)
    batch_rng = torch.Generator(device="cpu").manual_seed(args.seed)
    flow_rng = torch.Generator(device=device).manual_seed(args.seed + 1000003)
    batch_hash = hashlib.sha256()
    system.audio_encoder.train()
    for step in range(args.steps):
        if any(p.requires_grad for n, p in system.named_parameters() if not n.startswith("audio_encoder.")):
            raise RuntimeError("A non-audio parameter became trainable")
        cpu_ids = torch.randint(len(data["train"]), (int(cfg["training"].get("batch_size", 8)),), generator=batch_rng)
        batch_hash.update(cpu_ids.numpy().tobytes())
        ids = cpu_ids.to(device)
        q = select(queries["train"], ids)
        loss, parts = task_objective(system, q, select(bases["train"], ids), select(identities, q["speaker_id"]),
                                     select(targets["train"], ids), scales, args.objective, flow_rng)
        norm = optimize(loss, optimizer, params)
        local_grad = system.audio_encoder.control_head.weight.grad
        if step % 100 == 0 or step + 1 == args.steps:
            if state_hash(frozen()) != frozen_hash or any(p.grad is not None for n, p in system.named_parameters() if not n.startswith("audio_encoder.")):
                raise RuntimeError("Frozen model changed or accumulated gradients")
            record = {"time": time.time(), "step": step + 1, "loss": float(loss), "grad_norm": norm,
                      "local_head_grad_norm": float(local_grad.norm()) if local_grad is not None else 0.0,
                      **{name: float(value) for name, value in parts.items()}}
            with (args.output / "training.jsonl").open("a", encoding="utf8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
    if state_hash(frozen()) != frozen_hash:
        raise RuntimeError("Frozen model state changed")
    report.update(after={name: summarize(system, q, targets[name], scales) for name, q in queries.items()},
                  minibatch_sha256=batch_hash.hexdigest(), frozen_state_unchanged=True)
    write_json(args.output / "summary.json", report)
    torch.save({"model": system.state_dict(), "provenance": provenance, "stage": "audio", "steps": args.steps,
                "audio_stats": checkpoint["audio_stats"], "scales": checkpoint["scales"],
                "optimizer": optimizer.state_dict(), "minibatch_sha256": batch_hash.hexdigest()}, args.output / "audio.pt")


if __name__ == "__main__":
    main()
