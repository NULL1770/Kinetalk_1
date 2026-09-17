"""Reproducible real-data feasibility experiment (no VA, no query references).

This measures training fit separately from held-out sentences. It does not
declare publication readiness or equate nonzero dynamics with correct emotion.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset, stack_clips
from kinetalk_b0.semantic_losses import style_contrastive
from kinetalk_b0.utils import freeze_module
from scripts.neutral_affect_metrics import motion_metrics, plot_motion_curves


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def device_batch(clips, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in stack_clips(clips).items()}


def observed(b):
    return b["valid"].unsqueeze(-1) & b["channel_mask"].unsqueeze(1)


def affect_residual(b, identity):
    """Unobserved channels remain missing after subtracting the identity bias."""
    return torch.where(observed(b), b["residual"] - identity["baseline"][:, None], 0)


def masked_mse(x, y, mask):
    return (x - y)[mask.expand_as(x)].square().mean()


def semantics(a, b):
    loss = F.cross_entropy(a["emotion_logits"], b["emotion_id"])
    ok = b["intensity_valid"] & (b["intensity_id"] >= 0)
    if ok.any():
        loss = loss + F.cross_entropy(a["intensity_logits"][ok], b["intensity_id"][ok])
    return loss


def derivative_loss(x, y, b):
    pair = observed(b)[:, 1:] & observed(b)[:, :-1]
    # Per-frame displacement at the audited 25 Hz native clock. No invalid gaps.
    return masked_mse(x[:, 1:] - x[:, :-1], y[:, 1:] - y[:, :-1], pair)


def select(mapping, ids):
    return {k: v[ids] if torch.is_tensor(v) else [v[int(i)] for i in ids] for k, v in mapping.items()}


def optimize(loss, optimizer, params):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def prepare(system, data, device):
    q = device_batch(data.queries, device)
    base = system.base(q["content"], q["valid"])
    # Missing channels have no measurement; never encode zero-filled errors.
    q["residual"] = torch.where(observed(q), q["motion"] - base["b0"], 0)
    return q, base


def cached_identity(system, enrollment):
    return system.encode_identity(enrollment["residual"], enrollment["valid"])


def distill(audio, teacher, scales):
    global_loss = ((audio["global"] - teacher["global"]) / scales["global"]).square().mean()
    controls = ((audio["controls"] - teacher["controls"]) / scales["controls"]).square()
    weight = teacher["control_weight"].unsqueeze(-1)
    control_loss = (controls * weight).sum() / (weight.sum() * controls.shape[-1]).clamp_min(1)
    return global_loss + control_loss


@torch.no_grad()
def eval_split(system, q, base, identities, out, name, steps, seed, plot=False):
    system.eval()
    identity = select(identities, q["speaker_id"])
    teacher = system.encode_motion(affect_residual(q, identity), q["valid"])
    audio = system.encode_audio(q["audio"], q["valid"])
    rng = torch.Generator(device=q["motion"].device).manual_seed(seed)
    noise = torch.randn(q["motion"].shape, device=q["motion"].device, generator=rng)
    result = {"n": len(q["motion"]), "clips": q["clip_id"], "conditions": {}}
    saved = {"target": q["motion"].cpu(), "b0": base["b0"].cpu(), "valid": q["valid"].cpu(),
             "times": q["times"].cpu(), "channel_mask": q["channel_mask"].cpu(), "clips": q["clip_id"],
             "teacher_controls": teacher["controls"].cpu(), "audio_controls": audio["controls"].cpu()}
    # Neutral origin and B0 comparisons use identical GT but are articulation
    # baselines, not alleged ground-truth emotional counterfactuals.
    result["b0"] = motion_metrics(base["b0"], q["motion"], base["b0"], q["valid"], q["channel_mask"], q["times"])
    for source, affect in (("teacher", teacher), ("audio", audio)):
        result[source + "_emotion_accuracy"] = float((affect["emotion_logits"].argmax(-1) == q["emotion_id"]).float().mean())
        ok = q["intensity_valid"]
        result[source + "_level_accuracy"] = float((affect["intensity_logits"].argmax(-1)[ok] == q["intensity_id"][ok]).float().mean()) if ok.any() else None
        result[source + "_control_std"] = float(affect["controls"][affect["control_mask"]].std())
        full_motion = None
        for intervention in ("full", "zero", "mean", "reverse"):
            controlled = dict(affect)
            if intervention == "zero":
                controlled["local"] = torch.zeros_like(affect["local"])
            elif intervention == "mean":
                # Centering stride controls does not guarantee the interpolated
                # field's mean is exactly zero, especially for partial bins.
                mask = q["valid"].unsqueeze(-1)
                mean = (affect["local"] * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
                controlled["local"] = mean.expand_as(affect["local"]) * mask
            elif intervention == "reverse":
                local = affect["local"].clone()
                for i in range(len(local)):
                    valid = q["valid"][i]
                    local[i, valid] = local[i, valid].flip(0)
                controlled["local"] = local
            prediction = system.generate(q["content"], q["valid"], identity, controlled,
                                         initial_noise=noise, steps=steps, base=base)["motion"]
            key = source + "_" + intervention
            metrics = motion_metrics(prediction, q["motion"], base["b0"], q["valid"], q["channel_mask"], q["times"])
            if intervention == "full":
                full_motion = prediction
            else:
                metrics["change_from_full_mse"] = float(masked_mse(prediction, full_motion, observed(q)))
            # Per-clip paired MSE permits later inspection/uncertainty analysis.
            mask = observed(q)
            metrics["per_clip_mse"] = ((prediction - q["motion"]).square().masked_fill(~mask, 0).sum((1, 2)) / mask.sum((1, 2))).cpu().tolist()
            result["conditions"][key] = metrics
            saved[key] = prediction.cpu()
        if plot:
            plot_motion_curves(full_motion.cpu(), q["motion"].cpu(), base["b0"].cpu(), q["valid"].cpu(),
                               q["times"].cpu(), out / (name + "_" + source + ".png"),
                               sample_index=0, title=name + ": " + source, channel_mask=q["channel_mask"].cpu())
    valid = teacher["control_mask"]
    result["audio_teacher_control_mse"] = float((audio["controls"] - teacher["controls"])[valid].square().mean())
    result["zero_teacher_control_mse"] = float(teacher["controls"][valid].square().mean())
    result["audio_teacher_global_mse"] = float((audio["global"] - teacher["global"]).square().mean())
    write_json(out / (name + ".json"), result)
    torch.save(saved, out / (name + "_curves.pt"))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/neutral_affect_pilot.yaml"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    t = cfg["training"]
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(cfg.get("device", "cuda"))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "training.jsonl"
    if log_path.exists():
        raise FileExistsError("Use a fresh output directory to keep experiments unambiguous")

    def log(stage, step, **kw):
        record = {"time": time.time(), "stage": stage, "step": step, **kw}
        with log_path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    train = NeutralAffectDataset(args.data / "train.pt")
    heldout = NeutralAffectDataset(args.data / "heldout.pt")
    system = NeutralAffectSystem(cfg).to(device)
    ck = torch.load(args.stage1, map_location="cpu", weights_only=False)
    system.stage1.load_state_dict(ck["model"], strict=True)
    freeze_module(system.stage1)
    provenance = {"schema": "neutral_affect_v10_real_pilot", "config": cfg,
                  "stage1_path": str(args.stage1), "stage1_sha256": sha(args.stage1),
                  "train_sha256": sha(args.data / "train.pt"), "heldout_sha256": sha(args.data / "heldout.pt"),
                  "config_sha256": sha(args.config), "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                  "interpretation": "Heldout is new sentences of enrolled people; B0 may have seen these in its earlier pretraining. No unseen-person claim.",
                  "source": {str(f): sha(f) for f in (Path(__file__), Path("kinetalk_b0/models/neutral_affect.py"), Path("kinetalk_b0/neutral_data.py"), Path("kinetalk_b0/models/dit.py"))}}
    write_json(out / "provenance.json", provenance)
    if args.resume:
        system.load_state_dict(torch.load(args.resume, map_location=device, weights_only=False)["model"], strict=True)
    tr, tr_base = prepare(system, train, device)
    va, va_base = prepare(system, heldout, device)
    refs = train.identity_references
    speaker_ids = sorted(refs)
    if speaker_ids != list(range(len(speaker_ids))):
        raise ValueError("Pilot requires contiguous enrollment IDs")
    ref_count = len(refs[0])
    all_refs = device_batch([r for s in speaker_ids for r in refs[s]], device)
    ref_base = system.base(all_refs["content"], all_refs["valid"])
    residual = torch.where(observed(all_refs), all_refs["motion"] - ref_base["b0"], 0)
    enrollment = {"residual": residual.reshape(len(refs), ref_count, *residual.shape[1:]),
                  "valid": all_refs["valid"].reshape(len(refs), ref_count, -1)}
    log("preflight", 0, train=len(train), heldout=len(heldout), speakers=len(refs),
        observed_channels=int(tr["channel_mask"].all(0).sum()),
        parameter_count=sum(p.numel() for p in system.parameters()))
    batch_size = int(t.get("batch_size", 8))

    # 1. Fit the identity baseline using two disjoint neutral reference sets.
    params = list(system.identity_encoder.parameters()) + list(system.identity_bias.parameters())
    opt = torch.optim.AdamW(params, lr=float(t.get("identity_lr", 0.001)))
    system.train()
    for step in range(int(t.get("identity_steps", 200))):
        a = system.encode_identity(enrollment["residual"][:, :2], enrollment["valid"][:, :2])
        b = system.encode_identity(enrollment["residual"][:, 2:], enrollment["valid"][:, 2:])
        loss_baseline = (F.mse_loss(a["baseline"], b["neutral_mean"].detach()) + F.mse_loss(b["baseline"], a["neutral_mean"].detach())) / system.residual_scale ** 2
        contrast = style_contrastive(a["code"], b["code"], torch.arange(len(refs), device=device))
        loss = loss_baseline + 0.05 * contrast
        norm = optimize(loss, opt, params)
        if step % 50 == 0:
            log("identity", step, loss=float(loss), baseline=float(loss_baseline), contrast=float(contrast), grad_norm=norm)
    freeze_module(system.identity_encoder)
    freeze_module(system.identity_bias)
    with torch.no_grad():
        identities = cached_identity(system, enrollment)
        a = system.encode_identity(enrollment["residual"][:, :2], enrollment["valid"][:, :2])
        b = system.encode_identity(enrollment["residual"][:, 2:], enrollment["valid"][:, 2:])
        scores = F.normalize(a["code"], dim=-1) @ F.normalize(b["code"], dim=-1).T
        identity_report = {"training_reference_retrieval": float((scores.argmax(-1) == torch.arange(len(refs), device=device)).float().mean()),
                           "chance": 1 / len(refs), "mean_baseline_mse": float(F.mse_loss(identities["baseline"], identities["neutral_mean"])),
                           "scope": "Both reference sets participated in identity training; this is training-view fit, not heldout retrieval."}
    write_json(out / "identity.json", identity_report)
    decode_steps = int(t.get("decode_steps", 12))
    initial = eval_split(system, tr, tr_base, identities, out, "before_train", decode_steps, seed)
    result = {"identity": identity_report, "before_train": initial}

    # 2. Joint motion teacher + renderer. Classification alone cannot teach
    # local dynamics; reconstruction backpropagates through its control points.
    params = list(system.motion_teacher.parameters()) + list(system.local_projection.parameters()) + list(system.renderer.parameters())
    opt = torch.optim.AdamW(params, lr=float(t.get("teacher_lr", 0.0005)), weight_decay=1e-5)
    system.train()
    for step in range(int(t.get("teacher_steps", 1000))):
        ids = torch.randint(len(train), (batch_size,), device=device)
        b, base = select(tr, ids), select(tr_base, ids)
        identity = select(identities, b["speaker_id"])
        affect = system.encode_motion(affect_residual(b, identity), b["valid"])
        times = torch.rand(batch_size, device=device)
        times[torch.rand(batch_size, device=device) < 0.2] = 0
        output = system.flow(b["motion"], b["content"], b["valid"], identity, affect, time=times, base=base)
        flow = masked_mse(output["prediction"], output["velocity_target"], observed(b))
        semantic = semantics(affect, b)
        velocity = derivative_loss(output["motion"], b["motion"], b) / system.residual_scale ** 2
        loss = flow + 0.1 * semantic + 0.1 * velocity
        norm = optimize(loss, opt, params)
        if step % 100 == 0:
            log("teacher", step, loss=float(loss), flow=float(flow), semantic=float(semantic), velocity=float(velocity), grad_norm=norm,
                local_grad=float(system.motion_teacher.control_head.weight.grad.norm()),
                control_std=float(affect["controls"][affect["control_mask"]].std()))
    for name in ("train", "heldout"):
        q, base = (tr, tr_base) if name == "train" else (va, va_base)
        result["after_teacher_" + name] = eval_split(system, q, base, identities, out, "after_teacher_" + name, decode_steps, seed, plot=True)
    torch.save({"model": system.state_dict(), "provenance": provenance, "stage": "motion_teacher"}, out / "teacher.pt")

    # 3. Freeze motion and generator; audio learns exactly the teacher's global
    # and local coordinates. Normalization uses training teacher statistics only.
    for module in (system.motion_teacher, system.local_projection, system.renderer):
        freeze_module(module)
    with torch.no_grad():
        identity = select(identities, tr["speaker_id"])
        teacher = system.encode_motion(affect_residual(tr, identity), tr["valid"])
        scales = {"global": teacher["global"].std(0).clamp_min(0.05),
                  "controls": teacher["controls"][teacher["control_mask"]].std(0).clamp_min(0.05)}
    params = list(system.audio_encoder.parameters())
    opt = torch.optim.AdamW(params, lr=float(t.get("audio_lr", 0.0005)), weight_decay=1e-5)
    system.train()
    for step in range(int(t.get("audio_steps", 800))):
        ids = torch.randint(len(train), (batch_size,), device=device)
        b, target = select(tr, ids), select(teacher, ids)
        audio = system.encode_audio(b["audio"], b["valid"])
        latent = distill(audio, target, scales)
        semantic = semantics(audio, b)
        loss = latent + 0.1 * semantic
        norm = optimize(loss, opt, params)
        if step % 100 == 0:
            log("audio", step, loss=float(loss), distill=float(latent), semantic=float(semantic), grad_norm=norm)
    for name in ("train", "heldout"):
        q, base = (tr, tr_base) if name == "train" else (va, va_base)
        result["final_" + name] = eval_split(system, q, base, identities, out, "final_" + name, decode_steps, seed, plot=True)
    torch.save({"model": system.state_dict(), "provenance": provenance, "stage": "audio", "audio_stats": train.cache["audio_stats"]}, out / "audio.pt")
    before_mse = initial["conditions"]["teacher_full"]["masked_mse"]
    tc = result["final_train"]["conditions"]
    hc = result["final_heldout"]["conditions"]
    result["checks"] = {
        "teacher_fit_improved": tc["teacher_full"]["masked_mse"] < before_mse,
        "teacher_dynamic_better_than_static_train": tc["teacher_full"]["masked_mse"] < tc["teacher_mean"]["masked_mse"],
        "teacher_dynamic_better_than_static_heldout": hc["teacher_full"]["masked_mse"] < hc["teacher_mean"]["masked_mse"],
        "audio_dynamic_better_than_static_heldout": hc["audio_full"]["masked_mse"] < hc["audio_mean"]["masked_mse"],
        "audio_distillation_beats_zero_heldout": result["final_heldout"]["audio_teacher_control_mse"] < result["final_heldout"]["zero_teacher_control_mse"],
        "note": "Exploratory checks, not significance tests or proof of emotion disentanglement; no all-pass gate is implied."}
    write_json(out / "report.json", result)
    log("finished", 0, checks=result["checks"])


if __name__ == "__main__":
    main()
