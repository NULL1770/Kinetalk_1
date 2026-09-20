"""Train an isolated audio-dynamics receiver with final-output temporal losses.

This is a causal follow-up to ``train_full_staged.py``.  It warm-starts from
the completed five-stage run, freezes B0/identity/global/audio, and updates
only the local audio dynamic head plus upper innovation flow.  The additional
losses are evaluated on the *deployment* multi-step rollout:

* flow matching keeps the conditional distribution;
* predicted slow state and final output state must agree with motion targets;
* final frame velocity and per-region temporal standard deviation are matched.

The endpoint of each noise sample is never forced to one exact target.  This
keeps the experiment diagnostic and avoids treating one-to-many motion as a
deterministic regression problem.  No test targets are loaded and the output
is always isolated from historical/default checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_full_staged import (
    GROUPS,
    UPPER_INDICES,
    UpperFlow,
    batch_identity,
    cache_current_base,
    evaluate,
    huber,
    identity_cache,
    masked_slow_state,
    obs,
    optimize,
    readout_slow_state,
    subset,
    targets,
    upper_motion,
)
from scripts.full_staged_data import load_training_inputs
from scripts.train_full_staged import SlowStateAffect
from scripts.train_formal_predictable_projection import save_checkpoint, save_json
from scripts.train_predictable_renderer import state_hash


SCHEMA = "audio_dynamics_rollout_v3"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regional_std_loss(pred, truth, valid, anchors, scales, channel_mask):
    """Compare normalized per-channel temporal std with fixed valid masks.

    ``ddof=0`` and a minimum of two valid frames are fixed here.  Regions are
    reduced equally; no validation statistics are used to select weights.
    """
    losses = []
    for channels in (GROUPS["brows"], GROUPS["eyes_expression"]):
        cc = torch.as_tensor(channels, device=pred.device, dtype=torch.long)
        p = (pred[..., cc] - anchors[:, None, cc]) / scales[cc]
        y = (truth[..., cc] - anchors[:, None, cc]) / scales[cc]
        m = valid[..., None] & channel_mask[:, None, cc]
        count = m.sum(1)
        good = count >= 2
        if not good.any():
            continue
        p_sum = torch.where(m, p, 0.).sum(1)
        y_sum = torch.where(m, y, 0.).sum(1)
        p_mean = p_sum / count.clamp_min(1)
        y_mean = y_sum / count.clamp_min(1)
        p_var = torch.where(m, (p - p_mean[:, None]) ** 2, 0.).sum(1) / count.clamp_min(1)
        y_var = torch.where(m, (y - y_mean[:, None]) ** 2, 0.).sum(1) / count.clamp_min(1)
        losses.append((p_var.clamp_min(0).sqrt()[good] - y_var.clamp_min(0).sqrt()[good]).square().mean())
    return torch.stack(losses).mean() if losses else pred.sum() * 0.


def output_state_loss(pred_upper, b, scales, stride):
    """Read the final upper trajectory back through the same state operator."""
    full = b["b0"].clone()
    full[..., list(UPPER_INDICES)] = pred_upper
    observed = obs(b) & b["anchor_valid"][:, None]
    raw, raw_mask = readout_slow_state(full, observed, b["anchors"], scales)
    state = masked_slow_state(raw, raw_mask, stride=stride)
    return huber(state["state"], b["target_state"], state["state_mask"] & b["target_state_mask"])


def velocity_loss(pred_upper, b, scales):
    cc = torch.as_tensor(UPPER_INDICES, device=pred_upper.device, dtype=torch.long)
    p = pred_upper[..., :] / scales[cc]
    y = b["motion"][..., cc] / scales[cc]
    pair = obs(b)[:, 1:, cc] & obs(b)[:, :-1, cc]
    return huber(p[:, 1:] - p[:, :-1], y[:, 1:] - y[:, :-1], pair)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "audio", "targets", "enrollment", "native-root", "warmstart", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--decode-steps", type=int, default=12)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--smoke", action="store_true")
    return p


def main(args):
    if args.output.exists():
        raise FileExistsError("Fresh isolated output required")
    if args.epochs < 1 or args.batch_size < 1 or args.decode_steps < 1:
        raise ValueError("epochs, batch-size and decode-steps must be positive")
    args.output.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True

    data = load_training_inputs(args.source_run, args.audio, args.targets, args.enrollment, args.native_root)
    device = torch.device(args.device)
    system = data["system"].to(device).eval()
    cfg = data["config"]
    audio = SlowStateAffect(data["feature_stats"]["mean"], data["feature_stats"]["std"], stride=args.stride).to(device).eval()
    upper = UpperFlow(cfg, stride=args.stride).to(device).eval()
    local_audio = copy.deepcopy(audio).to(device).eval()

    warm = torch.load(args.warmstart, map_location="cpu", weights_only=False)
    for module, key in ((system, "system"), (audio, "audio"), (upper, "upper"), (local_audio, "local_audio")):
        module.load_state_dict(warm[key], strict=True)
    del warm
    cache_current_base(system, data, device)
    identities = identity_cache(system, data, device)
    scales = data["target_scales"].to(device)

    # Bind the exact source code and warm-start artifact before training.
    root = Path(__file__).resolve().parents[1]
    source_files = [root / "scripts" / "train_dynamics_v3.py", root / "scripts" / "train_full_staged.py",
                    root / "scripts" / "full_staged_data.py", root / "kinetalk_b0" / "models" / "slow_state_affect.py",
                    root / "kinetalk_b0" / "models" / "dit.py"]
    recipe = {
        "schema": SCHEMA, "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "decode_steps": args.decode_steps, "stride": args.stride, "smoke": args.smoke,
        "loss_weights": {"flow": 1.0, "state": 0.5, "output_state": 0.5, "velocity": 0.25, "std": 0.1, "domain": 0.05},
        "warmstart": str(args.warmstart.resolve()), "warmstart_sha256": sha(args.warmstart),
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in source_files},
        "data_provenance": data["provenance"], "test_loaded": False, "default_replaced": False,
        "scope": "inner development diagnostic; warm-started receiver objective ablation",
    }
    save_json(args.output / "provenance.json", recipe)

    for module in (system, audio, upper, local_audio):
        module.requires_grad_(False); module.eval(); module.zero_grad(set_to_none=True)
    local_audio.requires_grad_(True); upper.requires_grad_(True)
    params = list(local_audio.parameters()) + list(upper.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
    items = torch.arange(len(data["splits"]["train"]["valid"]))
    if args.smoke:
        items = items[:min(len(items), args.batch_size * 2)]
    epochs = 1 if args.smoke else args.epochs
    started = time.monotonic(); total_steps = 0
    save_json(args.output / "status.json", {"status": "training", "stage": "dynamics_v3", "test_loaded": False})

    for epoch in range(epochs):
        order = items[torch.randperm(len(items))]
        sums = {k: [] for k in ("total", "flow", "state", "output_state", "velocity", "std", "domain")}
        begin = time.monotonic()
        for bi, ix in enumerate(order.split(args.batch_size)):
            b = subset(data["splits"]["train"], ix, device)
            # These global/identity inputs are frozen; only the dynamic receiver learns.
            identity = batch_identity(identities, b)
            base = {"b0": b["b0"], "h0": b["h0"]}
            with torch.no_grad():
                affect = audio(b["audio_features"], b["valid"])
            dynamic = local_audio(b["audio_features"], b["valid"])
            truth, innovation = targets(b, scales, args.stride)
            b["target_state"] = truth["state"]
            b["target_state_mask"] = truth["state_mask"]
            noise = torch.randn(b["motion"].shape, device=device)
            flow_time = torch.rand(len(ix), device=device)
            state = dynamic["state"]
            state_loss = huber(state, truth["state"], truth["state_mask"])
            flow = upper.flow_loss(innovation, b, identity, affect, dynamic["local"], state,
                                   noise[..., list(UPPER_INDICES)], flow_time)
            residual = upper.decode(b, identity, affect, dynamic["local"], state,
                                    noise[..., list(UPPER_INDICES)], args.decode_steps)
            generated_upper = upper_motion(b, scales, state, residual)
            generated_full = b["b0"].clone()
            generated_full[..., list(UPPER_INDICES)] = generated_upper
            output_state = output_state_loss(generated_upper, b, scales, args.stride)
            velocity = velocity_loss(generated_upper, b, scales)
            std = regional_std_loss(generated_full, b["motion"], b["valid"], b["anchors"], scales,
                                    b["channel_mask"] & b["anchor_valid"])
            domain = (F.relu(-generated_upper[b["valid"]]) + F.relu(generated_upper[b["valid"]] - 1.)).mean()
            loss = flow + .5 * state_loss + .5 * output_state + .25 * velocity + .1 * std + .05 * domain
            norm = optimize(loss, optimizer, params)
            total_steps += 1
            values = {"total": loss, "flow": flow, "state": state_loss, "output_state": output_state,
                      "velocity": velocity, "std": std, "domain": domain}
            for key, value in values.items(): sums[key].append(float(value.detach()))
            if bi % 25 == 0:
                print(json.dumps({"event": "batch", "epoch": epoch + 1, "batch": bi + 1,
                                  "batches": len(order.split(args.batch_size)), "loss": float(loss.detach()),
                                  "grad_norm": norm}), flush=True)
        record = {"stage": "dynamics_v3", "epoch": epoch + 1,
                  "batches": math.ceil(len(items) / args.batch_size), "samples": len(items),
                  "seconds": time.monotonic() - begin,
                  "losses": {key: sum(value) / len(value) for key, value in sums.items()},
                  "total_steps": total_steps}
        save_json(args.output / f"epoch{epoch + 1:03d}.json", record)
        save_checkpoint(args.output / "last.pt", {"schema": SCHEMA, "recipe": recipe,
                        "epoch": epoch + 1, "total_steps": total_steps,
                        "system": system.state_dict(), "audio": audio.state_dict(),
                        "upper": upper.state_dict(), "local_audio": local_audio.state_dict(),
                        "optimizer": optimizer.state_dict()})
        save_json(args.output / "status.json", {**record, "status": "training", "test_loaded": False})
        print(json.dumps({"event": "epoch_complete", **record}), flush=True)

    # Full inner-development evaluation uses the same fixed seeds/modes as the
    # original run, making the comparison directly auditable.
    eval_args = argparse.Namespace(device=device, batch_size=args.batch_size,
                                   decode_steps=args.decode_steps, stride=args.stride)
    report, curves = evaluate(system, audio, upper, local_audio, data, identities,
                              "dynamics", eval_args, full=True)
    save_json(args.output / "evaluation.json", report)
    save_checkpoint(args.output / "curves.pt", curves)
    save_checkpoint(args.output / "final.pt", {"schema": SCHEMA, "recipe": recipe,
                     "system": system.state_dict(), "audio": audio.state_dict(),
                     "upper": upper.state_dict(), "local_audio": local_audio.state_dict(),
                     "config": cfg, "scales": data["target_scales"], "test_loaded": False})
    save_json(args.output / "summary.json", {"schema": SCHEMA, "status": "complete", "epochs": epochs,
              "total_steps": total_steps, "elapsed_seconds": time.monotonic() - started,
              "test_loaded": False, "default_replaced": False})
    save_json(args.output / "status.json", {"status": "complete", "summary": "summary.json",
              "elapsed_seconds": time.monotonic() - started, "test_loaded": False})
    print("AUDIO_DYNAMICS_V3_COMPLETE", flush=True)


if __name__ == "__main__":
    main(parser().parse_args())
