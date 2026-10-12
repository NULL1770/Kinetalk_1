"""Matched MEAD continuation; full objective, resumable64-update checkpoints.

The inherited joint decoder remains present. Low/high adapters are a temporal
bias, not proven phonetic/emotional disentanglement. Smoke is a16-update run
of this exact runner on the true TRAIN fold, never a speaker-id index shortcut.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.load_candidate_model import load_candidate_model
from scripts.train_expression_response import (
    configure, development_fold, load_runtime, save, sha, state_digest, write,
)
from scripts.train_unified_v4 import (
    attach_base, build_base_cache, evaluate, make_refs, motion_targets,
    objective, reference_ids, select_balanced,
)
from scripts.train_v5_mixed import next_ids
from kinetalk_v4.rollout_statistics import expression_mean_loss, pooled_state_loss
from kinetalk_v4.state_posture import posture_supervision
from kinetalk_v4.unified_model_v4 import v4_losses
from kinetalk_v2.unified_model import anchor_matrix
from kinetalk_v5.dual_timescale_model import DualTimescaleKineTalkV5, v5_losses


def digest_list(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def full_objective(model, batch, refs, targets, channels, rollout_weight=.5, posture_weight=1.):
    loss, out, logs = objective(model, batch, refs, targets, warmup=False, channels=channels)
    if isinstance(model, DualTimescaleKineTalkV5):
        extra = v5_losses(model, out, targets, batch)
        legacy = v4_losses(model, out, targets, batch)
        loss = loss + extra["total"] - legacy["total"]
        logs.update({key: float(extra[key].detach()) for key in
            ("envelope_distill", "envelope_direct", "confidence", "factor_regularizer",
             "teacher_reliable_fraction")})
    posture = posture_supervision(out["student"]["state"], out["teacher"]["state"],
        targets["state"], batch["valid"], targets["quality"])
    loss = loss + posture_weight * posture["posture_direct"]
    if rollout_weight:
        kwargs = dict(geometry_audio=out["geometry_audio_context"])
        if isinstance(model, DualTimescaleKineTalkV5):
            kwargs["times"] = batch["times"]
        rollout = model.decode(batch["h0"], batch["b0"], batch["valid"], out["conditioned"],
            out["style"], out["sampled_schedule"]["events"].detach(), **kwargs)["motion"]
        mean = expression_mean_loss(rollout, batch["motion"], batch["valid"],
            batch["channel_mask"], model.scales)
        state = (rollout - batch["b0"]) @ anchor_matrix().to(rollout)
        state_mean = pooled_state_loss(state, targets["state"], batch["valid"], targets["quality"])
        loss = loss + rollout_weight * (mean + state_mean)
        logs.update(rollout_mean=float(mean.detach()), rollout_state_mean=float(state_mean.detach()))
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite full objective")
    return loss, logs


def run(args):
    if args.updates < 1 or args.batch_size < 1:
        raise ValueError("Positive update and batch budgets required")
    root = Path(args.output)
    if root.exists() and not args.resume:
        raise FileExistsError("Fresh output required; do not replace evidence")
    root.mkdir(parents=True, exist_ok=bool(args.resume))
    configure(args.seed)
    device = torch.device(args.device)
    binding = json.loads(Path(args.binding).read_text())
    data, base = load_runtime(binding, device)
    frozen = state_digest(base.stage1)
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if parent["config"]["teacher_guidance_mode"] != "none":
        raise ValueError("Pilot requires raw-direct parent without full-frame teacher guidance")
    if args.arm == "control":
        model = load_candidate_model(parent, device)
    else:
        model = DualTimescaleKineTalkV5.from_v4_checkpoint(parent, device)
        model.cfg = replace(model.cfg,
            envelope_weight=.35 if args.arm == "guided" else 0.,
            confidence_weight=.1 if args.arm == "guided" else 0.)
        model.backbone.cfg = model.cfg
        model.decoder.cfg = model.cfg
    original = model.backbone.state_dict() if hasattr(model, "backbone") else model.state_dict()
    for key, tensor in parent["model"].items():
        if not torch.equal(original[key].cpu(), tensor):
            raise ValueError("Source tensor differs: " + key)
    # Architecture construction may consume a different amount of global RNG.
    configure(args.seed)
    q = data["splits"]["train"]
    fold = development_fold(q, data["fit_sids"])
    fit = fold["train"]
    held = select_balanced(q, fold["speaker_dev"] + fold["sentence_dev"], args.eval_limit, args.seed)
    pool = {}
    for i in fit:
        pool.setdefault(int(q["speaker_id"][i]), []).append(i)
    rng = random.Random(args.seed)
    refs = {i: reference_ids(q, [i], pool, rng)[0] for i in fit}
    required = sorted(set(fit + held))
    print(json.dumps({"event": "cache_start", "clips": len(required), "arm": args.arm}), flush=True)
    cache = build_base_cache(q, base, required, device)
    def rebuild():
        order = list(fit)
        rng.shuffle(order)
        return order
    order, position = rebuild(), 0
    protocol = dict(schema="dual_timescale_mead_pilot_v1", arm=args.arm, seed=args.seed,
        updates=args.updates, batch_size=args.batch_size, lr=args.lr,
        source_checkpoint_sha256=sha(args.checkpoint), binding_sha256=sha(args.binding),
        source_sha256=sha(__file__), model_source_sha256=sha(sys.modules[DualTimescaleKineTalkV5.__module__].__file__),
        fit_indices_sha256=digest_list(fit), training_references_sha256=digest_list(sorted(refs.items())),
        initial_parent_parameters_sha256=state_digest(load_candidate_model(parent)),
        geometry_channels=parent["protocol"]["geometry_channels"], neutral_digest=frozen,
        mead_fit_clips=len(fit), internal_dev_clips=len(held), config=asdict(model.cfg),
        hdtf_enabled=False, text_enabled=False, test_loaded=False, paper_result=False,
        objective="legacy complete motion/style/observed-event likelihood/sampled execution; direct posture1; rollout means.5; v5 supported-envelope and low/high adapter losses",
        residual_semantics="inherited joint residual plus low/high mixed-factor adapters; no exclusive articulation/emotion claim",
        confidence="TRAIN teacher-advantage fraction; uncalibrated learned score in guided arm only")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    step, sequence, history = 0, [], []
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved["protocol"] != protocol:
            raise ValueError("Resume protocol differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step, sequence, history = saved["step"], saved["sequence"], saved["history"]
        order, position = saved["order"], saved["position"]
        rng.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
    else:
        write(root / "protocol.json", protocol)
    begin = time.monotonic()
    heads_verified = args.arm == "control"
    model.train()
    while step < args.updates:
        ids, order, position = next_ids(order, position, args.batch_size, rebuild)
        if not set(ids).issubset(fit) or not set(refs[i] for i in ids).issubset(fit):
            raise ValueError("Query/reference outside TRAIN internal fold")
        sequence.extend(ids)
        batch = attach_base(q.batch(torch.tensor(ids), device), ids, cache)
        reference = make_refs(q, [refs[i] for i in ids], device)
        targets = motion_targets(batch)
        optimizer.zero_grad(set_to_none=True)
        loss, logs = full_objective(model, batch, reference, targets, protocol["geometry_channels"])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        if args.arm != "control" and not heads_verified:
            for head in (model.envelope_head, model.decoder.affect_adapter, model.decoder.phonetic_adapter):
                grad = head.weight.grad
                if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum().gt(0):
                    raise RuntimeError("New head receives no finite nonzero gradient")
            heads_verified = True
        optimizer.step()
        step += 1
        row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), **logs)
        history.append(row)
        if step == 1 or step % 20 == 0 or step == args.updates:
            state = dict(status="training", arm=args.arm, total_updates=args.updates,
                elapsed_seconds=time.monotonic()-begin, **row)
            write(root / "state.json", state)
            print(json.dumps(state), flush=True)
        if step % 64 == 0 or step == args.updates:
            if state_digest(base.stage1) != frozen or any(p.grad is not None for p in base.stage1.parameters()):
                raise RuntimeError("Frozen B0 changed")
            saved = dict(model=model.state_dict(), config=asdict(model.cfg), protocol=protocol,
                optimizer=optimizer.state_dict(), step=step, sequence=sequence, history=history,
                order=order, position=position, python_rng=rng.getstate(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else [])
            save(root / "resume.pt", saved)
    dev = evaluate(model, q, held, cache, data, device, args.seed, protocol["geometry_channels"])
    save(root / "final.pt", {key:value for key,value in saved.items() if key!="optimizer"})
    write(root / "history.json", history)
    write(root / "complete.json", dict(status="complete", updates=step, protocol=protocol,
        internal_dev=dev, frozen_b0_sha256=frozen, head_gradients_verified=heads_verified,
        mead_training_sequence_sha256=digest_list(sequence), checkpoint_sha256=sha(root/"final.pt"),
        elapsed_seconds=time.monotonic()-begin, unique_mead_clips=len(set(sequence)), paper_result=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--binding", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--arm", choices=("control", "direct", "guided"), required=True)
    p.add_argument("--updates", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--eval-limit", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume")
    run(p.parse_args())
