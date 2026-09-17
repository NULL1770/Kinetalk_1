"""Production v9 trainer for semantic/style experiments."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from kinetalk_b0.models.semantic import (
    SemanticGenerator,
    MotionSemanticReadout,
    MultiReferenceStyleEncoder,
    SemanticAudioEncoder,
)
from kinetalk_b0.semantic_data import SemanticMotionDataset, semantic_collate
from kinetalk_b0.semantic_losses import (
    semantic_supervision,
    style_contrastive,
    cross_style_objective,
)
from kinetalk_b0.utils import load_yaml, move_to_device, freeze_module, seed_everything


def infinite_batches(loader: Iterable):
    epoch = 0
    while True:
        if hasattr(getattr(loader, "dataset", None), "set_epoch"):
            loader.dataset.set_epoch(epoch)
        count = 0
        for batch in loader:
            count += 1
            yield epoch, batch
        if count == 0:
            raise ValueError("empty training loader")
        epoch += 1


def _content(gen, b):
    c = b.get("content_context")
    return (
        c
        if c is not None and c.shape[-2] == gen.stage1.native_aggregator.context
        else b["content"]
    )


def observed(b):
    motion = b["motion"]
    mask = b["valid"].bool()
    channels = b.get("channel_mask")
    if channels is None or channels.shape != motion.shape[:-2] + (motion.shape[-1],):
        raise ValueError("explicit channel_mask required")
    out = mask.unsqueeze(-1) & channels.bool().unsqueeze(-2)
    if not out.flatten(1).any(-1).all():
        raise ValueError("clip has no observed channels")
    return out


@torch.no_grad()
def base_residual(gen, b):
    motion = b["motion"]
    lead = motion.shape[:-2]
    c = _content(gen, b)
    flat_c = c.reshape(-1, *c.shape[len(lead) :])
    flat_m = b["valid"].reshape(-1, motion.shape[-2])
    b0 = gen.stage1(flat_c, flat_m)["b0"].reshape_as(motion)
    r = torch.where(observed(b), motion - b0, torch.zeros_like(motion))
    if not torch.isfinite(r).all():
        raise FloatingPointError("nonfinite residual")
    return b0, r


def style(gen, enc, b):
    _, r = base_residual(gen, b)
    return enc(r, b["valid"])


def semargs(b):
    return {
        k: b[k]
        for k in (
            "emotion_id",
            "intensity_id",
            "va",
            "va_valid",
            "va_confidence",
            "intensity_valid",
        )
    }


def masked_flow(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape or not mask.any():
        raise ValueError("masked flow needs matching tensors with observations")
    difference = prediction[mask] - target[mask]
    if not torch.isfinite(difference).all():
        raise FloatingPointError("nonfinite observed flow")
    return difference.square().mean()


def step(loss, opt, params, clip):
    params = list(params)
    if loss.ndim or not torch.isfinite(loss):
        raise FloatingPointError("nonfinite loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
    opt.step()
    if any(not torch.isfinite(p).all() for p in params):
        raise FloatingPointError("nonfinite parameters")
    return float(loss.detach())


@torch.no_grad()
def validate(models, va, d, cfg, audio=False):
    gen, read, sty = models["generator"], models["readout"], models["style_encoder"]
    modules = [gen, read, sty, models["audio"]]
    modes = [m.training for m in modules]
    for model in modules:
        model.eval()
    correct = {"emotion": 0, "intensity": 0}
    counts = {"emotion": 0, "intensity": 0}
    confusion = {}
    moments = torch.zeros(7, 2, dtype=torch.float64)
    anchors = []
    gallery = []
    ids = []
    limit = int(cfg.get("validation", {}).get("max_batches", 0))
    try:
        for index, raw in enumerate(va):
            if limit and index >= limit:
                break
            b = move_to_device(raw, d)
            q = b["query"]
            _, r = base_residual(gen, q)
            outputs = (
                models["audio"](q["audio"], q["valid"])
                if audio
                else read(r, q["valid"])
            )
            for key in ("emotion", "intensity"):
                labels = q[key + "_id"]
                valid = (labels >= 0) & q["valid"].any(-1)
                if key == "intensity":
                    valid &= q["intensity_valid"]
                guess = outputs[key + "_logits"].argmax(-1)
                correct[key] += int((guess[valid] == labels[valid]).sum())
                counts[key] += int(valid.sum())
                if key == "emotion":
                    for target, predicted in zip(
                        labels[valid].tolist(), guess[valid].tolist()
                    ):
                        confusion[(target, predicted)] = (
                            confusion.get((target, predicted), 0) + 1
                        )
            mask = q["valid"] & q["va_valid"] & (q["va_confidence"] > 0)
            x = outputs["va"][mask].double().cpu()
            y = q["va"][mask].double().cpu()
            w = q["va_confidence"][mask].double().cpu()[:, None]
            if not torch.isfinite(x).all():
                raise FloatingPointError("nonfinite validation prediction")
            moments += torch.stack(
                [
                    w.expand_as(x).sum(0),
                    (w * x).sum(0),
                    (w * y).sum(0),
                    (w * x * x).sum(0),
                    (w * y * y).sum(0),
                    (w * x * y).sum(0),
                    (w * (x - y).abs()).sum(0),
                ]
            )
            if not audio:
                anchors.append(sty(r, q["valid"]).cpu())
                gallery.append(style(gen, sty, b["positive_style_references"]).cpu())
                ids.append(q["speaker_id"].cpu())
    finally:
        for model, mode in zip(modules, modes):
            model.train(mode)
    if not counts["emotion"]:
        raise ValueError("empty labelled validation set")
    weight, sx, sy, sxx, syy, sxy, error = moments
    denom = weight.clamp_min(1e-12)
    mx, my = sx / denom, sy / denom
    vx, vy, cov = sxx / denom - mx * mx, syy / denom - my * my, sxy / denom - mx * my
    ccc = (
        2
        * cov
        / (vx.clamp_min(0) + vy.clamp_min(0) + (mx - my).square()).clamp_min(1e-12)
    )
    meaningful = (weight > 0) & (vy > 1e-8)
    labels = sorted({pair[0] for pair in confusion})
    recalls = [
        confusion.get((label, label), 0)
        / sum(n for (target, _), n in confusion.items() if target == label)
        for label in labels
    ]
    result = {
        "emotion_accuracy": correct["emotion"] / counts["emotion"],
        "emotion_balanced_accuracy": sum(recalls) / len(recalls),
        "intensity_accuracy": (
            correct["intensity"] / counts["intensity"] if counts["intensity"] else None
        ),
        "known_intensity_examples": counts["intensity"],
        "examples": counts["emotion"],
        "va_mae": float((error / denom).mean()) if (weight > 0).all() else None,
        "va_ccc": [float(ccc[i]) if meaningful[i] else None for i in range(2)],
        "va_ccc_mean": float(ccc.mean()) if meaningful.all() else None,
        "va_target_variance": vy.tolist(),
    }
    if not audio:
        x, y, speakers = torch.cat(anchors), torch.cat(gallery), torch.cat(ids)
        retrieval = []
        for start in range(0, len(x), 256):
            scores = x[start : start + 256] @ y.T
            # Another query's same-speaker gallery may contain this query
            # itself. Only this query's independently sampled positive set is
            # an eligible same-speaker candidate; all other-speaker sets remain.
            allowed = speakers[start : start + len(scores), None] != speakers[None]
            allowed[torch.arange(len(scores)), torch.arange(start, start + len(scores))] = True
            nearest = scores.masked_fill(~allowed, -torch.inf).argmax(-1)
            retrieval.append(
                (speakers[nearest] == speakers[start : start + 256]).float()
            )
        frequencies = torch.unique(speakers, return_counts=True)[1].float() / len(
            speakers
        )
        result.update(
            style_retrieval=(
                float(torch.cat(retrieval).mean()) if len(frequencies) > 1 else None
            ),
            style_speakers=len(frequencies),
            style_retrieval_chance=float(frequencies.square().sum()),
            style_gallery_policy="own independent cross-emotion/cross-sentence positive set plus other-speaker sets; other same-speaker sets excluded",
        )
    return result


def critic_gate(metrics, cfg):
    validation = cfg.get("validation", {})
    failures = []
    for name, actual, threshold, higher in [
        (
            "emotion",
            metrics.get("emotion_balanced_accuracy"),
            float(validation.get("min_emotion_accuracy", 0.35)),
            True,
        ),
        (
            "va_ccc",
            metrics.get("va_ccc_mean"),
            float(validation.get("min_va_ccc", 0.15)),
            True,
        ),
        (
            "va_mae",
            metrics.get("va_mae"),
            float(validation.get("max_va_mae", 0.35)),
            False,
        ),
        (
            "style",
            metrics.get("style_retrieval"),
            float(validation.get("min_style_retrieval", 0.75)),
            True,
        ),
    ]:
        if (
            actual is None
            or not math.isfinite(actual)
            or (actual < threshold if higher else actual > threshold)
        ):
            failures.append(f"{name}: {actual!r}; threshold {threshold}")
    return {
        "critics_accepted": not failures,
        "visual_path_accepted": False,
        "failures": failures,
        "metrics": metrics,
    }


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf8",
    )


def loader(cfg, split, training):
    ds = SemanticMotionDataset(cfg, split=split, random_crop=training)
    t = cfg["training"]
    return DataLoader(
        ds,
        batch_size=int(t.get("batch_size", 8)),
        shuffle=training,
        num_workers=int(t.get("num_workers", 0)),
        collate_fn=semantic_collate,
        drop_last=False,
    )


def provenance(cfg, tr, va):
    def sha(p):
        h = hashlib.sha256()
        with Path(p).open("rb") as f:
            for x in iter(lambda: f.read(1 << 20), b""):
                h.update(x)
        return h.hexdigest()

    assets = {}
    for data_loader in (tr, va):
        for row in data_loader.dataset.items:
            for key in ("motion_path", "content_path", "audio_path", "va_path"):
                if key not in row:
                    continue
                path = str(Path(row[key]).resolve())
                if path not in assets:
                    assets[path] = sha(path)
                expected = row.get("source_artifact_sha256") if key == "motion_path" else None
                if expected and assets[path] != expected:
                    raise RuntimeError(f"native artifact checksum differs from manifest: {row['clip_id']}")
    return {
        "config_sha256": sha(cfg["_config_path"]),
        "train_manifest_sha256": sha(tr.dataset.manifest),
        "val_manifest_sha256": sha(va.dataset.manifest),
        "stage1_sha256": sha(cfg["paths"]["stage1_ckpt"]),
        "assets_sha256": hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest(),
        "verified_asset_files": len(assets),
        "torch": torch.__version__,
    }


def save(
    path, models, cfg, prov, stage, step, validation, optimizer=None, exploratory=False
):
    p = path.with_suffix(path.suffix + ".tmp")
    data = {
        "architecture_version": 9,
        "stage": stage,
        "step": step,
        "config": cfg,
        "provenance": prov,
        "validation": validation,
        "exploratory": exploratory,
        **{k: v.state_dict() for k, v in models.items()},
    }
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, p)
    os.replace(p, path)


def train_critics(models, tr, va, d, cfg, out, prov):
    gen, read, sty = models["generator"], models["readout"], models["style_encoder"]
    freeze_module(gen.stage1)
    gen.eval()
    read.train()
    sty.train()
    params = list(read.parameters()) + list(sty.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg["training"].get("critic_lr", 3e-4)),
                            weight_decay=float(cfg["training"].get("weight_decay", 1e-5)))
    it = infinite_batches(tr)
    steps = int(cfg["training"].get("critic_steps", 400))
    for i in range(1, steps + 1):
        ep, raw = next(it)
        b = move_to_device(raw, d)
        q = b["query"]
        _, r = base_residual(gen, q)
        sem = semantic_supervision(read(r, q["valid"]), q)
        anchor = sty(r, q["valid"])
        neutral_refs = b.get("neutral_style_references", b["same_style_references"])
        pos = torch.stack(
            [
                style(gen, sty, neutral_refs),
                style(gen, sty, b["positive_style_references"]),
            ],
            1,
        )
        con = style_contrastive(
            anchor,
            pos,
            q["speaker_id"],
            temperature=float(cfg["training"].get("style_temperature", 0.1)),
        )
        loss = (
            sem["total"]
            + float(cfg.get("loss", {}).get("style_contrastive", 1.0)) * con
        )
        value = step(loss, opt, params, float(cfg["training"].get("grad_clip", 1.0)))
        _log(
            out,
            "critics",
            {
                "step": i,
                "epoch": ep,
                "loss": value,
                "style": float(con.detach()),
                **{"semantic_" + key: float(value.detach()) for key, value in sem.items()},
                "speakers": int(q["speaker_id"].unique().numel()),
            },
        )
    metrics = validate(models, va, d, cfg)
    gate = critic_gate(metrics, cfg)
    save(out / "critics.pt", models, cfg, prov, "critics", steps, gate, opt)
    _json(out / "critics_validation.json", gate)
    return gate


def generator_step(models, b, cfg, d, with_cross=True):
    gen, read, sty = models["generator"], models["readout"], models["style_encoder"]
    q = b["query"]
    # Identity is anchored by neutral references when the manifest provides
    # them; the dataset falls back to ordinary same-speaker views otherwise.
    own_refs = b.get("neutral_style_references", b["same_style_references"])
    own = style(gen, sty, own_refs).detach()
    m = observed(q)
    b0, _ = base_residual(gen, q)
    target = torch.where(m, q["motion"], b0)
    o = gen.forward_flow(target, _content(gen, q), q["valid"], style=own, **semargs(q))
    flow = masked_flow(o["prediction"], o["velocity_target"], m)
    total = flow
    if not with_cross:
        return total, {"flow": flow}
    with torch.no_grad():
        donor = style(gen, sty, b["donor_references"])
        negatives = torch.cat([own, donor])
        neg_ids = torch.cat([q["speaker_id"], b["donor_anchor"]["speaker_id"]])
    noise = torch.randn_like(q["motion"])
    prior_mode = gen.training
    gen.eval()  # Independent integration has fixed noise, without dropout noise.
    try:
        generated = gen.generate(
            _content(gen, q), q["valid"], style=donor, initial_noise=noise,
            steps=int(cfg["training"].get("decode_steps", 4)), **semargs(q),
        )
    finally:
        gen.train(prior_mode)
    rr = torch.where(m, generated["residual"], torch.zeros_like(generated["residual"]))
    cs = read(rr, q["valid"])
    gs = sty(rr, q["valid"])
    timing = [
        int(i)
        for i in cfg["model"].get("timing_indices", [])
        if q["channel_mask"][:, int(i)].all()
    ]
    cross = cross_style_objective(
        generated["motion"],
        o["b0"],
        cs,
        q,
        gs,
        donor,
        negatives,
        timing,
        donor_speaker_ids=b["donor_anchor"]["speaker_id"],
        negative_speaker_ids=neg_ids,
        mouth_weight=float(cfg.get("loss", {}).get("mouth_timing", 0.1)),
    )
    total = total + float(cfg.get("loss", {}).get("cross", 0.25)) * cross["total"]
    return total, {
        "flow": flow,
        "cross": cross["total"],
        "style": cross["style"],
        "semantic": cross["semantic"],
        "mouth_timing": cross["mouth_timing"],
    }


def train_generator(models, tr, va, d, cfg, out, prov, acceptance, exploratory):
    if not acceptance.get("critics_accepted", False) and not exploratory:
        raise RuntimeError(
            "critic gate failed; use --exploratory only for labelled pilot"
        )
    freeze_module(models["readout"])
    freeze_module(models["style_encoder"])
    gen = models["generator"]
    gen.train()
    params = [p for p in gen.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["training"].get("generator_lr", 3e-4)),
                            weight_decay=float(cfg["training"].get("weight_decay", 1e-5)))
    it = infinite_batches(tr)
    steps = int(cfg["training"].get("generator_steps", 300))
    before = evaluate(models, va, d, cfg, out / "before_generator", acceptance)
    _json(out / "generator_diagnostic_before.json", before)
    every = int(cfg["training"].get("cross_every", 2))
    if every < 1:
        raise ValueError("cross_every must be positive")
    for i in range(1, steps + 1):
        ep, raw = next(it)
        loss, parts = generator_step(
            models,
            move_to_device(raw, d),
            cfg,
            d,
            with_cross=(i == 1 or i % every == 0),
        )
        value = step(loss, opt, params, float(cfg["training"].get("grad_clip", 1.0)))
        _log(
            out,
            "generator",
            {
                "step": i,
                "epoch": ep,
                "loss": value,
                "exploratory": exploratory,
                **{k: float(v.detach()) for k, v in parts.items()},
            },
        )
    validation = {**acceptance, "visual_path_accepted": False}
    save(
        out / "generator.pt",
        models,
        cfg,
        prov,
        "generator",
        steps,
        validation,
        opt,
        exploratory,
    )
    report = evaluate(models, va, d, cfg, out, acceptance)
    validation["visual_path_accepted"] = bool(
        report["visual_path_accepted"] and not exploratory
    )
    save(
        out / "generator.pt",
        models,
        cfg,
        prov,
        "generator",
        steps,
        validation,
        opt,
        exploratory,
    )
    return validation


def require_audio_gate(state):
    if (
        state.get("stage") not in ("generator", "audio")
        or not state.get("validation", {}).get("visual_path_accepted", False)
        or state.get("exploratory", False)
    ):
        raise RuntimeError("audio requires accepted nonexploratory visual generator")


def train_audio(models, tr, va, d, cfg, out, prov, state):
    require_audio_gate(state)
    for k in ("generator", "readout", "style_encoder"):
        freeze_module(models[k])
    audio = models["audio"]
    audio.train()
    opt = torch.optim.AdamW(
        audio.parameters(),
        lr=float(
            cfg["training"].get("audio_lr", cfg["training"].get("critic_lr", 3e-4))
        ),
    )
    it = infinite_batches(tr)
    steps = int(cfg["training"].get("audio_steps", 300))
    for i in range(1, steps + 1):
        ep, raw = next(it)
        q = move_to_device(raw["query"], d)
        loss = semantic_supervision(audio(q["audio"], q["valid"]), q)["total"]
        _log(
            out,
            "audio",
            {
                "step": i,
                "epoch": ep,
                "loss": step(
                    loss,
                    opt,
                    audio.parameters(),
                    float(cfg["training"].get("grad_clip", 1.0)),
                ),
            },
        )
    validation = {
        **state["validation"],
        "audio_metrics": validate(models, va, d, cfg, audio=True),
    }
    save(out / "audio.pt", models, cfg, prov, "audio", steps, validation, opt)
    _json(out / "audio_validation.json", validation)


def evaluate(models, va, d, cfg, out, acceptance):
    from scripts.diagnose_semantic import diagnose

    report = diagnose(
        models["generator"],
        models["style_encoder"],
        models["readout"],
        va,
        d,
        steps=int(cfg["training"].get("decode_steps", 4)),
        max_batches=int(cfg.get("validation", {}).get("diagnostic_batches", 2)),
        seed=int(cfg.get("seed", 42)),
        output_dir=out / "diagnostic_csv",
    )
    report["critics_accepted"] = bool(acceptance.get("critics_accepted", False))
    report["visual_path_accepted"] = bool(
        report["critics_accepted"]
        and cfg.get("validation", {}).get("visual_generation_approved", False)
    )
    report["acceptance_source"] = (
        "recorded critic metrics plus explicit visual_generation_approved after reviewing diagnostics"
    )
    _json(out / "generator_diagnostic.json", report)
    return report


def _log(out, stage, record):
    out.mkdir(parents=True, exist_ok=True)
    value = {"time": time.time(), "stage": stage, **record}
    line = json.dumps(value, ensure_ascii=False, allow_nan=False)
    with (out / f"{stage}.jsonl").open("a", encoding="utf8") as f:
        f.write(line + "\n")
    if record.get("step", 1) == 1 or record.get("step", 0) % 20 == 0:
        print(line, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    p.add_argument(
        "--stage",
        choices=("preflight", "critics", "generator", "audio", "evaluate", "pilot"),
        default="preflight",
    )
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--exploratory", action="store_true")
    a = p.parse_args()
    cfg = load_yaml(a.config)
    cfg["_config_path"] = str(a.config.resolve())
    seed_everything(int(cfg.get("seed", 42)))
    torch.set_num_threads(int(cfg["training"].get("cpu_threads", 4)))
    if any(
        int(cfg["training"].get(k, 1)) < 1
        for k in (
            "batch_size",
            "critic_steps",
            "generator_steps",
            "audio_steps",
            "decode_steps",
        )
    ):
        raise ValueError("batch size and stage/decode steps must be positive")
    d = torch.device(cfg.get("device", "cpu"))
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; set device=cpu")
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tr, va = loader(cfg, "train", True), loader(cfg, "val", False)
    training_speakers = {(x["dataset"], x["speaker"]) for x in tr.dataset.items}
    validation_speakers = {(x["dataset"], x["speaker"]) for x in va.dataset.items}
    if training_speakers & validation_speakers:
        raise RuntimeError("speaker leakage between train/val")
    if len(training_speakers) < 2 or len(validation_speakers) < 2:
        raise RuntimeError("train and validation each need multiple speakers")
    loaded = {}
    for name, data_loader in (("train", tr), ("val", va)):
        loaded[name] = 0
        for index, batch in enumerate(data_loader):
            preflight_limit = int(cfg.get("validation", {}).get("preflight_batches", 0))
            if preflight_limit and index >= preflight_limit:
                break
            for branch in batch.values():
                if isinstance(branch, dict) and "motion" in branch:
                    if not torch.isfinite(branch["motion"][observed(branch)]).all():
                        raise FloatingPointError("nonfinite preflight input")
            loaded[name] += len(batch["query"]["emotion_id"])
    prov = provenance(cfg, tr, va)
    gen = SemanticGenerator(cfg).to(d)
    ck = torch.load(cfg["paths"]["stage1_ckpt"], map_location="cpu", weights_only=False)
    st = ck.get("model", ck)
    if int(st.get("architecture_version", -1)) != 7:
        raise RuntimeError("stage1 checkpoint must be v7")
    gen.stage1.load_state_dict(st, strict=True)
    freeze_module(gen.stage1)
    audit = {
        "passed": True,
        "train_clips": len(tr.dataset),
        "val_clips": len(va.dataset),
        "train_speakers": len(training_speakers),
        "val_speakers": len(validation_speakers),
        "loaded_examples": loaded,
        "stage1_strict_load": True,
        "frame_mask_contract": "valid is intersection of native motion/audio/content coverage and trusted visual VA; B0 and all critic/generator paths use the same mask",
        "complete_dataset_iteration": not bool(
            cfg.get("validation", {}).get("preflight_batches", 0)
        ),
    }
    _json(out / "preflight.json", audit)
    if a.stage == "preflight":
        print(json.dumps(audit), flush=True)
        return
    models = {
        "generator": gen,
        "readout": MotionSemanticReadout(cfg).to(d),
        "style_encoder": MultiReferenceStyleEncoder(cfg).to(d),
        "audio": SemanticAudioEncoder(cfg).to(d),
    }
    if a.stage in ("generator", "audio", "evaluate") and a.checkpoint is None:
        raise ValueError("--checkpoint required")
    state = (
        torch.load(a.checkpoint, map_location=d, weights_only=False)
        if a.checkpoint
        else {}
    )
    if a.checkpoint:
        if int(state.get("architecture_version", -1)) != 9:
            raise RuntimeError("v9 checkpoint required")
        for key in ("train_manifest_sha256", "val_manifest_sha256", "stage1_sha256", "assets_sha256"):
            if state.get("provenance", {}).get(key) != prov[key]:
                raise RuntimeError(f"checkpoint provenance mismatch: {key}")
        for k, m in models.items():
            m.load_state_dict(state[k], strict=True)
    if a.stage in ("critics", "pilot"):
        state["validation"] = train_critics(models, tr, va, d, cfg, out, prov)
    if a.stage in ("generator", "pilot"):
        train_generator(
            models,
            tr,
            va,
            d,
            cfg,
            out,
            prov,
            state.get("validation", {}),
            a.exploratory,
        )
    if a.stage == "audio":
        train_audio(models, tr, va, d, cfg, out, prov, state)
    if a.stage == "evaluate":
        report = evaluate(models, va, d, cfg, out, state.get("validation", {}))
        validation = {
            **state.get("validation", {}),
            "visual_path_accepted": bool(
                report["visual_path_accepted"] and not state.get("exploratory", False)
            ),
        }
        save(
            out / "generator_evaluated.pt",
            models,
            cfg,
            prov,
            "generator",
            state.get("step", 0),
            validation,
            exploratory=state.get("exploratory", False),
        )


if __name__ == "__main__":
    main()
