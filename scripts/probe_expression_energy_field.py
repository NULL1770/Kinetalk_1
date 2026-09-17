"""Probe an interpretable low-rate expression-energy field.

The field is built from a label-fitted expression basis (fit on training
sentences only).  Each frame is projected into that basis and its L2 norm is
used as a scalar expression energy.  The per-clip temporal mean is removed so
the target matches the existing low-rate affect contract.  This is a
diagnostic target only; it does not change the generator or add a new model
path. ``energy_velocity`` adds one centered finite-difference axis to test
whether onset/offset timing is more predictable than level alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.probe_dynamic_predictability import (
    bin_frames, center_controls, fit_target_scale, fresh_student,
    sentence_split, target_metrics, training_only_audio, weighted_mse,
)
from scripts.probe_label_expression_dynamics import fit_label_basis
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_pilot import (
    affect_residual, cached_identity, device_batch, observed, prepare, select,
    sha, write_json,
)


def fit_expression_energy_target(
    residual: torch.Tensor,
    valid: torch.Tensor,
    channel_mask: torch.Tensor,
    speaker_id: torch.Tensor,
    emotion_id: torch.Tensor,
    intensity_id: torch.Tensor,
    train_ids: torch.Tensor,
    stride: int,
    rank: int = 3,
    mode: str = "label_energy",
):
    """Return centered scalar energy and the fixed training-only basis.

    The basis is the same neutral-anchored, label-associated basis used by
    ``probe_label_expression_dynamics``.  Energy is computed before temporal
    centering, therefore it remains invariant to basis sign and is not a
    disguised class label.
    """
    if mode not in ("label_energy", "upper_l1", "face_l1", "energy_velocity"):
        raise ValueError("mode must be label_energy, upper_l1, face_l1, or energy_velocity")
    if mode != "label_energy":
        selected = torch.arange(residual.shape[-1], device=residual.device)
        if mode in ("upper_l1", "energy_velocity"):
            selected = torch.tensor([i for i in list(range(14)) + list(range(41, 46))
                                    if i < residual.shape[-1]], device=residual.device)
        clean = torch.where(valid[..., None], residual[:, :, selected], 0)
        binned, weight = bin_frames(clean, valid, stride)
        # L1 magnitude is the MEDTalk-style intensity proxy.  Centering keeps
        # the existing low-rate affect contract, whose controls have zero
        # temporal mean and let the global head carry clip-level intensity.
        raw_energy = binned.abs().mean(-1, keepdim=True)
        energy = center_controls(raw_energy, weight)
        if mode == "energy_velocity":
            # A compact dense target: level plus first temporal difference.
            # Both axes come from the same upper-face field; no regional
            # routing or additional teacher is introduced.
            velocity = torch.zeros_like(energy)
            velocity[:, 1:] = energy[:, 1:] - energy[:, :-1]
            # The model's local controls are clip-centered.  Remove the
            # derivative's tiny boundary/DC offset as well so both axes obey
            # the same contract while retaining onset/offset timing.
            velocity = center_controls(velocity, weight)
            pair_weight = weight
            return torch.cat([energy, velocity], -1), pair_weight, {
                "basis": None, "projected": None, "raw_energy": raw_energy,
                "target_name": mode, "channel_indices": selected,
                "axis_names": ["energy", "velocity"],
            }
        return energy, weight, {"basis": None, "projected": None,
                                "raw_energy": raw_energy,
                                "target_name": mode, "channel_indices": selected}
    if rank != 3:
        raise ValueError("The pilot label basis currently requires rank=3")
    _, weight, state = fit_label_basis(
        residual, valid, channel_mask, speaker_id, emotion_id, intensity_id,
        train_ids, stride,
    )
    selected = state["channel_indices"]
    clean = torch.where(valid[..., None], residual[:, :, selected], 0)
    binned, weight = bin_frames(clean, valid, stride)
    # Speaker neutral anchors define the origin of expression coordinates.
    neutral = torch.zeros((len(residual), selected.numel()), dtype=residual.dtype,
                          device=residual.device)
    for speaker, value in state["neutral_means"].items():
        neutral[speaker_id == int(speaker)] = value.to(residual.device, residual.dtype)
    projected = (binned - neutral[:, None]) @ state["basis"]
    raw_energy = projected.square().sum(-1, keepdim=True).sqrt()
    energy = center_controls(raw_energy, weight)
    return energy, weight, {
        "basis": state,
        "projected": projected,
        "raw_energy": raw_energy,
        "target_name": "label_expression_energy",
    }


def energy_metrics(prediction, target, weight, query, indices, scale):
    """Compact scalar-field metrics used by the CLI and tests."""
    q = select(query, indices)
    pred, truth, w = prediction[indices], target[indices], weight[indices]
    out = target_metrics(pred, truth, w, q["clip_id"], q["sentence_id"], scale)
    mse = weighted_mse(pred * scale, truth * scale, w)
    zero = weighted_mse(torch.zeros_like(truth) * scale, truth * scale, w)
    out.update({
        "native_mse": float(mse),
        "native_zero_mse": float(zero),
        "native_r2_against_zero": float(1 - mse / zero) if float(zero) > 1e-12 else None,
        "prediction_energy": float(weighted_mse(pred * scale, torch.zeros_like(pred), w)),
    })
    return out


def axis_metrics(prediction, target, weight, query, indices, scale):
    """Return one compact metric row per dynamic axis for rank>1 probes."""
    return [energy_metrics(prediction[..., axis:axis + 1], target[..., axis:axis + 1],
                           weight, query, indices, scale[axis:axis + 1])
            for axis in range(target.shape[-1])]


def augment_audio_context(audio: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Add delta and short smoothed branches without changing the target.

    The three branches approximate the local/mid-scale acoustic paths used by
    recent audio-driven facial animation work. Padding is masked before and
    after the temporal operation so invalid frames never become supervision.
    """
    if audio.ndim != 3 or valid.shape != audio.shape[:2]:
        raise ValueError("audio must be [batch,time,features] with matching valid mask")
    clean = torch.where(valid[..., None], audio, 0)
    delta = torch.diff(clean, dim=1, prepend=clean[:, :1])
    weight = valid.to(audio.dtype).unsqueeze(1)
    smooth = F.avg_pool1d(clean.transpose(1, 2), kernel_size=5, stride=1, padding=2,
                          count_include_pad=False).transpose(1, 2)
    return torch.where(valid[..., None], torch.cat([clean, delta, smooth], -1), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--target", choices=("label_energy", "upper_l1", "face_l1", "energy_velocity"), default="label_energy")
    parser.add_argument("--audio-features", choices=("acoustic", "acoustic_context"), default="acoustic")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "audio":
        raise ValueError("Use a frozen audio-stage checkpoint")
    source_hash = sha(args.data / "train.pt")
    if source_hash != checkpoint.get("provenance", {}).get("train_sha256"):
        raise ValueError("Training cache hash differs from checkpoint")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(cfg.get("device", "cuda"))
    dataset = NeutralAffectDataset(args.data / "train.pt")
    train_cpu, heldout_cpu = sentence_split([q["sentence_id"] for q in dataset.queries], args.seed)
    train_ids, heldout_ids = train_cpu.to(device), heldout_cpu.to(device)
    system = NeutralAffectSystem(cfg).to(device).eval()
    system.load_state_dict(checkpoint["model"], strict=True)
    freeze_module(system)
    with torch.no_grad():
        query, _ = prepare(system, dataset, device)
        query, feature_stats = training_only_audio(query, dataset.cache["audio_stats"], train_ids)
        if args.audio_features == "acoustic_context":
            query = {**query, "audio": augment_audio_context(query["audio"], query["valid"])}
            feature_stats = {**feature_stats, "augmentation": "current_delta_smooth5", "input_dim": int(query["audio"].shape[-1])}
        refs = dataset.identity_references
        reference = device_batch([r for speaker in sorted(refs) for r in refs[speaker]], device)
        ref_residual = torch.where(observed(reference), reference["motion"] - system.base(reference["content"], reference["valid"])["b0"], 0)
        identities = cached_identity(system, {"residual": ref_residual.reshape(len(refs), len(refs[0]), *ref_residual.shape[1:]), "valid": reference["valid"].reshape(len(refs), len(refs[0]), -1)})
        residual = affect_residual(query, select(identities, query["speaker_id"]))
        energy, weight, target_state = fit_expression_energy_target(
            residual, query["valid"], query["channel_mask"], query["speaker_id"],
            query["emotion_id"], query["intensity_id"], train_ids,
            system.motion_teacher.stride, mode=args.target,
        )
        scale = fit_target_scale(energy, weight, train_ids)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg_student = dict(cfg)
    cfg_student["model"] = dict(cfg["model"])
    target_rank = int(energy.shape[-1])
    cfg_student["model"]["affect_rank"] = target_rank
    student = fresh_student(cfg_student, args.seed, query["audio"].shape[-1], device)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=float(cfg["training"].get("audio_lr", 5e-4)), weight_decay=1e-5)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    target = energy / scale
    student.train()
    losses = []
    for _ in range(args.steps):
        sampled = torch.randint(len(train_cpu), (int(cfg["training"].get("batch_size", 8)),), generator=rng)
        ids = train_cpu[sampled].to(device)
        out = student(query["audio"][ids], query["valid"][ids])["controls"]
        pred = out[..., :target_rank]
        w = weight[ids]
        loss = ((pred - target[ids]).square() * w[..., None]).sum() / (w.sum() + 1e-8)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    student.eval()
    with torch.no_grad():
        pred = student(query["audio"], query["valid"])["controls"][..., :target_rank]
    summary = {
        "schema": "expression_energy_field_probe_v1", "seed": args.seed,
        "target_mode": args.target,
        "steps": args.steps, "rank": target_rank, "stride": system.motion_teacher.stride,
        "train": energy_metrics(pred, target, weight, query, train_ids, scale),
        "heldout": energy_metrics(pred, target, weight, query, heldout_ids, scale),
        "train_axes": axis_metrics(pred, target, weight, query, train_ids, scale),
        "heldout_axes": axis_metrics(pred, target, weight, query, heldout_ids, scale),
        "target_native_rms": float(scale[0]), "initial_state_sha256": state_hash(student.state_dict()),
        "basis_singular_values": (target_state["basis"]["singular_values"].detach().cpu().tolist()
                                  if target_state["basis"] is not None else None),
        "source_sha256": {"script": sha(Path(__file__))},
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "provenance.json", {"schema": "expression_energy_field_probe_v1", "checkpoint_sha256": sha(args.checkpoint), "train_cache_sha256": source_hash, "feature_stats": {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in feature_stats.items()}, "target_mode": args.target, "target": "training-only expression intensity proxy, binned stride4, per-clip mean removed", "train_indices": train_cpu.tolist(), "heldout_indices": heldout_cpu.tolist()})
    torch.save({"prediction": pred.cpu(), "target": target.cpu(), "weight": weight.cpu()}, args.output / "curves.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
