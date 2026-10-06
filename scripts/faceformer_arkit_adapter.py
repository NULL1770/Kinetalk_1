"""Auditable MEAD-ARKit52 FaceFormer adapter.

This runner consumes the prepared train/validation cache and never opens test
targets.  It keeps the upstream FaceFormer autoregressive Transformer decoder,
periodic positional encoding, and aligned temporal attention; only the input
representation (locked 1540-D audio cache) and output space (52 ARKit
coefficients) are adapted.  Identity is supplied by independent neutral
enrollment anchors, which makes validation speaker-disjoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

# Direct CLI execution places ``scripts/`` rather than the repository root on
# sys.path.  Add the root explicitly so the canonical paper-data loader and
# model modules resolve identically under ``python scripts/...`` and
# ``python -m scripts...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # Works both as ``python -m scripts...`` and direct script invocation.
    from scripts.faceformer_arkit_model import FaceFormerARKit, FaceFormerARKitConfig, masked_mse
except ModuleNotFoundError:  # pragma: no cover - direct CLI path
    from faceformer_arkit_model import FaceFormerARKit, FaceFormerARKitConfig, masked_mse


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prepared(path: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    """Load a prepared cache, accepting either the full object or a torch file.

    The canonical object is returned by ``prepare_paper_full_data.load_paper_data``
    or by ``full_staged_data.load_training_inputs``.  No fallback scans the
    sealed-test directory and no test split is accepted.
    """
    path = Path(path)
    if path.is_dir():
        from scripts.packed_trainval_cache import is_packed, load_packed
        if is_packed(path):
            payload = load_packed(path, materialize=False)
            if smoke:
                payload = dict(payload)
                payload["splits"] = {role: split.batch(torch.arange(min(8, len(split["_lengths"]))))
                                     for role, split in payload["splits"].items()}
        else:
            # Use the canonical paper-data loader directly.  This avoids
            # writing a second multi-gigabyte torch cache containing padded
            # train/validation tensors while preserving the same train-only
            # statistics and split provenance.
            from scripts.prepare_paper_full_data import load_paper_data
            payload = load_paper_data(path, smoke=smoke)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if "splits" not in payload:
        raise ValueError("Prepared FaceFormer input must contain train/validation splits")
    if set(payload["splits"]) != {"train", "validation"}:
        raise ValueError("FaceFormer adapter accepts only train and validation splits")
    if payload.get("provenance", {}).get("test_loaded", False) or payload.get("test_loaded", False):
        raise ValueError("Refuse a prepared object with test targets loaded")
    for role in ("train", "validation"):
        split = payload["splits"][role]
        for key in ("audio_features", "motion", "valid", "anchors"):
            if key not in split:
                raise ValueError(f"Missing {role}.{key} in prepared cache")
        if split["audio_features"].ndim not in (2, 3) or split["audio_features"].shape[-1] != 1540:
            raise ValueError("Expected locked 1540-D audio features")
        if split["motion"].shape[-1] != 52 or split["anchors"].shape[-1] != 52:
            raise ValueError("Expected MEAD-ARKit52 motion and anchors")
        if split["valid"].dtype != torch.bool:
            raise ValueError("valid must be boolean")
    train_speakers = set(map(str, payload["splits"]["train"].get("speaker", [])))
    val_speakers = set(map(str, payload["splits"]["validation"].get("speaker", [])))
    if train_speakers and val_speakers and train_speakers & val_speakers:
        raise ValueError("Train and validation speakers overlap")
    if smoke:
        # Keep one clip per speaker and emotion while preserving the complete
        # validation role.  This is for interface checks only, never a score.
        payload = dict(payload)
        payload["splits"] = {role: _smoke_split(split) for role, split in payload["splits"].items()}
    return payload


def _smoke_split(split: dict[str, Any], limit: int = 8) -> dict[str, Any]:
    if hasattr(split, "batch"):
        return split
    n = min(limit, len(split["valid"]))
    return {k: (v[:n] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == len(split["valid"]) else v)
            for k, v in split.items()}


def validate_batch(split: dict[str, Any]) -> None:
    if hasattr(split, "batch"):
        n = min(2, len(split["_lengths"]))
        validate_batch(split.batch(torch.arange(n), "cpu"))
        return
    audio, target, anchor, valid = (split["audio_features"], split["motion"],
                                    split["anchors"], split["valid"])
    if audio.shape[:2] != target.shape[:2] or valid.shape != target.shape[:2]:
        raise ValueError("Audio, motion and valid frame clocks differ")
    if anchor.shape != (len(valid), 52):
        raise ValueError("anchors must be [N,52]")
    if not torch.isfinite(audio[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError("Non-finite observed data")


def _batch(split: dict[str, Any], ids: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    if hasattr(split, "batch"):
        return split.batch(ids, device)
    out = {}
    for key in ("audio_features", "motion", "anchors", "valid"):
        value = split[key][ids]
        out[key] = value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
    if "channel_mask" in split:
        out["channel_mask"] = split["channel_mask"][ids].to(device)
    return out


def train(args: argparse.Namespace) -> tuple[FaceFormerARKit, dict[str, Any]]:
    from scripts.baseline_training import train_baseline
    return train_baseline(args, 'faceformer')


@torch.no_grad()
def infer(args: argparse.Namespace) -> dict[str, Any]:
    data = load_prepared(args.data, smoke=args.smoke)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = FaceFormerARKitConfig(**checkpoint["config"])
    model = FaceFormerARKit(cfg).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    split = data["splits"][args.split]; validate_batch(split)
    outputs = []
    for ids in torch.arange(len(split["valid"])).split(args.batch_size):
        batch = _batch(split, ids, args.device)
        outputs.append(model.predict(batch["audio_features"], batch["anchors"]).cpu())
    prediction = torch.cat(outputs)
    report = {"schema": "faceformer_arkit52_predictions_v1", "split": args.split,
              "prediction_shape": list(prediction.shape), "test_loaded": False,
              "checkpoint_sha256": _sha(Path(args.checkpoint))}
    if args.output:
        output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
        torch.save({"prediction": prediction, "clip_id": split.get("clip_id", []),
                    "valid": split["valid"], "report": report}, output / "predictions.pt")
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--infer-checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.infer_checkpoint:
        print(json.dumps(infer(args), indent=2))
    else:
        _, report = train(args)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
