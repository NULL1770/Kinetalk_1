from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from kinetalk_b0.data import B0ResidualDataset, collate_b0_residual
from kinetalk_b0.models import Stage1Model
from kinetalk_b0.utils import load_checkpoint, load_yaml, move_to_device, seed_everything


def _emotion_name(record: dict) -> str:
    aliases = {"disgusted": "disgust", "fearful": "fear", "surprised": "surprise"}
    return aliases.get(str(record.get("emotion", "neutral")).lower(), str(record.get("emotion", "neutral")).lower())


def _masked_stats(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, active: torch.Tensor) -> tuple[float, float, float, float, float]:
    weight = (mask.unsqueeze(-1) & active.view(1, 1, -1)).to(pred.dtype)
    count = weight.sum().clamp_min(1.0)
    diff = (pred - target) * weight
    mae = diff.abs().sum() / count
    rmse = diff.square().sum().sqrt() / count.sqrt()
    target_abs = (target.abs() * weight).sum() / count
    pred_abs = (pred.abs() * weight).sum() / count
    valid = weight[0].bool()
    x, y = pred[0][valid], target[0][valid]
    x = x - x.mean()
    y = y - y.mean()
    correlation = (x * y).sum() / (x.square().sum().sqrt() * y.square().sum().sqrt()).clamp_min(1e-8)
    return float(mae), float(rmse), float(target_abs), float(pred_abs), float(correlation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick Stage1 b0 evaluation by emotion")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--per-emotion", type=int, default=32)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    cfg["data"]["num_workers"] = min(int(cfg["data"].get("num_workers", 0)), 2)
    dataset = B0ResidualDataset(cfg, split="train", random_crop=False)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.items):
        grouped[_emotion_name(record)].append(index)
    selected: list[int] = []
    for emotion in cfg["data"]["emotion_classes"]:
        indices = grouped.get(str(emotion).lower(), [])
        if len(indices) > args.per_emotion:
            step = max(1, len(indices) // args.per_emotion)
            indices = indices[::step][: args.per_emotion]
        selected.extend(indices)
    if not selected:
        raise RuntimeError("No evaluation samples selected")

    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=int(cfg["data"].get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg["data"].get("num_workers", 0)) > 0,
        collate_fn=collate_b0_residual,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage1Model(cfg).to(device)
    payload = load_checkpoint(cfg["paths"]["stage1_ckpt"], model, map_location=device)
    model.eval()
    active = torch.zeros(int(cfg["data"]["motion_dim"]), dtype=torch.bool, device=device)
    active[list(cfg["data"]["neutral_output_indices"])] = True
    inactive = ~active

    sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0])
    pair_sum = [0.0, 0.0, 0]
    inactive_max = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            query, emotion = batch["query"], batch["emotion_pair"]
            query_out = model(query["content"], query["mask"])["b0"]
            emotion_out = model(emotion["content"], emotion["mask"])["b0"]
            for branch, output in (("query", query_out), ("emotion_partner", emotion_out)):
                source = query if branch == "query" else emotion
                labels = source["emotion_id"].tolist()
                for row, label in enumerate(labels):
                    mask = source["mask"][row : row + 1]
                    mae, rmse, target_abs, pred_abs, correlation = _masked_stats(
                        output[row : row + 1], source["b0_gt"][row : row + 1], mask, active
                    )
                    slot = sums[(branch, int(label))]
                    slot[0] += mae
                    slot[1] += rmse
                    slot[2] += target_abs
                    slot[3] += pred_abs
                    slot[4] += correlation
                    slot[5] += 1
                inactive_max = max(inactive_max, float(output[..., inactive].abs().max().item()))
            pair_mask = query["mask"] & emotion["mask"]
            pair_weight = pair_mask.unsqueeze(-1).to(query_out.dtype)
            pair_count = pair_weight.expand_as(query_out).sum().clamp_min(1.0)
            pair_sum[0] += float(((query_out - emotion_out).abs() * pair_weight).sum().item())
            pair_sum[1] += float((((query_out - emotion_out).square()) * pair_weight).sum().item())
            pair_sum[2] += float(pair_count.item())

    names = [str(item).lower() for item in cfg["data"]["emotion_classes"]]
    print(f"checkpoint_epoch={payload.get('epoch', 'unknown')} samples={len(selected)} device={device}")
    print("branch emotion count mae rmse target_abs pred_abs corr nmae")
    for branch in ("query", "emotion_partner"):
        for label, name in enumerate(names):
            total_mae, total_rmse, total_abs, total_pred_abs, total_corr, count = sums[(branch, label)]
            if not count:
                continue
            mae, rmse = total_mae / count, total_rmse / count
            target_abs, pred_abs = total_abs / count, total_pred_abs / count
            correlation = total_corr / count
            nmae = mae / max(target_abs, 1e-8)
            print(f"{branch} {name} {count} {mae:.6f} {rmse:.6f} {target_abs:.6f} {pred_abs:.6f} {correlation:.4f} {nmae:.4f} gain={1.0 - nmae:.4f}")
    pair_mae = pair_sum[0] / max(pair_sum[2], 1.0)
    pair_rmse = (pair_sum[1] / max(pair_sum[2], 1.0)) ** 0.5
    print(f"pair_consistency mae={pair_mae:.6f} rmse={pair_rmse:.6f}")
    print(f"inactive_channel_abs_max={inactive_max:.8f}")


if __name__ == "__main__":
    random.seed(42)
    main()
