"""Fit a real-BS emotion probe, then freeze it for held-out/Stage4 evaluation.

Run `fit` first (train and val only). Run `evaluate --split test` once after
freezing the protocol. Old Stage4 checkpoints remain diagnostic generators.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_probe import MotionEmotionProbe, classification_metrics, center_motion, motion_features
from kinetalk_b0.protocol import audit_config, load_split, pair_path, sha256
from kinetalk_b0.utils import load_yaml, seed_everything, move_to_device


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def real_features(cfg, split):
    rows, _ = load_split(cfg["data"], split, stage1=True)
    names = cfg["data"]["emotion_classes"]
    features, labels = [], []
    for row in rows:
        motion, mask = center_motion(row, pair_path(row, cfg["data"]), cfg["data"]["window"], cfg["data"]["motion_dim"])
        features.append(motion_features(motion, mask))
        labels.append(names.index(row["emotion"]))
    return rows, torch.stack(features), torch.tensor(labels)


def fit(args, cfg, protocol, device):
    if args.checkpoint.exists() or args.output.exists():
        raise FileExistsError("Use fresh probe checkpoint and output paths; never overwrite a frozen probe")
    _, train_x, train_y = real_features(cfg, "train")
    _, val_x, val_y = real_features(cfg, "val")
    names = cfg["data"]["emotion_classes"]
    if set(train_y.tolist()) != set(range(len(names))):
        raise ValueError("Train split is missing emotion classes")
    model = MotionEmotionProbe(train_x.shape[1], args.hidden, len(names))
    model.fit_normalization(train_x)  # Only train contributes preprocessing statistics.
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    # Class weights are also estimated only from train.
    counts = torch.bincount(train_y, minlength=len(names)).float()
    weights = (len(train_y) / (len(names) * counts)).to(device)
    best_score, best_epoch, history = -1., 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device), weight=weights)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite probe loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(val_x.to(device)).argmax(1).cpu()
        metrics = classification_metrics(val_y, prediction, names)
        history.append({"epoch": epoch, **metrics})
        # Fixed selection rule: val macro F1; ties retain earliest epoch.
        if metrics["macro_f1"] > best_score:
            best_score, best_epoch = metrics["macro_f1"], epoch
            payload = {"kind": "independent_real_bs_statistics_probe_v1",
                       "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       "hidden": args.hidden, "feature_dim": train_x.shape[1], "classes": names,
                       "window": cfg["data"]["window"], "motion_dim": cfg["data"]["motion_dim"],
                       "epoch": epoch, "seed": args.seed, "lr": args.lr,
                       "fit_splits": ["train", "val"], "test_used_for_selection": False,
                       "selection": "val macro_F1, earliest tie", "protocol": protocol,
                       "feature_schema": "all BS: mean, std, q10, q90, abs velocity mean, velocity std; teacher_mask center crop"}
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.checkpoint)
        print(f"probe epoch={epoch} val_macro_f1={metrics['macro_f1']:.4f} best_epoch={best_epoch}", flush=True)
    write_json(args.output, {"mode": "fit_real_motion_only", "protocol": protocol,
                            "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
                            "selection": "val macro_F1, earliest tie", "best_epoch": best_epoch,
                            "validation": history[best_epoch-1], "history": history,
                            "test_features_accessed": False, "generator_outputs_used_for_training": False})


def load_probe(args, cfg, protocol, device):
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kind") != "independent_real_bs_statistics_probe_v1" or payload.get("test_used_for_selection") is not False:
        raise ValueError("Not a frozen independent probe")
    for key in ("stage1", "stage2_4"):
        for split in ("train", "val", "test"):
            if payload["protocol"][key]["manifests"][split]["sha256"] != protocol[key]["manifests"][split]["sha256"]:
                raise ValueError("Manifest changed after probe fitting")
    if payload["classes"] != cfg["data"]["emotion_classes"] or payload["window"] != cfg["data"]["window"] or payload["motion_dim"] != cfg["data"]["motion_dim"]:
        raise ValueError("Probe input schema changed")
    model = MotionEmotionProbe(payload["feature_dim"], payload["hidden"], len(payload["classes"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, payload


def load_generator(cfg, device):
    spec = importlib.util.spec_from_file_location("stage4_semantics", Path(__file__).with_name("09_eval_stage4_semantics.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_stage4(cfg, device)


@torch.inference_mode()
def evaluate(args, cfg, protocol, device):
    if args.output.exists():
        raise FileExistsError("Evaluation output already exists; do not overwrite final-test evidence")
    model, payload = load_probe(args, cfg, protocol, device)
    rows, features, labels = real_features(cfg, args.split)
    prediction = model(features.to(device)).argmax(1).cpu()
    result = {"mode": "independent_motion_emotion_probe", "split": args.split, "protocol": protocol,
              "probe": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint), "selected_epoch": payload["epoch"],
                        "feature_schema": payload["feature_schema"], "frozen": True},
              "real_motion": classification_metrics(labels, prediction, payload["classes"]),
              "generator_heldout_generalization_established": False,
              "limitation": "A real-BS statistics probe, not human emotion ground truth. Old generator checkpoints saw held-out speakers; generated scores are diagnostic only."}
    predictions = [{"clip_id": r["clip_id"], "speaker": r["speaker"], "target": int(y), "prediction": int(p)}
                   for r, y, p in zip(rows, labels, prediction)]
    if args.stage4:
        from kinetalk_b0.data import B0ResidualDataset, collate_b0_residual
        base = B0ResidualDataset(cfg, args.split, random_crop=False)
        generator, checkpoints = load_generator(cfg, device)
        gt, generated, prior, targets, pairs = [], [], [], [], []
        for index, row in enumerate(base.items):
            # Per-query stable seed preserves the existing dataset reference
            # policy and makes selection independent of batch size/workers.
            seed = int.from_bytes(__import__("hashlib").sha256(f"{args.seed}:{row['clip_id']}".encode()).digest()[:4], "big")
            random.seed(seed)
            item = base[index]
            if not item["relations"]["style"]:
                raise ValueError(f"Missing deployment reference: {row['clip_id']}")
            batch = move_to_device(collate_b0_residual([item]), device)
            conditions = generator.conditions(batch)
            final, _ = generator.render(batch, conditions, steps=args.steps, stochastic=False)
            real, mask = center_motion(row, pair_path(row, cfg["data"]), cfg["data"]["window"], cfg["data"]["motion_dim"])
            n = len(real)
            if not torch.allclose(real, item["query"]["motion"][:n], atol=1e-6):
                raise ValueError("Real and generated evaluation clocks differ")
            pred = []
            for motion in (real, final[0, :n].cpu(), conditions["b0_pred"][0, :n].cpu()):
                pred.append(int(model(motion_features(motion, mask).unsqueeze(0).to(device)).argmax(1)))
            gt.append(pred[0]); generated.append(pred[1]); prior.append(pred[2]); targets.append(payload["classes"].index(row["emotion"]))
            pairs.append({"clip_id": row["clip_id"], "reference_row_id": item["style_reference"]["clip_id"], "target": targets[-1],
                          "real_prediction": pred[0], "generated_prediction": pred[1], "b0_prediction": pred[2]})
            if (index+1) % 50 == 0:
                print(f"generated emotion evaluation {index+1}/{len(base)}", flush=True)
        y, g, real_p = torch.tensor(targets), torch.tensor(generated), torch.tensor(gt)
        result["stage4"] = {"checkpoints": checkpoints, "steps": args.steps, "stochastic": False,
                             "seed": args.seed, "reference_policy": "existing dataset policy; deterministic seed per query; same split",
                             "paired_real": classification_metrics(y, real_p, payload["classes"]),
                             "generated": classification_metrics(y, g, payload["classes"]),
                             "b0_only": classification_metrics(y, torch.tensor(prior), payload["classes"]),
                             "generated_vs_real_agreement": float((g == real_p).float().mean()),
                             "generated_accuracy_where_probe_gets_real_correct": float((g[real_p == y] == y[real_p == y]).float().mean()) if (real_p == y).any() else None,
                             "predictions": pairs}
    result["real_predictions"] = predictions
    result["per_speaker_real"] = {speaker: classification_metrics(labels[[r["speaker"] == speaker for r in rows]], prediction[[r["speaker"] == speaker for r in rows]], payload["classes"])
                                  for speaker in sorted({r["speaker"] for r in rows})}
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "real": result["real_motion"], "generated": result.get("stage4", {}).get("generated")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--stage4", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden, args.steps) < 1 or args.lr <= 0:
        parser.error("epochs/batch-size/hidden/steps/lr must be positive")
    seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cfg = load_yaml(args.config)
    protocol = audit_config(cfg["data"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (fit if args.mode == "fit" else evaluate)(args, cfg, protocol, device)


if __name__ == "__main__":
    main()
