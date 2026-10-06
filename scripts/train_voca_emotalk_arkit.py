"""Train VOCA/EmoTalk-style MEAD-ARKit52 adapters on train/validation only."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.faceformer_arkit_adapter import load_prepared, validate_batch, _batch
from scripts.voca_emotalk_arkit_models import (VocaARKit, EmoTalkARKit, masked_mse,
                                                masked_velocity_mse)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def run(args):
    from scripts.baseline_training import train_baseline
    _, report = train_baseline(args, args.method)
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--method",choices=("voca","emotalk"),required=True); p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda"); p.add_argument("--epochs",type=int,default=80); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--hidden",type=int,default=256); p.add_argument("--layers",type=int,default=2); p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--seed",type=int,default=42); p.add_argument("--smoke",action="store_true")
    p.add_argument("--threads",type=int,default=2); p.add_argument("--resume",action="store_true")
    p.add_argument("--adapter",choices=("legacy","core_arkit_v1"),default="core_arkit_v1")
    run(p.parse_args())
