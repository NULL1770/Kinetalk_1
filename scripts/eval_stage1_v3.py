"""Compatibility entry point for the corrected Stage1 evaluator.

The former script read ``canonical_motion`` directly from a pair artifact.
That array can be source-side emotional motion and is not a neutral Stage1
target. This entry point delegates to the dataset contract, which loads the
materialized canonical neutral target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinetalk_b0.utils import load_yaml
from scripts.eval_stage1_comparable import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--pair-root",
        default=None,
        help="Retained for CLI compatibility; no longer read as a target source.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    cfg["_eval_split"] = args.split
    if args.manifest:
        data_cfg = cfg.setdefault("data", {})
        data_cfg[{"train": "stage1_manifest", "val": "stage1_val_manifest", "test": "stage1_test_manifest"}[args.split]] = args.manifest
    result = evaluate(cfg, args.ckpt)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
