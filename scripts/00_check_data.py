from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinetalk_b0.data import B0ResidualDataset
from kinetalk_b0.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v3 source motion and neutral-partner pairing")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args()
    config = load_yaml(args.config)
    dataset = B0ResidualDataset(config, split="train", random_crop=False)
    counts = {"neutral": 0, "emotion": 0, "style": 0}
    checked = min(len(dataset), args.limit)
    for index in range(checked):
        item = dataset[index]
        for key in counts:
            counts[key] += int(item["relations"][key])
        query = item["query"]
        assert query["motion"].shape[-1] == config["data"]["motion_dim"]
        assert query["b0_gt"].shape == query["motion"].shape
        assert query["residual_gt"].shape == query["motion"].shape
        assert query["mask"].shape == query["residual_mask"].shape
        if item["relations"]["emotion"]:
            partner = item["emotion_pair"]
            assert query["speaker"] == partner["speaker"]
            assert query["sentence_id"] == partner["sentence_id"]
            assert query["crop_start"] == partner["crop_start"]
            assert query["emotion_id"] != partner["emotion_id"]
    print(f"records={len(dataset)} checked={checked}")
    print("relation coverage:", {key: f"{value}/{checked} ({value / max(checked, 1):.1%})" for key, value in counts.items()})
    print("contract: raw BS; b0_gt=K0*aligned neutral BS; residual_gt=motion-b0_gt; no second DTW")


if __name__ == "__main__":
    main()
