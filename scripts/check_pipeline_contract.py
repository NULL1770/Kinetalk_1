"""Validate the four-stage data/model tensor contracts on a real dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kinetalk_b0.data import B0ResidualDataset, CanonicalStage1Dataset, collate_b0_residual, collate_stage1
from kinetalk_b0.models import Stage1Model, Stage2Model, Stage3Model, Stage4Model
from kinetalk_b0.utils import load_yaml, move_to_device
from kinetalk_b0.protocol import audit_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    protocol = audit_config(cfg["data"])
    print(f"protocol={protocol['version']} stage1={[(s, protocol['stage1']['splits'][s]['n']) for s in ('train','val','test')]} stage2_4={[(s, protocol['stage2_4']['splits'][s]['n']) for s in ('train','val','test')]}")
    device = torch.device(str(cfg.get("device", "cuda")) if torch.cuda.is_available() else "cpu")

    stage1_ds = CanonicalStage1Dataset(cfg, split="train", random_crop=False)
    stage1_batch = move_to_device(collate_stage1([stage1_ds[0]]), device)
    stage1 = Stage1Model(cfg).to(device).eval()
    with torch.no_grad():
        stage1_out = stage1(stage1_batch["content"], stage1_batch["mask"])
    assert stage1_out["b0"].shape == stage1_batch["target"].shape
    assert stage1_out["h0"].shape[:2] == stage1_batch["content"].shape[:2]
    print(f"stage1 records={len(stage1_ds)} content={tuple(stage1_batch['content'].shape)} b0={tuple(stage1_out['b0'].shape)}")

    ds = B0ResidualDataset(cfg, split="train", random_crop=False)
    batch = move_to_device(collate_b0_residual([ds[0]]), device)
    stage2 = Stage2Model(cfg).to(device).eval()
    query = batch["query"]
    with torch.no_grad():
        q1 = stage1(query["content"], query["mask"])
        reference = batch["style_reference"]
        r1 = stage1(reference["content"], reference["mask"])
        reference_residual = reference["motion"] - r1["b0"]
        factors = stage2.encode_factors(
            query["residual_gt"], query["residual_mask"],
            query.get("audio_emotion"), style_residual=query["motion"] - q1["b0"],
        )
        ref_style = stage2.style(reference_residual, reference["mask"], reference.get("audio_emotion"))
        flow_pred, flow_target = stage2.flow_prediction(query["residual_gt"], q1["h0"], factors, query["residual_mask"])
    assert flow_pred.shape == flow_target.shape == query["residual_gt"].shape
    assert ref_style.shape == factors["style"].shape
    print(f"stage2-4 records={len(ds)} query_motion={tuple(query['motion'].shape)} reference_residual={tuple(reference_residual.shape)} flow={tuple(flow_pred.shape)}")

    stage3 = Stage3Model(cfg, stage2).to(device).eval()
    with torch.no_grad():
        audio_factors = stage3(query["audio_emotion"], query["mask"])
        teacher = stage3.teacher(query["residual_gt"], query["residual_mask"])
    assert audio_factors["global"].shape == teacher["global"].shape
    stage4 = Stage4Model(cfg, stage1, stage2, stage3).to(device).eval()
    with torch.no_grad():
        conditions = stage4.conditions(batch, training_target=True)
    assert conditions["b0_pred"].shape == query["motion"].shape
    assert conditions["style"].shape[0] == query["motion"].shape[0]
    print(f"stage3 global={tuple(audio_factors['global'].shape)} stage4 b0={tuple(conditions['b0_pred'].shape)} style={tuple(conditions['style'].shape)}")
    print("PIPELINE_CONTRACT_OK")


if __name__ == "__main__":
    main()
