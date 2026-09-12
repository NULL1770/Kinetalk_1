import json

import pytest
import torch

from kinetalk_b0.protocol import audit_splits, load_split, enrich_pair, require_training_protocol
from kinetalk_b0.emotion_probe import motion_features, MotionEmotionProbe, classification_metrics
from kinetalk_b0.data import B0ResidualDataset, CanonicalStage1Dataset
from kinetalk_b0.utils import canonical_intensity, safe_intensity


def row(split="train", clip="mead_M003_happy_L2_001", emotion="happy", intensity=2, speaker="mead_M003"):
    return {
        "clip_id": clip,
        "source_clip_id": clip,
        "reference_clip_id": f"{speaker}_neutral_L1_001",
        "speaker": speaker,
        "sentence_id": "text_1",
        "split": split,
        "emotion": emotion,
        "intensity": intensity,
        "intensity_id": 0 if emotion == "neutral" else intensity,
        "reference_speaker": speaker,
        "reference_split": split,
        "reference_sentence_id": "text_1",
        "reference_emotion": "neutral",
        "teacher_artifact": "/tmp/pair.npz",
    }


def test_missing_intensity_is_not_silently_neutral():
    with pytest.raises(ValueError):
        safe_intensity(None, 4)
    assert canonical_intensity(1, "neutral", 4) == 0
    assert canonical_intensity(3, "happy", 4) == 3


def test_split_loader_requires_explicit_test_manifest(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row()) + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="test_manifest"):
        load_split({"train_manifest": str(path)}, "test")


def test_split_audit_rejects_speaker_overlap():
    with pytest.raises(ValueError, match="Split leakage"):
        audit_splits({"train": [row()], "val": [row("val", "mead_M003_sad_L1_001", "sad", 1)], "test": [row("test", "mead_M005_sad_L1_001", "sad", 1)]})


@pytest.mark.parametrize("value", [None, "", "bad", -1, 4, 1.5, True, float("nan"), float("inf")])
def test_invalid_intensity_is_rejected(value):
    with pytest.raises(ValueError):
        safe_intensity(value, 4)


@pytest.mark.parametrize("dataset", [B0ResidualDataset, CanonicalStage1Dataset])
def test_datasets_cannot_fallback_to_val(dataset, tmp_path):
    with pytest.raises(ValueError, match="test_manifest"):
        dataset({"data": {"val_manifest": str(tmp_path / "val.jsonl")}}, "test")


def test_test_manifest_must_contain_test_rows(tmp_path):
    path = tmp_path / "wrong.jsonl"
    path.write_text(json.dumps(row("val")) + "\n")
    with pytest.raises(ValueError, match="expected test"):
        load_split({"test_manifest": str(path)}, "test")


def test_distinct_files_do_not_make_duplicate_clips_independent(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    path.write_text((json.dumps(row()) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_split({"train_manifest": str(path)}, "train")


def test_valid_split_loads_and_counts(tmp_path):
    data = {}
    splits = {}
    for split, speaker in zip(("train", "val", "test"), ("mead_M003", "mead_M005", "mead_W003")):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(row(split, f"{speaker}_happy_L2_001", speaker=speaker)) + "\n")
        data[f"{split}_manifest"] = str(path)
        splits[split], _ = load_split(data, split)
    report = audit_splits(splits)
    assert report["splits"]["test"]["n"] == 1
    assert all(not x["speakers"] for x in report["overlap"].values())


def test_enrichment_does_not_silently_rewrite_labels():
    pair = row()
    source = dict(clip_id=pair["clip_id"], speaker=pair["speaker"], sentence_id="text_1", split="train", emotion="happy", intensity=2, dataset="mead")
    reference = dict(source, clip_id=pair["reference_clip_id"], emotion="neutral", intensity=1)
    records = {source["clip_id"]: source, reference["clip_id"]: reference}
    enriched = enrich_pair(pair, records)
    assert enriched["intensity_id"] == 2
    with pytest.raises(ValueError, match="intensity_id disagreement"):
        enrich_pair(dict(pair, intensity_id=0), records)


def test_probe_ignores_masked_motion_and_does_not_mutate_stats():
    motion = torch.rand(12, 52)
    mask = torch.ones(12, dtype=torch.bool)
    mask[-2:] = False
    first = motion_features(motion, mask)
    motion[-2:] = float("nan")
    assert torch.equal(first, motion_features(motion, mask))
    model = MotionEmotionProbe(len(first), 8, 8).eval()
    model.fit_normalization(torch.stack([first, first+1]))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model(first.unsqueeze(0) * 100)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_classification_reports_absent_class_and_macro_f1():
    result = classification_metrics(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1, 2]), ["a", "b", "c"])
    assert result["accuracy"] == .5
    assert result["balanced_accuracy"] == .5
    assert result["per_class"]["c"]["recall"] is None
    assert result["macro_f1"] == pytest.approx((2/3 + .5)/3)


def test_old_checkpoint_cannot_enter_new_training_chain():
    with pytest.raises(ValueError, match="no split provenance"):
        require_training_protocol({"model": {}}, {})
