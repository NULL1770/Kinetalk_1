"""Frozen, independent neutral identity evaluation for the small MEAD pilot.

Gallery: the four training enrollment references per identity. Queries: only
heldout neutral clips, never used to fit the new identity module. The original
B0 pretraining may have included these clips, so this is not a full-system
unseen-data or unseen-person evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from scripts.train_neutral_affect_pilot import device_batch, observed, sha, write_json


def check_independence(train, heldout):
    train_clips = {q["clip_id"] for q in train.queries}
    heldout_clips = {q["clip_id"] for q in heldout.queries}
    train_sentences = {q["sentence_id"] for q in train.queries}
    heldout_sentences = {q["sentence_id"] for q in heldout.queries}
    if train_clips & heldout_clips or train_sentences & heldout_sentences:
        raise ValueError("Training and heldout queries overlap by clip or sentence")
    gallery = [q for refs in train.identity_references.values() for q in refs]
    gallery_clips = {q["clip_id"] for q in gallery}
    gallery_sentences = {q["sentence_id"] for q in gallery}
    if gallery_clips & (train_clips | heldout_clips):
        raise ValueError("A query clip is used for enrollment")
    if gallery_sentences & (train_sentences | heldout_sentences):
        raise ValueError("An enrollment sentence is used by a query")
    if len(gallery_clips) != len(gallery):
        raise ValueError("Enrollment contains duplicate clips")
    for speaker, references in train.identity_references.items():
        if len({q["sentence_id"] for q in references}) != len(references):
            raise ValueError("A person's enrollment references repeat a sentence")
        heldout_refs = heldout.identity_references.get(speaker)
        if heldout_refs is None or {q["clip_id"] for q in references} != {q["clip_id"] for q in heldout_refs}:
            raise ValueError("Training and heldout have different enrollment galleries")
    return {"train_heldout_clip_disjoint": True, "train_heldout_sentence_disjoint": True,
            "all_query_enrollment_clip_disjoint": True, "all_query_enrollment_sentence_disjoint": True,
            "distinct_enrollment_sentences_per_speaker": True,
            "enrollment_clips": len(gallery), "train_query_clips": len(train.queries),
            "heldout_query_clips": len(heldout.queries)}


def residual_and_mean(system, clips, device):
    batch = device_batch(clips, device)
    base = system.base(batch["content"], batch["valid"])["b0"]
    mask = observed(batch)
    residual = torch.where(mask, batch["motion"] - base, 0)
    count = mask.sum(1)
    mean = residual.sum(1) / count.clamp_min(1)
    return batch, residual, mean, count > 0


def source_record(clip):
    return {"clip_id": clip["clip_id"], "speaker_id": int(clip["speaker_id"]),
            "sentence_id": clip["sentence_id"], "artifact": clip["metadata"]["artifact"],
            "artifact_sha256": clip["metadata"]["artifact_sha256"],
            "crop_start": clip["metadata"]["crop_start"],
            "observed_frames": clip["metadata"]["observed_frames"]}


@torch.no_grad()
def evaluate(args):
    cfg = yaml.safe_load(args.config.read_text(encoding="utf8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_cfg = checkpoint.get("provenance", {}).get("config")
    if saved_cfg is not None and saved_cfg != cfg:
        raise ValueError("Evaluation configuration differs from checkpoint provenance")
    provenance = checkpoint.get("provenance", {})
    for split in ("train", "heldout"):
        expected = provenance.get(split + "_sha256")
        if expected is not None and sha(args.data / (split + ".pt")) != expected:
            raise ValueError(f"{split} cache differs from checkpoint training provenance")
    device = torch.device(cfg.get("device", "cuda"))
    torch.set_num_threads(4)
    train = NeutralAffectDataset(args.data / "train.pt")
    heldout = NeutralAffectDataset(args.data / "heldout.pt")
    isolation = check_independence(train, heldout)
    system = NeutralAffectSystem(cfg).to(device)
    system.load_state_dict(checkpoint["model"], strict=True)
    system.requires_grad_(False)
    system.eval()

    gallery_ids = sorted(train.identity_references)
    gallery, gallery_masks = [], []
    enrollment_sources = []
    for speaker in gallery_ids:
        clips = train.identity_references[speaker]
        batch, residual, _, valid_channels = residual_and_mean(system, clips, device)
        encoded = system.encode_identity(residual[None], batch["valid"][None])
        gallery.append({key: encoded[key][0] for key in ("code", "baseline", "neutral_mean")})
        gallery_masks.append(valid_channels.all(0))
        enrollment_sources.extend(source_record(clip) for clip in clips)
    gallery_codes = torch.stack([entry["code"] for entry in gallery])
    gallery_baselines = torch.stack([entry["baseline"] for entry in gallery])
    gallery_raw_means = torch.stack([entry["neutral_mean"] for entry in gallery])
    gallery_masks = torch.stack(gallery_masks)

    # The pooled comparison uses only independent neutral training queries;
    # each clip has equal weight and no heldout observation fits this baseline.
    train_neutral = [clip for clip in train.queries if int(clip["emotion_id"]) == 0]
    if not train_neutral:
        raise ValueError("Training neutral queries are required for the pooled baseline")
    _, _, training_means, training_channels = residual_and_mean(system, train_neutral, device)
    pooled_mean = (training_means * training_channels).sum(0) / training_channels.sum(0).clamp_min(1)
    pooled_channels = training_channels.any(0)

    queries = [clip for clip in heldout.queries if int(clip["emotion_id"]) == 0]
    if not queries:
        raise ValueError("No heldout neutral queries")
    batch, residual, target_means, target_channels = residual_and_mean(system, queries, device)
    query_codes = system.encode_identity(residual, batch["valid"])["code"]
    similarities = F.normalize(query_codes, dim=-1) @ F.normalize(gallery_codes, dim=-1).T
    predictions = similarities.argmax(-1)
    # Evaluation groups only; the architecture retains one shared motion output.
    regions = {"all_observed": tuple(range(52)), "eyes": tuple(range(14)),
               "brows": tuple(range(41, 46)), "eyes_brows": tuple(range(14)) + tuple(range(41, 46))}
    baseline_names = ("learned_enrollment_baseline", "raw_enrollment_mean", "train_pooled_neutral_mean", "zero")
    sums = {name: {region: [0.0, 0] for region in regions} for name in baseline_names}
    output_queries = []
    for index, query in enumerate(queries):
        true_id = int(query["speaker_id"])
        if true_id not in gallery_ids:
            raise ValueError("Heldout query identity has no enrollment gallery")
        gallery_index = gallery_ids.index(true_id)
        mask = target_channels[index] & gallery_masks[gallery_index] & pooled_channels
        candidates = {"learned_enrollment_baseline": gallery_baselines[gallery_index],
                      "raw_enrollment_mean": gallery_raw_means[gallery_index],
                      "train_pooled_neutral_mean": pooled_mean, "zero": torch.zeros_like(pooled_mean)}
        metrics = {}
        for name, prediction in candidates.items():
            metrics[name] = {}
            for region, channels in regions.items():
                ids = torch.tensor(channels, device=device)
                region_mask = mask[ids]
                errors = (prediction[ids] - target_means[index, ids]).square()[region_mask]
                count = errors.numel()
                metrics[name][region + "_mse"] = float(errors.mean()) if count else None
                sums[name][region][0] += float(errors.sum())
                sums[name][region][1] += count
        output_queries.append({**source_record(query), "true_id": true_id,
            "predicted_id": gallery_ids[int(predictions[index])],
            "cosine_scores_in_gallery_order": similarities[index].cpu().tolist(),
            "observed_channel_indices": mask.nonzero(as_tuple=True)[0].cpu().tolist(),
            "baseline_errors": metrics})
    correct = sum(query["true_id"] == query["predicted_id"] for query in output_queries)
    aggregate = {name: {region + "_mse": values[0] / values[1] if values[1] else None
                        for region, values in grouped.items()} for name, grouped in sums.items()}
    return {"schema": "neutral_identity_independent_eval_v1", "isolation_checks": isolation,
        "gallery_id_order": gallery_ids, "query_count": len(queries), "correct": correct,
        "accuracy": correct / len(queries), "chance": 1 / len(gallery_ids),
        "queries": output_queries, "baseline_errors": aggregate,
        "baseline_target": "Valid-frame temporal mean of heldout neutral motion minus frozen B0; no query motion fits a gallery baseline",
        "pooled_baseline_source": "Equal-weight per-clip means of neutral training-query residuals",
        "identity_retrieval": "Frozen single-reference heldout query code versus four-reference enrollment gallery code, cosine similarity",
        "enrollment_sources": enrollment_sources,
        "pooled_baseline_sources": [source_record(clip) for clip in train_neutral],
        "hashes": {"checkpoint": sha(args.checkpoint), "config": sha(args.config),
                   "train_cache": sha(args.data / "train.pt"), "heldout_cache": sha(args.data / "heldout.pt"),
                   "evaluation_script": sha(__file__), "identity_model_source": sha(Path("kinetalk_b0/models/neutral_affect.py"))},
        "checkpoint_provenance": checkpoint.get("provenance"),
        "limitations": ["Only enrolled speakers; no unseen-person generalization claim.",
                        "Original frozen B0 may have seen these clips in earlier pretraining; heldout applies to the new modules.",
                        "Four default queries give a coarse feasibility result; report correct/query_count, not statistical significance.",
                        "Cosine retrieval uses heldout neutral motion as an evaluation probe; generation still uses enrollment identity plus audio.",
                        "Expression coefficients can contain tracking/camera biases; retrieval does not establish geometric identity or emotion disentanglement.",
                        "Artifact hashes are recorded from validated materialization metadata; this evaluation checks cache hashes and does not reread source NPZ files."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    args = parser.parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({"correct": result["correct"], "queries": result["query_count"],
                      "accuracy": result["accuracy"], "baseline_errors": result["baseline_errors"]}, indent=2))


if __name__ == "__main__":
    main()
