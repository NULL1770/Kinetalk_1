import copy

import pytest
import torch

from scripts.train_projection_schedule_ablation import (
    ARMS, canonical_hash, draws, schedule_probability, validate_internal_split,
)


def test_schedules_have_exact_prespecified_probabilities():
    assert schedule_probability("constant_teacher", 17, 5, 6) == .5
    assert schedule_probability("audio_only", 0, 0, 6) == 0
    assert schedule_probability("decay_teacher", 0, 0, 6) == .5
    assert schedule_probability("decay_teacher", 4, 5, 6) == 0
    assert schedule_probability("decay_teacher", 17, 5, 6) == 0
    with pytest.raises(ValueError):
        schedule_probability("other", 0, 0, 6)


def test_all_schedules_keep_identical_batch_noise_time_and_choice_rng():
    records = []
    for arm in ARMS:
        generator = torch.Generator().manual_seed(46)
        values = []
        for epoch in range(6):
            order = torch.randperm(7, generator=generator)
            for batch_index, ids in enumerate(order.split(3)):
                noise, flow_time, choice = draws(generator, len(ids), (4, 2))
                _ = choice < schedule_probability(arm, epoch, batch_index, 3)
                values += [ids, noise, flow_time, choice]
        records.append(values)
    for left, right in zip(records[0], records[1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(records[0], records[2]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_internal_split_accepts_shared_scripts_but_rejects_speaker_or_outer_leakage():
    fit = ["mead_M003_a", "mead_M005_a"]
    dev = ["mead_M007_a", "mead_M009_a", "mead_M011_a"]
    cache = {"splits": {"train": {"q": {"clip_id": fit, "sentence_id": ["s", "s"]}},
                        "validation": {"q": {"clip_id": dev, "sentence_id": ["s", "s", "s"]}}}}
    lock = {"schema": "projection_schedule_internal_split_v1", "source_role": "formal_training_queries_only",
            "outer_development_loaded": False, "new_identity_development_loaded": False,
            "fit_clip_ids": fit, "validation_clip_ids": dev, "heldout_speakers": ["mead_M007", "mead_M009", "mead_M011"]}
    bundle = {"provenance": {"internal_split_lock_sha256": canonical_hash(lock)}}
    assert validate_internal_split(cache, bundle, lock, fit, dev)["shared_sentence_count"] == 1
    changed = copy.deepcopy(lock)
    changed["outer_development_loaded"] = True
    with pytest.raises(ValueError, match="excluded"):
        validate_internal_split(cache, bundle, changed, fit, dev)
    bad_cache = copy.deepcopy(cache)
    bad_cache["splits"]["validation"]["q"]["speaker"] = ["mead_M003", "mead_M009", "mead_M011"]
    with pytest.raises(ValueError, match="heldout identities"):
        validate_internal_split(bad_cache, bundle, lock, fit, dev)
