import copy

import pytest
import torch

from scripts.prepare_cross_identity_development import validate_cross_identity_protocol


def protocol():
    rows = [{"clip_id": "q", "sentence": "known_training_script", "speaker": "mead_V", "speaker_id": 22,
             "dataset": "mead", "split": "val", "pilot_split": "new_identity_validation",
             "emotion": 5, "emotion_id": 5, "valid_frames": 70}]
    refs = [{"clip_id": f"r{i}", "sentence": f"ref{i}", "speaker": "mead_V", "speaker_id": 22,
             "dataset": "mead", "split": "val", "pilot_split": "new_identity_enrollment",
             "emotion": 0, "emotion_id": 0, "valid_frames": 70} for i in range(2)]
    selection = {"schema": "formal_predictable_metadata_lock_v1",
                 "new_identity_validation_speaker_to_id": {"mead_V": 22},
                 "train_speaker_to_id": {"mead_A": 0}, "sealed_test_speaker_to_id": {"mead_T": 25},
                 "sentence_roles": {"enrollment": ["ref0", "ref1"], "reserved_prior_test": ["reserved"]},
                 "min_references": 2, "reference_counts": {"val": {"mead_V": 2}}}
    return rows, refs, selection


def test_cross_identity_allows_shared_script_with_two_independent_references():
    rows, refs, selection = protocol()
    grouped = validate_cross_identity_protocol(rows, refs, selection)
    assert list(grouped) == [22] and len(grouped[22]) == 2


@pytest.mark.parametrize("change", ["test_source", "identity_overlap", "reference_query", "reserved_query", "remapped_emotion", "wrong_id"])
def test_cross_identity_rejects_protocol_leaks(change):
    rows, refs, selection = protocol()
    if change == "test_source":
        rows[0]["split"] = "test"
    elif change == "identity_overlap":
        selection["train_speaker_to_id"]["mead_V"] = 22
    elif change == "reference_query":
        rows[0]["sentence"] = "ref0"
    elif change == "reserved_query":
        rows[0]["sentence"] = "reserved"
    elif change == "remapped_emotion":
        rows[0]["emotion_id"] = 2
    else:
        refs[0]["speaker_id"] = 0
    with pytest.raises(ValueError):
        validate_cross_identity_protocol(rows, refs, selection)


def test_cross_identity_requires_locked_reference_count_and_unique_clips():
    rows, refs, selection = protocol()
    with pytest.raises(ValueError):
        validate_cross_identity_protocol(rows, refs[:1], selection)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_cross_identity_protocol(rows + copy.deepcopy(rows), refs, selection)


def test_restored_head_uses_exact_saved_scales_and_outputs_without_training_data():
    from scripts.evaluate_cross_identity_projection import RestoredFixedAudioHead
    from scripts.train_predictable_renderer import PredictableAudioHead, state_hash
    torch.manual_seed(5)
    basis, _ = torch.linalg.qr(torch.randn(12, 8))
    state = {"basis": basis, "motion_channel_indices": list(range(12)),
             "std": torch.rand(17) + .1, "weights": torch.randn(17, 8)}
    training = torch.randn(4, 6, 52)
    weight = torch.ones(4, 6)
    original = PredictableAudioHead(state, training, weight)
    restored = RestoredFixedAudioHead(original.state_dict())
    features = torch.randn(3, 6, 17)
    test_weight = torch.ones(3, 6)
    torch.testing.assert_close(original(features, test_weight), restored(features, test_weight), rtol=0, atol=0)
    torch.testing.assert_close(original.teacher(training, weight), restored.teacher(training, weight), rtol=0, atol=0)
    assert state_hash(original.state_dict()) == state_hash(restored.state_dict())
    assert not any(p.requires_grad for p in restored.parameters())
    broken = dict(original.state_dict(), target_scale=torch.zeros(8))
    with pytest.raises(ValueError, match="scales"):
        RestoredFixedAudioHead(broken)


def test_selected_run_rejects_ineligible_or_swapped_checkpoint():
    from scripts.evaluate_cross_identity_projection import validate_selected_run
    from scripts.train_formal_predictable_projection import SCHEMA, canonical_hash
    from scripts.train_predictable_renderer import state_hash
    recipe = {"audio_activity_gate": "1-softmax(frozen_audio_logits)[neutral]",
              "trainable": ["local_projection.weight"], "noise_seeds": [42, 123, 2026], "new_test_loaded": False}
    head = {"test": torch.tensor([3.])}
    digest = canonical_hash(recipe)
    provenance = {"recipe": recipe, "recipe_sha256": digest, "head_sha256": state_hash(head),
                  "frozen_state_sha256": "frozen"}
    selected = {"schema": SCHEMA, **provenance, "head": head, "latest_selection": {"eligible": True},
                "completed_epochs": 12, "best_epoch": 12}
    summary = {"recipe_sha256": digest, "has_eligible_checkpoint": True,
               "selected_checkpoint_sha256": "selected", "best_epoch": 12}
    assert validate_selected_run(provenance, summary, selected, "selected") == recipe
    with pytest.raises(ValueError, match="differs"):
        validate_selected_run(provenance, summary, selected, "swapped")
    selected["latest_selection"]["eligible"] = False
    with pytest.raises(ValueError, match="eligible"):
        validate_selected_run(provenance, summary, selected, "selected")
