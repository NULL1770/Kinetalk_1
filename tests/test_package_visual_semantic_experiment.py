import copy

import numpy as np
import pytest

from scripts import package_visual_semantic_experiment as pack


def example():
    baseline = np.arange(8 * 52, dtype=np.float32).reshape(8, 52) / 500
    common = np.array([False, True, True, False, True, True, False, False])
    native = np.array([True, True, True, False, True, True, True, False])
    return {"baseline52": baseline, "valid": common, "native_valid": native,
            "times": np.arange(8) / 25., "target": np.ones((8, 9), np.float32),
            "samples": {arm: np.stack([np.full((8, 9), 2., np.float32),
                                       np.full((8, 9), .7, np.float32)]) for arm in pack.ARMS},
            "seeds": [123, 42], "metadata": {"split": "holdout", "emotion": 0}}


def test_composition_uses_seed42_preserves_other43_and_continues_without_visual_teacher():
    case = example()
    before = copy.deepcopy(case)
    values, common, native, times, draw = pack.compose_case(case)
    assert draw == 1
    for arm in pack.ARMS:
        assert np.all(values[arm][np.ix_(native, pack.UPPER)] == np.float32(.7))
        np.testing.assert_array_equal(values[arm][:, pack.OTHER], case["baseline52"][:, pack.OTHER])
        np.testing.assert_array_equal(values[arm][~native], case["baseline52"][~native])
    np.testing.assert_array_equal(values["reference"][~common], case["baseline52"][~common])
    np.testing.assert_array_equal(case["baseline52"], before["baseline52"])


def test_composition_never_raw_clamps_and_rejects_mask_leak():
    case = example()
    for arm in pack.ARMS:
        case["samples"][arm][1] = 1.3
    values, common, *_ = pack.compose_case(case)
    assert np.all(values["va_audio"][np.ix_(common, pack.UPPER)] == np.float32(1.3))
    case["valid"][3] = True
    with pytest.raises(ValueError, match="nested masks"):
        pack.compose_case(case)


def test_metadata_selection_cannot_choose_by_prediction_or_train_examples():
    clips = {"z": example(), "a": example(), "train": example(), "b": example()}
    clips["train"]["metadata"]["split"] = "train"
    clips["b"]["metadata"]["emotion"] = 2
    assert pack.select_examples(clips) == ["a", "b"]


def test_missing_seed_or_non_native_clock_rejected():
    case = example()
    case["seeds"] = [123, 2026]
    with pytest.raises(ValueError, match="seed42"):
        pack.compose_case(case)
    case = example()
    case["times"] *= 2
    with pytest.raises(ValueError, match="25Hz"):
        pack.compose_case(case)
