import numpy as np
import pytest
from scipy.special import expit

from scripts import diagnose_sparse_brow_controls as d


def test_fixed_model_and_schedule_preserve47_and_saved_global_level():
    model = d.synthetic_model()
    assert model["components"][0]["loc"][:5] == [0., 0., .1, .07, .07]
    assert model["components"][1]["loc"][:5] == [.1, .1, 0., 0., 0.]
    base = np.linspace(-.1, .5, 52, dtype=np.float32)
    level = np.linspace(-7., -1., 9)
    out, state, chunks = d._make_curve(base, level, 42, "fixed", 250, model)
    protected = np.delete(np.arange(52), d.BROW5)
    np.testing.assert_array_equal(out[:, protected], np.broadcast_to(base[protected], (250, 47)))
    np.testing.assert_array_equal(out[0, d.BROW5], expit(level[:5]).astype(np.float32))
    assert [row["mode"] for row in state.controls] == ["HOLD", "RELEASE", "RUN"]
    assert [row["frame"] for row in state.controls] == [100, 125, 150]
    assert sum(len(chunk["values"]) for chunk in chunks) == 250
    assert len(state.events) > 0
    for row in state.events:
        assert row["durations"] == [12, 8, 18]
    with pytest.raises(ValueError, match="250"):
        d._make_curve(base, level, 42, "fixed", 249, model)


def test_long_synthetic_controls_are_auditable_and_finite():
    state, out, checks = d._long_control_checks(d.synthetic_model(), np.full(9, -7.), 0, "fixed", 1500)
    assert checks["passed"]
    assert 0 < np.mean(out["phase"] == "HOLD") < 1
    assert checks["amplitude_excursions"] == pytest.approx([0., .05, .1, .15])
    assert checks["activity_counts"][0] < checks["activity_counts"][-1]
    assert all(row["return_level_offset"] == [0.] * 5 for row in state.events)
