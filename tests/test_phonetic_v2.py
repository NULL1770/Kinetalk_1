import numpy as np
import pytest
import torch

from kinetalk_v2.phonetic import (
    CORE, RELEASE, DIAGNOSTIC_SOURCE, LipConstraintAsset, PhoneticEvidence,
    PhoneticCalibration, align_bilabial_phases, build_constraints,
    calibrate_intervals, diagnostic_neutral_protection,
)


def asset():
    j = np.zeros((2, 52), np.float32); j[0, 1] = 1.; j[1, 2] = 1.
    return LipConstraintAsset(j, np.array([1., 2.], np.float32), tuple(map(str, range(52))),
                              np.ones(2, np.float32), "a" * 64)


def test_external_alignment_requires_provenance_and_uses_half_open_clock():
    t = np.arange(5, dtype=float) * .04
    e = align_bilabial_phases(t, [{"phone":"p", "phase":"core", "start":.04, "end":.12, "confidence":.9}],
                              source="external_closure_release", provenance="forced-align-v1")
    assert e.phase.tolist() == [0, CORE, CORE, 0, 0]
    with pytest.raises(ValueError):
        align_bilabial_phases(t, [], source="external_closure_release", provenance="")


def test_diagnostic_is_explicit_and_not_accepted_as_production():
    b = np.zeros((1, 8, 52), np.float32); b[0, :, 17] = .01
    e = diagnostic_neutral_protection(b, np.ones((1, 8), bool), np.arange(8)[None] * .04)
    assert e.source == DIAGNOSTIC_SOURCE and not (e.confidence > .5).any()
    cal = type("C", (), {"asset_digest":"a"*64, "diagnostic_only":True,
        "evidence_source":DIAGNOSTIC_SOURCE,
        "lower_gap_mm":np.ones((3,2)), "upper_gap_mm":np.ones((3,2))*2,
        "reliability":np.ones((3,2)), "base_mismatch_mm":np.zeros((3,2))})()
    with pytest.raises(ValueError):
        build_constraints(asset(), cal, e, torch.ones(1,8,dtype=torch.bool))


def test_missing_evidence_has_zero_constraint_weight():
    c = build_constraints(asset(), None, None, torch.ones(2, 4, dtype=torch.bool))
    assert torch.count_nonzero(c["weight"]) == 0
    assert c["jacobian"].shape == (2, 52)


def test_asset_observation_includes_rest_offset():
    assert np.allclose(asset().observe(np.zeros((3,52))), [[1.,2.]]*3)


def training_arrays():
    motion = np.zeros((2,10,52),np.float32)
    motion[:,:,1] = np.linspace(0,.1,10)
    motion[:,:,2] = np.linspace(.2,.3,10)
    valid = np.ones((2,10),bool)
    phase = np.full((2,10),CORE,np.int64); phase[:,5:] = RELEASE
    e = PhoneticEvidence(phase,np.ones((2,10)),"external_closure_release","mock external boundary")
    return motion,valid,e


def calibration(base_error=0.,channels=None):
    motion,valid,e = training_arrays()
    base = motion.copy();base[:,:,1:3] += base_error
    return calibrate_intervals(asset(),motion,base,valid,e,split="train",training_ids=["a","b"],
        fit_confidence=np.ones((2,10)),channel_mask=np.ones((2,52),bool) if channels is None else channels,
        min_samples=5)


def test_train_only_tolerances_and_mismatch_weighting():
    good,bad = calibration(),calibration(base_error=3.)
    assert good.reliability[CORE].min() == 1.
    assert bad.reliability[CORE].max() < good.reliability[CORE].min()
    motion,valid,e = training_arrays()
    with pytest.raises(ValueError,match="TRAIN-only"):
        calibrate_intervals(asset(),motion,motion,valid,e,split="validation",training_ids=["a","b"],
            fit_confidence=np.ones((2,10)),channel_mask=np.ones((2,52),bool))


def test_unobserved_rig_channels_do_not_impute_zero_targets():
    channels = np.ones((2,52),bool);channels[:,1] = False
    cal = calibration(channels=channels)
    assert cal.sample_count[CORE,0] == 0 and cal.reliability[CORE,0] == 0
    assert cal.sample_count[CORE,1] == 10 and cal.reliability[CORE,1] > 0


def test_calibration_roundtrip_and_absolute_bounds(tmp_path):
    a,cal = asset(),calibration()
    path = tmp_path / "calibration.npz";cal.save(path)
    loaded = PhoneticCalibration.load(path)
    motion,valid,e = training_arrays()
    constraints = build_constraints(a,loaded,e,torch.tensor(valid))
    expected = loaded.lower_gap_mm[CORE]-a.rest_gap_mm
    np.testing.assert_allclose(constraints["lower"][0,0].numpy(),expected)
    assert constraints["weight"][0,0,0] > constraints["weight"][0,5,0]
    assert not constraints["weight"].requires_grad


def test_runtime_validity_confidence_and_base_mismatch_gate():
    a,cal = asset(),calibration()
    motion,valid,e = training_arrays()
    confidence = np.ones(valid.shape);confidence[0,1] = .1
    valid[0,0] = False
    e = PhoneticEvidence(e.phase,confidence,e.source,e.provenance)
    c = build_constraints(a,cal,e,torch.tensor(valid),base=torch.tensor(motion))
    wrong = torch.tensor(motion); wrong[:,:,1:3] += 10
    d = build_constraints(a,cal,e,torch.tensor(valid),base=wrong)
    assert c["weight"][0,:2].sum() == 0
    assert d["weight"].sum() < c["weight"].sum()


def test_composer_preserves_nullspace_and_retains_gradients():
    from kinetalk_v2.unified_model import FeasibleResidualComposer
    proposal = torch.zeros(1,2,52,requires_grad=True)
    with torch.no_grad():
        proposal[...,1] = 3.;proposal[...,2] = 2.;proposal[...,40] = .7
    c = {"jacobian":torch.tensor(asset().jacobian_mm),"lower":torch.zeros(1,2,2),
         "upper":torch.ones(1,2,2)*.2,"weight":torch.ones(1,2,2)}
    result = FeasibleResidualComposer()(proposal,c)
    assert (result[...,1:3] < proposal[...,1:3]).all()
    assert torch.equal(result[...,40],proposal[...,40])
    result.square().sum().backward()
    assert proposal.grad is not None and torch.isfinite(proposal.grad).all()
    assert proposal.grad[...,40].abs().sum() > 0


def test_diagnostic_does_not_bridge_clock_gaps():
    base = np.zeros((1,6,52),np.float32);valid = np.ones((1,6),bool)
    times = np.array([[0,.04,.08,.40,.44,.48]])
    e = diagnostic_neutral_protection(base,valid,times,min_run=4)
    assert np.count_nonzero(e.phase) == 0


def test_whole_phone_or_missing_confidence_is_not_closure_evidence():
    t = np.arange(5)*.04
    segments = [{"phone":"b","start":0,"end":1,"confidence":1.},
                {"phone":"p","phase":"core","start":0,"end":1}]
    e = align_bilabial_phases(t,segments,source="external_closure_release",provenance="test")
    assert np.count_nonzero(e.phase) == 0
