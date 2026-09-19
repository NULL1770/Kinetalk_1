import numpy as np

from scripts.evaluate_paper_motion_probes import Probe, motion_features


def test_features_respect_frame_and_channel_masks_and_use_native_dt():
    x = np.zeros((6, 52), dtype=float)
    x[:, 0] = np.arange(6, dtype=float)
    x[:, 1] = np.arange(6, dtype=float) * 2
    mask = np.ones_like(x, dtype=bool)
    mask[2, 0] = False
    mask[:, 10] = False
    valid = np.ones(6, dtype=bool)
    valid[4] = False
    f = motion_features(x, mask, valid).reshape(52, 6)
    # Channel 0 has gaps, so only adjacent observed pairs contribute to speed.
    assert f[0, 4] == 25.0
    assert f[1, 4] == 50.0
    # An unobserved channel is kept as NaN, never filled from a target value.
    assert np.isnan(f[10]).all()


def test_ridge_probe_is_fixed_alpha_and_reports_unknown_labels():
    base = np.zeros(312)
    x = np.stack([base, base + 1, base + 2, base + 3])
    probe = Probe().fit(x, ["a", "b", "a", "b"])
    report = probe.report(x, ["a", "b", "unseen", "a"])
    assert report["known_n"] == 3
    assert report["unknown_label_n"] == 1
    assert report["uniform_chance"] == 0.5


def test_masks_cannot_silently_invent_support():
    import pytest
    from scripts.evaluate_paper_motion_probes import frame_mask, channel_mask
    with pytest.raises(ValueError):frame_mask({},3)
    with pytest.raises(ValueError):channel_mask({},3)


def test_outer_uses_canonical_labels_and_distinct_arms(tmp_path):
    import torch
    from scripts.evaluate_paper_motion_probes import outer_rows,STAGES,UPPER
    r={"clip_id":"mead_M003_happy_L1_001","speaker":"0","emotion":"5","sentence":"s","valid":np.ones(3,bool),"channel_mask":np.ones((3,52),bool),"target52":np.zeros((3,52)),"baseline52":np.ones((3,52))}
    r["channel_mask"][:,0]=False
    for i,(arm,stage) in enumerate(STAGES.items()):
        path=tmp_path/stage/"holdout";path.mkdir(parents=True)
        torch.save({"clips":{r["clip_id"]:{"native_valid":r["valid"],"score_mask":np.ones(3,bool),"target":np.zeros((3,9)),"samples":np.full((4,3,9),i*.1),"seeds":[42,123,2026,77]}}},path/"curves.pt")
    groups,sources=outer_rows(tmp_path,{r["clip_id"]:r},[r["clip_id"]])
    assert len(groups)==26 and len(groups["reference"])==len(groups["base"])==1
    assert groups["audio/seed42"][0]["speaker"]=="0"
    assert not groups["audio/seed42"][0]["channel_mask"][:,0].any()
    np.testing.assert_allclose(groups["audio/seed42"][0]["target52"][:,UPPER],.1)
    np.testing.assert_allclose(groups["prior/seed42"][0]["target52"][:,UPPER],0)


def test_allmissing_and_constant_dimensions_do_not_carry_signal():
    X=np.array([[0.,np.nan,8.],[1.,np.nan,8.],[2.,np.nan,8.],[3.,np.nan,8.]])
    p=Probe().fit(X,["a","a","b","b"])
    assert np.isfinite(p.w).all()
    np.testing.assert_allclose(p.w[2:],0,atol=1e-12)
