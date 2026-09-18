"""Teacher construction, native-clock interventions and frozen-head contracts."""
import copy

import numpy as np
import pytest
import torch

from scripts import activity_condition_core as core


def test_affine_motion_recovers_exact_group_rms_speed_and_stationary_zero():
    slopes = np.arange(1, 10, dtype=np.float64)/1000
    upper = .3+np.arange(32)[:, None]*slopes
    expected = np.array([25*np.sqrt(np.mean(slopes[list(group)]**2)) for group in core.GROUPS])
    np.testing.assert_allclose(core.activity_energy(upper), expected, atol=2e-15, rtol=1e-12)
    np.testing.assert_array_equal(core.activity_energy(np.full((32, 9), .37)), np.zeros(4))
    np.testing.assert_allclose(core.activity_energy(upper+100), expected, atol=1e-12)


def test_motion_energy_reads_only_fixed_smoothed_central_core():
    upper = np.zeros((32, 9))
    upper[11:20, 2] = np.arange(9)
    expected_delta = np.diff((upper[10:19]+upper[11:20]+upper[12:21])/3, axis=0)
    expected = 25*np.sqrt(np.square(expected_delta[:, [2, 3, 4]]).mean())
    assert core.activity_energy(upper)[0] == expected
    altered = upper.copy(); altered[:10] = 13; altered[21:] = -7
    np.testing.assert_array_equal(core.activity_energy(altered), core.activity_energy(upper))


def test_windows_keep_original_clock_gaps_and_short_run_coverage():
    valid = np.zeros(115, dtype=bool)
    valid[0:41] = True; valid[45:75] = True; valid[80:115] = True
    assert core.window_locations(valid) == [(0, 32), (8, 32), (9, 32), (80, 32), (83, 32)]
    for start, count in core.window_locations(valid):
        assert valid[start:start+count].all()
    coverage = core.window_coverage(valid)
    assert coverage['short_runs'] == [(45, 75)]
    assert coverage['short_run_valid_frames'] == 30 and coverage['valid_frames'] == 106
    assert coverage['context_covered_frames'] == 76 and coverage['target_covered_transitions'] == 28
    raw = np.zeros((len(valid), 9)); raw[~valid] = np.nan; raw[80:] = .8
    for start, count in core.window_locations(valid):
        np.testing.assert_array_equal(core.activity_energy(raw[start:start+count]), np.zeros(4))
    with pytest.raises(ValueError, match='fully observed'):
        core.activity_energy(raw[30:62])


def test_window_boundaries_no_duplicates_no_compression():
    for length, expected in [(0, []), (31, []), (32, [(0, 32)]),
                             (40, [(0, 32), (8, 32)]), (42, [(0, 32), (8, 32), (10, 32)])]:
        assert core.window_locations(torch.ones(length, dtype=torch.bool)) == expected
    assert core.window_coverage(np.zeros(7, dtype=bool))['context_fraction'] == 0


def test_clip_balanced_weighted_quantiles_differ_from_window_weighting_and_preserve_targets():
    # Nine low-motion windows belong to one clip; one high-motion window to a
    # second clip. Equal clip mass makes the65th percentile high, not low.
    energy = np.tile(np.r_[np.ones(9), 10.][:, None], (1, 4))
    balanced = np.r_[np.full(9, 1/9), 1.]
    threshold = core.fit_thresholds(energy, balanced)
    np.testing.assert_array_equal(threshold, np.full(4, 10.))
    np.testing.assert_array_equal(core.fit_thresholds(energy, np.ones(10)), np.ones(4))
    before = threshold.copy()
    query = np.array([[0., 10., 10.01, 100.], [20., 9., 0., 2.]])
    result = core.activity_targets(query, threshold)
    np.testing.assert_array_equal(result, query > 10.)
    np.testing.assert_array_equal(core.activity_targets(query[:1], threshold), result[:1])
    np.testing.assert_array_equal(threshold, before)
    floor = core.fit_thresholds(np.zeros((3, 4)), np.ones(3))
    np.testing.assert_array_equal(floor, np.full(4, 1e-6))
    assert not core.activity_targets(np.zeros((2, 4)), floor).any()


def acoustic_fixture():
    torch.manual_seed(29)
    local = torch.randn(83, 4, dtype=torch.float64)
    valid = torch.ones(83, dtype=torch.bool); valid[35:41] = False
    local[~valid] = float('nan')
    full = torch.zeros(83, 1540, dtype=local.dtype); full[:, core.PROSODY] = local
    return full, valid


def test_descriptor_uses_original_clip_mean_fixed_bins_and_four_channels_only():
    features, valid = acoustic_fixture()
    mean = core.prosody_mean(features, valid)
    std = torch.tensor([1., 2., 3., 4.], dtype=features.dtype)
    windows = torch.stack([features[start:start+32] for start, _ in core.window_locations(valid)])
    expected = ((windows[..., core.PROSODY]-mean)/std).reshape(-1, 8, 4, 4).mean(2).reshape(-1, 32)
    result = core.prosody_descriptor(windows, mean, std)
    torch.testing.assert_close(result, expected)
    assert result[0].mean().abs() > 1e-5  # No per-window recentering.
    windows[:, :, :1536] = float('nan')
    torch.testing.assert_close(core.prosody_descriptor(windows, mean, std), result, rtol=0, atol=0)
    torch.testing.assert_close(core.prosody_descriptor(windows[..., core.PROSODY], mean, std), result, rtol=0, atol=0)


def test_reverse_and_shift_do_not_cross_gaps_or_change_original_mean():
    features, valid = acoustic_fixture()
    before = features.clone()
    mean = core.prosody_mean(features, valid)
    for mode in ('real', 'static', 'reverse', 'shift'):
        changed = core.intervene_prosody(features, valid, mode)
        assert changed.shape == (len(valid), 4) and (changed[~valid] == 0).all()
        torch.testing.assert_close(core.prosody_mean(changed, valid), mean, rtol=1e-14, atol=1e-14)
        if mode == 'reverse':
            torch.testing.assert_close(changed[:35], features[:35, core.PROSODY].flip(0))
            torch.testing.assert_close(changed[41:], features[41:, core.PROSODY].flip(0))
        elif mode == 'shift':
            torch.testing.assert_close(changed[:35], features[:35, core.PROSODY].roll(17, 0))
            torch.testing.assert_close(changed[41:], features[41:, core.PROSODY].roll(21, 0))
        elif mode == 'static':
            torch.testing.assert_close(changed[valid], mean.expand(int(valid.sum()), -1), rtol=0, atol=0)
    torch.testing.assert_close(features, before, equal_nan=True, rtol=0, atol=0)


def test_mismatch_resamples_then_preserves_recipient_mean_and_unmodified_globals():
    features, valid = acoustic_fixture()
    global_ = torch.randn(65, dtype=features.dtype); before_global = global_.clone()
    donor = torch.arange(20, dtype=features.dtype)[:, None]*torch.tensor([1., 2., 3., 4.])+100
    donor_valid = torch.ones(20, dtype=torch.bool)
    before, before_donor = features.clone(), donor.clone()
    changed = core.intervene_prosody(features, valid, 'mismatch', donor_features=donor, donor_valid=donor_valid)
    torch.testing.assert_close(core.prosody_mean(changed, valid), core.prosody_mean(features, valid), rtol=1e-12, atol=1e-12)
    # Resampling keeps donor temporal direction within each recipient run.
    assert (torch.diff(changed[:35], dim=0) > 0).all() and (torch.diff(changed[41:], dim=0) > 0).all()
    torch.testing.assert_close(global_, before_global, rtol=0, atol=0)
    torch.testing.assert_close(features, before, equal_nan=True, rtol=0, atol=0)
    torch.testing.assert_close(donor, before_donor, rtol=0, atol=0)
    donor_valid[7] = False
    with pytest.raises(ValueError, match='continuous'):
        core.intervene_prosody(features, valid, 'mismatch', donor_features=donor, donor_valid=donor_valid)


def test_activity_prediction_static_null_and_no_motion_target_input():
    torch.manual_seed(731)
    model = core.FrozenActivityResidual(core.StaticActivityHead())
    context, descriptor = torch.randn(3, 69), torch.randn(3, 32)
    torch.testing.assert_close(model(context, descriptor), model.base(context), rtol=0, atol=0)
    with torch.no_grad():
        model.correction[-1].weight.normal_()
        model.correction[-1].bias.normal_()
    torch.testing.assert_close(model(context, torch.zeros_like(descriptor)), model.base(context), rtol=0, atol=0)
    predictions = model(context, descriptor).detach().clone()
    # Teacher construction may change labels, never acoustic model inputs.
    threshold = core.fit_thresholds(np.ones((5, 4)), np.ones(5))
    assert core.activity_targets(core.activity_energy(np.arange(32)[:, None]*np.ones((1, 9))), threshold).all()
    assert not core.activity_targets(core.activity_energy(np.ones((32, 9))), threshold).any()
    torch.testing.assert_close(model(context, descriptor), predictions, rtol=0, atol=0)
    assert model.residual_logits(descriptor).abs().max() <= 2


def test_optimizer_updates_only_correction_and_reasserts_frozen_base():
    torch.manual_seed(77)
    model = core.FrozenActivityResidual(core.StaticActivityHead())
    before = copy.deepcopy(model.base.state_dict())
    context, descriptor = torch.randn(12, 69), torch.randn(12, 32)
    targets = (torch.arange(48).reshape(12, 4) % 3 == 0).float()
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=.01)
    model.train()
    model.base.train().requires_grad_(True)  # forward must restore the contract.
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(context, descriptor), targets)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    assert not model.base.training
    assert all(param.grad is None and not param.requires_grad for param in model.base.parameters())
    for key, value in model.base.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert model.correction[-1].weight.abs().sum() > 0
    torch.testing.assert_close(model(context, torch.zeros_like(descriptor)), model.base(context), rtol=0, atol=0)


def test_invalid_observed_inputs_fail_and_invalid_prosody_frames_do_not_poison_mean():
    features, valid = acoustic_fixture()
    assert torch.isfinite(core.prosody_mean(features, valid)).all()
    features[0, 1536] = float('nan')
    with pytest.raises(ValueError, match='Finite observed'):
        core.prosody_mean(features, valid)
    with pytest.raises(ValueError, match='positive fitting weights'):
        core.fit_thresholds(np.zeros((2, 4)), np.array([1., 0.]))
    with pytest.raises(ValueError, match='Boolean'):
        core.window_locations(np.ones(32))
