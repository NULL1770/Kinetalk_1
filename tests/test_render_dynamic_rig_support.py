"""Display excludes unsupported coefficients without changing raw artifacts."""
import hashlib

import numpy as np
import pytest

from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input


def save_input(path, *, channel_mask=None, channels=None, motions=None):
    data = dict(channels=np.asarray(channels or ARKIT_NAMES),
                times=np.arange(4, dtype=np.float64) / 25,
                valid=np.asarray([True, False, True, True]),
                mode_names=np.asarray(['GT', 'Prediction']),
                motions=np.full((2, 4, 52), .25, dtype=np.float32) if motions is None else motions)
    if channel_mask is not None:
        data['channel_mask'] = channel_mask
    np.savez_compressed(path, **data)


def test_missing_tongue_is_disabled_and_nonfinite_values_are_audited(tmp_path):
    path = tmp_path / 'raw.npz'
    values = np.full((2, 4, 52), .25, dtype=np.float32)
    values[0, :, 51] = 0
    values[1, :, 51] = [1.6, np.nan, -.5, np.nan]
    mask = np.ones(52, dtype=bool)
    mask[51] = False
    save_input(path, channel_mask=mask, motions=values)
    before = hashlib.sha256(path.read_bytes()).digest()
    *_, display, report = inspect_input(path, 25)
    assert np.isfinite(display).all()
    assert np.count_nonzero(display[..., 51]) == 0
    np.testing.assert_array_equal(display[..., :51], .25)
    assert report['unsupported_channels'] == ['tongueOut']
    pred = report['mode_statistics']['Prediction']
    assert pred['observed_value_count'] == 3 * 51
    assert pred['unsupported_raw_value_count'] == 3
    assert pred['unsupported_raw_finite_nonzero_count'] == 2
    assert pred['unsupported_raw_nonfinite_count'] == 1
    assert pred['unsupported_raw_finite_abs_max'] == pytest.approx(1.6)
    assert pred['clamped_fraction_by_channel']['tongueOut'] is None
    assert hashlib.sha256(path.read_bytes()).digest() == before


def test_support_uses_named_input_order_and_frame_support(tmp_path):
    path = tmp_path / 'raw.npz'
    channels = [ARKIT_NAMES[51], *ARKIT_NAMES[:51]]
    mask = np.ones((4, 52), dtype=bool)
    mask[:, 0] = False
    mask[2, channels.index('jawOpen')] = False
    values = np.full((2, 4, 52), .25, dtype=np.float32)
    values[:, :, 0] = np.nan
    values[:, 2, channels.index('jawOpen')] = np.nan
    save_input(path, channel_mask=mask, channels=channels, motions=values)
    *_, display, report = inspect_input(path, 25)
    assert np.isfinite(display).all()
    assert np.count_nonzero(display[..., 0]) == 0
    assert np.count_nonzero(display[:, 2, channels.index('jawOpen')]) == 0
    assert report['partially_supported_channels'] == ['jawOpen']
    assert report['display_source_index'] == [0, 0, 2, 3]


def test_supported_nan_still_fails_and_nonboolean_support_is_rejected(tmp_path):
    path = tmp_path / 'raw.npz'
    values = np.full((2, 4, 52), .25, dtype=np.float32)
    values[1, 2, ARKIT_NAMES.index('jawOpen')] = np.nan
    save_input(path, channel_mask=np.ones(52, dtype=bool), motions=values)
    with pytest.raises(ValueError, match='finite'):
        inspect_input(path, 25)
    save_input(path, channel_mask=np.ones(52, dtype=np.int64))
    with pytest.raises(ValueError, match='channel_mask'):
        inspect_input(path, 25)


def test_legacy_inputs_keep_all_channels_and_original_display_policy(tmp_path):
    path = tmp_path / 'legacy.npz'
    values = np.full((2, 4, 52), .25, dtype=np.float32)
    values[:, :, 51] = 1.5
    save_input(path, motions=values)
    *_, display, report = inspect_input(path, 25)
    assert np.all(display[..., 51] == 1.)
    assert report['channel_mask_provided'] is False
    assert report['unsupported_channels'] == []
    assert report['mode_statistics']['Prediction']['observed_clamped_count'] == 3
