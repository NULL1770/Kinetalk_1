import copy

import numpy as np
import pytest

from scripts.clocked_motion_dictionary import (
    SCHEMA, extract_windows, fit_shape_dictionary, overlap_add,
    pairwise_distances, target_distances,
)


def clip(name, frames=96, amplitude=.1, offset=.4):
    clock = np.arange(frames)[:, None]
    return {'clip_id': name,
        'upper': offset+amplitude*np.sin(clock*.17+np.arange(9)[None]*.3),
        'valid': np.ones(frames, bool), 'global': np.linspace(.1, .5, 4)}


@pytest.fixture(scope='module')
def fitted():
    clips = [clip('a'), clip('b', frames=64, amplitude=.2), clip('untouched')]
    windows = extract_windows(clips, [0, 1])
    return clips, windows, fit_shape_dictionary(windows, k=4)


def test_full_native_fixed_clock_includes_initial_and_excludes_gap_tail():
    source = clip('a', frames=105)
    source['valid'][45:51] = False
    source['upper'][45:51] = np.nan
    windows = extract_windows([source], [0])
    assert [w['start'] for w in windows] == [0, 51, 67]
    for window in windows:
        start = window['start']
        assert source['valid'][start:start+32].all()
        assert window['horizon'] == 32 and window['hop'] == 16
        np.testing.assert_allclose(window['raw_mean'], source['upper'][source['valid']].mean(0), atol=1e-16)
        assert window['centering_scope'] == 'training_clip_observed_frames'
        np.testing.assert_allclose(window['future']+window['raw_mean'], source['upper'][start:start+32])
    assert extract_windows([clip('short', frames=31)], [0]) == []
    assert len(extract_windows([clip('exact', frames=32)], [0])) == 1


def test_allowlist_does_not_access_unselected_targets_and_no_mutation(fitted):
    clips, windows, _ = fitted
    original = copy.deepcopy(clips)
    # The omitted source need not even provide a motion/global field.
    changed = copy.deepcopy(clips)
    changed[2] = {'clip_id': 'heldout', 'upper': object()}
    other = extract_windows(changed, [0, 1])
    for before, after in zip(windows, other):
        np.testing.assert_array_equal(before['future'], after['future'])
    for before, after in zip(original, clips):
        np.testing.assert_array_equal(before['upper'], after['upper'])
    windows[0]['global'][0] = 50
    assert clips[0]['global'][0] == .1
    windows[0]['global'][0] = .1
    with pytest.raises(ValueError, match='duplicate'):
        extract_windows(clips, [0, 0])
    with pytest.raises(ValueError, match='outside'):
        extract_windows(clips, [3])
    with pytest.raises(ValueError):
        extract_windows(clips, [-1])


def test_extraction_is_padding_and_dc_invariant():
    source = clip('a')
    base = extract_windows([source], [0])
    moved = copy.deepcopy(source)
    moved['upper'] += np.arange(9)[None]*.1+2.
    moved['upper'] = np.r_[moved['upper'], np.full((30, 9), np.nan)]
    moved['valid'] = np.r_[moved['valid'], np.zeros(30, bool)]
    result = extract_windows([moved], [0])
    assert [w['start'] for w in base] == [w['start'] for w in result]
    for a, b in zip(base, result):
        np.testing.assert_allclose(a['future'], b['future'], atol=1e-15)
    constant = clip('constant')
    constant['upper'][:] = np.arange(9)*.1
    for window in extract_windows([constant], [0]):
        np.testing.assert_array_equal(window['future'], 0.)


def test_dictionary_keeps_real_clip_centered_medoids_including_window_mean(fitted):
    _, windows, dictionary = fitted
    assert dictionary['schema'] == SCHEMA
    assert dictionary['source_clip_ids'] == ['a', 'b']
    assert dictionary['horizon'] == 32 and dictionary['hop'] == 16
    assert dictionary['shapes'].shape == (4, 32, 9)
    assert len(set(dictionary['medoid_window_indices'])) == 4
    assert np.any(np.abs(dictionary['shapes'].mean(1)) > .01)
    assert dictionary['centering_scope'] == 'training_clip_observed_frames'
    for value, index, source in zip(dictionary['shapes'], dictionary['medoid_window_indices'], dictionary['medoid_sources']):
        np.testing.assert_array_equal(value, windows[index]['future'])
        assert source == {key: windows[index][key] for key in ('clip_id', 'start')}
    repeated = fit_shape_dictionary(windows, k=4)
    np.testing.assert_array_equal(repeated['shapes'], dictionary['shapes'])
    with pytest.raises(ValueError, match='no larger'):
        fit_shape_dictionary(windows[:1], k=2)


def test_scales_are_clip_balanced_and_static_floor():
    windows = extract_windows([clip('a', frames=96), clip('b', frames=48, amplitude=.4)], [0, 1])
    original = fit_shape_dictionary(windows, k=2)
    a = [w for w in windows if w['clip_id'] == 'a']
    b = [w for w in windows if w['clip_id'] == 'b']
    repeated = fit_shape_dictionary(a*4+b, k=2)
    np.testing.assert_allclose(original['scales'], repeated['scales'], atol=1e-15)
    expected = np.sqrt((np.mean(np.square([w['future'] for w in a]), (0, 1))+
        np.mean(np.square([w['future'] for w in b]), (0, 1)))/2)
    np.testing.assert_allclose(original['scales'], np.maximum(expected, .005))
    for source in ('a', 'b'):
        mass = sum(weight for weight, window in zip(original['window_weights'], windows) if window['clip_id'] == source)
        assert mass == pytest.approx(.5)
    constant = clip('static')
    constant['upper'][:] = .4
    static_dictionary = fit_shape_dictionary(extract_windows([constant], [0]), k=1)
    np.testing.assert_array_equal(static_dictionary['scales'], np.full(9, .005))


def test_overlap_constant_preservation_positive_endpoints_and_tail():
    chunks = np.full((3, 32, 9), 2.5)
    output, coverage = overlap_add(chunks, [0, 16, 32], 49)
    assert coverage.all()
    np.testing.assert_allclose(output, 2.5)
    # A final chunk with only one visible frame still owns positive weight.
    final, support = overlap_add(chunks[:1], [48], 49)
    assert support.sum() == 1 and support[-1]
    np.testing.assert_array_equal(final[-1], chunks[0, 0])
    np.testing.assert_array_equal(final[:-1], 0.)


def test_overlap_preserves_consistent_global_curve_and_blends_without_hard_seam():
    timeline = np.arange(64)[:, None]*np.linspace(.1, .9, 9)[None]
    chunks = np.stack([timeline[start:start+32] for start in (0, 16, 32)])
    output, coverage = overlap_add(chunks, [0, 16, 32], 64)
    assert coverage.all()
    np.testing.assert_allclose(output, timeline, atol=1e-14)
    # Deliberately discontinuous constant chunks become a monotone crossfade,
    # with neither duplicated endpoint nor post-hoc recentering.
    output, support = overlap_add(np.stack([np.zeros((32, 9)), np.ones((32, 9))]), [0, 16], 48)
    assert support.all()
    np.testing.assert_array_equal(output[:16], 0.)
    np.testing.assert_array_equal(output[32:], 1.)
    np.testing.assert_allclose(output[16:32, 0], np.arange(1, 17)/17)
    assert np.max(np.diff(output[:, 0])) == pytest.approx(1/17)


def test_clip_centering_preserves_slow_motion_in_exact_shape_oracle():
    # This is representation-only teacher reconstruction. Query clip means are
    # never supplied to deployment; the controller predicts a separate level.
    source = clip('slow', frames=320)
    source['upper'] = .5+.2*np.sin(2*np.pi*np.arange(320)[:, None]/250+np.arange(9)[None]*.1)
    windows = extract_windows([source], [0])
    chunks = np.stack([w['future'] for w in windows])
    mean = windows[0]['raw_mean']
    output, coverage = overlap_add(chunks, [w['start'] for w in windows], 320)
    assert coverage.all()
    np.testing.assert_allclose(output+mean, source['upper'], atol=1e-15)
    np.testing.assert_allclose(np.std(output, axis=0), np.std(source['upper'], axis=0), atol=1e-15)
    # Previously proposed per-window centering would discard most slow motion.
    local_centered = chunks-chunks.mean(1, keepdims=True)
    removed, _ = overlap_add(local_centered, [w['start'] for w in windows], 320)
    assert np.std(removed[:, 0])/np.std(source['upper'][:, 0]) < .15


def test_clip_mean_shared_across_gaps_and_ignores_unobserved_values():
    source = clip('two_runs', frames=80)
    source['valid'][32:48] = False
    source['upper'][:32] = .2
    source['upper'][32:48] = np.nan
    source['upper'][48:] = .8
    windows = extract_windows([source], [0])
    assert [w['start'] for w in windows] == [0, 48]
    for window in windows:
        np.testing.assert_allclose(window['raw_mean'], .5, atol=1e-15)
    np.testing.assert_allclose(windows[0]['future'], -.3, atol=1e-15)
    np.testing.assert_allclose(windows[1]['future'], .3, atol=1e-15)


def test_overlap_empty_gap_and_no_clipping_or_target_normalization():
    chunks = np.full((2, 32, 9), -3.)
    output, support = overlap_add(chunks, [0, 48], 80)
    assert not support[32:48].any()
    np.testing.assert_array_equal(output[support], -3.)
    np.testing.assert_array_equal(output[~support], 0.)
    empty, support = overlap_add(np.empty((0, 32, 9)), [], 0)
    assert empty.shape == (0, 9) and support.shape == (0,)


def test_distance_helpers_equal_explicit_rms_and_do_not_center_targets(fitted):
    _, _, dictionary = fitted
    shapes, scales = dictionary['shapes'], dictionary['scales']
    pair = pairwise_distances(shapes, scales)
    expected = np.sqrt(np.square((shapes[:, None]-shapes[None])/scales).mean((2, 3)))
    np.testing.assert_allclose(pair, expected, atol=1e-14)
    np.testing.assert_array_equal(np.diag(pair), 0.)
    np.testing.assert_array_equal(pair, pair.T)
    shifted = shapes[:2]+.1
    result = target_distances(shifted, shapes, scales)
    assert result.shape == (2, 4)
    explicit = np.sqrt(np.square((shifted[:, None]-shapes[None])/scales).mean((2, 3)))
    np.testing.assert_allclose(result, explicit, atol=1e-14)
    assert result[0, 0] > 0
    assert target_distances(np.empty((0, 32, 9)), shapes, scales).shape == (0, 4)


@pytest.mark.parametrize('mutation', ['nan_observed', 'float_mask', 'bad_global'])
def test_invalid_extraction_inputs_fail(mutation):
    source = clip('bad')
    if mutation == 'nan_observed':
        source['upper'][0, 0] = np.nan
    elif mutation == 'float_mask':
        source['valid'] = source['valid'].astype(float)
    else:
        source['global'][0] = np.inf
    with pytest.raises(ValueError):
        extract_windows([source], [0])


def test_validation_of_clock_dictionary_and_overlap(fitted):
    clips, windows, dictionary = fitted
    for kwargs in ({'horizon': 0}, {'hop': True}, {'hop': 1.5}):
        with pytest.raises(ValueError):
            extract_windows(clips, [0], **kwargs)
    invalid = copy.deepcopy(windows)
    invalid[0].pop('centering_scope')
    with pytest.raises(ValueError, match='centering provenance'):
        fit_shape_dictionary(invalid, k=2)
    invalid = copy.deepcopy(windows)
    invalid[0]['hop'] = 7
    with pytest.raises(ValueError, match='fixed'):
        fit_shape_dictionary(invalid, k=2)
    for starts in ([0, 0], [16, 0], [-1, 16], [0, 64], [0]):
        with pytest.raises(ValueError):
            overlap_add(np.zeros((2, 32, 9)), starts, 64)
    with pytest.raises(ValueError):
        overlap_add(np.full((1, 32, 9), np.nan), [0], 32)
    with pytest.raises(ValueError):
        target_distances(dictionary['shapes'], dictionary['shapes'], np.zeros(9))
