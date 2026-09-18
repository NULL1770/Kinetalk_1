"""Saved-review contracts: no seed/query selection, native gaps, no fake video."""
import copy
import json

import numpy as np
import pytest
import torch

from scripts import package_controlled_prior_review as review


def fixture():
    seeds = [42, 123, 2026, 77, 91, 301, 509, 997]
    protocol = {'schema': review.SCHEMA, 'smoke': False, 'test_loaded': False,
        'extra_expression_reference': True, 'audio_timing_claim': False,
        'seeds': seeds, 'activity_gains': [0., .5, 1., 1.5], 'fit_clip_ids': ['ref0', 'ref1'],
        'query_clip_ids': [f'query{i:02d}' for i in range(32)]}
    status = {'schema': review.SCHEMA, 'status': 'complete', 'smoke': False,
        'test_loaded': False, 'default_replaced': False, 'audio_timing_proven': False, 'queries': 32}
    picks, references, curves = [], {}, {}
    valid = np.ones(50, bool); valid[10:15] = False
    for cid in protocol['query_clip_ids']:
        meta = {'clip_id': cid, 'speaker': 0, 'sentence': cid, 'emotion': 0}
        picks.append(meta)
        curves[cid] = {'metadata': meta, 'valid': valid, 'target': np.full((50, 9), .4),
            'samples': {arm: np.full((8, 50, 9), .3) for arm in review.ARMS},
            'gain_controls_seed42': {gain: np.full((50, 9), .3) for gain in review.GAIN_KEYS}}
        references[cid] = {'reference_ids': ['ref0', 'ref1'],
            'reference_metadata': [{'clip_id': f'ref{i}', 'speaker': 0, 'sentence': f'r{i}', 'emotion': 0} for i in range(2)],
            'references': [{'source_clip_id': f'ref{i}', 'style': [.2, .2, .2, .2, .5]} for i in range(2)]}
    predictions = {'schema': review.SCHEMA, 'curves': curves, 'long': {'query00': {
        **{'gain_'+key: np.full((1500, 9), .3) for key in review.GAIN_KEYS}, 'controls': np.full((512, 9), .3)}}}
    reports = {arm: {'summary': {'clips': 32}, 'rows': [dict(clip_id=cid, sample_count=8)
                for cid in protocol['query_clip_ids']]} for arm in review.ARMS}
    long_rows = {'0': [dict(seed=seed, gain=gain, kind='stationary_60s')
        for seed in seeds for gain in protocol['activity_gains']]+[
            dict(seed=seed, kind='run_hold_release_resume_swap') for seed in seeds]}
    return protocol, status, {'queries': picks}, references, reports, predictions, long_rows


def test_all_queries_seeds_and_independent_references_are_bound():
    args = fixture()
    selected = review.validate_predictions(*args)
    assert [row['clip_id'] for row in selected] == args[0]['query_clip_ids']
    for index, mutation, message in [
        (0, lambda p: p['seeds'].reverse(), 'All32'),
        (2, lambda p: p['queries'].reverse(), 'All32'),
        (3, lambda p: p['query00']['reference_metadata'][0].update(sentence='query00'), 'Independent'),
        (4, lambda p: p['process']['rows'][0].update(sample_count=1), 'all fixed'),
        (6, lambda p: p['0'].pop(), 'All8'),
    ]:
        changed = copy.deepcopy(args); mutation(changed[index])
        with pytest.raises(ValueError, match=message):
            review.validate_predictions(*changed)


def test_group_curves_average_channels_only_and_mask_gaps():
    raw = np.tile(np.arange(9), (20, 1)).astype(float)
    raw[10:] += 10
    valid = np.ones(20, bool); valid[8:12] = False
    result = review.group_curves(raw, valid)
    np.testing.assert_array_equal(result[0], [3, .5, 6, 7])
    assert np.isnan(result[8:12]).all()
    np.testing.assert_array_equal(result[12], [13, 10.5, 16, 17])
    with pytest.raises(ValueError, match='one raw'):
        review.group_curves(np.zeros((8, 20, 9)))


def test_video_is_only_linked_when_file_exists(tmp_path):
    output = tmp_path/'review'; output.mkdir()
    fullface = tmp_path/'fullface'
    missing, record = review.video_link(output, 'clip', fullface)
    assert record is None and '尚未生成' in missing and '<a ' not in missing
    target = fullface/'clip'/'comparison.mp4'; target.parent.mkdir(parents=True)
    target.write_bytes(b'not-an-actual-movie')
    # This function asserts only file presence/hashes, not decode or naturalness.
    link, record = review.video_link(output, 'clip', fullface)
    assert '../fullface/clip/comparison.mp4' in link
    assert record['sha256'] == review.sha(target)
    assert '<video controls preload="none"' in link and 'autoplay' not in link
    assert record['poster'] is None and ' poster=' not in link
    poster = target.with_name('preview.png'); poster.write_bytes(b'not-an-actual-png')
    link, record = review.video_link(output, 'clip', fullface)
    assert 'poster="../fullface/clip/preview.png"' in link
    assert record['poster']['sha256'] == review.sha(poster)
    with pytest.raises(ValueError, match='safe basename'):
        review.video_link(output, '../escape', fullface)


def test_plot_writes_all_arm_and_gain_labels_at_fixed_seed(tmp_path):
    row = fixture()[5]['curves']['query00']
    row['samples']['process'][0, :, 2] = np.linspace(.1, .8, 50)
    output = tmp_path/'query.svg'
    review.plot_query(output, 'query00', row)
    svg = output.read_text(encoding='utf8')
    for label in ('Tracked GT', 'Bounded medoid, seed42', 'Continuous process, seed42',
                  'Reference swap, seed42', 'Gain 0.0, seed42', 'Gain 1.5, seed42'):
        assert label in svg
    assert 'no sample averaging' in svg


def test_package_preserves_full_membership_and_limits_without_videos(tmp_path, monkeypatch):
    protocol, status, selection, references, reports, predictions, long_rows = fixture()
    status.update(fit_clips=2, control_exact=True, finite_bounded=True)
    for reference in references.values():
        reference.update(predicted_emotion=0, reference_emotion_matched=True)
    for report in reports.values():
        report['summary'].update(joint_fair_es={'raw': .5, 'centered': .4}, rms_ratio=[1.]*4)
        report['acceleration'] = {'rms_ratio': 1.}
    for row in long_rows['0']:
        if row['kind'] == 'stationary_60s':
            row.update(group_rms=[.1]*4, velocity_p95=.2, acceleration_p95=.3,
                above_3hz_energy_fraction=.01, activity_fraction_speed_norm_gt_005=.8)
        else:
            row.update(hold_speed_max=0., release_equilibrium_max_error=0.,
                boundary_velocity_p95=.2, boundary_acceleration_p95=.3)
    fit = {'quantiles': {key: {str(q): q for q in (.1, .5, .9, .95)}
        for key in ('velocity_p95', 'acceleration_p95')}}
    root = tmp_path/'run'; root.mkdir()
    for name, value in [('protocol', protocol), ('status', status), ('query_selection', selection),
                        ('references', references), ('reports', reports), ('long_controls', long_rows),
                        ('fit_dynamics_reference', fit)]:
        (root/(name+'.json')).write_text(json.dumps(value), encoding='utf8')
    torch.save(predictions, root/'predictions.pt')
    def placeholder(path, *args):
        path.write_text('<svg></svg>', encoding='utf8')
    monkeypatch.setattr(review, 'plot_query', placeholder)
    monkeypatch.setattr(review, 'plot_long', placeholder)
    output = tmp_path/'review'
    manifest = review.package(root, output)
    assert [row['clip_id'] for row in manifest['query_examples']] == protocol['query_clip_ids']
    assert all(row['seed'] == 42 and row['video'] is None for row in manifest['query_examples'])
    assert len(manifest['output_sha256']) == 34  #32query +1long +index
    assert manifest['full_face_verified'] is False and manifest['audio_timing_proven'] is False
    page = (output/'index.html').read_text(encoding='utf8')
    assert page.count('此片全脸对比视频尚未生成') == 32
    assert '不宣称音频动态预测成功' in page and '不挑 seed' in page and '43 个非眉眼通道' in page
    assert '0 / 32 段已渲染' in page and '另外 32 段尚未渲染' in page
    with pytest.raises(FileExistsError, match='Fresh'):
        review.package(root, output)
