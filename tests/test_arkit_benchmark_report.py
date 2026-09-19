import json

import numpy as np
import pytest
import torch

from scripts.arkit_benchmark_report import score_fullface, build_report
from tests.test_evaluate_continuous_motion_latent import fixture, StubAE, StubFlow
from scripts.evaluate_continuous_motion_latent import evaluate_generation


def test_raw_mask_required_and_missing_tongue_never_becomes_ground_truth():
    clip, _ = fixture()
    target = clip['target52'].numpy()
    samples = target[None].copy()
    assert score_fullface(samples, clip)['metrics']['arkit_mbe']['status'] == 'pending'
    clip['channel_mask'] = torch.ones_like(clip['target52'], dtype=torch.bool)
    clip['channel_mask'][:, 51] = False
    samples[:, :, 51] = 100
    row = score_fullface(samples, clip)
    assert row['metrics']['arkit_mbe']['value'] == 0
    assert 'tongueOut' not in row['observed_channel_names']
    clip['target52'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        score_fullface(samples, clip)


def test_generation_emits_benchmark_and_masks_display_without_changing_draws(tmp_path):
    clip, stats = fixture()
    clip['channel_mask'] = torch.ones_like(clip['target52'], dtype=torch.bool)
    clip['channel_mask'][:, 51] = False
    before = evaluate_generation(StubFlow(), StubAE(), [clip], stats, tmp_path/'run', seeds=(42, 77), steps=2)
    report = json.loads((tmp_path/'run/arkit_benchmark.json').read_text())
    assert report['summary']['arkit_mbe']['status'] == 'computed'
    assert report['summary']['fd']['value'] is None
    assert report['summary']['fd']['status'] == 'pending'
    assert 'browInnerUp' not in report['region_names']['main_fdd']
    assert 'browInnerUp' in report['region_names']['supp_upper9']
    with np.load(tmp_path/'run/npz/clip_a.npz') as z:
        assert z['reference_display_baseline_filled'][:, 51].all()
    assert before['numerical_gate']['passed']


def test_clip_equal_aggregation_and_pending_external():
    clip, _ = fixture()
    clip['channel_mask'] = torch.ones_like(clip['target52'], dtype=torch.bool)
    y = clip['target52'].numpy()
    a = score_fullface(y[None], clip)
    b = score_fullface(y[None]+1, clip)
    result = build_report([a, b], scope='test')
    assert result['summary']['arkit_mbe']['value'] == pytest.approx(np.sqrt(52)/2)
    assert result['summary']['av_offset']['value'] is None
