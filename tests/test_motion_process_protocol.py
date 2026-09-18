"""Cross-module protocol contracts independent of the real heldout population."""
import copy
import inspect

import numpy as np
import pytest
import torch

from scripts import train_motion_process as p


def _metadata():
    return [(speaker, f'sentence-{sentence}') for speaker in range(9) for sentence in range(28)]


def test_fixed_split_is_metadata_bound_disjoint_complete_and_order_invariant():
    rows = _metadata()
    split = p.fixed_split([r[0] for r in rows], [r[1] for r in rows])
    actual = {key: {rows[i] for i in ids} for key, ids in split.items()}
    assert sum(map(len, actual.values())) == len(rows)
    for a, ca in actual.items():
        for b, cb in actual.items():
            if a != b:
                assert ca.isdisjoint(cb)
    fit_speakers = {r[0] for r in actual['fit']}
    fit_sentences = {r[1] for r in actual['fit']}
    assert all(r[0] in fit_speakers and r[1] not in fit_sentences for r in actual['sentence'])
    assert all(r[0] not in fit_speakers and r[1] in fit_sentences for r in actual['speaker'])
    assert all(r[0] not in fit_speakers and r[1] not in fit_sentences for r in actual['joint'])
    reverse = rows[::-1]
    other = p.fixed_split([r[0] for r in reverse], [r[1] for r in reverse])
    assert actual == {key: {reverse[i] for i in ids} for key, ids in other.items()}


def test_fixed_split_rejects_silent_zip_truncation_and_empty_cells():
    rows = _metadata()
    with pytest.raises(ValueError):
        p.fixed_split([r[0] for r in rows][:-1], [r[1] for r in rows])
    with pytest.raises(ValueError):
        p.fixed_split([0] * 20, list(range(20)))


def _clip(state, valid=None, anchor=None):
    state = np.asarray(state, dtype=np.float64)
    valid = np.ones(len(state), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return {'state': state, 'valid': torch.tensor(valid),
        'groupmask': np.repeat(valid[:, None], 4, axis=1),
        'anchor': np.zeros(4) if anchor is None else np.asarray(anchor),
        'features': torch.zeros(len(state), 1540)}


def test_feature_statistics_ignore_holdout_payload_and_invalid_padding():
    fit = _clip(np.zeros((4, 4)), [True, True, False, False])
    fit['features'][0] = 2
    fit['features'][1] = 4
    fit['features'][2:] = float('nan')
    held = _clip(np.zeros((3, 4)))
    held['features'].fill_(1e8)
    mean, std = p.fit_feature_stats([fit, held], [0])
    torch.testing.assert_close(mean, torch.full((1540,), 3.))
    torch.testing.assert_close(std, torch.ones(1540))
    held['features'].fill_(-1e8)
    m2, s2 = p.fit_feature_stats([fit, held], [0])
    assert torch.equal(m2, mean) and torch.equal(s2, std)


def test_plan_targets_censor_terminal_and_reset_history_only_at_real_gap():
    x = np.tile(np.sin(np.arange(91) * .8)[:, None], (1, 4))
    valid = np.ones(91, dtype=bool); valid[41:46] = False
    clips = [_clip(x, valid, anchor=np.array([.1, -.2, .3, -.4])) for _ in range(4)]
    scales, _, _ = p.prepare_plans(clips, dict(zip(p.CELLS, [[0], [1], [2], [3]])))
    c = clips[0]
    expected = []
    for group, segments in enumerate(c['plan']['segments']):
        previous = None
        for s in segments:
            if not s['right_censored']:
                delta_before = 0. if previous is None or previous['run_id'] != s['run_id'] else (
                    previous['end_value'] - previous['start_value']) / scales[group]
                duration_before = 0 if previous is None or previous['run_id'] != s['run_id'] else previous['duration']
                expected.append([s['start'], group, (s['start_value']-c['anchor'][group])/scales[group],
                    delta_before, duration_before, p.DURATIONS.index(s['duration']),
                    (s['end_value']-s['start_value'])/scales[group]])
            previous = s
    assert len(expected) > 0
    np.testing.assert_allclose(c['events'], np.asarray(expected), rtol=2e-7, atol=1e-6)
    np.testing.assert_allclose(c['initial'], (x[[0, 46]]-c['anchor'])/scales, rtol=2e-7)
    assert all(not (41 <= e[0] < 46) for e in c['events'])


def test_reconstruction_gate_cannot_count_between_run_offsets_as_motion():
    # Both trajectories have zero within-run dynamics, but the second run has
    # another DC level. A clip-centered implementation incorrectly gives corr=1.
    state = np.r_[np.zeros((8, 4)), np.full((3, 4), np.nan), np.ones((8, 4)) * 10]
    valid = np.r_[np.ones(8, bool), np.zeros(3, bool), np.ones(8, bool)]
    c = _clip(state, valid)
    c['reconstructed'] = np.where(valid[:, None], state, 0.)
    c['plan'] = {'segments': [[{'duration': 7}, {'duration': 7}] for _ in range(4)]}
    summary = p.reconstruction_summary([c], [0], 'reconstructed')
    for row in summary['groups'].values():
        assert row['correlation'] is None or row['correlation'] == 0.
        assert row['rms_ratio'] is None or row['rms_ratio'] == 0.


def test_distribution_centering_removes_only_each_run_mean_and_variogram_never_crosses_gap():
    valid = np.r_[np.ones(7, bool), np.zeros(3, bool), np.ones(9, bool)]
    t = np.arange(len(valid), dtype=np.float64)
    target = np.tile(np.sin(t * .2)[:, None], (1, 4))
    samples = np.stack([target + k * .01 for k in range(8)])
    changed = samples.copy()
    changed[:, :7] += 100
    changed[:, 10:] -= 100
    before = p.distribution_metrics(samples, target, valid)
    after = p.distribution_metrics(changed, target, valid)
    for key in ('centered_es', 'rms_ratio', 'correlation', 'variogram'):
        np.testing.assert_allclose(after[key], before[key], rtol=0, atol=2e-7, err_msg=key)
    assert min(after['raw_es']) > 50
    padding = np.pad(valid, (0, 40))
    padded = p.distribution_metrics(np.pad(samples, ((0, 0), (0, 40), (0, 0)), constant_values=np.nan),
        np.pad(target, ((0, 40), (0, 0)), constant_values=np.nan), padding)
    for key in before:
        np.testing.assert_allclose(padded[key], before[key], rtol=0, atol=1e-12)


class _DeterministicPrior:
    def __init__(self):
        self.calls = []
        self.durations = torch.tensor(p.DURATIONS)

    def initial_distribution(self, pooled):
        value = torch.zeros(len(pooled), 4, dtype=pooled.dtype, device=pooled.device)
        return {'loc': value, 'log_scale': torch.full_like(value, -100.)}

    def event_distribution(self, hidden, pooled, group, level, previous_delta, previous_duration):
        self.calls.extend(zip(hidden[:, 0].tolist(), group.tolist(), previous_delta.tolist(), previous_duration.tolist()))
        logits = torch.full((len(group), 7), -100., dtype=hidden.dtype, device=hidden.device)
        logits[:, 0] = 100.
        return {'duration_logits': logits, 'loc': torch.full_like(logits, .4),
            'log_scale': torch.full_like(logits, -100.)}


def test_free_rollout_shared_endpoints_real_gap_reset_and_padding_invariance():
    assert not {'target', 'motion', 'state', 'events', 'teacher'} & set(inspect.signature(p.sample_rollout).parameters)
    valid = torch.ones(1, 17, dtype=torch.bool); valid[:, 6:9] = False
    hidden = torch.arange(17, dtype=torch.float32)[None, :, None]
    model = _DeterministicPrior()
    actual = p.sample_rollout(model, hidden, torch.zeros(1, 1), valid, 42)
    expected = torch.tensor([0., .1, .2, .3, .4, .5, 0., 0., 0., 0., .1, .2, .3, .4, .5, .6, .7])
    torch.testing.assert_close(actual[0], expected[:, None].expand(-1, 4))
    boundaries = sorted(set(t for t, _, _, _ in model.calls))
    assert boundaries == [0., 4., 9., 13.]
    assert all(delta == 0 and duration == 0 for t, _, delta, duration in model.calls if t in (0., 9.))
    extended = p.sample_rollout(_DeterministicPrior(), torch.nn.functional.pad(hidden, (0, 0, 0, 21)),
        torch.zeros(1, 1), torch.nn.functional.pad(valid, (0, 21)), 42)
    assert torch.equal(extended[:, :17], actual)
    assert torch.count_nonzero(extended[:, 17:]) == 0


def test_free_rollout_randomness_is_keyed_per_clip_not_batch_composition():
    from kinetalk_b0.models.motion_process_prior import MotionProcessPrior
    torch.manual_seed(18)
    model = MotionProcessPrior(feature_dim=3, hidden=8).double().eval()
    hidden = torch.randn(2, 29, 8, dtype=torch.float64)
    pooled = torch.randn(2, 8, dtype=torch.float64)
    valid = torch.ones(2, 29, dtype=torch.bool); valid[0, 10:13] = False
    both = p.sample_rollout(model, hidden, pooled, valid, 42, ['first', 'second'])
    separate = p.sample_rollout(model, hidden[1:], pooled[1:], valid[1:], 42, ['second'])
    torch.testing.assert_close(both[1:], separate, atol=2e-6, rtol=0)
    longer = p.sample_rollout(model, torch.nn.functional.pad(hidden, (0, 0, 0, 32)), pooled,
        torch.nn.functional.pad(valid, (0, 32)), 42, ['first', 'second'])
    torch.testing.assert_close(both, longer[:, :29], atol=0, rtol=0)


def test_free_rollout_leading_padding_does_not_consume_an_observed_run_initial_draw():
    from kinetalk_b0.models.motion_process_prior import MotionProcessPrior
    torch.manual_seed(28)
    model = MotionProcessPrior(feature_dim=3, hidden=8).double().eval()
    hidden = torch.randn(1, 19, 8, dtype=torch.float64)
    pooled = torch.randn(1, 8, dtype=torch.float64)
    valid = torch.ones(1, 19, dtype=torch.bool); valid[:, 8:10] = False
    original = p.sample_rollout(model, hidden, pooled, valid, 42, ['clip'])
    padded = p.sample_rollout(model, torch.nn.functional.pad(hidden, (0, 0, 5, 0)), pooled,
        torch.nn.functional.pad(valid, (5, 0)), 42, ['clip'])
    torch.testing.assert_close(padded[:, 5:], original, atol=0, rtol=0)


def _reports():
    result = {arm: {} for arm in ('global_only', 'local_audio')}
    for arm in result:
        for cell in ('sentence', 'speaker', 'joint'):
            rows = [{'clip_id': f'{cell}-{i}', 'sentence': f's{i}', 'event_nll': 2. if arm == 'global_only' else 1.9}
                for i in range(4)]
            result[arm][cell] = {'rows': rows, 'summary': {
                'centered_es': [1.] * 4, 'raw_es': [1.] * 4, 'variogram': [1.] * 4,
                'rms_ratio': [1.] * 4, 'out_of_domain': [0.] * 4}}
    return result


def test_audio_gate_requires_id_pairing_and_declared_gain():
    reports = _reports()
    assert p.audio_gate(reports)['passed']
    reports['local_audio']['sentence']['rows'][0]['clip_id'] = 'different'
    with pytest.raises(ValueError):
        p.audio_gate(reports)


@pytest.mark.parametrize('key,value', [('centered_es', float('nan')), ('raw_es', 1.2),
    ('rms_ratio', .01), ('rms_ratio', 4.), ('out_of_domain', .2)])
def test_audio_gate_rejects_nonfinite_or_physically_invalid_free_rollout(key, value):
    reports = _reports()
    reports['local_audio']['sentence']['summary'][key][1] = value
    assert not p.audio_gate(reports)['passed']


def test_audio_gate_rejects_absent_event_support_without_empty_bootstrap():
    reports = _reports()
    for arm in reports:
        for row in reports[arm]['sentence']['rows']:
            row['event_nll'] = None
    assert not p.audio_gate(reports)['passed']
