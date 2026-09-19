"""Bounded contracts for the event predictor stage; no remote/data dependency."""
import copy
import json

import numpy as np
import torch
from torch.nn import functional as F

from kinetalk_b0.models.audio_event_schedule import AudioEventSchedulePredictor, DURATIONS
from scripts import train_event_schedule_pipeline as pipeline


def label_fixture():
    schedule = np.zeros((18, 12), dtype=np.float32)
    schedule[4:11, 0] = 1.
    schedule[4:11, 4] = np.linspace(0., 1., 7)
    schedule[4:11, 8] = 7/25
    known = np.ones((18, 4), bool); known[:2] = False; known[-2:] = False
    onset = np.zeros((18, 4), bool); onset[4, 0] = True
    return {'schedule': schedule, 'known': known, 'onset': onset,
            'events': [{'start': 4, 'end': 11, 'group_index': 0}]}


def clip_fixture():
    values = torch.arange(18*6).reshape(18, 6).float()
    valid = torch.ones(18, dtype=torch.bool); valid[7:9] = False
    return {'clip_id': 'fixture', 'features': values, 'context': torch.tensor([2., 3.]),
            'valid': valid, 'motion9': torch.randn(18, 9),
            'motion_mask': torch.ones(18, 9, dtype=torch.bool)}


def stats_fixture():
    return {'audio_mean': torch.zeros(6), 'audio_scale': torch.ones(6),
            'context_mean': torch.zeros(2), 'context_scale': torch.ones(2)}


def test_target_risk_is_idle_or_onset_and_duration_only_at_event_start():
    label = label_fixture(); onset, risk, duration = pipeline.target_arrays(label)
    assert risk[4, 0] and not risk[5:11, 0].any()
    assert not risk[:2].any() and not risk[-2:].any()
    assert risk[3, 0] and risk[11, 0]
    assert duration[4, 0] == np.argmin(np.abs(np.asarray(DURATIONS)-7))
    assert onset.sum() == 1


def test_process_likelihood_uses_same_risk_denominator_for_onset_and_duration():
    logits = torch.zeros(2, 3, 4, requires_grad=True)
    duration_logits = torch.zeros(2, 3, 4, 7, requires_grad=True)
    y = torch.zeros(2, 3, 4); y[0, 1, 2] = 1.
    risk = torch.ones_like(y, dtype=torch.bool); risk[1] = False
    duration = torch.zeros_like(y, dtype=torch.long); duration[0, 1, 2] = 3
    out = {'onset_logits': logits, 'duration_logits': duration_logits}
    loss = pipeline.process_nll(out, y, risk, duration)
    expected = np.log(2.) + np.log(7.)/12.
    torch.testing.assert_close(loss, torch.tensor(expected, dtype=loss.dtype))
    loss.backward()
    assert not logits.grad[1].any() and not duration_logits.grad[1].any()
    assert duration_logits.grad[0, 1, 2].abs().sum() > 0
    assert duration_logits.grad[0, 0].abs().sum() == 0


def test_acoustic_input_never_reads_motion_and_interventions_preserve_native_clock():
    clip = clip_fixture(); stats = stats_fixture()
    baseline = pipeline.acoustic_input(clip, stats, 'real')
    changed = {**clip, 'motion9': None, 'motion_mask': None, 'baseline52': None}
    for a, b in zip(baseline, pipeline.acoustic_input(changed, stats, 'real')):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    reverse, context, valid = pipeline.acoustic_input(clip, stats, 'reverse')
    static, _, _ = pipeline.acoustic_input(clip, stats, 'static')
    assert torch.equal(valid, clip['valid']) and not reverse[~valid].any()
    for a, b in ((0, 7), (9, 18)):
        torch.testing.assert_close(reverse[a:b], clip['features'][a:b].flip(0), atol=0, rtol=0)
        torch.testing.assert_close(static[a:b], clip['features'][a:b].mean(0, keepdim=True).expand(b-a, -1))


def test_predictor_crop_clock_does_not_depend_on_target_known_mask():
    clip = clip_fixture(); label = label_fixture(); labels = {'fixture': label}
    rows = pipeline.prepare_audio([clip], labels, stats_fixture(), 'real')
    altered = copy.deepcopy(rows); altered[0]['risk'][:] = False
    left = pipeline.predictor_batch(rows, np.random.default_rng(17), torch.device('cpu'), batch_size=3, frames=5)
    right = pipeline.predictor_batch(altered, np.random.default_rng(17), torch.device('cpu'), batch_size=3, frames=5)
    for index in (0, 1, 2): torch.testing.assert_close(left[index], right[index], atol=0, rtol=0)


def test_matched_static_and_audio_initialize_and_sample_same_examples():
    clip = clip_fixture(); labels = {'fixture': label_fixture()}
    audio = pipeline.prepare_audio([clip], labels, stats_fixture(), 'real')
    static = pipeline.prepare_audio([clip], labels, stats_fixture(), 'static')
    torch.manual_seed(123); a = AudioEventSchedulePredictor(6, 2, hidden=8)
    torch.manual_seed(123); b = AudioEventSchedulePredictor(6, 2, hidden=8)
    pipeline.initialize_rates(a, audio); pipeline.initialize_rates(b, static)
    for key, value in a.state_dict().items(): torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
    a_batch = pipeline.predictor_batch(audio, np.random.default_rng(9), torch.device('cpu'), batch_size=4, frames=5)
    b_batch = pipeline.predictor_batch(static, np.random.default_rng(9), torch.device('cpu'), batch_size=4, frames=5)
    for index in (1, 2, 3, 4, 5): torch.testing.assert_close(a_batch[index], b_batch[index], atol=0, rtol=0)


def test_paired_comparison_weights_sentences_equally():
    first = [{'clip_id': 'a', 'sentence': 's1', 'brier': 0.},
             {'clip_id': 'b', 'sentence': 's1', 'brier': 0.},
             {'clip_id': 'c', 'sentence': 's2', 'brier': .9}]
    other = [{**r, 'brier': .3} for r in first]
    report = pipeline.paired_delta(first, other)
    assert abs(report['audio_minus_control']-.15) < 1e-12
    assert report['sentences'] == 2


def test_crop_excludes_only_artificial_boundary_targets():
    class FixedRng:
        def __init__(self, crop): self.crop = crop; self.calls = 0
        def integers(self, low, high=None, size=None):
            if size is not None: return np.zeros(size, dtype=int)
            self.calls += 1
            return 0 if self.calls == 1 else self.crop

    row = {'clip_id': 'a', 'feature': torch.randn(100, 6), 'context': torch.zeros(2),
           'valid': torch.ones(100, dtype=torch.bool), 'onset': torch.zeros(100, 4),
           'risk': torch.ones(100, 4, dtype=torch.bool), 'duration': torch.zeros(100, 4, dtype=torch.long)}
    for crop, expected in ((0, (0, 35)), (25, (15, 35)), (50, (15, 50))):
        batch = pipeline.predictor_batch([row], FixedRng(crop), torch.device('cpu'), batch_size=1, frames=50)
        support = batch[4][0].any(-1)
        wanted = torch.zeros(50, dtype=torch.bool); wanted[slice(*expected)] = True
        assert torch.equal(support, wanted)
        assert batch[2].all()  # Audio halo is retained, only likelihood masked.


def test_evaluation_joint_nll_matches_training_and_duration_null_is_explicit():
    class ZeroModel:
        def __call__(self, features, context, valid):
            return {'onset_logits': torch.zeros(*valid.shape, 4),
                    'duration_logits': torch.zeros(*valid.shape, 4, 7)}

    clip = clip_fixture(); clip['sentence'] = 's'
    label = label_fixture(); labels = {'fixture': label}
    rows = pipeline.evaluate_predictor(ZeroModel(), [clip], labels, stats_fixture(), 'real', torch.device('cpu'))
    y, risk, duration = pipeline.target_arrays(label)
    out = ZeroModel()(None, None, torch.ones(1, 18, dtype=torch.bool))
    expected = pipeline.process_nll(out, torch.from_numpy(y)[None], torch.from_numpy(risk)[None],
                                    torch.from_numpy(duration)[None])
    assert abs(rows[0]['joint_nll']-float(expected)) < 1e-6
    assert abs(rows[0]['duration_nll']-np.log(7.)) < 1e-6
    quiet = copy.deepcopy(label); quiet['onset'][:] = False; quiet['schedule'][:] = 0.; quiet['events'] = []
    quiet_rows = pipeline.evaluate_predictor(ZeroModel(), [clip], {'fixture': quiet}, stats_fixture(), 'real', torch.device('cpu'))
    assert quiet_rows[0]['duration_nll'] is None
    report = pipeline.summarize_predictor(rows + quiet_rows)
    assert report['duration_scored_clips'] == 1 and report['scored_onsets'] == 1
    paired = pipeline.paired_delta(quiet_rows, quiet_rows, metric='duration_nll')
    assert paired['audio_minus_control'] is None and paired['omitted_clips'] == ['fixture']


def test_protocol_sources_include_executed_training_split_and_metrics():
    assert {'scripts/train_continuous_motion_latent.py',
            'scripts/probe_motion_condition_predictability.py',
            'scripts/joint_motion_metrics.py',
            'scripts/prepare_continuous_motion_dataset.py',
            'scripts/train_prior_audio_adapter.py'} <= set(pipeline.SOURCES)


def test_shared_event_score_keeps_gt_support_for_missing_prediction():
    n = 60
    motion = torch.full((n, 9), .1)
    motion[15:40, 2:5] += .3*torch.sin(torch.linspace(0, torch.pi, 25))[:, None]
    clip = {'clip_id': 'one', 'sentence': 's', 'motion9': motion,
            'valid': torch.ones(n, dtype=torch.bool), 'motion_mask': torch.ones(n, 9, dtype=torch.bool)}
    teacher = pipeline.fit_teacher([clip])
    truth = pipeline.schedule_for(clip, teacher)
    assert truth['events']
    exact = pipeline.shared_event_score(clip, teacher, truth['schedule'][None, :, :4])
    # An unrecognized/censored generated excursion is represented as inactive,
    # as evaluate_receiver does; it must not remove GT positives from scoring.
    missing = pipeline.shared_event_score(clip, teacher, np.zeros((4, n, 4), np.float32))
    assert exact['target_known_positions'] == missing['target_known_positions'] == int(truth['known'].sum())
    assert missing['target_active_positions'] > 0
    assert exact['brier'] == 0. and missing['brier'] > 0.
    assert missing['mean_probability_on_true_events'] == 0.


def test_real_cpu_pipeline_smoke_with_bound_source_and_full_reports(tmp_path, monkeypatch):
    """Run every actual stage with tiny source models and one update per arm.

    Only ODE integration steps are reduced (24->2). Data splitting, teacher,
    optimizer steps, sampling, metrics/artifacts and final decision are real.
    The tiny source is an integration fixture, not a quality-qualified prior.
    """
    from kinetalk_b0.models.continuous_upper_motion import ContinuousUpperAE, ContinuousLatentFlow
    from scripts.prepare_continuous_motion_dataset import fit_statistics

    torch.manual_seed(711)
    n = 60; clips = []
    for i in range(8):
        baseline = torch.full((n, 52), .12)
        target = baseline.clone()
        motion = baseline[:, pipeline.UPPER].clone()
        wave = .24*torch.sin(torch.linspace(0, torch.pi, 25))
        motion[15:40, 2:5] += wave[:, None]
        target[:, pipeline.UPPER] = motion
        clips.append({'clip_id': f'c{i}', 'sentence': f's{i}', 'speaker': str(i % 2),
                      'emotion': str(i % 2), 'split': 'train' if i < 7 else 'holdout',
                      'features': torch.randn(n, 1540), 'context': torch.randn(8),
                      'valid': torch.ones(n, dtype=torch.bool),
                      'motion9': motion, 'motion_mask': torch.ones(n, 9, dtype=torch.bool),
                      'baseline52': baseline, 'target52': target,
                      'channel_mask': torch.ones(n, 52, dtype=torch.bool),
                      'b9': torch.full((9,), .12), 'times': torch.arange(n)/25})
    dataset = tmp_path/'dataset.pt'
    torch.save({'schema': 'continuous_motion_dataset_v1', 'clips': clips}, dataset)
    train, valid = pipeline.split_train_pool(clips)
    assert len(train) == 2 and len(valid) == 5
    reference = {'schema': 'bounded_audio_experiment_v1', 'dataset_sha256': pipeline.sha(dataset),
                 'inner_train_ids': [c['clip_id'] for c in train],
                 'inner_validation_ids': [c['clip_id'] for c in valid],
                 'validation_sentences': sorted({c['sentence'] for c in valid})}
    source = tmp_path/'source'; source.mkdir()
    pipeline.common._write(source/'protocol.json', reference)
    ae = ContinuousUpperAE(hidden=8, depth=1, latent_dim=4)
    prior = ContinuousLatentFlow(context_dim=8, hidden=8, depth=1, latent_dim=4)
    stats = fit_statistics(train)
    stats.update(latent_mean=torch.zeros(4), latent_scale=torch.ones(4))
    protocol_sha = pipeline.common._value_sha(reference)
    torch.save(stats, source/'fit_stats.pt')
    torch.save({'config': ae.config, 'state': ae.state_dict(),
                'binding': {'protocol_sha256': protocol_sha}}, source/'ae_final.pt')
    torch.save({'config': prior.config, 'state': prior.state_dict(),
                'binding': {'protocol_sha256': protocol_sha,
                            'ae_sha256': pipeline.common._value_sha(ae.state_dict()),
                            'stats_sha256': pipeline.common._value_sha(stats)}}, source/'prior_final.pt')
    original_sample = ContinuousLatentFlow.sample
    def fast_sample(self, *args, **kwargs):
        kwargs['steps'] = 2
        return original_sample(self, *args, **kwargs)
    monkeypatch.setattr(ContinuousLatentFlow, 'sample', fast_sample)
    output = tmp_path/'run'
    args = pipeline.parser().parse_args(['--dataset', str(dataset), '--source-run', str(source),
        '--output', str(output), '--device', 'cpu', '--smoke', '--receiver-steps', '1', '--predictor-steps', '1'])
    old_threads = torch.get_num_threads()
    try: pipeline.run(args)
    finally: torch.set_num_threads(old_threads)
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'complete'
    decision = json.loads((output/'decision.json').read_text())
    assert decision['smoke'] and decision['default_replaced'] is False
    assert decision['outer_evaluated'] is False
    assert isinstance(decision['audio_predictability_passed'], bool)
    assert isinstance(decision['receiver_control_path_passed'], bool)
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['fit_stats_sha256'] == pipeline.sha(source/'fit_stats.pt')
    assert protocol['outer_tensor_indexed'] is False and 'c7' not in protocol['train_ids'] + protocol['validation_ids']
    assert len(protocol['validation_ids']) == 4
    reports = json.loads((output/'predictor_results.json').read_text())['42']
    assert np.isfinite(reports['summary']['audio']['joint_nll'])
    assert np.isfinite(reports['summary']['audio']['duration_nll'])
    for name in ('oracle', 'empty', 'shifted', 'null'):
        assert (output/'control'/name/'curves.pt').is_file()
        assert (output/'control'/name/'arkit_benchmark.json').is_file()
    for name in ('audio', 'matched_static', 'reverse'):
        path = output/'generation'/name
        assert (path/'arkit_benchmark.json').is_file()
        saved = torch.load(path/'curves.pt', map_location='cpu', weights_only=False)
        assert len(saved['clips']) == 4
        for row in saved['clips'].values():
            assert row['samples'].shape == (4, n, 9)
            assert row['raw_coefficient_prediction'] is True
    checkpoints = [torch.load(output/f'receiver_{name}_final.pt', weights_only=False) for name in ('event', 'null')]
    assert all(c['step'] == 1 for c in checkpoints)
    assert checkpoints[0]['initial_state_sha256'] == checkpoints[1]['initial_state_sha256']
    assert checkpoints[0]['order_sha256'] == checkpoints[1]['order_sha256']
