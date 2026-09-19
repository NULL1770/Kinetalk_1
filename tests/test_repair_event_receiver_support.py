import copy
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from kinetalk_b0.models.continuous_upper_motion import ContinuousUpperAE, ContinuousLatentFlow
from kinetalk_b0.models.event_conditioned_flow import EventConditionedFlow
from scripts import repair_event_receiver_support as repair


def fixture():
    lengths = [4, 5, 9, 10, 200, 201, 205]
    n = sum(lengths)+3*(len(lengths)-1)
    clip = {'clip_id': 'fit', 'split': 'train', 'sentence': 'fit_sentence',
            'speaker': 'person', 'emotion': 'happy', 'valid': torch.zeros(n, dtype=torch.bool),
            'motion_mask': torch.ones(n, 9, dtype=torch.bool), 'motion9': torch.randn(n, 9),
            'b9': torch.zeros(9), 'features': torch.randn(n, 1540), 'context': torch.randn(3)}
    start = 0
    for length in lengths:
        clip['valid'][start:start+length] = True
        start += length+3
    stats = {'audio_mean': torch.zeros(1540), 'audio_scale': torch.ones(1540),
             'context_mean': torch.zeros(3), 'context_scale': torch.ones(3),
             'residual_scale': torch.ones(9), 'latent_mean': torch.zeros(4), 'latent_scale': torch.ones(4)}
    return clip, stats


def test_restored_support_keeps_source_tails_and_zero12_without_teacher():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    clip, stats = fixture()
    original = copy.deepcopy(clip)
    ae = ContinuousUpperAE(hidden=8, depth=1, latent_dim=4).eval().requires_grad_(False)
    items, coverage = repair.prepare_full_null_segments([clip], stats, ae, torch.device('cpu'))
    lengths = [len(x['audio']) for x in items]
    assert lengths == [5, 9, 10, 200, 200, 200, 5]
    assert coverage['totals']['kept_frames'] == 629
    assert coverage['totals']['dropped_short_frames'] == 5
    assert not coverage['teacher_read_for_training']
    assert coverage['conditions_all_zero'] and coverage['condition_width'] == 12
    observed = set()
    for item in items:
        a, b = item['metadata']['start'], item['metadata']['end']
        assert clip['valid'][a:b].all()
        assert not observed.intersection(range(a, b))
        observed.update(range(a, b))
        assert item['audio'].shape == (b-a, 12) and not item['audio'].any()
        assert item['z'].shape == ((b-a+4)//5, 4)
    for key in clip:
        if torch.is_tensor(clip[key]):
            assert torch.equal(clip[key], original[key])
    with pytest.raises(ValueError, match='fit-only'):
        repair.prepare_full_null_segments([{**clip, 'split': 'holdout'}], stats, ae, torch.device('cpu'))


def test_common_encoding_matches_old_receiver_on_identical_intervals_and_trains_only_copy(tmp_path):
    torch.manual_seed(8)
    clip, stats = fixture()
    ae = ContinuousUpperAE(hidden=8, depth=1, latent_dim=4).eval().requires_grad_(False)
    prior = ContinuousLatentFlow(3, hidden=8, depth=1, latent_dim=4).eval().requires_grad_(False)
    frozen = repair.common._value_sha([ae.state_dict(), prior.state_dict(), stats])
    items, _ = repair.prepare_full_null_segments([clip], stats, ae, torch.device('cpu'))
    n = len(clip['valid'])
    labels = {'fit': {'known': np.ones((n, 4), bool), 'schedule': np.zeros((n, 12), np.float32)}}
    old = repair.event.receiver_segments([clip], labels, stats, ae, torch.device('cpu'))
    new_by_span = {(r['metadata']['start'], r['metadata']['end']): r for r in items}
    for row in old:
        same = new_by_span[row['metadata']['start'], row['metadata']['end']]
        torch.testing.assert_close(same['z'], row['z'], atol=0, rtol=0)
        torch.testing.assert_close(same['context'], row['context'], atol=0, rtol=0)
        assert torch.equal(same['audio'], row['audio'])
    torch.manual_seed(repair.SEED)
    receiver = EventConditionedFlow.from_prior(prior)
    initial = repair.common._value_sha(receiver.state_dict())
    record = repair.common._train_stage(receiver, items, stats, stage='fixture', budget=2,
        batch_size=3, device=torch.device('cpu'), output=tmp_path, binding={'fixture': True},
        seed=repair.SEED, use_audio=True)
    assert record['step'] == 2 and record['initial_state_sha256'] == initial
    assert all(np.isfinite(record['losses']))
    assert repair.common._value_sha(receiver.state_dict()) != initial
    assert repair.common._value_sha([ae.state_dict(), prior.state_dict(), stats]) == frozen


def test_formal_coverage_gate_requires_exact_613_and_66272_and_disjoint_validation():
    fit = [{'clip_id': f'f{i}', 'sentence': 'fit'} for i in range(613)]
    validation = [{'clip_id': f'v{i}', 'sentence': 'valid'} for i in range(206)]
    coverage = {'retained_clip_ids': [c['clip_id'] for c in fit], 'totals': {'kept_frames': 66272}}
    repair.verify_production_coverage(coverage, fit, validation)
    for bad in ({**coverage, 'totals': {'kept_frames': 14898}},
                {**coverage, 'retained_clip_ids': coverage['retained_clip_ids'][:-1]}):
        with pytest.raises(ValueError, match='66272'):
            repair.verify_production_coverage(bad, fit, validation)
    with pytest.raises(ValueError, match='sentences overlap'):
        repair.verify_production_coverage(coverage, fit, [{**v, 'sentence': 'fit'} for v in validation])


def test_reference_table_guard_rejects_metric_changes():
    row = {'arm': 'null', 'rms': [.6, .7], 'value': {'x': None, 'computed': True}}
    repair.verify_reference_row(row, copy.deepcopy(row))
    with pytest.raises(ValueError, match='numerical'):
        repair.verify_reference_row({**row, 'rms': [.8, .7]}, row)


def test_bound_training_and_all_validation_pipeline(tmp_path, monkeypatch):
    """Real source/null checkpoints, hash/table binding, two updates and export.

    Only production population/coverage constants and the update count are
    reduced; actual preparation, optimization, sampling and scoring execute.
    """
    from scripts.prepare_continuous_motion_dataset import fit_statistics
    torch.manual_seed(221)
    torch.set_num_threads(1)
    clips = []
    n = 60
    for i in range(7):
        base = torch.full((n, 52), .12)
        motion = base[:, repair.audit.native.UPPER].clone()
        motion[15:40, 2:5] += .24*torch.sin(torch.linspace(0, torch.pi, 25))[:, None]
        target = base.clone(); target[:, repair.audit.native.UPPER] = motion
        clips.append({'clip_id': f'c{i}', 'split': 'train', 'sentence': f's{i}',
                      'speaker': str(i % 2), 'emotion': str(i % 2),
                      'features': torch.randn(n, 1540), 'context': torch.randn(3),
                      'valid': torch.ones(n, dtype=torch.bool), 'motion9': motion,
                      'motion_mask': torch.ones(n, 9, dtype=torch.bool), 'baseline52': base,
                      'target52': target, 'channel_mask': torch.ones(n, 52, dtype=torch.bool),
                      'b9': torch.full((9,), .12), 'times': torch.arange(n)/25})
    dataset = tmp_path/'dataset.pt'
    torch.save({'schema': 'continuous_motion_dataset_v1', 'clips': clips}, dataset)
    fit, valid = repair.audit.split_train_pool(clips)
    code = Path(repair.__file__).resolve().parents[1]
    source = tmp_path/'source'; source.mkdir()
    event_run = tmp_path/'event'; event_run.mkdir()
    reference = tmp_path/'reference'; reference.mkdir()
    output = tmp_path/'repair'
    source_protocol = {'schema': 'bounded_audio_experiment_v1',
        'dataset_sha256': repair.audit.sha(dataset),
        'inner_train_ids': [c['clip_id'] for c in fit],
        'inner_validation_ids': [c['clip_id'] for c in valid],
        'validation_sentences': sorted({c['sentence'] for c in valid}),
        'source_sha256': {p: repair.audit.sha(code/p) for p in repair.event.SOURCES}}
    repair.common._write(source/'protocol.json', source_protocol)
    ae = ContinuousUpperAE(hidden=8, depth=1, latent_dim=4).eval().requires_grad_(False)
    prior = ContinuousLatentFlow(3, hidden=8, depth=1, latent_dim=4).eval()
    stats = fit_statistics(fit)
    stats.update(latent_mean=torch.zeros(4), latent_scale=torch.ones(4))
    ae_ck = {'config': ae.config, 'state': ae.state_dict(),
             'binding': {'protocol_sha256': repair.common._value_sha(source_protocol)}}
    prior_ck = {'config': prior.config, 'state': prior.state_dict(),
                'binding': {'protocol_sha256': repair.common._value_sha(source_protocol),
                            'ae_sha256': repair.common._value_sha(ae.state_dict()),
                            'stats_sha256': repair.common._value_sha(stats)}}
    torch.save(ae_ck, source/'ae_final.pt')
    torch.save(prior_ck, source/'prior_final.pt')
    torch.save(stats, source/'fit_stats.pt')
    teacher = repair.event.fit_teacher(fit)
    repair.common._write(event_run/'teacher.json', teacher)
    labels = {c['clip_id']: repair.event.schedule_for(c, teacher) for c in fit}
    old_segments = repair.event.receiver_segments(fit, labels, stats, ae, torch.device('cpu'))
    old_segments = [{**r, 'audio': torch.zeros_like(r['audio'])} for r in old_segments]
    old_coverage = {'clips': len({s['metadata']['clip_id'] for s in old_segments}),
                    'frames': sum(len(s['audio']) for s in old_segments), 'segments': len(old_segments)}
    assert old_coverage['frames'] > 0
    repair.common._write(event_run/'receiver_coverage.json', old_coverage)
    event_protocol = {'schema': repair.event.SCHEMA, 'smoke': False,
        'dataset_sha256': repair.audit.sha(dataset), 'source_protocol_sha256': repair.audit.sha(source/'protocol.json'),
        'source_sha256': {p: repair.audit.sha(code/p) for p in repair.event.SOURCES},
        'train_ids': [c['clip_id'] for c in fit], 'validation_ids': [c['clip_id'] for c in valid],
        'receiver_seed': repair.SEED, 'receiver_steps_per_arm': 2,
        **{key: repair.audit.sha(source/file) for key, file in
           [('ae_sha256', 'ae_final.pt'), ('prior_sha256', 'prior_final.pt'), ('fit_stats_sha256', 'fit_stats.pt')]}}
    repair.common._write(event_run/'protocol.json', event_protocol)
    torch.manual_seed(repair.SEED)
    old_null = EventConditionedFlow.from_prior(prior)
    repair.common._train_stage(old_null, old_segments, stats, stage='receiver_null', budget=2,
        batch_size=repair.BATCH_SIZE, device=torch.device('cpu'), output=event_run,
        binding={'protocol_sha256': repair.audit.sha(event_run/'protocol.json'),
                 'teacher_sha256': repair.audit.sha(event_run/'teacher.json')}, seed=repair.SEED, use_audio=True)
    source_curves_dir = source/'prior_generation/inner_validation'
    repair.audit.native.evaluate_generation(prior, ae, valid, stats, source_curves_dir,
                                            use_audio=False, seeds=repair.audit.SEEDS)
    empty = {c['clip_id']: [np.zeros((n, 12), np.float32)] for c in valid}
    repair.event.evaluate_receiver(old_null, ae, valid, stats, empty, reference/'event_null',
                                    torch.device('cpu'), 'event_null', teacher)
    rows = []
    for name, path in [('source_prior', source_curves_dir), ('event_null', reference/'event_null')]:
        curve = repair.common._load(path/'curves.pt')['clips']
        scored, arkit = repair.audit.rescore_arm(curve, valid, stats, reference/name,
                                                deterministic=False, scope='fixture')
        rows.append(repair.audit.table_row(name, scored, arkit))
    repair.common._write(reference/'summary.json', {'table': rows})
    repair.common._write(reference/'status.json', {'state': 'complete'})
    repair.common._write(reference/'protocol.json', {'schema': 'event_validation_scope_audit_v1',
        'source_protocol_sha256': repair.audit.sha(source/'protocol.json'),
        'event_protocol_sha256': repair.audit.sha(event_run/'protocol.json'),
        'dataset_sha256': repair.audit.sha(dataset), 'selected_ids': [c['clip_id'] for c in valid],
        'seeds': list(repair.audit.SEEDS), 'steps': 24,
        'receiver_sha256': {'null': repair.audit.sha(event_run/'receiver_null_final.pt')},
        'source_curves_sha256': {'source_prior': repair.audit.sha(source_curves_dir/'curves.pt')},
        'code_sha256': {p: repair.audit.sha(code/p) for p in repair.event.SOURCES}})
    _, coverage = repair.prepare_full_null_segments(fit, stats, ae, torch.device('cpu'))
    for key, value in {'BUDGET': 2, 'EXPECTED_FIT_CLIPS': len(fit), 'EXPECTED_VALID_CLIPS': len(valid),
        'EXPECTED_FIT_FRAMES': coverage['totals']['kept_frames'], 'EXPECTED_OLD_CLIPS': old_coverage['clips'],
        'EXPECTED_OLD_FRAMES': old_coverage['frames']}.items():
        monkeypatch.setattr(repair, key, value)
    args = Namespace(dataset=dataset, source_run=source, event_run=event_run,
                     reference_audit=reference, output=output, device='cpu')
    before = repair.preserved_hashes(args)
    repair.run(args)
    assert repair.audit.read(output/'status.json')['state'] == 'complete'
    decision = repair.audit.read(output/'decision.json')
    assert decision['restored_support_verified'] and decision['reference_table_reproduced']
    assert not decision['promoted'] and not decision['audio_timing_success_claimed']
    assert before == repair.preserved_hashes(args)
    assert len(repair.audit.read(output/'summary.json')['table']) == 3
