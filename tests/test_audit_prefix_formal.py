"""Small saved-record audits with synthetic tensors and no dataset reads."""
import json
from pathlib import Path

import pytest
import torch

from scripts import audit_prefix_formal as audit


@pytest.fixture(autouse=True)
def small_audit(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(audit, 'DEVELOPMENT_CLIPS', 2)
    monkeypatch.setattr(audit, 'FIT_CLIPS', 16)
    yield
    torch.set_num_threads(threads)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf8')


def curves_fixture():
    torch.manual_seed(5)
    target = torch.rand(2, 96, 52)
    valid = torch.ones(2, 96, dtype=torch.bool)
    valid[0, 27:29] = False
    valid[1, 70:] = False
    meta = {'clip_id': ['synthetic_a', 'synthetic_b'], 'target': target, 'valid': valid,
            'channel_mask': torch.ones(2, 52, dtype=torch.bool),
            'times': torch.arange(96, dtype=torch.float64)[None].expand(2, -1).clone()*.04,
            'b0': torch.zeros_like(target), 'emotion_id': torch.tensor([0, 1]),
            'speaker_id': torch.tensor([3, 7]), 'noise_seeds': list(audit.SEEDS)}
    baselines = {str(seed)+'/base': torch.rand_like(target) for seed in audit.SEEDS}
    for value in baselines.values():
        value[~valid] = float('nan')
    return meta, {**meta, 'predictions': baselines}


def evaluation_for(curves, arm):
    prefix = arm == 'scheduled_prefix'
    modes, oracle_modes = {}, {}
    for key, pred in curves['predictions'].items():
        mode = key.split('/')[1]
        scored = prefix and mode in ('full', 'oracle_history')
        metrics = audit.actual_prefix_continuation(pred, curves['target'], curves['valid'], curves['channel_mask'],
                     curves['target'] if mode == 'oracle_history' else pred) if scored else None
        record = {'oracle_target_history': mode == 'oracle_history', 'prefix_enabled': prefix,
                  'generated_emotion_accuracy_nonindependent': .5,
                  'actual_prefix_continuation': {'scored': scored,
                      'source': ('tracked_reference_GT' if mode == 'oracle_history' else 'generated_past') if scored else None,
                      'metrics': metrics},
                  'same_gt_endpoint_output_diagnostic': {
                      'GT_was_supplied_as_history': prefix and mode == 'oracle_history',
                      'metrics': audit.actual_prefix_continuation(pred, curves['target'], curves['valid'],
                                                                curves['channel_mask'], curves['target'])}}
        (oracle_modes if mode == 'oracle_history' else modes)[key] = record
    distribution = audit.summarize(curves, curves['emotion_id'])
    controls = distribution.pop('single_seed_interventions')
    oracle = controls.pop('42/oracle_history')
    return {'schema': audit.SCHEMA, 'arm': arm, 'recipe_sha256': curves['recipe_sha256'], 'smoke': False,
            'clips': 2, 'noise_seeds': list(audit.SEEDS), 'decode_steps': 12,
            'evaluation_role': 'internal_development', 'prefix_enabled': prefix, 'test_loaded': False,
            'default_replaced': False, 'nonupper_exact': True, 'invalid_baseline_exact': True,
            'first_chunk_history_interventions_equal': True,
            'no_prefix_history_interventions_equal': None if prefix else True,
            'modes': modes, 'distribution': distribution, 'deployable_interventions_seed42': controls,
            'ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY': {'modes': oracle_modes, 'paired_metrics_vs_full': oracle}}


def write_run(root):
    run = root/'prefix12'; run.mkdir()
    meta, baseline = curves_fixture()
    baseline_path = root/'baseline.pt'; torch.save(baseline, baseline_path)
    digest = 'a'*64
    recipe = {'schema': audit.SCHEMA, 'epochs': 12, 'requested_epochs': 12, 'batch_size': 16, 'seed': 79,
              'smoke': False, 'fixed_final_epoch': True, 'test_loaded': False, 'default_replaced': False,
              'local_frozen': True, 'chunk': 16, 'history': 8, 'decode_steps': 12,
              'teacher_probabilities': [audit.teacher_probability(e, 12) for e in range(12)],
              'explicit_discarded_source_keys': sorted(audit.DISCARDED),
              'frozen': {key: digest for key in ('system', 'audio', 'local')},
              'initial': digest, 'initial_file_sha256': digest, 'warmstart_sha256': digest,
              'warmstart_recipe_sha256': digest, 'baseline_curves_sha256': audit.sha(baseline_path),
              'scales': [1.]*9,
              'data_provenance': {'fit_clips': 16, 'development_clips': 2, 'test_loaded': False,
                  'outer_development_loaded': False, 'default_replaced': False, 'input_sha256': {'x': digest}},
              'pilot_binding': {'gate': {'passed': True}, 'original_warmstart_sha256': digest,
                  'selected_fit_ids': ['fit_'+str(i) for i in range(8)]},
              'code_sha256': {name: audit.sha(Path(audit.__file__).resolve().parents[1]/name) for name in audit.CODE_FILES}}
    matched = {}
    for arm in audit.ARMS:
        folder = run/arm; folder.mkdir()
        arm_recipe = {**recipe, 'arm': arm}
        rd = audit.canonical_hash(arm_recipe)
        predictions = {}
        for seed in audit.SEEDS:
            current = baseline['predictions'][str(seed)+'/base'].clone()
            for channel in audit.CC:
                current[..., channel] = torch.where(meta['valid'], meta['target'][..., channel]+.03, current[..., channel])
            predictions[str(seed)+'/full'] = current
        for mode in audit.MODES[1:]:
            value = predictions['42/full'].clone()
            if arm == 'scheduled_prefix' or mode in ('static', 'reverse'):
                for channel in audit.CC:
                    valid = meta['valid'][:, 16:]
                    value[:, 16:, channel] = torch.where(valid, value[:, 16:, channel]+.02, value[:, 16:, channel])
            predictions['42/'+mode] = value
        curves = {**meta, 'schema': audit.CURVE_SCHEMA, 'formal_schema': audit.SCHEMA, 'arm': arm,
                  'recipe_sha256': rd, 'decode_steps': 12, 'prefix_enabled': arm == 'scheduled_prefix',
                  'oracle_prediction_keys': ['42/oracle_history'], 'predictions': predictions}
        torch.save(curves, folder/'curves.pt')
        save_json(folder/'evaluation.json', evaluation_for(curves, arm))
        save_json(folder/'provenance.json', {'recipe': arm_recipe, 'recipe_sha256': rd})
        save_json(folder/'complete.json', {'recipe_sha256': rd, 'completed_epochs': 12,
                  'frozen': recipe['frozen'], 'curves_sha256': audit.sha(folder/'curves.pt'), 'final_sha256': digest})
        draws = []
        for epoch in range(1, 13):
            probability = recipe['teacher_probabilities'][epoch-1]
            chosen = 2 if probability else 0
            draw = audit.canonical_hash({'epoch': epoch}); draws.append(draw)
            save_json(folder/f'epoch{epoch:03d}.json', {'arm': arm, 'epoch': epoch,
                'total_steps': epoch, 'teacher_probability': probability,
                'no_prefix_ignores_history': arm == 'no_prefix', 'loss': .1, 'seconds': 1., 'draw_sha256': draw,
                'teacher_draw_count': chosen, 'valid_history_count': 3,
                'teacher_eligible_count': chosen, 'teacher_used_count': chosen if arm == 'scheduled_prefix' else 0})
        matched[arm] = {'initial': digest, 'draws': draws}
    save_json(run/'matched_audit.json', {'equal': True, 'arms': matched})
    save_json(run/'status.json', {'status': 'complete', 'epochs_per_arm': 12, 'smoke': False, 'seconds': 30.})
    return run, baseline_path


def rewrite_json(path, update):
    value = audit.read(path); update(value); save_json(path, value)


def rewrite_curves(folder, update):
    curves = torch.load(folder/'curves.pt', map_location='cpu', weights_only=False)
    update(curves); torch.save(curves, folder/'curves.pt')
    rewrite_json(folder/'complete.json', lambda complete: complete.update(curves_sha256=audit.sha(folder/'curves.pt')))


def test_full_local_audit_passes_without_checkpoints_and_does_not_overclaim(tmp_path):
    run, base = write_run(tmp_path)
    before = {str(path): audit.sha(path) for path in tmp_path.rglob('*') if path.is_file()}
    report = audit.audit(run, base)
    assert report['status'] == 'passed' and report['test_loaded'] is False
    assert all(report['checks'].values())
    for binding in report['bindings'].values():
        assert binding['initial_tensor_and_file_locally_verified'] is False
        assert binding['final_checkpoint_file_hash_verified_at_audit_location'] is False
        assert binding['final_checkpoint_tensors_loaded'] is False
        assert binding['frozen_source_tensors_locally_verified'] is False
    after = {str(path): audit.sha(path) for path in tmp_path.rglob('*') if path.is_file()}
    assert before == after


@pytest.mark.parametrize('failure,match', [
    ('curve_hash', 'Curve file hash'), ('pair_draws', 'Initial or draw'),
    ('teacher_final', 'Teacher eligibility|generated-only'), ('frozen_record', 'Frozen runtime'),
    ('oracle_in_deployment', 'Oracle must remain separate'), ('metric_tamper', 'numeric value'),
    ('recipe_tamper', 'Recipe digest'), ('missing_epoch', None)])
def test_saved_record_corruptions_are_rejected(tmp_path, failure, match):
    run, base = write_run(tmp_path); folder = run/'scheduled_prefix'
    if failure == 'curve_hash':
        rewrite_json(folder/'complete.json', lambda c: c.update(curves_sha256='b'*64))
    elif failure == 'pair_draws':
        rewrite_json(folder/'epoch002.json', lambda c: c.update(draw_sha256='b'*64))
    elif failure == 'teacher_final':
        rewrite_json(folder/'epoch012.json', lambda c: c.update(teacher_used_count=1))
    elif failure == 'frozen_record':
        rewrite_json(folder/'complete.json', lambda c: c['frozen'].update(local='b'*64))
    elif failure == 'oracle_in_deployment':
        rewrite_json(folder/'evaluation.json', lambda c: c['modes'].update({'42/oracle_history': {}}))
    elif failure == 'metric_tamper':
        rewrite_json(folder/'evaluation.json', lambda c: c['distribution']['populations']['all']['groups']['brows']['mean_over_three_seeds'].update(raw_mse=10.))
    elif failure == 'recipe_tamper':
        rewrite_json(folder/'provenance.json', lambda c: c['recipe'].update(seed=8))
    else:
        (folder/'epoch005.json').unlink()
    with pytest.raises(FileNotFoundError if match is None else ValueError, match=match):
        audit.audit(run, base)


@pytest.mark.parametrize('failure,match', [
    ('nonupper', 'Nonupper protection'), ('invalid', 'Invalid baseline protection'),
    ('first_chunk', 'History affected first chunk'), ('no_prefix_history', 'No-prefix output'),
    ('metadata', 'Cross-run metadata')])
def test_curve_contract_violations_are_detected_after_file_hash_is_rebound(tmp_path, failure, match):
    run, base = write_run(tmp_path)
    folder = run/('no_prefix' if failure == 'no_prefix_history' else 'scheduled_prefix')
    def change(curves):
        if failure == 'metadata':
            curves['speaker_id'][0] += 1
        else:
            key = '42/empty' if failure in ('first_chunk', 'no_prefix_history') else '42/full'
            frame = 0 if failure in ('first_chunk', 'nonupper') else (27 if failure == 'invalid' else 20)
            channel = audit.NOT_UPPER[0] if failure == 'nonupper' else audit.CC[0]
            curves['predictions'][key][0, frame, channel] = .991
    rewrite_curves(folder, change)
    with pytest.raises(ValueError, match=match):
        audit.audit(run, base)


def test_actual_prefix_semantics_rejects_gt_source_for_deployment(tmp_path):
    run, base = write_run(tmp_path)
    path = run/'scheduled_prefix/evaluation.json'
    rewrite_json(path, lambda report: report['modes']['42/full']['actual_prefix_continuation'].update(source='tracked_reference_GT'))
    with pytest.raises(ValueError, match='Actual-prefix metric source'):
        audit.audit(run, base)


def test_optional_final_checkpoint_is_hashed_without_loading_tensor(tmp_path):
    run, base = write_run(tmp_path)
    for arm in audit.ARMS:
        path = run/arm/'final.pt'; path.write_bytes(b'opaque synthetic checkpoint bytes')
        rewrite_json(path.with_name('complete.json'), lambda c: c.update(final_sha256=audit.sha(path)))
    result = audit.audit(run, base)
    assert all(row['final_checkpoint_file_hash_verified_at_audit_location'] for row in result['bindings'].values())
    (run/'no_prefix/final.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Final checkpoint file hash'):
        audit.audit(run, base)
