import argparse
import json

import pytest

from scripts import run_event_mouth_sequence as sequence


def fixture(tmp_path):
    code = tmp_path / 'code' / 'scripts'
    code.mkdir(parents=True)
    for name in ('train_event_schedule_pipeline.py', 'train_mouth_residual_candidate.py'):
        (code / name).write_text('')
    (tmp_path / 'dataset.pt').write_bytes(b'fixture')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'protocol.json').write_text(json.dumps({'schema': 'bounded_audio_experiment_v1'}))
    return argparse.Namespace(root=tmp_path, code_root=tmp_path / 'code',
        dataset=tmp_path / 'dataset.pt', source_run=source, output=tmp_path / 'new_output',
        event_receiver_steps=2, event_predictor_steps=3, mouth_steps=4, device='cpu', smoke=False)


def test_stage_commands_budget_and_smoke_label(tmp_path):
    args = fixture(tmp_path)
    full = sequence.stages(args)
    assert full[0][1][-4:] == ['--receiver-steps', '2', '--predictor-steps', '3']
    assert full[1][1][-4:] == ['--seeds', '42', '123', '2026']
    args.smoke = True
    smoke = sequence.stages(args)
    assert smoke[0][1][-1] == '--smoke'
    assert smoke[1][1][-2:] == ['--seeds', '42']


def test_failure_continues_independent_stage_and_final_nonzero(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    calls = []
    def fake_stage(name, command, expected_state, passed_args, shared):
        calls.append(name)
        result = {'state': 'failed' if name == 'event' else 'completed', 'exit_code': 7 if name == 'event' else 0}
        shared['stages'][name] = result
        return result
    monkeypatch.setattr(sequence, 'execute_stage', fake_stage)
    assert sequence.run(args) == 1
    assert calls == ['event', 'mouth']
    result = json.loads((args.output / 'status.json').read_text())
    assert result['failed_stages'] == ['event']
    assert result['state'] == 'failed'
    assert result['default_replaced'] is False
    with pytest.raises(FileExistsError):
        sequence.run(args)


def test_exit_zero_with_stopped_status_is_failed(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    args.output.mkdir()
    (args.output / 'supervisor').mkdir()
    (args.output / 'event').mkdir()
    (args.output / 'event/status.json').write_text(json.dumps({'state': 'stopped', 'reason': 'no events'}))
    class Process:
        pid = 123
        returncode = 0
        def poll(self):
            return 0
    monkeypatch.setattr(sequence.subprocess, 'Popen', lambda *a, **kw: Process())
    shared = {'stages': {}}
    row = sequence.execute_stage('event', ['fixture'], 'complete', args, shared)
    assert row['state'] == 'failed'
    assert row['exit_code'] == 0
    assert row['failure_reason'] == 'missing_or_incomplete_child_status'


def test_completed_scientific_gate_rejection_is_reported_without_false_success(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    args.output.mkdir()
    (args.output / 'supervisor').mkdir()
    (args.output / 'event').mkdir()
    (args.output / 'event/status.json').write_text(json.dumps({'state': 'complete'}))
    decision = {'audio_predictability_passed': False, 'default_replaced': False}
    (args.output / 'event/decision.json').write_text(json.dumps(decision))
    class Process:
        pid = 123
        returncode = 0
        def poll(self):
            return 0
    monkeypatch.setattr(sequence.subprocess, 'Popen', lambda *a, **kw: Process())
    row = sequence.execute_stage('event', ['fixture'], 'complete', args, {'stages': {}})
    assert row['state'] == 'completed'
    assert row['scientific_decision']['audio_predictability_passed'] is False


def test_rejects_output_outside_experiment_root(tmp_path):
    args = fixture(tmp_path)
    args.output = tmp_path.parent / 'outside'
    with pytest.raises(ValueError, match='fresh child'):
        sequence.validate(args)
