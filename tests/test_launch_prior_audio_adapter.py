"""Use fake CPU children; never start CUDA training in launcher tests."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import launch_prior_audio_adapter as launch


def arguments(tmp_path, **overrides):
    dataset = tmp_path / 'dataset.pt'; dataset.write_bytes(b'fake dataset')
    result = dict(dataset=dataset, output=tmp_path / 'run', code_root=tmp_path,
                  device='cpu', ae_steps=40, prior_steps=140, adapter_steps=30,
                  milestones=[10, 30], batch_size=4, valid_sentences=5, max_delta=.35,
                  resume=False, check_only=False, supervise=False)
    result.update(overrides)
    return SimpleNamespace(**result)


def fake_child(output, state='complete', exit_code=0, *, write=True):
    script = 'import json,os,pathlib,sys; '
    if write:
        script += ('pathlib.Path(sys.argv[1]).write_text(json.dumps({"state":sys.argv[2],'
                   '"reason":"quality gate"}),encoding="utf8"); ')
    script += 'print("fake-child",os.environ.get("OMP_NUM_THREADS")); sys.exit(int(sys.argv[3]))'
    return [sys.executable, '-c', script, str(output / 'status.json'), state, str(exit_code)]


@pytest.mark.parametrize('runner_state,exit_code,expected,result', [
    ('complete', 0, 'complete', 0), ('stopped', 0, 'stopped', 0), ('failed', 0, 'failed', 1),
    ('complete', 7, 'failed', 7), ('running', 0, 'failed', 1), ('stopped', 3, 'failed', 3),
])
def test_supervisor_checks_actual_exit_and_fresh_terminal_state(tmp_path, runner_state, exit_code, expected, result):
    args = arguments(tmp_path)
    assert launch.supervise(args, command=fake_child(args.output, runner_state, exit_code)) == result
    status = json.loads((args.output / 'pipeline_status.json').read_text())
    assert status['state'] == expected and status['exit_code'] == exit_code
    assert status['runner_state'] == runner_state
    assert status['pid'] > 0 and status['child_pid'] > 0 and status['end'] >= status['start']
    assert status['elapsed_seconds'] >= 0
    assert 'fake-child 4' in (args.output / 'training.log').read_text()
    assert not list(args.output.glob('*.tmp'))


@pytest.mark.parametrize('stale', [False, True])
def test_missing_or_unchanged_status_cannot_complete(tmp_path, stale):
    args = arguments(tmp_path, resume=stale); args.output.mkdir()
    if stale: (args.output / 'status.json').write_text('{"state":"complete"}')
    assert launch.supervise(args, command=fake_child(args.output, write=False)) == 1
    status = json.loads((args.output / 'pipeline_status.json').read_text())
    assert status['state'] == 'failed' and 'fresh status' in status['reason']


def test_spawn_error_is_recorded(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    def fail(*_args, **_kwargs): raise OSError('fake spawn failure')
    monkeypatch.setattr(launch.subprocess, 'Popen', fail)
    assert launch.supervise(args) == 1
    status = json.loads((args.output / 'pipeline_status.json').read_text())
    assert status['state'] == 'failed' and 'fake spawn failure' in status['reason']


def test_pid_ownership_matches_new_modules_and_exact_output(tmp_path):
    proc = tmp_path / 'proc'; record = proc / '102'; record.mkdir(parents=True)
    output = tmp_path / 'run'
    def cmd(tokens): (record / 'cmdline').write_bytes(b'\0'.join(str(x).encode() for x in tokens) + b'\0')
    for module in (launch.RUNNER_MODULE, launch.LAUNCHER_MODULE, '/x/train_prior_audio_adapter.py'):
        cmd(['python', '-m', module, '--output', output])
        assert launch.process_matches_output(102, output, proc)
    for module in ('unrelated.module', 'scripts.train_continuous_motion_latent'):
        cmd(['python', '-m', module, '--output', output])
        assert not launch.process_matches_output(102, output, proc)
    cmd(['python', '-m', launch.RUNNER_MODULE, '--output', str(output)+'_other'])
    assert not launch.process_matches_output(102, output, proc)
    cmd(['python', '-m', launch.RUNNER_MODULE, '--note', output])
    assert not launch.process_matches_output(102, output, proc)
    cmd(['python', '-m', launch.RUNNER_MODULE, '--output='+str(output)])
    assert launch.process_matches_output(102, output, proc)
    assert not launch.process_matches_output(999, output, proc)


def test_check_only_validates_and_never_creates_output(tmp_path, monkeypatch):
    args = arguments(tmp_path, check_only=True, device='cuda'); calls = []
    monkeypatch.setattr(launch, 'require_cuda', lambda device: calls.append(device) or {'cuda_checked': True})
    result = launch.launch(args)
    assert result['state'] == 'checked' and result['started'] is False
    assert calls == ['cuda'] and not args.output.exists()


def test_disk_and_cuda_fail_before_output_creation(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(launch.shutil, 'disk_usage', lambda _: SimpleNamespace(free=launch.MIN_FREE_BYTES-1))
    with pytest.raises(RuntimeError, match='1 GiB'): launch.preflight(args)
    assert not args.output.exists()
    monkeypatch.setattr(launch.shutil, 'disk_usage', lambda _: SimpleNamespace(free=2*launch.MIN_FREE_BYTES))
    def unavailable(_): raise RuntimeError('CUDA unavailable')
    monkeypatch.setattr(launch, 'require_cuda', unavailable)
    with pytest.raises(RuntimeError, match='CUDA'): launch.preflight(args)
    assert not args.output.exists()


def test_resume_requires_existing_directory_and_rejects_live_owner(tmp_path, monkeypatch):
    args = arguments(tmp_path, resume=True)
    with pytest.raises(FileNotFoundError, match='existing'): launch.preflight(args)
    args.output.mkdir(); args.resume = False
    with pytest.raises(FileExistsError, match='resume'): launch.preflight(args)
    args.resume = True
    launch.write_json(args.output / 'launch.json', {'pid': 12})
    launch.write_json(args.output / 'pipeline_status.json', {'supervisor_pid': 12, 'child_pid': 13})
    monkeypatch.setattr(launch, 'process_matches_output', lambda pid, _: pid == 13)
    with pytest.raises(RuntimeError, match='still alive'): launch.preflight(args)
    monkeypatch.setattr(launch, 'process_matches_output', lambda *_: False)
    assert launch.preflight(args)['resume'] is True


def test_detached_args_and_ownership_record(tmp_path, monkeypatch):
    args = arguments(tmp_path, device='cuda:0'); calls = []
    monkeypatch.setattr(launch, 'require_cuda', lambda _: {'cuda_checked': True})
    def fake_popen(command, **kwargs): calls.append((command, kwargs)); return SimpleNamespace(pid=789)
    monkeypatch.setattr(launch.subprocess, 'Popen', fake_popen)
    result = launch.launch(args)
    command, options = calls[0]
    assert result['pid'] == 789 and result['state'] == 'launched'
    assert '--supervise' in command and command[command.index('--adapter-steps')+1] == '30'
    assert command[command.index('--milestones')+1:command.index('--milestones')+3] == ['10', '30']
    assert command[command.index('--valid-sentences')+1] == '5'
    assert command[command.index('--max-delta')+1] == '0.35'
    assert command[command.index('--device')+1] == 'cuda:0'
    assert options['start_new_session'] and options['stdin'] == subprocess.DEVNULL
    assert options['env']['MKL_NUM_THREADS'] == '4'
    if sys.platform == 'win32': assert options['creationflags'] == subprocess.CREATE_NO_WINDOW
    record = json.loads((args.output / 'launch.json').read_text())
    assert record['pid'] == 789 and record['completion_not_yet_verified']
    assert not (args.output / 'pipeline_status.json').exists()


def test_parser_and_runner_match_production_protocol(tmp_path):
    args = launch.parser().parse_args(['--dataset',str(tmp_path/'d.pt'),'--output',str(tmp_path/'out')])
    assert (args.ae_steps,args.prior_steps,args.adapter_steps,args.batch_size) == (4000,14000,3000,24)
    assert args.milestones == [1000,3000] and args.valid_sentences == 5 and args.max_delta == .35
    args.resume = True
    command = launch.runner_command(args)
    assert command[:4] == [sys.executable,'-u','-m',launch.RUNNER_MODULE]
    assert '--resume' in command and '--supervise' not in command


@pytest.mark.parametrize('option', ['--smoke','--skip-quality-gates','--gate-override'])
def test_no_production_gate_bypass_flags(option):
    with pytest.raises(SystemExit): launch.parser().parse_args(['--dataset','d','--output','o',option])


@pytest.mark.parametrize('overrides', [
    {'ae_steps':0}, {'prior_steps':-1}, {'adapter_steps':0}, {'batch_size':0}, {'valid_sentences':0},
    {'max_delta':0}, {'max_delta':float('nan')}, {'milestones':[]}, {'milestones':[10,10,30]},
    {'milestones':[30,10]}, {'milestones':[10,29]}, {'milestones':[10,31]},
])
def test_invalid_protocol_fails_before_launch(tmp_path, overrides):
    args = arguments(tmp_path, **overrides)
    with pytest.raises(ValueError): launch.preflight(args)
    assert not args.output.exists()
