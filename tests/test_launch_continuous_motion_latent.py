"""Launcher checks use CPU fake children, never an actual training or CUDA job."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import launch_continuous_motion_latent as launch


def arguments(tmp_path, **kwargs):
    dataset = tmp_path / 'dataset.pt'; dataset.write_bytes(b'fake dataset')
    values = dict(dataset=dataset, output=tmp_path / 'run', code_root=tmp_path,
                  device='cpu', ae_steps=40, prior_steps=80, adapt_steps=60,
                  batch_size=4, resume=False, check_only=False, supervise=False)
    values.update(kwargs)
    return SimpleNamespace(**values)


def child_command(output, state='complete', exit_code=0, *, write=True):
    script = 'import json,os,pathlib,sys; '
    if write:
        script += ('pathlib.Path(sys.argv[1]).write_text(json.dumps({"state":sys.argv[2],'
                   '"reason":"quality gate"}),encoding="utf8"); ')
    script += 'print("fake-child",os.environ.get("OMP_NUM_THREADS")); sys.exit(int(sys.argv[3]))'
    return [sys.executable, '-c', script, str(output / 'status.json'), state, str(exit_code)]


@pytest.mark.parametrize('runner_state,exit_code,expected,supervisor_code', [
    ('complete', 0, 'complete', 0), ('stopped', 0, 'stopped', 0),
    ('failed', 0, 'failed', 1), ('complete', 7, 'failed', 7),
    ('running', 0, 'failed', 1), ('stopped', 3, 'failed', 3),
])
def test_supervisor_uses_actual_exit_and_runner_state(tmp_path, runner_state, exit_code, expected, supervisor_code):
    args = arguments(tmp_path)
    code = launch.supervise(args, command=child_command(args.output, runner_state, exit_code))
    status = json.loads((args.output / 'pipeline_status.json').read_text())
    assert code == supervisor_code and status['state'] == expected
    assert status['exit_code'] == exit_code and status['runner_state'] == runner_state
    assert status['pid'] > 0 and status['child_pid'] > 0
    assert status['end'] >= status['start'] and status['elapsed_seconds'] >= 0
    assert 'fake-child 4' in (args.output / 'training.log').read_text()
    assert not list(args.output.glob('*.tmp'))


@pytest.mark.parametrize('stale', [False, True])
def test_missing_or_stale_runner_status_is_failure_even_on_zero_exit(tmp_path, stale):
    args = arguments(tmp_path, resume=stale)
    args.output.mkdir()
    if stale:
        (args.output / 'status.json').write_text('{"state":"complete"}')
    code = launch.supervise(args, command=child_command(args.output, write=False))
    status = json.loads((args.output / 'pipeline_status.json').read_text())
    assert code == 1 and status['state'] == 'failed'
    assert 'fresh status' in status['reason']


def test_supervisor_records_spawn_exception(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    def fail(*_args, **_kwargs): raise OSError('fake spawn failure')
    monkeypatch.setattr(launch.subprocess, 'Popen', fail)
    assert launch.supervise(args) == 1
    status = json.loads((args.output / 'pipeline_status.json').read_text())
    assert status['state'] == 'failed' and 'fake spawn failure' in status['reason']


def test_linux_proc_pid_reuse_and_exact_output_token(tmp_path):
    proc = tmp_path / 'proc'; record = proc / '102'; record.mkdir(parents=True)
    output = tmp_path / 'run'
    def cmd(tokens): (record / 'cmdline').write_bytes(b'\0'.join(str(x).encode() for x in tokens) + b'\0')
    cmd(['python', '-m', launch.RUNNER_MODULE, '--output', output])
    assert launch.process_matches_output(102, output, proc)
    cmd(['python', '-m', 'unrelated.module', '--output', output])
    assert not launch.process_matches_output(102, output, proc)
    cmd(['python', '-m', launch.LAUNCHER_MODULE, '--output', str(output) + '_different'])
    assert not launch.process_matches_output(102, output, proc)
    cmd(['python', '-m', launch.RUNNER_MODULE, '--note', str(output)])
    assert not launch.process_matches_output(102, output, proc)
    cmd(['python', '-m', launch.RUNNER_MODULE, '--output=' + str(output)])
    assert launch.process_matches_output(102, output, proc)
    assert not launch.process_matches_output(999, output, proc)


def test_preflight_check_only_does_not_create_output(tmp_path, monkeypatch):
    args = arguments(tmp_path, check_only=True, device='cuda')
    called = []
    monkeypatch.setattr(launch, 'require_cuda', lambda device: called.append(device) or {'cuda_checked': True})
    result = launch.launch(args)
    assert result['state'] == 'checked' and result['started'] is False
    assert called == ['cuda'] and not args.output.exists()


def test_preflight_fails_without_disk_or_cuda_before_creating_output(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(launch.shutil, 'disk_usage', lambda _: SimpleNamespace(free=launch.MIN_FREE_BYTES - 1))
    with pytest.raises(RuntimeError, match='1 GiB'): launch.preflight(args)
    assert not args.output.exists()
    monkeypatch.setattr(launch.shutil, 'disk_usage', lambda _: SimpleNamespace(free=2 * launch.MIN_FREE_BYTES))
    def unavailable(_): raise RuntimeError('CUDA is unavailable')
    monkeypatch.setattr(launch, 'require_cuda', unavailable)
    with pytest.raises(RuntimeError, match='CUDA'): launch.preflight(args)
    assert not args.output.exists()


def test_resume_is_explicit_and_live_ownership_blocks_duplicate(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.output.mkdir()
    with pytest.raises(FileExistsError, match='resume'): launch.preflight(args)
    args.resume = True
    launch.write_json(args.output / 'launch.json', {'pid': 12})
    launch.write_json(args.output / 'pipeline_status.json', {'pid': 12, 'child_pid': 13})
    monkeypatch.setattr(launch, 'process_matches_output', lambda pid, _: pid == 13)
    with pytest.raises(RuntimeError, match='still alive'): launch.preflight(args)
    monkeypatch.setattr(launch, 'process_matches_output', lambda *_: False)
    assert launch.preflight(args)['resume'] is True


def test_detached_launch_preserves_args_and_records_pid(tmp_path, monkeypatch):
    args = arguments(tmp_path, device='cuda:0')
    monkeypatch.setattr(launch, 'require_cuda', lambda _: {'cuda_checked': True})
    calls = []
    def fake_popen(command, **kwargs):
        calls.append((command, kwargs)); return SimpleNamespace(pid=789)
    monkeypatch.setattr(launch.subprocess, 'Popen', fake_popen)
    result = launch.launch(args)
    command, kwargs = calls[0]
    assert result['pid'] == 789 and result['state'] == 'launched'
    assert '--supervise' in command and command[command.index('--ae-steps')+1] == '40'
    assert command[command.index('--device')+1] == 'cuda:0'
    assert kwargs['start_new_session'] is True and kwargs['stdin'] == subprocess.DEVNULL
    assert kwargs['env']['MKL_NUM_THREADS'] == '4'
    record = json.loads((args.output / 'launch.json').read_text())
    assert record['pid'] == 789 and record['completion_not_yet_verified'] is True
    assert not (args.output / 'pipeline_status.json').exists()


def test_runner_command_and_parser_defaults_match_protocol(tmp_path):
    args = launch.parser().parse_args(['--dataset', str(tmp_path/'d.pt'), '--output', str(tmp_path/'out')])
    assert (args.ae_steps, args.prior_steps, args.adapt_steps, args.batch_size) == (4000, 8000, 6000, 24)
    args.resume = True
    command = launch.runner_command(args)
    assert command[:4] == [sys.executable, '-u', '-m', launch.RUNNER_MODULE]
    assert '--resume' in command and '--supervise' not in command
