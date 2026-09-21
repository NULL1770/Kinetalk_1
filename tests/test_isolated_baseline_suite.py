import json
from pathlib import Path
import sys

import pytest

import scripts.run_isolated_baseline_suite as suite


def harness(tmp_path,monkeypatch,*,kinetalk_status='stopped_quality_gate',exit_codes=None,
            inventory_sequence=None,mutate=None):
    data=tmp_path/'data';data.mkdir();(data/'index.json').write_text('{}')
    source=tmp_path/'source.pt';source.write_bytes(b'checkpoint')
    output=tmp_path/'suite';calls=[]
    monkeypatch.setattr(sys,'argv',['run_isolated_baseline_suite','--data',str(data),
        '--source',str(source),'--output',str(output)])
    original={'scripts/runner.py':'fixed'}
    inventories=iter(inventory_sequence or [original]*8)
    last=original
    def inventory(_):
        nonlocal last
        last=next(inventories,last)
        return last
    monkeypatch.setattr(suite,'source_inventory',inventory)
    class Process:
        def __init__(self,command,**kwargs):
            self.command=command;self.pid=100+len(calls);calls.append(command)
            self.stage='kinetalk' if 'run_isolated_state_queue.py' in command[2] else 'facediffuser'
            self.destination=Path(command[command.index('--output')+1])
            self.destination.mkdir(parents=True)
        def wait(self):
            if mutate is not None:mutate(self.stage,source,data)
            result={'status':kinetalk_status if self.stage=='kinetalk' else 'complete','test_loaded':False}
            (self.destination/'status.json').write_text(json.dumps(result))
            return (exit_codes or {}).get(self.stage,0)
    monkeypatch.setattr(suite.subprocess,'Popen',Process)
    return calls,output,original


def test_state_quality_stop_skips_residual_inside_queue_but_still_launches_baseline(tmp_path,monkeypatch):
    calls,output,_=harness(tmp_path,monkeypatch)
    suite.main()
    assert len(calls)==2
    assert calls[1][2].endswith('train_facediffuser_arkit.py')
    assert calls[1][calls[1].index('--epochs')+1]=='100'
    status=json.loads((output/'status.json').read_text())
    assert status['status']=='complete'
    assert status['quality_pass_claimed'] is False and status['review_required'] is True
    assert json.loads((output/'kinetalk/status.json').read_text())['status']=='stopped_quality_gate'


def test_kinetalk_process_error_stops_without_running_baseline(tmp_path,monkeypatch):
    calls,output,_=harness(tmp_path,monkeypatch,exit_codes={'kinetalk':7})
    with pytest.raises(RuntimeError,match='kinetalk failed'):suite.main()
    assert len(calls)==1
    status=json.loads((output/'status.json').read_text())
    assert status['status']=='failed' and status['records']==[{'stage':'kinetalk','exit_code':7}]


def test_zero_exit_without_final_child_status_stops_suite(tmp_path,monkeypatch):
    calls,output,_=harness(tmp_path,monkeypatch,kinetalk_status='running')
    with pytest.raises(RuntimeError,match='final status'):suite.main()
    assert len(calls)==1
    assert json.loads((output/'status.json').read_text())['status']=='failed'


def test_code_change_before_next_job_rejected(tmp_path,monkeypatch):
    fixed={'scripts/runner.py':'fixed'};changed={'scripts/runner.py':'changed'}
    calls,output,_=harness(tmp_path,monkeypatch,inventory_sequence=[fixed,fixed,changed])
    with pytest.raises(RuntimeError,match='code changed'):suite.main()
    assert len(calls)==1
    assert json.loads((output/'status.json').read_text())['status']=='failed'


def test_code_change_during_final_job_rejected(tmp_path,monkeypatch):
    fixed={'scripts/runner.py':'fixed'};changed={'scripts/runner.py':'changed'}
    calls,output,_=harness(tmp_path,monkeypatch,inventory_sequence=[fixed,fixed,fixed,fixed,changed])
    with pytest.raises(RuntimeError,match='code changed'):suite.main()
    assert len(calls)==2
    assert json.loads((output/'status.json').read_text())['status']=='failed'


@pytest.mark.parametrize('which',['source','index'])
def test_changed_source_or_data_index_rejected_before_baseline(tmp_path,monkeypatch,which):
    def mutate(stage,source,data):
        if stage=='kinetalk':
            path=source if which=='source' else data/'index.json'
            path.write_bytes(b'changed')
    calls,output,_=harness(tmp_path,monkeypatch,mutate=mutate)
    with pytest.raises(RuntimeError):suite.main()
    assert len(calls)==1
    assert json.loads((output/'status.json').read_text())['status']=='failed'
