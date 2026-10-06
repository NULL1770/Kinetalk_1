"""Reject changed recipe values, unbound bytes and false path equivalence."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

helper = Path(__file__).resolve().parents[1] / '.codex-finalizer/phase29_report_contract.py'
spec = importlib.util.spec_from_file_location('phase29_report_contract', helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pair(tmp_path):
    paths = [tmp_path / name for name in ('control.json', 'candidate.json')]
    for path in paths:
        path.write_bytes(b'{"std":[0.25,0.75]}')
    checksum = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    return [{'mode': 'train-residual-std', 'normalized_std': [1, 3],
             'statistics': {'path': str(path), 'sha256': checksum, 'train_only': True}}
            for path in paths]


def test_equivalent_copy_and_no_mutation(tmp_path):
    sources = pair(tmp_path)
    original = copy.deepcopy(sources)
    identities = module.assert_same_flow_source(*sources)
    assert sources == original
    assert identities[0]['sha256'] == identities[1]['sha256']
    assert identities[0]['path'] != identities[1]['path']


@pytest.mark.parametrize('field', ['mode', 'normalized_std', 'statistics_metadata'])
def test_recipe_change_rejected(tmp_path, field):
    sources = pair(tmp_path)
    if field == 'statistics_metadata':
        sources[1]['statistics']['train_only'] = False
    else:
        sources[1][field] = 'changed'
    with pytest.raises(AssertionError):
        module.assert_same_flow_source(*sources)


@pytest.mark.parametrize('update_record', [False, True])
def test_changed_bytes_rejected_even_when_recorded(tmp_path, update_record):
    sources = pair(tmp_path)
    Path(sources[1]['statistics']['path']).write_bytes(b'changed')
    if update_record:
        sources[1]['statistics']['sha256'] = hashlib.sha256(b'changed').hexdigest()
    with pytest.raises(AssertionError):
        module.assert_same_flow_source(*sources)
