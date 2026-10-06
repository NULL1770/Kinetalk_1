"""Strict source-statistics identity allowing only a verified location change."""
import copy
import hashlib
from pathlib import Path


def assert_same_flow_source(control, candidate):
    sources = [copy.deepcopy(x) for x in (control, candidate)]
    identities = []
    for source in sources:
        statistics = source['statistics']
        path = Path(statistics['path'])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == statistics['sha256'], f'Statistics bytes mismatch: {path}'
        identities.append({'path': str(path), 'sha256': actual})
        del statistics['path']
    assert sources[0] == sources[1], 'Flow source differs beyond statistics location'
    return identities
