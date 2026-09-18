"""Reject review metadata and nuisance diagnostics from another tracking run."""
import json

import pytest

from scripts import package_brow_states_v3 as p


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf8')


def manifest(root):
    write_json(root/'manifest.json', {
        path.name: {'sha256': p.sha(path), 'bytes': path.stat().st_size}
        for path in root.iterdir() if path.is_file() and path.name != 'manifest.json'
    })


def source_fixture(tmp_path):
    tracking, nuisance = tmp_path/'tracking', tmp_path/'nuisance'
    tracking.mkdir(); nuisance.mkdir()
    write_json(tracking/'selection.json', {'clips': [{'clip_id': 'clip_a'}]})
    write_json(tracking/'provenance.json', {
        'selection_sha256': p.sha(tracking/'selection.json'), 'ffmpeg_sha256': 'decoder',
    })
    manifest(tracking)
    write_json(nuisance/'protocol.json', {
        'source_manifest_sha256': p.sha(tracking/'manifest.json'),
        'selection_sha256': p.sha(tracking/'selection.json'),
        'source_provenance_sha256': p.sha(tracking/'provenance.json'),
    })
    manifest(nuisance)
    return tracking, nuisance


def test_matching_metadata_accepted(tmp_path):
    tracking, nuisance = source_fixture(tmp_path)
    tm, nm, selection, provenance = p.checked_source_metadata(tracking, nuisance)
    assert set(tm) == {'selection.json', 'provenance.json'}
    assert set(nm) == {'protocol.json'}
    assert selection['clips'][0]['clip_id'] == 'clip_a'
    assert provenance['selection_sha256'] == p.sha(tracking/'selection.json')


@pytest.mark.parametrize('root_name,name', [
    ('tracking', 'selection.json'), ('tracking', 'provenance.json'), ('nuisance', 'protocol.json'),
])
@pytest.mark.parametrize('corruption', ['unmanifested', 'changed'])
def test_metadata_requires_manifest_entry_and_matching_content(tmp_path, root_name, name, corruption):
    tracking, nuisance = source_fixture(tmp_path)
    root = tracking if root_name == 'tracking' else nuisance
    if corruption == 'unmanifested':
        records = json.loads((root/'manifest.json').read_text(encoding='utf8'))
        del records[name]
        write_json(root/'manifest.json', records)
    else:
        with (root/name).open('a', encoding='utf8') as handle:
            handle.write(' ')
    with pytest.raises(ValueError, match='manifest|hash/size mismatch'):
        p.checked_source_metadata(tracking, nuisance)


@pytest.mark.parametrize('field', [
    'source_manifest_sha256', 'selection_sha256', 'source_provenance_sha256',
])
def test_internally_consistent_nuisance_from_other_tracking_run_rejected(tmp_path, field):
    tracking, nuisance = source_fixture(tmp_path)
    protocol = json.loads((nuisance/'protocol.json').read_text(encoding='utf8'))
    protocol[field] = '0'*64
    write_json(nuisance/'protocol.json', protocol)
    manifest(nuisance)
    with pytest.raises(ValueError, match='nuisance tracking source mismatch: '+field):
        p.checked_source_metadata(tracking, nuisance)


def test_tracking_provenance_must_refer_to_selected_clips(tmp_path):
    tracking, nuisance = source_fixture(tmp_path)
    provenance = json.loads((tracking/'provenance.json').read_text(encoding='utf8'))
    provenance['selection_sha256'] = '0'*64
    write_json(tracking/'provenance.json', provenance)
    manifest(tracking)
    with pytest.raises(ValueError, match='tracking provenance selection mismatch'):
        p.checked_source_metadata(tracking, nuisance)


def test_package_rejects_source_mismatch_before_creating_review(tmp_path):
    tracking, nuisance = source_fixture(tmp_path)
    protocol = json.loads((nuisance/'protocol.json').read_text(encoding='utf8'))
    protocol['source_manifest_sha256'] = '0'*64
    write_json(nuisance/'protocol.json', protocol)
    manifest(nuisance)
    states = tmp_path/'states'; states.mkdir()
    write_json(states/'clips.json', [])
    manifest(states)
    audit = tmp_path/'audit.json'
    write_json(audit, {'schema': p.SCHEMA+'_audit', 'source_manifest_sha256': p.sha(states/'manifest.json')})
    output = tmp_path/'review'
    with pytest.raises(ValueError, match='nuisance tracking source mismatch'):
        p.package(states, audit, tracking, nuisance, tmp_path/'nonexistent_ffmpeg', output)
    assert not output.exists()
