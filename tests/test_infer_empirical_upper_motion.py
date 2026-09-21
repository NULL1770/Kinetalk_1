import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kinetalk_b0.models.empirical_upper_motion import fit_empirical_bank
from scripts import infer_empirical_upper_motion as inference
from scripts.render_dynamic_rig_comparison import inspect_input


# Reuse the reference deployment fixture without executing its test suite.
spec = importlib.util.spec_from_file_location('reference_cli_fixture',
    Path(__file__).with_name('test_infer_reference_intensity_decoder.py'))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def fixture(tmp_path, length=40):
    reference_path, input_path, reference_checkpoint, values, *_ = fixture_module.fixture(tmp_path)
    time = torch.arange(length)[None, :, None]
    channel = torch.arange(9)[None, None]
    upper = .4+.1*torch.sin(time*.3+channel*.2+torch.arange(3)[:, None, None])
    bank = fit_empirical_bank(upper, torch.ones(3, length, dtype=torch.bool),
        torch.arange(3.)[:, None].expand(3, 64), torch.arange(3.)[:, None].expand(3, 128),
        ['speakerA', 'speakerB', 'speakerC'], ['sentenceA', 'sentenceB', 'sentenceC'], ['a', 'b', 'c'])
    protocol = {'schema': 'empirical_upper_motion_feasibility_v1', 'window': 5, 'top_k': 32,
                'reference_final_sha256': inference.reference._sha(reference_path), 'source_sha256': None}
    checkpoint = {'schema': 'empirical_upper_motion_v1', 'bank': bank, 'temperature': 1.,
                  'protocol': protocol, 'protocol_sha256': inference.reference._canonical_hash(protocol)}
    bank_path = tmp_path/'bank.pt'
    torch.save(checkpoint, bank_path)
    return bank_path, reference_path, input_path, checkpoint, values


def test_cli_writes_protected_renderer_output_donors_hashes_and_default_seed(tmp_path, monkeypatch):
    bank_path, checkpoint, source, _, values = fixture(tmp_path)
    output = tmp_path/'empirical.npz'
    monkeypatch.setattr('sys.argv', ['infer', '--bank', str(bank_path), '--checkpoint', str(checkpoint),
        '--input', str(source), '--output', str(output), '--speaker', 'speakerA', '--sentence', 'sentenceB'])
    inference.main()
    channels, times, valid, modes, display, _ = inspect_input(output, 25)
    assert modes == list(inference.MODE_NAMES) and display.shape == (5, len(times), 52)
    report = json.loads(output.with_suffix('.report.json').read_text(encoding='utf8'))
    assert report['seed'] == 42 and report['default_replaced'] is False
    assert report['bank_sha256'] == inference.reference._sha(bank_path)
    assert report['reference_final_sha256'] == inference.reference._sha(checkpoint)
    assert report['output_sha256'] == inference.reference._sha(output)
    assert report['exclusion']['unknown_metadata'] == []
    assert report['query_motion_read'] is False and report['query_target_read'] is False and report['query_emotion_read'] is False
    for runs in report['donors'].values():
        assert runs and all(record['donor_clip_id'] == 'c' for record in runs)
    assert all(check['passed'] for check in report['protection'].values())
    with np.load(output, allow_pickle=False) as generated:
        assert generated['empirical_seed'].item() == 42
        for motion in generated['motions']:
            np.testing.assert_array_equal(motion[:, inference.reference.NOT_UPPER].view(np.int32),
                                           values['prior'][:, inference.reference.NOT_UPPER].view(np.int32))
    with pytest.raises(FileExistsError):
        inference.main()


def test_unknown_metadata_reported_and_seed_repeatable(tmp_path):
    bank, checkpoint, source, *_ = fixture(tmp_path)
    first, report = inference.infer(bank, checkpoint, source, seed=17)
    second, report_second = inference.infer(bank, checkpoint, source, seed=17)
    np.testing.assert_array_equal(first['motions'].view(np.int32), second['motions'].view(np.int32))
    assert report['donors'] == report_second['donors']
    assert report['exclusion']['unknown_metadata'] == ['speaker', 'sentence']
    assert 'cannot enforce' in report['exclusion']['limitation']
    assert report['missing_donor_policy'].startswith('raise')


@pytest.mark.parametrize('change', ['hash', 'reference_binding', 'source_binding'])
def test_bank_protocol_and_reference_binding_enforced(tmp_path, change):
    bank, checkpoint, source, saved, _ = fixture(tmp_path)
    if change == 'hash':
        saved['protocol_sha256'] = 'tampered'
    elif change == 'reference_binding':
        saved['protocol']['reference_final_sha256'] = '0'*64
        saved['protocol_sha256'] = inference.reference._canonical_hash(saved['protocol'])
    else:
        saved['protocol']['source_sha256'] = 'different'
        saved['protocol_sha256'] = inference.reference._canonical_hash(saved['protocol'])
    torch.save(saved, bank)
    with pytest.raises(ValueError):
        inference.infer(bank, checkpoint, source)


def test_missing_donor_raises_before_cli_artifact_write(tmp_path, monkeypatch):
    bank, checkpoint, source, *_ = fixture(tmp_path, length=3)
    output = tmp_path/'cannot_sample.npz'
    monkeypatch.setattr('sys.argv', ['infer', '--bank', str(bank), '--checkpoint', str(checkpoint),
        '--input', str(source), '--output', str(output)])
    with pytest.raises(ValueError, match='No eligible native donor run'):
        inference.main()
    assert not output.exists() and not output.with_suffix('.report.json').exists()
