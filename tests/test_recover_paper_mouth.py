import copy
import json
import sys

import pytest
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts import recover_paper_mouth as recovery
from tests.test_full_staged_runner import config


def test_input_interventions_preserve_native_clock_and_ignore_unknown_payloads():
    content = torch.tensor([[[1.], [float('nan')], [3.], [9.], [float('inf')]]])
    valid = torch.tensor([[True, False, True, True, False]])
    real = recovery.intervene(content, valid, 'real')
    static = recovery.intervene(content, valid, 'static')
    reverse = recovery.intervene(content, valid, 'reverse')
    torch.testing.assert_close(real, torch.tensor([[[1.], [0.], [3.], [9.], [0.]]]))
    torch.testing.assert_close(reverse, torch.tensor([[[9.], [0.], [3.], [1.], [0.]]]))
    torch.testing.assert_close(static[valid], torch.full((3, 1), 13 / 3))
    assert static[~valid].count_nonzero() == 0
    assert real.shape == static.shape == reverse.shape == content.shape
    assert torch.isnan(content[0, 1, 0])
    with pytest.raises(ValueError, match='intervention'):
        recovery.intervene(content, valid, 'shift')


def tiny_data():
    cfg = config()
    cfg['data']['neutral_output_indices'] = list(recovery.MOUTH)
    torch.manual_seed(753)
    system = NeutralAffectSystem(cfg).eval()
    rng = torch.Generator().manual_seed(127)
    def partition(prefix):
        valid = torch.ones(4, 9, dtype=torch.bool)
        valid[1, 3] = False; valid[2, 7:] = False
        motion = torch.rand(4, 9, 52, generator=rng) * .5
        channel_mask = torch.ones(4, 52, dtype=torch.bool); channel_mask[:, 51] = False
        return {'motion': motion, 'content': torch.randn(4, 9, 8, generator=rng),
                'valid': valid, 'channel_mask': channel_mask,
                'emotion_id': torch.tensor([0, 0, 1, 1]),
                'clip_id': [prefix + str(i) for i in range(4)],
                'sentence_id': ['sentence_' + str(i % 2) for i in range(4)],
                'times': torch.arange(9, dtype=torch.float64)[None].expand(4, -1) / 25}
    return {'system': system, 'config': cfg,
            'splits': {'train': partition('fit_'), 'validation': partition('val_')},
            'target_scales': torch.ones(52) * .2,
            'provenance': {'manifest_sha256': 'synthetic_manifest_only'}}


def test_recovery_smoke_checkpoint_resume_and_no_downstream_training(tmp_path, monkeypatch):
    data = tiny_data()
    source = tmp_path / 'source.pt'
    torch.save({'system': data['system'].state_dict(), 'stage': 'articulation',
                'config': copy.deepcopy(data['config']), 'completed_epochs': 12,
                'data_manifest_sha256': data['provenance']['manifest_sha256']}, source)
    monkeypatch.setattr(recovery, 'load_paper_data', lambda *a, **kw: copy.deepcopy(data))
    monkeypatch.setattr(recovery.subprocess, 'call', lambda *a, **kw: pytest.fail('Smoke must never launch downstream training'))
    output = tmp_path / 'recovery'
    argv = ['recover_paper_mouth', '--data', str(tmp_path / 'unused'), '--source', str(source),
            '--output', str(output), '--device', 'cpu', '--batch-size', '2', '--smoke', '--continue-protected']
    previous_threads = torch.get_num_threads()
    try:
        monkeypatch.setattr(sys, 'argv', argv)
        recovery.main()
        first = torch.load(output / 'final.pt', map_location='cpu', weights_only=False)
        first_digest = recovery.sha(output / 'final.pt')
        protocol = json.loads((output / 'protocol.json').read_text())
        baseline = json.loads((output / 'baseline.json').read_text())
        assert protocol['train_clips'] == protocol['validation_clips'] == 4
        assert protocol['epochs'] == 1
        assert first['completed_epochs'] == 13
        assert first['recovery_passed'] is False
        assert first['config']['model']['motion_support'][-1] is False
        assert not any(first['config']['model']['residual_support'][i] for i in recovery.MOUTH)
        changed = [k for k in first['system'] if not torch.equal(first['system'][k], data['system'].state_dict()[k])]
        assert changed and all(k.startswith('stage1.') for k in changed)
        assert json.loads((output / 'status.json').read_text())['status'] == 'gate_rejected'
        monkeypatch.setattr(sys, 'argv', [*argv, '--resume'])
        recovery.main()
        second = torch.load(output / 'final.pt', map_location='cpu', weights_only=False)
        for key in first['system']:
            torch.testing.assert_close(first['system'][key], second['system'][key], rtol=0, atol=0)
        assert json.loads((output / 'baseline.json').read_text()) == baseline
        assert recovery.sha(output / 'final.pt') == first_digest
    finally:
        torch.set_num_threads(previous_threads)


def test_paired_controls_use_sentence_clusters_and_correct_direction():
    data = tiny_data(); q = data['splits']['validation']
    truth = q['motion']; inaccurate = truth.flip(1)
    interval = recovery.paired_interval(truth, inaccurate, q)
    assert interval['passed']
    assert interval['clusters'] == 2
    assert interval['clips'] == 4
    assert interval['real_minus_control'] < 0
    assert interval['sentence_cluster_95ci'][1] < 0
