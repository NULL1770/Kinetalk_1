import numpy as np
import torch
from scripts.packed_trainval_cache import _store_flat, _open_split
from scripts.train_full_staged import subset, cache_current_base, evaluate
from argparse import Namespace


def packed(tmp_path):
    clips = []
    for i, n in enumerate((5, 8, 3)):
        clips.append(dict(audio_features=torch.randn(n, 1540), motion=torch.randn(n, 52),
            times=torch.arange(n, dtype=torch.float64) / 25 + 7,
            valid=torch.ones(n, dtype=torch.bool), motion_valid=torch.ones(n, dtype=torch.bool),
            channel_mask=torch.ones(52, dtype=torch.bool), anchors=torch.zeros(52),
            anchor_valid=torch.ones(52, dtype=torch.bool), speaker_id=torch.tensor(0),
            emotion_id=torch.tensor(0), intensity_id=torch.tensor(0),
            intensity_valid=torch.tensor(True), dataset_id=torch.tensor(0),
            clip_id=f'clip{i}', sentence_id=f's{i}', speaker='p'))
    info = _store_flat(tmp_path, 'validation', clips)
    return _open_split(tmp_path, 'validation', info), clips


def test_packed_batch_native_values_metadata_clock_and_disk_unchanged(tmp_path):
    split, clips = packed(tmp_path)
    split.set_extra('b0', [torch.zeros(len(c['valid']), 52) for c in clips])
    batch = subset(split, torch.tensor([2, 0]), 'cpu')
    assert batch['clip_id'] == ['clip2', 'clip0']
    assert batch['audio_features'].shape == (2, 5, 1540)
    assert batch['times'].dtype == torch.float64
    assert (batch['times'][:, 1:] > batch['times'][:, :-1]).all()
    assert not batch['valid'][0, 3:].any()
    torch.testing.assert_close(batch['motion'][0, :3], clips[2]['motion'])
    minimal = subset(split, torch.tensor([2]), 'cpu', keys=('motion', 'valid', 'clip_id'))
    assert set(minimal) == {'motion', 'valid', 'clip_id'}
    split['_motion_flat'][0, 0] = 123
    assert np.load(tmp_path / 'validation' / 'motion.npy')[0, 0] != 123


def test_staged_base_cache_and_evaluate_support_different_batch_lengths(tmp_path):
    split, clips = packed(tmp_path)
    class System:
        def stage1(self, content, valid):
            return {'b0': content[..., :52], 'h0': content[..., :8]}
    system = System()
    data = {'splits': {'validation': split}, 'refs': {}}
    cache_current_base(system, data, 'cpu', batch_size=2)
    identities = {0: {'code': torch.zeros(1, 8), 'baseline': torch.zeros(1, 52)}}
    args = Namespace(batch_size=2, device='cpu', decode_steps=1, paper_data=tmp_path)
    report, curves = evaluate(system, None, data=data, identities=identities,
                              stage='articulation', args=args, full=True)
    assert report['clips'] == 3
    assert curves['clip_id'] == [c['clip_id'] for c in clips]
    assert curves['predictions']['42/full'].shape == (3, 8, 52)
    assert curves['times'].dtype == torch.float64
    assert not curves['predictions']['42/full'][2, 3:].any()
