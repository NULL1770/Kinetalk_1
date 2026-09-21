"""Early stage reports must not depend on the untrained motion renderer."""
from types import SimpleNamespace

import pytest
import torch

from scripts.train_full_staged import evaluate


@pytest.mark.parametrize('stage', ['articulation', 'identity'])
def test_early_stage_evaluation_never_calls_teacher_audio_or_renderer(stage):
    class Untrained:
        def encode_motion(self, *args, **kwargs):
            raise AssertionError('Motion teacher has not been trained')

        def generate(self, *args, **kwargs):
            raise AssertionError('Renderer has not been trained')

        def __call__(self, *args, **kwargs):
            raise AssertionError('Audio student has not been trained')

    rng = torch.Generator().manual_seed(3)
    valid = torch.tensor([[True, True, False, True, True]])
    b0 = torch.where(valid[..., None], torch.rand(1, 5, 52, generator=rng), 0.)
    bias = torch.linspace(-.03, .03, 52)[None]
    q = {'clip_id': ['query'], 'speaker_id': torch.tensor([4]),
         'content': torch.zeros(1, 5, 8), 'audio_features': torch.zeros(1, 5, 10),
         'h0': torch.zeros(1, 5, 8), 'b0': b0, 'valid': valid,
         'motion': torch.rand(1, 5, 52, generator=rng),
         'times': torch.arange(5, dtype=torch.float64)[None] / 25,
         'channel_mask': torch.ones(1, 52, dtype=torch.bool), 'emotion_id': torch.tensor([0])}
    identities = {4: {'code': torch.zeros(1, 8), 'baseline': bias}}
    unavailable = Untrained()
    report, curves = evaluate(unavailable, unavailable, unavailable, unavailable,
        {'splits': {'validation': q}}, identities, stage,
        SimpleNamespace(batch_size=1, device='cpu', decode_steps=2, paper_data=None), full=True)
    expected = b0 if stage == 'articulation' else torch.where(valid[..., None], b0 + bias[:, None], 0.)
    for pred in curves['predictions'].values():
        torch.testing.assert_close(pred, expected, rtol=0, atol=0)
    assert report['audio_or_teacher_emotion_accuracy'] is None
    assert report['motion_teacher_emotion_accuracy'] is None
    assert all(v['generated_teacher_emotion_accuracy_nonindependent'] is None for v in report['modes'].values())
    assert report['condition_source'] == ('audio_content_B0' if stage == 'articulation' else 'audio_content_B0_plus_neutral_identity')
