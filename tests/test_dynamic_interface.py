import copy
import torch

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.train_dynamic_interface import configure_trainable, dynamic_objective, GLOBAL_KEYS
from scripts.train_neutral_affect_audio_ablation import state_hash


def test_joint_interface_updates_local_but_preserves_global_and_zero_local():
    torch.set_num_threads(1)
    torch.manual_seed(46)
    cfg = {'data': {'content_dim': 8, 'motion_dim': 6, 'neutral_output_indices': [2, 3],
                    'emotion_classes': ['neutral', 'happy'], 'num_intensity_levels': 2, 'audio_emotion_dim': 5},
           'model': {'content_dim': 8, 'emotion_dim': 8, 'style_dim': 8, 'hidden_dim': 8,
                     'heads': 2, 'dropout': 0., 'dit_dim': 8, 'dit_depth': 1, 'residual_scale': .25,
                     'affect_stride': 4, 'affect_rank': 3, 'audio_control_refiner': True}}
    system = NeutralAffectSystem(cfg).eval()
    allowed = configure_trainable(system, True)
    frozen = lambda: {n: t for n, t in system.state_dict().items() if n not in allowed}
    before_hash = state_hash(frozen())
    mask = torch.ones(3, 12, dtype=torch.bool)
    q = {'motion': torch.randn(3, 12, 6), 'content': torch.randn(3, 12, 8),
         'audio': torch.randn(3, 12, 5), 'valid': mask, 'channel_mask': torch.ones(3, 6, dtype=torch.bool)}
    with torch.no_grad():
        base = system.base(q['content'], mask)
        identity = system.encode_identity(torch.randn(3, 2, 12, 6))
        teacher = system.encode_motion(q['motion'], mask)
        affect = system.encode_audio(q['audio'], mask)
        globals_before = {k: affect[k].clone() for k in GLOBAL_KEYS}
        zero = {**affect, 'local': torch.zeros_like(affect['local'])}
        noise = torch.randn_like(q['motion'])
        generated_before = system.generate(q['content'], mask, identity, zero, noise, 2, base=base)['motion']
    optimizer = torch.optim.Adam([p for p in system.parameters() if p.requires_grad], lr=.001)
    loss, _, _ = dynamic_objective(system, q, base, identity, teacher, torch.Generator().manual_seed(31), .5)
    loss.backward()
    assert system.audio_encoder.control_head.weight.grad.norm() > 0
    assert system.renderer.local_emotion.weight.grad.norm() > 0
    assert system.renderer.local_emotion.bias.grad is None
    optimizer.step()
    assert state_hash(frozen()) == before_hash
    with torch.no_grad():
        after = system.encode_audio(q['audio'], mask)
        for key in GLOBAL_KEYS:
            assert torch.equal(globals_before[key], after[key])
        assert not torch.equal(affect['local'], after['local'])
        generated_after = system.generate(q['content'], mask, identity, zero, noise, 2, base=base)['motion']
        assert torch.equal(generated_before, generated_after)
