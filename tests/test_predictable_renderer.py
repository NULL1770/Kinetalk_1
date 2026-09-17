"""Small contracts for the isolated experiment, not evidence of real quality."""
import pytest
import torch
from torch.nn import functional as F

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.predictable_motion import fit_predictable_motion, predict_controls, motion_target
from scripts.train_predictable_renderer import (
    PredictableAudioHead, cached_flow, center, configure_trainable,
    frozen_state, projected_affect, audio_activity_gate,
)
from scripts.train_neutral_affect_audio_ablation import state_hash


@pytest.fixture(autouse=True)
def seed():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(11)
    yield
    torch.set_num_threads(old)


def tiny_system():
    return NeutralAffectSystem({
        'data': {'content_dim': 8, 'motion_dim': 6, 'neutral_output_indices': [2, 3],
                 'emotion_classes': ['neutral', 'happy', 'sad'], 'num_intensity_levels': 3,
                 'audio_emotion_dim': 5},
        'model': {'content_dim': 8, 'emotion_dim': 8, 'style_dim': 8, 'hidden_dim': 8,
                  'heads': 2, 'dropout': 0., 'dit_dim': 8, 'dit_depth': 1,
                  'residual_scale': .25, 'affect_stride': 4, 'affect_rank': 3,
                  'global_condition_dropout': 0., 'style_condition_dropout': 0.}}).eval()


def test_head_initialization_reproduces_ridge_and_keeps_gradients():
    x, y = torch.randn(5, 4, 9), torch.randn(5, 4, 6)
    w = torch.tensor([[4., 4., 4., 1.]]).expand(5, -1).clone()
    state = fit_predictable_motion(x, y, w, [0, 1, 2, 3], alpha=1., rank=3)
    state['motion_channel_indices'] = list(range(6))
    head = PredictableAudioHead(state, y[:4], w[:4])
    prediction = head(x, w)
    expected = predict_controls(x, w, state) / head.target_scale.double()
    torch.testing.assert_close(prediction.double(), expected, atol=2e-6, rtol=2e-5)
    truth = head.teacher(y, w)
    torch.testing.assert_close(truth.double(), motion_target(y, w, state) / head.target_scale.double(), atol=2e-6, rtol=2e-5)
    (prediction - truth).square().mean().backward()
    assert head.linear.weight.grad.abs().sum() > 0
    assert all(not b.requires_grad for b in head.buffers())
    # Appended invalid/NaN bins cannot change valid controls or their gradient.
    padded = head(F.pad(x, (0, 0, 0, 2), value=float('nan')), F.pad(w, (0, 2)))
    torch.testing.assert_close(padded[:, :4], prediction)
    assert padded[:, 4:].count_nonzero() == 0


def test_flow_uses_cached_base_and_only_allowed_modules_receive_gradients(monkeypatch):
    system = tiny_system()
    valid = torch.ones(2, 12, dtype=torch.bool)
    content = torch.randn(2, 12, 8)
    with torch.no_grad():
        base = system.base(content, valid)
        identity = system.encode_identity(torch.randn(2, 2, 12, 6))
        affect = system.encode_audio(torch.randn(2, 12, 5), valid)
    batch = {'q': {'content': content.half().float(), 'motion': torch.randn(2, 12, 6), 'valid': valid},
             'base': base, 'identity': identity, 'affect': affect}
    before = state_hash(frozen_state(system))
    configure_trainable(system, seed=46)
    monkeypatch.setattr(system, 'base', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Recomputed cached B0')))
    controls = torch.randn(2, 3, 3, requires_grad=True)
    out = cached_flow(system, batch, controls, torch.full((2, 3), 4.), torch.zeros(2, 12, 6), torch.zeros(2))
    loss = (out['prediction'] - out['velocity_target']).square().mean()
    loss.backward()
    assert controls.grad.abs().sum() > 0
    assert system.local_projection.weight.grad.abs().sum() > 0
    for name, parameter in system.named_parameters():
        if parameter.grad is not None:
            assert name.startswith(('renderer.', 'local_projection.'))
    assert state_hash(frozen_state(system)) == before
    torch.testing.assert_close(out['target_residual'], batch['q']['motion'] - base['b0'] - identity['baseline'][:, None])


def test_zero_condition_ignores_controls_and_preserves_global():
    system = tiny_system()
    valid = torch.tensor([[True] * 9 + [False] * 3])
    affect = system.encode_audio(torch.randn(1, 12, 5), valid)
    batch = {'q': {'valid': valid}, 'affect': affect}
    w = torch.tensor([[4., 4., 1.]])
    first = projected_affect(system, batch, torch.randn(1, 3, 3), w, zero=True)
    second = projected_affect(system, batch, torch.full((1, 3, 3), float('nan')), w, zero=True)
    assert first['local'].count_nonzero() == 0
    torch.testing.assert_close(first['local'], second['local'], rtol=0, atol=0)
    for key in ('global', 'intensity_value'):
        torch.testing.assert_close(first[key], affect[key], rtol=0, atol=0)
    configure_trainable(system, seed=46, zero=True)
    assert all(not p.requires_grad for p in system.local_projection.parameters())


def test_differentiable_center_ignores_nan_padding():
    x = torch.tensor([[[1.], [3.], [float('nan')]]], requires_grad=True)
    w = torch.tensor([[3., 1., 0.]])
    result = center(x, w)
    torch.testing.assert_close(result, torch.tensor([[[-.5], [1.5], [0.]]]))
    result.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0, 2].count_nonzero() == 0


def test_projection_only_preserves_zero_local_decoder_after_update():
    system=tiny_system()
    valid=torch.ones(2,12,dtype=torch.bool)
    content=torch.randn(2,12,8)
    with torch.no_grad():
        base=system.base(content,valid)
        identity=system.encode_identity(torch.randn(2,2,12,6))
        affect=system.encode_audio(torch.randn(2,12,5),valid)
    batch={'q':{'content':content,'motion':torch.randn(2,12,6),'valid':valid},
           'base':base,'identity':identity,'affect':affect}
    controls=torch.randn(2,3,3);w=torch.full((2,3),4.)
    configure_trainable(system,seed=46,projection_only=True)
    frozen=state_hash(system.renderer.state_dict())
    zero=projected_affect(system,batch,controls,w,zero=True)
    noise=torch.randn(2,12,6)
    with torch.no_grad():before=system.generate(content,valid,identity,zero,noise,steps=2,base=base)['motion']
    opt=torch.optim.Adam(system.local_projection.parameters(),lr=.01)
    out=cached_flow(system,batch,controls,w,noise,torch.zeros(2))
    opt.zero_grad();out['prediction'].square().mean().backward();opt.step()
    assert system.local_projection.weight.count_nonzero()>0
    assert state_hash(system.renderer.state_dict())==frozen
    with torch.no_grad():after=system.generate(content,valid,identity,zero,noise,steps=2,base=base)['motion']
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    assert all(n=='local_projection.weight' for n,p in system.named_parameters() if p.requires_grad)


def test_audio_gate_shared_linear_equivalence_and_no_label_dependency():
    system=tiny_system();valid=torch.ones(2,12,dtype=torch.bool)
    affect=system.encode_audio(torch.randn(2,12,5),valid)
    affect['emotion_logits']=torch.tensor([[6.,0.,0.],[-6.,0.,0.]],requires_grad=True)
    batch={'q':{'valid':valid,'emotion_id':torch.tensor([2,0])},'affect':affect}
    controls=torch.randn(2,3,3,requires_grad=True);w=torch.full((2,3),4.)
    original=projected_affect(system,batch,controls,w)
    gated=projected_affect(system,batch,controls,w,audio_gate=True)
    g=audio_activity_gate(affect)
    torch.testing.assert_close(gated['local'],original['local']*g[:,None,None])
    assert g[0]<.01 and g[1]>.99
    batch['q']['emotion_id']=torch.tensor([0,2])
    torch.testing.assert_close(projected_affect(system,batch,controls,w,audio_gate=True)['local'],gated['local'])
    gated['local'].square().sum().backward()
    assert affect['emotion_logits'].grad is None
    assert controls.grad is not None
