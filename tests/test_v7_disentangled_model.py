import torch

from kinetalk_v7.disentangled_model import V7Config, V7DisentangledModel, v7_loss, _valid_velocity


def _batch():
    torch.manual_seed(7)
    cfg = V7Config(hidden=24, decoder_hidden=32, content_dim=16, style_dim=8,
                   emotion_dim=6, event_dim=4)
    model = V7DisentangledModel(cfg)
    b, t, r = 3, 13, 2
    valid = torch.ones(b, t, dtype=torch.bool)
    valid[-1, -2:] = False
    audio = torch.randn(b, t, 1540)
    motion = torch.tanh(torch.randn(b, t, 52))
    refs = {"motion": torch.tanh(torch.randn(b, r, t + 2, 52)),
            "valid": torch.ones(b, r, t + 2, dtype=torch.bool),
            "channel_mask": torch.ones(b, r, 52, dtype=torch.bool)}
    batch = {"audio_features": audio, "motion": motion, "valid": valid,
             "h0": torch.randn(b, t, 16), "emotion_id": torch.tensor([0, 1, -1]),
             "intensity": torch.tensor([.5, 1., .7])}
    return cfg, model, batch, refs


def test_v7_shapes_and_finite_loss():
    _, model, batch, refs = _batch()
    out = model(batch, refs)
    loss = v7_loss(model, out, batch)
    assert out["teacher_out"]["motion"].shape == batch["motion"].shape
    assert out["student_out"]["motion"].shape == batch["motion"].shape
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert model.teacher.trajectory.net[-1].weight.grad is not None
    assert model.student.trajectory.net[-1].weight.grad is not None
    assert model.decoder.population_head.weight.grad is not None


def test_v7_reference_is_independent_and_style_response_can_be_swapped():
    _, model, batch, refs = _batch()
    out = model(batch, refs)
    swapped = {"motion": refs["motion"].flip(1), "valid": refs["valid"].flip(1),
               "channel_mask": refs["channel_mask"].flip(1)}
    a = model.encode_style(refs)["mean"]
    b = model.encode_style(swapped)["mean"]
    assert a.shape == (3, 8)
    decoded = model.decode(out["content"], out["student"]["trajectory_mean"], b, batch["valid"])
    assert decoded["motion"].shape == batch["motion"].shape


def test_v7_flow_sampling_is_seeded_and_affect_intervention_changes_output():
    _, model, batch, refs = _batch()
    model.eval()
    out = model(batch, refs)
    style = out["style"]["mean"]
    a = model.sample_trajectory(out["student"], style, batch["valid"], seed=3, steps=3)
    b = model.sample_trajectory(out["student"], style, batch["valid"], seed=4, steps=3)
    assert a.shape == b.shape
    assert not torch.equal(a, b)
    interventions = model.interventions(batch, refs, emotion_ids=torch.tensor([0, 1]))
    assert not torch.equal(interventions["outputs"]["emotion_0"], interventions["outputs"]["emotion_1"])


def test_velocity_error_cannot_cancel_and_masks_invalid_edges():
    pred = torch.tensor([[[0.], [1.], [0.], [8.]]])
    gt = torch.zeros_like(pred)
    valid = torch.tensor([[True, True, True, False]])
    assert _valid_velocity(pred, gt, valid).item() == 1.


def test_response_has_exact_population_origin_and_is_linear_in_style():
    _, model, batch, refs = _batch()
    model.eval()
    out = model(batch, refs)
    c, e = out['content'], out['teacher']['trajectory']
    s = out['style']['mean']
    zero = model.decode(c, e, torch.zeros_like(s), batch['valid'])
    one = model.decode(c, e, s, batch['valid'])
    two = model.decode(c, e, 2*s, batch['valid'])
    assert torch.count_nonzero(zero['response']) == 0
    torch.testing.assert_close(two['response'], 2*one['response'])


def test_conditional_response_posterior_masks_and_reference_permutation():
    _, model, batch, refs = _batch()
    model.eval()
    refs['audio_features'] = torch.randn(*refs['motion'].shape[:3],1540)
    a = model.encode_style(refs)
    swapped = {k:v.flip(1) for k,v in refs.items()}
    b = model.encode_style(swapped)
    torch.testing.assert_close(a['mean'], b['mean'])
    assert torch.isfinite(a['logvar']).all()
    empty = {**refs, 'valid':torch.zeros_like(refs['valid'])}
    prior = model.encode_style(empty)
    torch.testing.assert_close(prior['mean'], torch.zeros_like(prior['mean']))
    torch.testing.assert_close(prior['logvar'], torch.zeros_like(prior['logvar']))


def test_unknown_hdtf_emotion_and_intensity_are_masked():
    _, model, batch, refs = _batch()
    batch['emotion_id'][:] = -1
    batch['intensity'][:] = float('nan')
    batch['intensity_valid'] = torch.zeros(3,dtype=torch.bool)
    losses = v7_loss(model, model(batch,refs),batch)
    assert losses['teacher_emotion'].item() == 0
    assert losses['intensity'].item() == 0
    assert torch.isfinite(losses['total'])


def test_canonical_flow_trajectory_is_invariant_to_style_for_fixed_seed():
    cfg, model, batch, refs = _batch()
    model.eval()
    student=model.encode_student(batch['audio_features'][...,768:],batch['valid'])
    s=torch.randn(3,cfg.style_dim)
    a=model.sample_trajectory(student,s,batch['valid'],seed=7,steps=3)
    b=model.sample_trajectory(student,s*3,batch['valid'],seed=7,steps=3)
    torch.testing.assert_close(a[...,:cfg.emotion_dim],b[...,:cfg.emotion_dim],rtol=0,atol=0)
