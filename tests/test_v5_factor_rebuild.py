"""Contracts for the retained-v5 continuation and privileged learning route."""
from dataclasses import asdict

import torch

from kinetalk_v5.factor_rebuild import RebuildConfig, V5FactorRebuild, trajectory_noise
from tests.test_unified_v4 import fixture
from scripts.pilot_v5_factor_rebuild import supervision, reference


def setup():
    old, cfg, valid, base, motion, audio, refs, _, _ = fixture()
    # Give the retained decoder an actual nonzero expression/style response.
    with torch.no_grad():
        old.decoder.output.weight.normal_(std=.01)
        old.decoder.geometry_delta.weight.normal_(std=.005)
    model = V5FactorRebuild.from_v5({'config': asdict(cfg), 'model': old.state_dict()},
                                   RebuildConfig(flow_hidden=24, flow_steps=4))
    content = torch.randn(2, 19, cfg.content_dim)
    refs.update(audio_features=audio.flip(1), h0=content.flip(1),
                h0_available=torch.ones(2, dtype=torch.bool))
    b = dict(audio_features=audio, h0=content, b0=base, motion=motion, valid=valid,
             channel_mask=torch.ones(2, 52, dtype=torch.bool))
    return old.eval(), model.eval(), b, refs


def test_new_modules_preserve_actual_v5_prediction_at_initialization():
    old, model, b, refs = setup()
    with torch.no_grad():
        a = old.predict(b['audio_features'], b['h0'], b['b0'], b['valid'], refs, seed=17)
        z = model.predict(b['audio_features'], b['h0'], b['b0'], b['valid'], refs, seed=17)
    assert torch.allclose(a['motion'], z['motion'], atol=2e-6, rtol=0.)
    assert torch.equal(a['events'], z['schedule']['events'])


def test_privileged_teacher_zero_start_and_real_observation_gradients():
    _, model, b, refs = setup()
    p = model.affect_audio(b['audio_features'], b['valid'], anchor=True)
    t = model.visual_teacher(b)
    assert torch.equal(t['state'], p['state'])
    y = model.decode(b, t, model.encode_style(refs), mode='modal')['motion']
    (y - b['motion']).square()[b['valid']].mean().backward()
    assert model.teacher.state.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.anchor_student.parameters())


def test_inference_never_reads_query_motion_or_privileged_teacher():
    _, model, b, refs = setup()
    model.visual_teacher = lambda _: (_ for _ in ()).throw(AssertionError('teacher used'))
    with torch.no_grad():
        z = model.predict(b['audio_features'], b['h0'], b['b0'], b['valid'], refs)
    assert z['motion'].shape == b['motion'].shape
    assert torch.equal(z['motion'][~b['valid']], torch.zeros_like(z['motion'][~b['valid']]))


def test_style_context_requires_genuine_reference_context():
    _, model, _, refs = setup()
    import pytest
    with pytest.raises(ValueError, match='OWN audio'):
        model.encode_style({k: v for k, v in refs.items() if k != 'audio_features'})
    with pytest.raises(ValueError, match='genuine'):
        model.encode_style({**refs, 'h0_available': torch.tensor([True, False])})


def test_style_can_learn_from_the_disjoint_query_target():
    _, model, b, refs = setup()
    y = model.decode(b, model.affect_audio(b['audio_features'], b['valid']),
                     model.encode_style(refs), mode='modal')['motion']
    (y - b['motion']).square()[b['valid']].mean().backward()
    assert model.reference_context.output.weight.grad.abs().sum() > 0


def test_checkpoint_migration_is_strict_and_roundtrips():
    _, model, b, refs = setup()
    restored = V5FactorRebuild.from_checkpoint(model.checkpoint()).eval()
    args = (b['audio_features'], b['h0'], b['b0'], b['valid'], refs)
    with torch.no_grad():
        assert torch.equal(model.predict(*args)['motion'], restored.predict(*args)['motion'])


def test_flow_has_seeded_variation_after_learning_and_no_style_semantic_path():
    _, model, b, _ = setup()
    model.flow.sample_gain.fill_(1.)
    with torch.no_grad():
        model.flow.output.weight.normal_(std=.03)
    p = model.affect_audio(b['audio_features'], b['valid'])
    c = model.core.condition(p, b['valid'])
    args = (p['state'], p['hidden'], c['emotion'], c['intensity'], b['valid'])
    a = model.flow.sample(*args, seed=11, steps=4)
    assert torch.equal(a, model.flow.sample(*args, seed=11, steps=4))
    assert not torch.equal(a, model.flow.sample(*args, seed=12, steps=4))
    assert torch.equal(a[~b['valid']], torch.zeros_like(a[~b['valid']]))
    loss = model.flow.loss(p['state'] + .1, p['state'], p['hidden'],
                           c['emotion'], c['intensity'], b['valid'])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.flow.output.weight.grad.abs().sum() > 0


def test_hdtf_has_no_emotion_or_intensity_gradients_even_with_bad_labels():
    emotion = torch.randn(2, 8, requires_grad=True)
    intensity = torch.randn(2, 4, requires_grad=True)
    b = dict(dataset_id=torch.ones(2, dtype=torch.long), emotion_id=torch.tensor([0, 1]),
             intensity=torch.ones(2), intensity_valid=torch.ones(2, dtype=torch.bool))
    loss = supervision(emotion, intensity, b)
    loss.backward()
    assert loss.item() == 0
    assert emotion.grad.abs().sum() == 0
    assert intensity.grad.abs().sum() == 0


def test_missing_second_support_uses_first_legal_reference_not_query():
    refs = {'valid': torch.tensor([[[True, True], [False, False]]]),
            'motion': torch.tensor([[[[1.], [2.]], [[0.], [0.]]]]),
            'emotion_id': torch.tensor([[4, -1]])}
    out = reference(refs, 1)
    assert torch.equal(out['motion'], refs['motion'][:, 0])
    assert out['emotion_id'].item() == 4


def test_personal_response_zero_start_has_actual_learning_gradient():
    _, model, b, refs = setup()
    p = model.affect_audio(b['audio_features'], b['valid'])
    out = model.decode(b, p, model.encode_style(refs), mode='modal')
    assert out['style_response'].abs().sum() == 0
    (out['motion'] - b['motion']).square()[b['valid']].mean().backward()
    assert model.style_response.basis[-1].weight.grad.abs().sum() > 0


def test_personal_response_can_affect_full_face_without_emotion_channel_masks():
    _, model, b, refs = setup()
    with torch.no_grad(): model.style_response.basis[-1].weight.normal_(std=.1)
    p = model.affect_audio(b['audio_features'], b['valid'])
    s = model.encode_style(refs)
    a = model.style_response(b['h0'], p['state'], s, b['valid'], model.core.scales)
    z = model.style_response(b['h0'], p['state'], torch.zeros_like(s), b['valid'], model.core.scales)
    assert (a - z)[b['valid']][:, 14:40].abs().sum() > 0
    assert (a - z)[b['valid']][:, 41:51].abs().sum() > 0


def test_correlated_gaussian_preserves_marginal_variance_in_short_runs():
    rng = torch.Generator().manual_seed(171)
    white = torch.randn(5000, 13, 1, generator=rng)
    valid = torch.ones(5000, 13, dtype=torch.bool)
    valid[:, 4:6] = False
    valid[:, 8:] = False
    out = trajectory_noise(white, valid, kernel=9)
    assert torch.equal(out[~valid], torch.zeros_like(out[~valid]))
    assert torch.allclose(out.var(0)[valid[0]], torch.ones_like(out.var(0)[valid[0]]), atol=.06)
    assert float((out[:, 0] * out[:, 1]).mean()) > .8
    assert torch.equal(trajectory_noise(white, valid, kernel=1),
                       torch.where(valid[..., None], white, 0.))


def test_correlated_base_does_not_cross_invalid_or_timestamp_gaps():
    white = torch.randn(1, 12, 3)
    valid = torch.ones(1, 12, dtype=torch.bool); valid[:, 4] = False
    times = torch.arange(12, dtype=torch.float64)[None] / 25
    times[:, 8:] += .12
    a = trajectory_noise(white, valid, times, kernel=9)
    modified = white.clone(); modified[:, 4:8] = 1000.
    b = trajectory_noise(modified, valid, times, kernel=9)
    assert torch.equal(a[:, :4], b[:, :4])
    assert torch.equal(a[:, 8:], b[:, 8:])
    modified = white.clone(); modified[:, 8:] = -1000.
    c = trajectory_noise(modified, valid, times, kernel=9)
    assert torch.equal(a[:, :8], c[:, :8])


def test_training_sampling_share_base_and_teacher_endpoint_is_unmodified():
    _, model, b, _ = setup(); model.flow.noise_kernel = 9
    model.flow.sample_gain.fill_(1.)
    prior = model.affect_audio(b['audio_features'], b['valid'])
    cond = model.core.condition(prior, b['valid'])
    times = torch.arange(19, dtype=torch.float64)[None].expand(2, -1) / 25
    seed = 34
    rng = torch.Generator().manual_seed(seed)
    start = model.flow.base_noise(prior['state'], b['valid'], times=times, generator=rng)
    y = model.flow.sample(prior['state'], prior['hidden'], cond['emotion'], cond['intensity'],
                          b['valid'], seed, 4, times=times)
    # Zero velocity at initialization makes the exact shared base observable.
    assert torch.equal(y, torch.where(b['valid'][..., None],
                       prior['state'] + start * model.flow.scale, 0.))
    target = prior['state'].detach() + torch.randn_like(prior['state']) * .1
    captured = {}
    model.flow.base_noise = lambda *args, **kwargs: start
    def capture(x, tau, *args):
        captured.update(x=x.detach(), tau=tau.detach())
        return x * 0.
    model.flow.forward = capture
    model.flow.loss(target, prior['state'], prior['hidden'], cond['emotion'], cond['intensity'],
                    b['valid'], times=times)
    endpoint = (captured['x'] - (1. - captured['tau']) * start) / captured['tau']
    expected = (target - prior['state'].detach()) / model.flow.scale
    assert torch.allclose(endpoint[b['valid']], expected[b['valid']], atol=1e-6)


def test_old_rebuild_checkpoint_keeps_white_base_distribution():
    _, model, _, _ = setup(); ck = model.checkpoint()
    ck['rebuild_config'].pop('flow_noise_kernel')
    ck['rebuild_config'].pop('flow_noise_correlation')
    restored = V5FactorRebuild.from_checkpoint(ck)
    assert restored.flow.noise_kernel == 1
    assert restored.rebuild_config.flow_noise_kernel == 1


def test_flow_covariance_interpolation_preserves_marginal_and_exact_endpoints():
    _, model, b, _ = setup(); model.flow.noise_kernel = 9
    like = torch.zeros(12000, 13, 1); valid = torch.ones(12000, 13, dtype=torch.bool)
    times = torch.arange(13, dtype=torch.float64)[None].expand(12000, -1) / 25
    for weight in (0., .25, .5, .75, 1.):
        rng1 = torch.Generator().manual_seed(89)
        rng2 = torch.Generator().manual_seed(89)
        sample = model.flow.mixed_base_noise(like, valid, times=times, generator=rng1,
                                             colored_weight=weight)
        assert torch.allclose(sample.var(0).squeeze(-1), torch.ones(13), atol=.04)
        assert torch.equal(sample, model.flow.mixed_base_noise(like, valid, times=times,
                          generator=rng2, colored_weight=weight))
    white = model.flow.mixed_base_noise(like, valid, times=times,
        generator=torch.Generator().manual_seed(10), colored_weight=0.)
    rng = torch.Generator().manual_seed(10)
    torch.randn(like.shape, generator=rng)
    expected_white = torch.randn(like.shape, generator=rng)
    assert torch.equal(white, expected_white)
    model.flow.noise_correlation = .5
    actual = model.flow.base_noise(like, valid, times=times,
                                   generator=torch.Generator().manual_seed(10))
    expected = model.flow.mixed_base_noise(like, valid, times=times,
        generator=torch.Generator().manual_seed(10), colored_weight=.5)
    assert torch.equal(actual, expected)


def test_mixed_covariance_train_sample_share_same_base():
    _, model, b, _ = setup(); model.flow.noise_kernel = 9
    model.flow.noise_correlation = .5
    model.flow.sample_gain.fill_(1.)
    prior = model.affect_audio(b['audio_features'], b['valid'])
    cond = model.core.condition(prior, b['valid'])
    a = model.flow.sample(prior['state'], prior['hidden'], cond['emotion'], cond['intensity'],
                          b['valid'], 33, 4)
    start = model.flow.base_noise(prior['state'], b['valid'], generator=torch.Generator().manual_seed(33))
    assert torch.equal(a, torch.where(b['valid'][..., None],
                       prior['state'] + start * model.flow.scale, 0.))
    import pytest
    with pytest.raises(ValueError, match='colored_weight'):
        model.flow.mixed_base_noise(prior['state'], b['valid'], colored_weight=1.5)
