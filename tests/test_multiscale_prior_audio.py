import copy
from argparse import Namespace

import numpy as np
import pytest
import torch

from kinetalk_b0.models.continuous_upper_motion import ContinuousLatentFlow, ContinuousUpperAE
from kinetalk_b0.models.prior_audio_multiscale import MultiScalePriorAudioResidual
from scripts import train_multiscale_prior_audio as run
from scripts.prepare_continuous_motion_dataset import fit_statistics


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture():
    torch.manual_seed(43)
    model = MultiScalePriorAudioResidual(ContinuousLatentFlow(3, latent_dim=4, hidden=8, depth=1),
                                         hidden=8, depth=2, fast_hidden=5)
    valid = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    frames = torch.tensor([[[1]*5, [1]*5, [1, 1, 0, 0, 0]],
                           [[1]*5, [1, 1, 1, 0, 0], [0]*5]], dtype=torch.bool)
    data = dict(noisy=torch.randn(2, 3, 4), time=torch.tensor([.2, .8]), valid=valid,
                context=torch.randn(2, 3), audio_blocks=torch.randn(2, 3, 5, 1540), audio_frame_valid=frames)
    return model, data


def test_exact_prior_at_init_null_after_training_and_frozen_gradients():
    model, b = fixture()
    base = model.prior.velocity(**b, use_audio=False)
    assert torch.equal(model.velocity(**b), base)
    before = copy.deepcopy(model.prior.state_dict())
    optimizer = torch.optim.AdamW(model.residual_parameters(), lr=.003)
    for _ in range(4):
        model.train(); assert not model.prior.training
        loss = model.flow_loss(b['noisy']*.4, b['valid'], b['context'], b['audio_blocks'],
                               b['noisy'], b['time'], audio_frame_valid=b['audio_frame_valid'])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    for module in (model.fast_frame_projection, model.slow_projection):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in model.prior.parameters())
    assert all(torch.equal(v, model.prior.state_dict()[k]) for k, v in before.items())
    zero = {**b, 'audio_blocks': torch.zeros_like(b['audio_blocks'])}
    assert torch.equal(model.velocity(**zero), base)
    assert torch.equal(model.velocity(**b, use_audio=False), base)
    assert not torch.equal(model.velocity(**b), base)
    assert model.residual(**b).abs().max() <= model.max_delta


def test_pooling_uses_all_four_prosody_coordinates_and_native_frame_counts():
    model, b = fixture()
    x = torch.zeros_like(b['audio_blocks'])
    for j in range(3):
        x[:, j, :, 1536:1540] = (j+1)*torch.tensor([1., 2., 3., 4.])
    clean, frames = model._validate(b['valid'], b['context'], x, x.dtype, b['audio_frame_valid'])
    captured = []
    hook = model.slow_projection.register_forward_pre_hook(lambda m, a: captured.append(a[0].detach()))
    model._fast_and_slow(b['valid'], clean, frames, x.dtype); hook.remove()
    torch.testing.assert_close(captured[0][0, 0], (5+10+6)/12*torch.tensor([1., 2., 3., 4.]))
    torch.testing.assert_close(captured[0][1, 0], (5+6)/8*torch.tensor([1., 2., 3., 4.]))


def test_padding_nan_gap_rejection_and_config_roundtrip():
    model, b = fixture()
    with torch.no_grad(): model.output.weight.normal_(std=.2)
    expected = model.residual(**b)
    poisoned = {**b, 'audio_blocks': torch.where(b['audio_frame_valid'][..., None], b['audio_blocks'], float('nan')),
                'noisy': torch.where(b['valid'][..., None], b['noisy'], float('nan'))}
    assert torch.equal(model.residual(**poisoned), expected)
    config = copy.deepcopy(model.config)
    restored = MultiScalePriorAudioResidual(ContinuousLatentFlow(**config.pop('prior')), **config)
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored.residual(**b), expected)
    bad = {**b, 'audio_frame_valid': b['audio_frame_valid'].clone()}; bad['audio_frame_valid'][0, 0, 1] = False
    with pytest.raises(ValueError, match='prefix'): model.residual(**bad)
    with pytest.raises(TypeError): model.velocity(**b, motion_mask=b['valid'])


def clips():
    result = []
    torch.manual_seed(23)
    for i in range(9):
        n = 12+i; baseline = torch.full((n, 52), .2)
        motion = .2+.03*torch.randn(n, 9)
        target = baseline.clone(); target[:, [41,42,43,44,45,5,6,12,13]] = motion
        result.append({'clip_id': f'c{i}', 'sentence': f's{i}', 'speaker': 'one', 'emotion': 'happy',
                       'split': 'train', 'features': torch.randn(n, 1540), 'context': torch.randn(3),
                       'valid': torch.ones(n, dtype=torch.bool), 'motion_mask': torch.ones(n,9,dtype=torch.bool),
                       'motion9': motion, 'b9': torch.full((9,), .2), 'baseline52': baseline,
                       'target52': target, 'channel_mask': torch.ones(n,52,dtype=torch.bool), 'times': torch.arange(n)/25})
    return result


def test_full_support_ignores_event_labels_and_donors_ignore_motion():
    fit = clips(); stats = fit_statistics(fit); ae = ContinuousUpperAE(hidden=8, depth=1, latent_dim=4)
    for c in fit: c['known'] = torch.zeros(len(c['valid']),4,dtype=torch.bool)
    real, static, coverage = run.prepare_training(fit, stats, ae, 'cpu')
    assert coverage['totals']['kept_frames'] == sum(len(c['valid']) for c in fit)
    assert len(coverage['retained_clip_ids']) == len(fit)
    for a,b in zip(real, static):
        assert torch.equal(a['z'],b['z']) and a['metadata'] == b['metadata']
        assert torch.allclose(b['audio'], b['audio'][:1].expand_as(b['audio']))
    bank = run.DonorBank(fit, stats)
    for c in fit: c['motion9'].fill_(float('nan')); c['motion_mask'].fill_(False)
    other = run.DonorBank(fit, stats)
    model, _ = fixture(); items = real[:2]
    stats.update(latent_mean=torch.zeros(4), latent_scale=torch.ones(4))
    values = run.common._flow_batch(model, items, stats, torch.device('cpu'), 4, 1)
    a = bank.batch(items, values[3], values[6], seed=8, step=2)
    b = other.batch(items, values[3], values[6], seed=8, step=2)
    assert torch.equal(a[0],b[0]) and a[2:] == b[2:]
    for item, donor in zip(items, a[2]):
        assert item['metadata']['sentence'] != bank.clips[donor]['sentence']


def test_smoke_selects_donor_pairs_before_reading_motion():
    members = [{'clip_id':f'{e}{i}', 'emotion':e, 'sentence':f's{i//2}', 'speaker':'one'}
               for e in ('happy','neutral') for i in range(4)]
    selected = run.smoke_population(members)
    assert len(selected) == 4
    for c in selected:
        assert any(d['emotion'] == c['emotion'] and d['sentence'] != c['sentence'] for d in selected)


def test_actual_end_to_end_smoke_and_resumption(tmp_path):
    fit_clips = clips(); dataset = tmp_path/'dataset.pt'; source = tmp_path/'source'; source.mkdir()
    torch.save({'schema':'continuous_motion_dataset_v1', 'clips':fit_clips}, dataset)
    fit, val = run.split_train_pool(fit_clips)
    # A singleton emotion/sentence must stay in full arms and be explicitly
    # excluded only from the no-valid-donor mismatch arm.
    val[0]['emotion'] = 'neutral'
    torch.save({'schema':'continuous_motion_dataset_v1', 'clips':fit_clips}, dataset)
    protocol = {'schema':'bounded_audio_experiment_v1', 'dataset_sha256':run.common._file_sha(dataset),
                'inner_train_ids':[c['clip_id'] for c in fit], 'inner_validation_ids':[c['clip_id'] for c in val],
                'validation_sentences': sorted({c['sentence'] for c in val})}
    run.common._write(source/'protocol.json',protocol)
    ae = ContinuousUpperAE(hidden=8, depth=1, latent_dim=4)
    prior = ContinuousLatentFlow(3,hidden=8,depth=1,latent_dim=4)
    stats = fit_statistics(fit); stats.update(latent_mean=torch.zeros(4),latent_scale=torch.ones(4))
    binding = {'protocol_sha256':run.common._value_sha(protocol)}
    torch.save({'config':ae.config,'state':ae.state_dict(),'binding':binding},source/'ae_final.pt')
    torch.save({'config':prior.config,'state':prior.state_dict(),'binding':{**binding,
                'ae_sha256':run.common._value_sha(ae.state_dict()), 'stats_sha256':run.common._value_sha(stats)}},source/'prior_final.pt')
    torch.save(stats,source/'fit_stats.pt')
    args = Namespace(dataset=dataset,source_run=source,output=tmp_path/'out',device='cpu',
                     pilot_steps=2,final_steps=2,batch_size=2,seed=5,resume=False,smoke=True)
    run.run(args)
    import json
    result = json.loads((args.output/'point_2/acceptance.json').read_text())
    assert set(result['comparisons']) == {'prior','static','matched_static','reverse','mismatch'}
    assert result['protections']['lbe_exact'] and result['protections']['numerics_and_other43']
    assert result['mismatch_excluded_no_donor'] == [val[0]['clip_id']]
    assert result['comparisons']['mismatch']['centered']['clip_count'] < result['clip_count']
    before = run.common._file_sha(args.output/'audio_last.pt')
    args.resume = True; run.run(args)
    assert run.common._file_sha(args.output/'audio_last.pt') == before
