"""Prove both student outputs are invariant to direct HuBERT inputs."""
import copy
import pytest
import torch
from kinetalk_b0.models.slow_state_affect import (
    SlowStateAffect, affect_audio_statistics, migrate_affect_audio_state,
)
from scripts.train_full_staged import parser


def model_and_audio():
    torch.manual_seed(31772)
    model = SlowStateAffect(torch.zeros(772), torch.ones(772), hidden=16,global_dim=8,local_dim=8).eval()
    with torch.no_grad():
        model.local_head.weight.normal_(0, .1)
    features = torch.randn(2, 9, 1540)
    valid = torch.arange(9)[None] < torch.tensor([9,6])[:, None]
    features[~valid] = float('nan')
    return model,features,valid


def test_global_and_ua_do_not_read_hubert_and_packed_equals_selected():
    model,features,valid = model_and_audio()
    original = features.clone()
    base = model(features,valid,include_legacy_state=False)
    selected = model(features[...,768:],valid,include_legacy_state=False)
    changed = features.clone();changed[...,:768] = float('nan')
    alternative = model(changed,valid,include_legacy_state=False)
    for key in base:
        assert torch.equal(base[key], selected[key])
        assert torch.equal(base[key], alternative[key])
    torch.testing.assert_close(features,original,rtol=0,atol=0,equal_nan=True)
    assert model.input.in_features == 772


def test_student_losses_have_zero_hubert_gradient_but_train_affect_features():
    model,features,valid = model_and_audio()
    features = features.nan_to_num().requires_grad_(True)
    output = model(features,valid,include_legacy_state=False)
    (output['global'].square().mean()+output['u_a'].square().mean()).backward()
    assert not features.grad[...,:768].any()
    assert torch.isfinite(features.grad).all()
    assert features.grad[...,768:][valid].abs().sum() > 0
    assert not features.grad[~valid].any()


def test_selected_train_statistics_are_copied_without_refitting_or_mutation():
    statistics = dict(mean=torch.arange(1540).float(),std=torch.arange(1540).float()+1,count=222)
    original = copy.deepcopy(statistics)
    selected = affect_audio_statistics(statistics,'affect-prosody')
    assert torch.equal(selected['mean'], original['mean'][768:])
    assert torch.equal(selected['std'], original['std'][768:])
    assert selected['count'] == 222 and selected['source_feature_width'] == 1540
    selected['mean'][0] = -1
    assert torch.equal(statistics['mean'], original['mean'])


def test_explicit_warm_migration_only_drops_declared_input_columns():
    torch.manual_seed(14)
    old = SlowStateAffect(torch.zeros(1540),torch.ones(1540),hidden=16,global_dim=8,local_dim=8).double().eval()
    new = SlowStateAffect(torch.zeros(772),torch.ones(772),hidden=16,global_dim=8,local_dim=8).double().eval()
    state = old.state_dict();original = {k:v.clone() for k,v in state.items()}
    converted = migrate_affect_audio_state(state,new.state_dict())
    new.load_state_dict(converted,strict=True)
    for key in state:
        assert torch.equal(state[key],original[key])
        if key not in ('input.weight','feature_mean','feature_std'):
            assert torch.equal(new.state_dict()[key],state[key])
    features = torch.randn(2,7,1540,dtype=torch.float64)
    features[...,:768] = 0 # TRAIN mean: no normalized content contribution
    valid = torch.ones(2,7,dtype=torch.bool)
    with torch.no_grad():
        base = old(features,valid,include_legacy_state=False)
        selected = new(features,valid,include_legacy_state=False)
    for key in base:
        torch.testing.assert_close(base[key],selected[key],atol=1e-10,rtol=1e-10)


def test_checkpoint_restore_selects_layout_and_rejects_wrong_warm_normalization():
    model,features,valid = model_and_audio()
    state = model.state_dict()
    restored = SlowStateAffect(state['feature_mean'],state['feature_std'],hidden=16,global_dim=8,local_dim=8).eval()
    restored.load_state_dict(state,strict=True)
    assert torch.equal(model(features,valid)['u_a'],restored(features,valid)['u_a'])
    old = SlowStateAffect(torch.zeros(1540),torch.ones(1540),hidden=16,global_dim=8,local_dim=8)
    incompatible = {k:v.clone() for k,v in state.items()};incompatible['feature_std'][0]=2
    with pytest.raises(ValueError,match='normalization'):
        migrate_affect_audio_state(old.state_dict(),incompatible)
    with pytest.raises(ValueError,match='1540-to-772'):
        migrate_affect_audio_state(state,state)


def test_new_training_default_is_affect_only_and_legacy_is_explicit():
    args = parser().parse_args(['--output','unused_test_output'])
    assert args.audio_feature_layout == 'affect-prosody'
    assert not args.allow_audio_input_migration
    legacy = parser().parse_args(['--output','unused_test_output','--audio-feature-layout','legacy-content-affect-prosody'])
    assert legacy.audio_feature_layout == 'legacy-content-affect-prosody'
