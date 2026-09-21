import copy

import pytest
import torch
from torch import nn

from scripts.train_isolated_residual import (SEEDS, MODES, acceptance_checks,
    condition_inputs, fixed_residual_target, compose_upper_residual,
    objective, validate_state_checkpoint)
from scripts.train_formal_predictable_projection import canonical_hash
from kinetalk_b0.models.isolated_audio_state import IsolatedAudioState
from kinetalk_b0.models.dc_protected_temporal_flow import DCProtectedTemporalFlow
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup():
    torch.manual_seed(373)
    b, t = 2, 17
    valid = torch.ones(b, t, dtype=torch.bool); valid[0, 5] = False; valid[1, 12:] = False
    scales = torch.full((52,), .2)
    model = IsolatedAudioState(torch.zeros(6), torch.ones(6), scales, torch.ones(4),
        global_dim=8, identity_dim=8, hidden=12, stride=4, dilations=(1, 2))
    model.state_branch.head.weight.data.normal_(std=.01)
    model.requires_grad_(False).eval()
    cfg = {'content_dim':8, 'emotion_dim':8, 'style_dim':8,
           'dit_dim':12, 'dit_depth':1, 'heads':3, 'dropout':0.}
    flow = DCProtectedTemporalFlow(cfg, stride=4)
    adapter = nn.Linear(12, 8)
    batch = {'valid':valid, 'h0':torch.randn(b,t,8), 'audio_features':torch.randn(b,t,6),
        'global_code':torch.randn(b,8), 'identity_code':torch.randn(b,8),
        'anchors':torch.full((b,52), .3), 'intensity_value':torch.ones(b,1),
        'channel_mask':torch.ones(b,52,dtype=torch.bool), 'motion':torch.rand(b,t,52)}
    return flow, adapter, model, batch, scales


def test_fixed_target_detached_zero_dc_and_mean_guard_only():
    _, adapter, model, b, scales = setup()
    b['motion'].requires_grad_()
    *_, out = condition_inputs(b, model, adapter, 'audio')
    target = fixed_residual_target(b, out['upper'], scales)
    assert not target.requires_grad
    torch.testing.assert_close(target.double().sum(1)/b['valid'].sum(1)[:,None],
                               torch.zeros(2,9,dtype=torch.float64),atol=1e-7,rtol=0)
    assert target[~b['valid']].count_nonzero() == 0
    changed = dict(b); changed['channel_mask'] = b['channel_mask'].clone(); changed['channel_mask'][:,41] = False
    with pytest.raises(ValueError,match='all nine'): fixed_residual_target(changed,out['upper'],scales)


def test_condition_interventions_cover_h0_local_and_state_without_query_targets():
    _, adapter, model, b, _ = setup()
    normal = condition_inputs(b,model,adapter,'audio')
    static = condition_inputs(b,model,adapter,'static')
    reverse = condition_inputs(b,model,adapter,'reverse')
    for i in range(len(b['valid'])):
        native = torch.nonzero(b['valid'][i]).flatten()
        torch.testing.assert_close(static[0]['h0'][i,native],normal[0]['h0'][i,native].mean(0).expand(len(native),-1))
        torch.testing.assert_close(static[3][i,native],static[3][i,native[0]].expand(len(native),-1))
        assert static[4]['state'][i,native].count_nonzero() == 0
        torch.testing.assert_close(reverse[0]['h0'][i,native],normal[0]['h0'][i,native.flip(0)])
        torch.testing.assert_close(reverse[3][i,native],normal[3][i,native.flip(0)])
        torch.testing.assert_close(reverse[4]['state'][i,native],normal[4]['state'][i,native.flip(0)])
    for result in (static, reverse):
        torch.testing.assert_close(result[4]['mean'],normal[4]['mean'],atol=0,rtol=0)
        torch.testing.assert_close(result[1]['code'],normal[1]['code'],atol=0,rtol=0)
        torch.testing.assert_close(result[2]['global'],normal[2]['global'],atol=0,rtol=0)
    dirty = dict(b); dirty['motion'] = torch.full_like(b['motion'],float('nan'))
    dirty['target_mean'] = torch.full((2,9),1e8)
    clean_out = condition_inputs(dirty,model,adapter,'audio')
    for key in normal[4]: torch.testing.assert_close(clean_out[4][key],normal[4][key],atol=0,rtol=0)


def test_training_updates_only_residual_and_adapter_even_with_rollout_objectives():
    flow, adapter, model, b, scales = setup()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    for key in ('h0','global_code','identity_code','audio_features','motion'):
        b[key].requires_grad_()
    optimizer = torch.optim.AdamW(list(flow.parameters())+list(adapter.parameters()),lr=1e-3)
    noise = torch.randn(2,17,9)
    loss, metrics = objective(flow,adapter,model,b,scales,'audio',noise,torch.tensor([.2,.7]),
                             second_noise=torch.randn_like(noise),decode_steps=2)
    optimizer.zero_grad(); loss.backward()
    assert set(metrics) == {'flow','raw_fair_es','centered_fair_es','domain'}
    assert all(torch.isfinite(x) for x in metrics.values())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in flow.parameters())
    assert adapter.weight.grad is not None and adapter.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters())
    assert all(b[key].grad is None for key in ('h0','global_code','identity_code','audio_features','motion'))
    optimizer.step()
    for key,value in model.state_dict().items(): torch.testing.assert_close(value,before[key],atol=0,rtol=0)


def test_residual_generation_preserves_upper_means_and_all_other43_coefficients():
    flow, adapter, model, b, scales = setup()
    q, identity, affect, local, out = condition_inputs(b,model,adapter,'audio')
    residual = flow.decode(q,identity,affect,local,out['state'],torch.randn(2,17,9),3)
    upper = compose_upper_residual(out['upper'],residual,scales,b['valid'])
    baseline = torch.rand(2,17,52)
    full = compose_upper_face(baseline,upper,b['valid'])
    cc = list(UPPER_INDICES); other = [i for i in range(52) if i not in cc]
    assert torch.equal(full[...,other],baseline[...,other])
    assert torch.equal(full[~b['valid']],baseline[~b['valid']])
    delta = (upper-out['upper']).double().sum(1)/b['valid'].sum(1)[:,None]
    torch.testing.assert_close(delta,torch.zeros_like(delta),atol=1e-7,rtol=0)


def test_state_provenance_fail_closed():
    data = {'provenance': {'manifest_sha256':'m','index_sha256':'i'}}
    protocol = {'schema':'isolated_mean_state_v1','source_sha256':'s',
        'data':data['provenance'],'test_loaded':False,'mode':'audio'}
    saved = {'protocol':protocol,'protocol_sha256':canonical_hash(protocol),
             'model':{},'config':{},'dynamic_scales':torch.ones(4)}
    validate_state_checkpoint(saved,data,'s')
    with pytest.raises(ValueError): validate_state_checkpoint(saved,data,'changed')
    bad=copy.deepcopy(saved);bad['protocol']['mode']='static'
    bad['protocol_sha256']=canonical_hash(bad['protocol'])
    with pytest.raises(ValueError):validate_state_checkpoint(bad,data,'s')


def report_fixture():
    keys=[f'{s}/{m}' for s in SEEDS for m in MODES]
    report={'mouth_protection':{k:{'passed':True} for k in keys},
        'nonupper43_exact':{k:True for k in keys},
        'generated_teacher_accuracy_nonindependent':{k:.65 for k in keys},
        'max_absolute_upper_mean_drift':{'full':1e-8,'static':2e-8,'reverse':1e-8}}
    benchmark={m:{'prediction_keys':[f'{s}/{m}' for s in SEEDS],
        'coefficient':{'arkit_mbe':{'value':.5 if m=='full' else .6}}} for m in MODES}
    return report,benchmark


def test_acceptance_requires_complete_seeds_and_detects_regressions():
    report, benchmark = report_fixture()
    assert all(acceptance_checks(report,benchmark).values())
    report['generated_teacher_accuracy_nonindependent']['42/full']=.4
    assert not acceptance_checks(report,benchmark)['teacher_drop_at_most_3pp']
    report,benchmark=report_fixture();benchmark['full']['coefficient']['arkit_mbe']['value']=.7
    assert not acceptance_checks(report,benchmark)['mbe_no_worse_than_stage4']
    report['mouth_protection'].pop('42/full')
    with pytest.raises(ValueError,match='coverage'):acceptance_checks(report,benchmark)
    report,benchmark=report_fixture();report['max_absolute_upper_mean_drift']['full']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):acceptance_checks(report,benchmark)
