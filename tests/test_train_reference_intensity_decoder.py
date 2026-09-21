from types import SimpleNamespace

import pytest
import torch

from scripts import train_reference_intensity_decoder as runner


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture():
    g = torch.Generator().manual_seed(31)
    n,t = 6,13
    valid = torch.ones(n,t,dtype=torch.bool);valid[:,5]=False;valid[1,-2:]=False
    phase = torch.arange(t)[None,:,None] * .3
    motion = (.3 + .04*torch.sin(phase+torch.arange(52)[None,None,:]*.2)).expand(n,-1,-1).clone()
    return {'audio_features':torch.randn(n,t,1540,generator=g), 'motion':motion,
            'anchors':torch.full((n,52),.2),'valid':valid,
            'channel_mask':torch.ones(n,52,dtype=torch.bool),'anchor_valid':torch.ones(n,52,dtype=torch.bool),
            'global_code':torch.randn(n,4,generator=g),'identity_code':torch.randn(n,6,generator=g),
            'clip_id':[f'c{i}' for i in range(n)],'speaker':[f's{i//2}' for i in range(n)],
            'sentence_id':[f't{i%2}' for i in range(n)]}


def setup():
    q=fixture();ids=torch.arange(6)
    stats=runner.fit_statistics(q,ids,'cpu',batch_size=2,pca_frames=40)
    cache=runner.prepare_cache(q,stats,batch_size=2)
    decoder,student=runner.new_models(stats,'cpu',21)
    return q,ids,stats,cache,decoder,student


def test_stats_and_pca_are_fit_only():
    q=fixture();fit=torch.tensor([0,1,2])
    original=runner.fit_statistics(q,fit,'cpu',pca_frames=30)
    for key in ('audio_features','motion','global_code','identity_code'):
        q[key][3:]=10000
    changed=runner.fit_statistics(q,fit,'cpu',pca_frames=30)
    for key in original:
        torch.testing.assert_close(original[key],changed[key],rtol=0,atol=0)


def test_objective_padding_is_safe_and_all_stages_use_identical_target():
    _,ids,stats,cache,_,_=setup()
    b=runner.batch(cache,ids,'cpu')
    upper=b['upper'].clone().requires_grad_()
    loss,parts=runner.objective(upper,b,stats['scales9'],b['intensity'])
    assert float(loss.detach())==0.
    for value in parts.values():assert float(value.detach())==0.
    b['upper'][~b['valid']]=float('nan');b['intensity'][~b['valid']]=float('nan')
    upper=(upper.detach()+.01).requires_grad_()
    with torch.no_grad():upper[~b['valid']]=float('nan')
    requested=torch.where(b['valid'][...,None],b['intensity'],torch.nan).requires_grad_()
    loss,_=runner.objective(upper,b,stats['scales9'],requested)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(upper.grad).all() and torch.isfinite(requested.grad).all()
    assert (upper.grad[~b['valid']]==0).all()


def changed(before,module):
    return any(not torch.equal(before[k],v) for k,v in module.state_dict().items())


@pytest.mark.parametrize('stage',['oracle','student','joint'])
def test_stage_optimizer_updates_only_authorized_modules_with_nonzero_gradients(stage):
    _,ids,stats,cache,decoder,student=setup()
    d0={k:v.clone() for k,v in decoder.state_dict().items()};s0={k:v.clone() for k,v in student.state_dict().items()}
    optimizer=runner.optimizer_for(decoder,student,stage)
    loss=runner.epoch(decoder,student,cache,ids,stats,optimizer,stage,'cpu',2,41)
    assert loss>0
    assert changed(d0,decoder)==(stage!='student')
    assert changed(s0,student)==(stage!='oracle')
    for module,enabled in ((decoder,stage!='student'),(student,stage!='oracle')):
        gradients=[p.grad for p in module.parameters() if p.grad is not None]
        assert bool(gradients)==enabled
        if enabled:assert sum(float(v.abs().sum()) for v in gradients)>0


def test_mismatch_donors_are_nonself_and_evaluation_only():
    _,_,_,cache,_,_=setup()
    cal=torch.tensor([2,3,5])
    donors=runner._evaluation_donors(cache,cal)
    assert set(donors)==set(cal.tolist())
    assert set(donors.values())<=set(cal.tolist())
    assert all(dst!=src for dst,src in donors.items())
    for i in range(6):cache['features'][i].fill_(i+1.)
    out=runner.mismatch_features(cache,cal,donors)
    for row,i in enumerate(cal.tolist()):
        torch.testing.assert_close(out[row,cache['valid'][i]],torch.full_like(out[row,cache['valid'][i]],donors[i]+1.))


def test_static_intensity_and_static_hidden_cannot_create_boundary_motion():
    _,ids,stats,cache,decoder,student=setup()
    for mode in ('static','static_intensity','oracle_static','oracle_zero'):
        upper,intensity=runner.predict(decoder,student,cache,ids,'cpu',2,mode)
        for row in range(len(ids)):
            u=upper[row,cache['valid'][row]];i=intensity[row,cache['valid'][row]]
            torch.testing.assert_close(u,u[:1].expand_as(u),rtol=0,atol=1e-7)
            torch.testing.assert_close(i,i[:1].expand_as(i),rtol=0,atol=1e-7)


def test_report_separates_mean_error_and_oracle_output_evidence():
    _,ids,stats,cache,decoder,student=setup()
    args=SimpleNamespace(device='cpu',batch_size=2)
    report,_=runner.evaluate_modes(decoder,student,cache,ids,stats,args,('oracle','oracle_static','oracle_reverse','oracle_zero'))
    acceptance=report['receiver_acceptance_diagnostic']
    assert acceptance['oracle_requested_intensity_self_correlation_used'] is False
    assert acceptance['paper_success_established'] is False
    assert 'centered_upper_mse_beats_oracle_reverse_both_cluster_ci' in acceptance['checks']
    for m in report['modes'].values():
        assert m['normalized_upper_mse']==pytest.approx(m['normalized_centered_upper_mse']+m['normalized_mean_upper_mse'],rel=1e-5)
    # Baselines inserted after mode inference must also receive paired tests.
    report['modes']['original_prior']=report['modes']['oracle_static']
    pairs=runner.paired_comparisons(report['modes'],cache,ids)
    assert 'original_prior' in pairs and 'mean_upper_mse' in pairs['original_prior']


def test_joint_zero_budget_restores_valid_pre_joint_weights(tmp_path):
    _,ids,stats,cache,decoder,student=setup()
    args=SimpleNamespace(device='cpu',batch_size=2,seed=41)
    original={k:v.clone() for k,v in decoder.state_dict().items()}
    selected=runner.select_stage(decoder,student,cache,ids[:4],ids[4:],stats,'joint',0,args,tmp_path)
    assert selected['epochs']==0 and selected['score']>0
    assert not changed(original,decoder)
