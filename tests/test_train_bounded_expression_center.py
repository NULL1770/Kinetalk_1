from types import SimpleNamespace

import pytest
import torch

from scripts import train_bounded_expression_center as runner


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture():
    gen=torch.Generator().manual_seed(92);n,t=18,9
    valid=torch.ones(n,t,dtype=torch.bool);valid[:,3]=False;valid[0,-1]=False
    motion=.3+.02*torch.randn(n,t,52,generator=gen)
    return {'global_code':torch.randn(n,64,generator=gen),'identity_code':torch.randn(n,128,generator=gen),
        'motion':motion,'anchors':torch.full((n,52),.3),'valid':valid,
        'channel_mask':torch.ones(n,52,dtype=torch.bool),'anchor_valid':torch.ones(n,52,dtype=torch.bool),
        'speaker':[f's{i//3}' for i in range(n)],'clip_id':[f'c{i}' for i in range(n)],
        'sentence_id':[f't{i%3}' for i in range(n)]}


def test_outer_loso_and_inner_folds_cover_every_query_without_speaker_overlap():
    q=fixture();seen=[]
    for fit,held in runner.outer_folds(q['speaker']):
        seen.extend(held.tolist())
        assert fit.dtype==held.dtype==torch.long
        a={q['speaker'][i] for i in fit};b={q['speaker'][i] for i in held}
        assert len(b)==1 and not a&b
        inner_seen=[]
        for inner,val in runner.speaker_folds(q['speaker'],fit):
            assert not {q['speaker'][i] for i in inner}&{q['speaker'][i] for i in val}
            assert set(inner.tolist())|set(val.tolist())==set(fit.tolist())
            inner_seen.extend(val.tolist())
        assert sorted(inner_seen)==sorted(fit.tolist())
    assert sorted(seen)==list(range(18))


def test_fit_statistics_and_model_inputs_exclude_held_targets():
    q=fixture();cache=runner.compact_targets(q);fit=torch.arange(12)
    before=runner.fit_statistics(cache,fit)
    inputs=runner.model_inputs(cache,torch.arange(18),before,'cpu')
    q['motion'][12:]=100.;cache2=runner.compact_targets(q)
    after=runner.fit_statistics(cache2,fit)
    for key in ('global_mean','global_std','identity_mean','identity_std','scales9'):
        assert torch.equal(before[key],after[key])
    for a,b in zip(inputs,runner.model_inputs(cache2,torch.arange(18),before,'cpu')):
        assert torch.equal(a,b)


def test_speaker_balancing_and_epoch_zero_real_baseline_selection():
    speakers=['a','a','a','b'];weights=runner.speaker_weights(speakers,torch.arange(4))
    assert float(weights[:3].sum())==pytest.approx(float(weights[3]))
    assert runner.select_epoch([.1,.3,.2])==0
    assert runner.select_epoch([.1,.1,.09,.09])==2
    assert runner.select_epoch([.1,.1,.1])==0
    with pytest.raises(ValueError):runner.select_epoch([.1,float('nan')])


def test_inner_selection_cannot_use_outer_target_and_records_fold_statistics(tmp_path):
    q=fixture();cache=runner.compact_targets(q)
    fit,held=runner.outer_folds(q['speaker'])[0]
    base=cache['target'].clone()  # perfect external baseline must win
    args=SimpleNamespace(epochs=1,smoke=False,device='cpu',batch_size=8)
    epoch,record=runner.inner_selection(cache,base,fit,args,tmp_path/'first',53)
    assert epoch==0
    cache['target'][held]=1000.;cache['relative_sq'][held]=1000.
    epoch2,record2=runner.inner_selection(cache,base,fit,args,tmp_path/'second',53)
    assert epoch2==0 and record==record2
    for row in record['inner_folds']:
        assert not set(row['fit_ids'])&set(row['validation_ids'])
        assert not set(held.tolist())&(set(row['fit_ids'])|set(row['validation_ids']))


def test_metrics_decomposition_padding_and_bypass_protection():
    q=fixture();cache=runner.compact_targets(q);ids=torch.arange(18)
    stats=runner.fit_statistics(cache,ids)
    full=q['motion'].clone();full[~q['valid']]=float('nan')
    base=full[...,runner.UPPER].clone()
    args=SimpleNamespace(device='cpu',batch_size=8)
    report,curves=runner.evaluate_head(q,cache,ids,base,full,None,stats,args)
    assert runner.bit_equal(base,curves['upper'])
    assert report['selected']['summary']['nonupper43_bit_exact']
    assert report['selected']['summary']['invalid_frames_bit_exact']
    for row in report['selected']['per_clip']:
        assert row['raw_total_mse']==pytest.approx(row['raw_mean_mse']+row['raw_centered_mse'])
        assert row['normalized_total_mse']==pytest.approx(row['normalized_mean_mse']+row['normalized_centered_mse'])
        assert row['outside_fraction']==0


def test_new_head_fits_center_without_local_audio_and_refit_zero_is_bypass(tmp_path):
    q=fixture();cache=runner.compact_targets(q);ids=torch.arange(18)
    args=SimpleNamespace(device='cpu',batch_size=8)
    model,stats=runner.refit(cache,ids,1,args,tmp_path/'fit.pt',31)
    before=runner.predict_centers(model,cache,ids,stats,'cpu')
    assert torch.isfinite(before).all() and ((before>=0)&(before<=1)).all()
    cache['target'].fill_(100)
    assert torch.equal(before,runner.predict_centers(model,cache,ids,stats,'cpu'))
    model,_=runner.refit(cache,ids,0,args,tmp_path/'zero.pt',31)
    assert model is None
    saved=torch.load(tmp_path/'zero.pt',weights_only=False)
    assert saved['bypass'] and saved['model'] is None
