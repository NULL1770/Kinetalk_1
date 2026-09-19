import copy
import json

import pytest
import torch

from scripts import train_prior_audio_adapter as r


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def clips():
    torch.manual_seed(513)
    out=[]
    upper=[41,42,43,44,45,5,6,12,13]
    for index in range(8):
        n=11; baseline=torch.full((n,52),.25)
        target=baseline.clone(); target[:,upper] += torch.sin(torch.linspace(0,5,n))[:,None]*.06
        out.append({'clip_id':f'c{index}','sentence':f's{index}','split':'train' if index<6 else 'holdout',
            'speaker':index%2,'emotion':index%2, 'features':torch.randn(n,1540),
            'context':torch.randn(202),'b9':torch.full((9,),.25),'valid':torch.ones(n,dtype=torch.bool),
            'motion_mask':torch.ones(n,9,dtype=torch.bool),'motion9':target[:,upper],
            'baseline52':baseline,'target52':target,'times':torch.arange(n)/25})
    return out


def test_sentence_split_stats_and_diagnostics_do_not_use_outer_or_inner_targets():
    source=clips(); train,valid,outer,held=r.inner_split(source,2)
    assert len(train)==4 and len(valid)==2 and len(outer)==2
    assert not {x['sentence'] for x in train}&set(held)
    changed=copy.deepcopy(source)
    for c in changed:
        if c['sentence'] in held or c['split']=='holdout':
            c['features'][:]=1e8;c['motion9'][:]=1e8
    a=r.fit_statistics(train); b=r.fit_statistics(r.inner_split(changed,2)[0])
    for k in a:
        if torch.is_tensor(a[k]):torch.testing.assert_close(a[k],b[k])
    assert len({c['sentence'] for c in r.diverse_diagnostics(source,6)})==6


def test_static_clock_uses_full_audio_runs_not_target_mask_or_chunks():
    source=clips()[0]; source['valid'][5]=False
    source['motion_mask'][2:4]=False
    transformed=r.static_features([source])[0]
    for start,stop in ((0,5),(6,11)):
        torch.testing.assert_close(transformed['features'][start:stop],source['features'][start:stop].mean(0).expand(stop-start,-1))
    torch.testing.assert_close(source['features'][5],transformed['features'][5])
    assert transformed['motion9'] is source['motion9']


def test_complete_two_milestone_pipeline_then_resume(tmp_path):
    dataset=tmp_path/'dataset.pt';torch.save({'schema':r.SCHEMA,'clips':clips()},dataset)
    output=tmp_path/'run';output.mkdir();(output/'launch.json').write_text('{}')
    args=r.parser().parse_args(['--dataset',str(dataset),'--output',str(output),'--device','cpu',
        '--ae-steps','1','--prior-steps','1','--adapter-steps','2','--milestones','1','2',
        '--valid-sentences','2','--batch-size','2','--smoke','--skip-quality-gates'])
    r.run(args)
    status=json.loads((output/'status.json').read_text())
    assert status['state']=='complete' and status['total_updates']==6
    for step in (1,2):
        for arm in ('audio','matched_static'):
            saved=r.common._load(output/f'point_{step}'/(arm+'.pt'))
            assert saved['step']==step
        assert all(json.loads((output/f'point_{step}/pairing.json').read_text()).values())
    assert (output/'outer/audio_generation/holdout/result.json').exists()
    final_time=(output/'audio_final.pt').stat().st_mtime_ns
    args.resume=True;r.run(args)
    assert (output/'audio_final.pt').stat().st_mtime_ns==final_time
    args.max_delta=.6
    with pytest.raises(ValueError,match='protocol'):r.run(args)


def test_earliest_passing_not_best_and_gate_override_restricted(tmp_path):
    assert r.select_candidate([{'step':3000,'accepted':True},{'step':1000,'accepted':True}])['step']==1000
    assert r.select_candidate([{'step':1000,'accepted':False}]) is None
    args=r.parser().parse_args(['--dataset','missing','--output',str(tmp_path),'--skip-quality-gates'])
    with pytest.raises(ValueError,match='smoke'):r.run(args)
