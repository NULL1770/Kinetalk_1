"""One-time history leakage, fixed-clock fairness and read-only diagnostics."""
import copy

import pytest
import torch
from torch import nn

from scripts import diagnose_initial_history as diagnostic
from scripts import train_context_mechanism as context
from kinetalk_b0.models.prefix_upper_flow import PrefixUpperFlow


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    gen = torch.Generator().manual_seed(693)
    valid = torch.ones(3, 96, dtype=torch.bool)
    valid[0, 10:12] = False
    valid[0, 31:33] = False
    valid[1, :4] = False
    valid[1, 65:68] = False
    valid[2, 80:] = False
    q = {'valid':valid, 'motion':torch.rand(3,96,52,generator=gen),
         'h0':torch.randn(3,96,4,generator=gen), 'prefix_local':torch.randn(3,96,3,generator=gen),
         'audio_global':torch.randn(3,3,generator=gen), 'audio_intensity':torch.ones(3,1),
         'static_upper':torch.rand(3,9,generator=gen), 'channel_mask':torch.ones(3,52,dtype=torch.bool),
         'clip_id':['a','b','c'], 'speaker_id':torch.arange(3),
         'times':torch.arange(96,dtype=torch.float64)[None].expand(3,-1)*.04}
    for key in ('motion','h0','prefix_local'):
        q[key][~valid] = float('nan')
    identities = {i:{'code':torch.tensor([[float(i),.5]]),'baseline':torch.zeros(1,52)} for i in range(3)}
    identity = {'code':torch.tensor([[0.,.5],[1.,.5],[2.,.5]])}
    scales = torch.arange(1,10).float()*.1
    noise = torch.randn(3,96,9,generator=gen)
    conditions = {key:q[key] for key in diagnostic.CONDITION_KEYS}
    prefix = (q['motion'][:,:16,diagnostic.p.CC]-q['static_upper'][:,None])/scales
    return q, identities, identity, scales, noise, conditions, prefix


class RecordingFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.calls = []
        self.eval()

    def decode_prefix(self, valid,h0,identity,affect,local,noise,*,known,known_mask,steps):
        assert steps == 12 and not torch.is_grad_enabled()
        self.calls.append({k:v.clone() for k,v in {'valid':valid,'h0':h0,'identity':identity,
            'local':local,'noise':noise,'known':known,'known_mask':known_mask,
            'global':affect['global'],'intensity':affect['intensity_value']}.items()})
        mean = known.sum(1)/known_mask.sum(1)[:,None].clamp_min(1)
        value = noise*.03 + mean[:,None]*.7 + local[:,:,:1]*.1 + h0[:,:,:1]*.01
        return torch.where(known_mask[...,None],known,torch.where(valid[...,None],value*self.weight,0.))


def test_normal_exactly_matches_original_context_receiver_calls_and_noise():
    q, _, identity, _, noise, conditions, _ = fixture()
    a,b = RecordingFlow(),RecordingFlow()
    expected = context.decode_context(a,conditions,identity,q['prefix_local'],noise,arm='chunk_teacher')
    actual = diagnostic.decode_initial_history(b,conditions,identity,q['prefix_local'],noise)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    for old,new in zip(a.calls,b.calls):
        for key in old:
            torch.testing.assert_close(old[key],new[key],atol=0,rtol=0,equal_nan=True)


@pytest.mark.parametrize('initialization',('gt_prefix16','gt_terminal_static'))
def test_gt_is_injected_once_with_original_masks_then_history_is_generated(initialization):
    q, _, identity, _, noise, conditions, prefix = fixture()
    model = RecordingFlow()
    output = diagnostic.decode_initial_history(model,conditions,identity,q['prefix_local'],noise,
        initialization=initialization,gt_prefix=prefix)
    assert len(model.calls) == 6
    for start,call in zip(range(0,96,16),model.calls):
        assert not call['known_mask'][:,8:].any() and not call['known'][:,8:].any()
        if start == 0:
            assert not call['known_mask'].any()
            continue
        ids = call['identity'][:,0].long()
        native_mask = q['valid'][ids,start-8:start]
        assert torch.equal(call['known_mask'][:,:8],native_mask)
        expected = prefix[ids,8:16] if start == 16 else output[ids,start-8:start]
        if start == 16 and initialization == 'gt_terminal_static':
            expected = prefix[ids,15:16].expand(-1,8,-1)
        torch.testing.assert_close(call['known'][:,:8][native_mask],expected[native_mask],atol=0,rtol=0)
        assert not call['known'][:,:8][~native_mask].any()
    assert not output[~q['valid']].any() and torch.isfinite(output).all()


def test_decoder_rejects_full_target_and_normal_gt_and_ignores_prefix0_to8():
    q, _, identity, _, noise, conditions, prefix = fixture()
    with pytest.raises(ValueError,match='rejects GT'):
        diagnostic.decode_initial_history(RecordingFlow(),conditions,identity,q['prefix_local'],noise,gt_prefix=prefix)
    with pytest.raises(ValueError,match='16-frame'):
        diagnostic.decode_initial_history(RecordingFlow(),conditions,identity,q['prefix_local'],noise,
            initialization='gt_prefix16',gt_prefix=torch.randn(3,96,9))
    with pytest.raises(ValueError,match='declared acoustic'):
        diagnostic.decode_initial_history(RecordingFlow(),{**conditions,'motion':q['motion']},identity,q['prefix_local'],noise)
    expected = diagnostic.decode_initial_history(RecordingFlow(),conditions,identity,q['prefix_local'],noise,
        initialization='gt_prefix16',gt_prefix=prefix)
    prefix[:,:8] = float('nan')
    actual = diagnostic.decode_initial_history(RecordingFlow(),conditions,identity,q['prefix_local'],noise,
        initialization='gt_prefix16',gt_prefix=prefix)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_local_static_only_changes_local_not_h0_global_clock_or_fixed_noise():
    q, _, identity, _, noise, conditions, prefix = fixture()
    a,b = RecordingFlow(),RecordingFlow()
    for model,condition in ((a,'full'),(b,'local_static')):
        diagnostic.decode_initial_history(model,conditions,identity,q['prefix_local'],noise,
            initialization='gt_prefix16',gt_prefix=prefix,condition=condition)
    for left,right in zip(a.calls,b.calls):
        for key in ('valid','h0','identity','noise','known_mask','global','intensity'):
            torch.testing.assert_close(left[key],right[key],atol=0,rtol=0,equal_nan=True)
    for row in range(3):
        full = q['prefix_local'][row,q['valid'][row]]
        expected = full.mean(0)
        for call in b.calls:
            selected = (call['identity'][:,0] == row).nonzero(as_tuple=True)[0]
            if not len(selected):
                continue
            index = int(selected[0])
            good = call['valid'][index]
            torch.testing.assert_close(call['local'][index,good],expected[None].expand(int(good.sum()),-1))
    torch.testing.assert_close(a.calls[1]['known'],b.calls[1]['known'],atol=0,rtol=0)


def test_static_terminal_oracle_reads_only_index15_not_other_real_prefix_values():
    q,_,identity,_,noise,conditions,prefix = fixture()
    before = diagnostic.decode_initial_history(RecordingFlow(),conditions,identity,q['prefix_local'],noise,
        initialization='gt_terminal_static',gt_prefix=prefix)
    changed = prefix.clone()
    changed[:,:15] = 10000.
    after = diagnostic.decode_initial_history(RecordingFlow(),conditions,identity,q['prefix_local'],noise,
        initialization='gt_terminal_static',gt_prefix=changed)
    torch.testing.assert_close(before,after,atol=0,rtol=0)


def test_eligibility_uses_fixed_index_never_moves_start_to_next_valid_frame():
    valid = torch.ones(5,96,dtype=torch.bool)
    valid[0,8:15] = False  # retained gaps are allowed, terminal15 observed
    valid[1,15] = False
    valid[2,16:32] = False
    valid[3,48:] = False
    mask = torch.ones(5,9,dtype=torch.bool)
    mask[4,2] = False
    eligible,reasons = diagnostic.eligibility(valid,mask)
    assert eligible.tolist() == [True,False,False,False,False]
    assert reasons['native_index15_invalid'].tolist() == [False,True,False,False,False]


def test_scoring_excludes_prefix_and_suffix_boundary_and_does_not_bridge_gaps():
    valid = torch.ones(1,96,dtype=torch.bool)
    valid[:,31] = False
    cmask = torch.ones(1,9,dtype=torch.bool)
    target = torch.arange(96).float()[None,:,None].expand(1,-1,9).clone()*.01
    pred = target.clone()
    pred[:,:16] = 10000.
    pred[:,31] = float('nan')
    report = diagnostic.score_window(pred,target,valid,cmask,16)
    for group,cc in diagnostic.GROUPS.items():
        row = report['groups'][group]
        assert row['paired']['raw_mse'] == 0
        assert row['speed']['displacement_mse'] == 0
        assert row['speed']['observed_channel_pairs'] == 77*len(cc)
        assert row['seams']['observed_channel_pairs'] == 3*len(cc)
    changed = target.clone()
    changed[:,:48] += 999.
    late = diagnostic.score_window(changed,target,valid,cmask,48)
    assert late['groups']['brows']['paired']['raw_mse'] == 0


def test_evaluation_future_target_and_prefix_first8_do_not_affect_any_prediction():
    q,identities,_,scales,_,_,_ = fixture()
    before = copy.deepcopy(q)
    state = torch.random.get_rng_state().clone()
    report,curves = diagnostic.evaluate_initial_history(RecordingFlow(),q,identities,scales,batch_size=2)
    assert torch.equal(state,torch.random.get_rng_state())
    changed = copy.deepcopy(q)
    changed['motion'][:,16:] += 50.
    changed['motion'][:,:8] -= 80.
    _,other = diagnostic.evaluate_initial_history(RecordingFlow(),changed,identities,scales,batch_size=1)
    assert len(curves['upper_predictions9']) == 18
    for key,value in curves['upper_predictions9'].items():
        torch.testing.assert_close(value,other['upper_predictions9'][key],atol=1e-6,rtol=1e-6)
        assert not value[~q['valid']].any()
    for key,value in q.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value,before[key],atol=0,rtol=0,equal_nan=True)
        else:
            assert value == before[key]
    assert report['future_GT_or_mean_input'] is False and report['test_loaded'] is False
    assert report['nonupper_scored'] is False and curves['not_fullface_render_input'] is True
    assert set(report['mean_over_three_seeds']) == {i+'/'+c for i in diagnostic.INITIALIZATIONS for c in diagnostic.CONDITIONS}


def test_smoke_selection_keeps_original_cohort_noise_indices_and_weights_unchanged():
    q,identities,_,scales,_,_,_ = fixture()
    q['valid'][0,15] = False
    model = RecordingFlow()
    before = copy.deepcopy(model.state_dict())
    report,curves = diagnostic.evaluate_initial_history(model,q,identities,scales,max_clips=1)
    assert report['cohort_indices'] == [1] and report['eligible_count_before_smoke_limit'] == 2
    expected = torch.randn(3,96,9,generator=torch.Generator().manual_seed(42))[1,0:16]
    torch.testing.assert_close(model.calls[0]['noise'][0,8:][q['valid'][1,:16]],expected[q['valid'][1,:16]],atol=0,rtol=0)
    for key in before:
        assert torch.equal(before[key],model.state_dict()[key])
    assert model.weight.grad is None
    assert curves['clip_id'] == ['b']


def test_fit_membership_uses_metadata_only_and_rejects_foreign_or_mismatched_rows():
    class MetadataOnly(dict):
        def __getitem__(self,key):
            assert key in ('clip_id','speaker_id','emotion_id')
            return super().__getitem__(key)
    train = MetadataOnly(clip_id=[f'fit_{i}' for i in range(2315)],
                         speaker_id=torch.zeros(2315,dtype=torch.long),emotion_id=torch.ones(2315,dtype=torch.long))
    selection = {'count':128,'selection_uses_motion':False,'clips':[
        {'clip_id':f'fit_{i}','original_fit_index':i,'speaker_id':0,'emotion_id':1} for i in range(128)]}
    assert diagnostic.historical_fit_indices(train,selection).tolist() == list(range(128))
    selection['clips'][0]['clip_id'] = 'sealed_test'
    with pytest.raises(ValueError,match='member metadata'):
        diagnostic.historical_fit_indices(train,selection)


def test_real_receiver_normal_replay_matches_original_and_parameters_stay_frozen():
    cfg = {'model':{'content_dim':4,'emotion_dim':3,'style_dim':2,
                    'dit_dim':8,'dit_depth':1,'heads':2,'dropout':0.}}
    with torch.random.fork_rng():
        torch.manual_seed(731)
        model = PrefixUpperFlow(cfg).eval().requires_grad_(False)
    q,_,identity,_,noise,conditions,prefix = fixture()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    expected = context.decode_context(model,conditions,identity,q['prefix_local'],noise,arm='chunk_teacher')
    actual = diagnostic.decode_initial_history(model,conditions,identity,q['prefix_local'],noise)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    oracle = diagnostic.decode_initial_history(model,conditions,identity,q['prefix_local'],noise,
        initialization='gt_prefix16',gt_prefix=prefix)
    assert torch.isfinite(oracle).all()
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert not oracle.requires_grad
