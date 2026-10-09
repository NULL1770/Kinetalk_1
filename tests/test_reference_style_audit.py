import numpy as np
import pytest
from scripts.audit_reference_style_swap import summary,matched_pairs,target_direction,export,change,canonical_batches


def test_smoke_preserves_saved_batch_context_and_native_order():
    assert canonical_batches(35,[34,0,17])==[list(range(16)),list(range(16,32)),[32,33,34]]
    assert canonical_batches(35,[17,31])==[list(range(16,32))]
    with pytest.raises(ValueError):canonical_batches(35,[35])


def test_stats_ignore_invalid_unsupported_and_nonadjacent_displacements():
    x=np.zeros((5,52));x[:,17]=[0,1,900,3,4];x[:,51]=np.nan
    valid=np.array([True,True,False,True,True]);channels=np.array([True]*51+[False])
    s=summary(x,valid,channels,np.arange(5)*.04)
    assert s[0,17]==2 and s[3,17]==1 and not s[:,51].any()
    # Timestamp gap also removes an otherwise valid adjacent difference.
    x[:,17]=[0,1,2,10,11];times=np.array([0,.04,.08,.5,.54])
    s=summary(x,np.ones(5,dtype=bool),channels,times)
    assert s[3,17]==1


def test_pairs_are_directed_label_matched_and_reject_ambiguity():
    p,g=matched_pairs(['a','a','b','a'],[1,1,1,1],[2,2,2,3],[0,1,2,2])
    assert p==[(0,1),(1,0)] and len(g)==3
    with pytest.raises(ValueError,match='Ambiguous'):
        matched_pairs(['a','a'],[1,1],[2,2],[0,0])


def test_target_direction_rewards_correct_donor_not_arbitrary_change():
    own=np.zeros((4,52));donor=own.copy();donor[:,17]=.2
    ch=np.zeros(52,dtype=bool);ch[17]=True
    good=target_direction(own,donor,own,donor,np.zeros(52),np.zeros(52),ch,np.ones(52))
    bad=target_direction(own,-donor,own,donor,np.zeros(52),np.zeros(52),ch,np.ones(52))
    assert good['mouth/mean_improvement']>0 and good['mouth/mean_delta_cosine']==1
    assert bad['mouth/mean_improvement']<0 and bad['mouth/mean_delta_cosine']==-1


def test_render_exports_longest_actual_span_preserving_native_start(tmp_path):
    valid=np.array([True,True,False,True,True,True,False]);times=.2+np.arange(7)*.04
    motions=np.arange(2*7*52).reshape(2,7,52)
    p=tmp_path/'render.npz'
    r=export(p,motions,valid,times,np.ones(52,dtype=bool),'query',['GT','donor'],['none','ref'])
    with np.load(p,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['motions'],motions[:,3:6])
        np.testing.assert_array_equal(z['times'],times[3:6])
        assert z['valid'].all() and not z['query_gt_used_for_deployment']
    assert r['native_start_frame']==3 and r['frames']==3


def test_timing_diagnostic_detects_native_shift_and_does_not_retime():
    x=np.zeros((60,52));x[:,17]=np.random.default_rng(4).uniform(.1,.7,60)
    y=x.copy();y[2:,17]=x[:-2,17];y[:2,17]=.3
    v=np.ones(60,dtype=bool);ch=np.array([True]*51+[False]);t=np.arange(60)*.04
    same=change(x,x,v,ch,t);shift=change(x,y,v,ch,t)
    assert same['jaw_best_lag_frames']==0 and shift['jaw_best_lag_frames']==2
    assert same['mouth/centered_delta_rms']==0


@pytest.mark.parametrize('mode',['temporal','statistics'])
def test_candidate_restore_replays_own_checkpoint_without_parent_or_correction(tmp_path,mode):
    import torch
    from types import SimpleNamespace
    from tests.test_reference_response import fixture
    from scripts.audit_reference_style_swap import load_receiver,compute
    from scripts.train_expression_response import sha
    _,model,b,refs=fixture(mode);model.eval().requires_grad_(False)
    path=tmp_path/'final.pt'
    torch.save(dict(model=model.state_dict(),config=model.checkpoint_config(),
        parent_checkpoint_sha256='parent',test_loaded=False),path)
    binding=dict(parent_checkpoint={'sha256':'parent'},
        style_audit={'candidate_checkpoint':{'sha256':sha(path)}})
    args=SimpleNamespace(candidate=path,candidate_curves='own_curves.pt',
        correction=None,parent_curves=None,latent_curves=None)
    restored,correction,curves=load_receiver(args,binding,'cpu')
    restored.eval().requires_grad_(False)
    with torch.no_grad():
        p=model.audio_prior(b['audio_features'],b['valid'])
        y,_=compute(restored,correction,b,p,refs)
        assert tuple(y)==('candidate',) and correction is None
        assert curves=={'candidate':'own_curves.pt'}
        torch.testing.assert_close(y['candidate'],model.predict(b['audio_features'],b['b0'],b['valid'],refs),rtol=0,atol=0)
    args.correction='forbidden_old_correction.pt'
    with pytest.raises(ValueError,match='cannot reuse'):load_receiver(args,binding,'cpu')
    args.correction=None;binding['style_audit']['candidate_checkpoint']['sha256']='wrong'
    with pytest.raises(AssertionError):load_receiver(args,binding,'cpu')
