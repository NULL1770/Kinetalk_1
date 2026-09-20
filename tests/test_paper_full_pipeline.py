import json
from pathlib import Path
import pytest
import torch
from scripts.prepare_paper_full_data import read_native,validate_manifest
from kinetalk_b0.models.audio_residual_flow import fair_trajectory_es,static_audio,AudioResidualFlow
from kinetalk_b0.models.slow_state_affect import project_upper_innovation


def test_sealed_target_read_fails_before_path_access(tmp_path):
    with pytest.raises(ValueError,match='Test artifact'):
        read_native({'source_split':'test','artifact':'not_there'},tmp_path)


def test_manifest_seals_and_reference_queries_checked_even_with_valid_hash():
    from scripts.train_formal_predictable_projection import canonical_hash
    def make():
        roles={}
        for role in ('train','val','test'):
            common={'speaker':role+'_person','source_split':role,'dataset':'mead','emotion':0}
            roles[role]={'query':[{**common,'clip_id':role+'_q','sentence':'query_sentence'}],
                'enrollment':[{**common,'clip_id':role+'_r'+str(i),'sentence':'ref'+str(i)} for i in range(2)]}
        return {'status':'approved_train_val_only','sealed_test_targets_loaded':False,
                'reserved_sentences':['sealed'],'roles':roles}
    m=make();m['manifest_sha256']=canonical_hash(m);validate_manifest(m)
    m=make();m['roles']['train']['query'][0]['sentence']='sealed';m['manifest_sha256']=canonical_hash(m)
    with pytest.raises(ValueError,match='sealed sentence'):validate_manifest(m)
    m=make();m['roles']['val']['query'][0]['sentence']='ref0';m['manifest_sha256']=canonical_hash(m)
    with pytest.raises(ValueError,match='speaker-sentence'):validate_manifest(m)


def test_complete_native_read_retains_frames_beyond_old96(tmp_path):
    import numpy as np
    from scripts.extract_emotion2vec_pilot import sha
    n=113;motion=np.zeros((n,52),dtype=np.float32);motion[-1,43]=.6
    path=tmp_path/'clip.npz'
    np.savez(path,motion=motion,content=np.zeros((n,768)),audio=np.zeros((n,83)),
             times=np.arange(n)/25,mask=np.ones(n,dtype=bool),channel_mask=np.ones(52,dtype=bool),
             provenance=json.dumps({'clock_evidence':'embedded_video','fps':25}))
    row={'source_split':'train','artifact':'clip.npz','artifact_sha256':sha(path),'clip_id':'clip',
         'frames':n,'valid_frames':n}
    result=read_native(row,tmp_path)
    assert len(result['motion'])==113 and result['motion'][-1,43]==pytest.approx(.6)


def test_fair_es_preserves_distribution_instead_of_collapsing_to_mean():
    # On Y ~ {-1,+1}, two iid draws in expectation attain the proper score,
    # whereas deterministic mean has the strictly worse value 1.
    valid=torch.ones(1,3,dtype=torch.bool)
    scores=[]
    for y in (-1.,1.):
        truth=torch.full((1,3,1),y)
        for a in (-1.,1.):
            for b in (-1.,1.):
                draws=torch.stack([torch.full_like(truth,a),torch.full_like(truth,b)])
                scores.append(fair_trajectory_es(draws,truth,valid))
    assert torch.stack(scores).mean().item()==pytest.approx(.5)
    assert fair_trajectory_es(torch.zeros(2,1,3,1),truth,valid).item()==pytest.approx(1.)


def test_es_mask_and_zero_distance_have_finite_gradients():
    valid=torch.tensor([[True,False,True]])
    y=torch.zeros(1,3,2);y[:,1]=float('nan')
    x=torch.zeros(2,1,3,2,requires_grad=True)
    loss=fair_trajectory_es(x,y,valid,centered=True);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert x.grad[:,:,1].count_nonzero()==0


def test_static_audio_erases_permutation_without_using_targets():
    x=torch.randn(2,9,6);valid=torch.ones(2,9,dtype=torch.bool)
    torch.testing.assert_close(static_audio(x,valid),static_audio(x.flip(1),valid))
    assert (static_audio(x,valid)[:,0]-static_audio(x,valid)[:,-1]).abs().max()==0


def test_new_flow_does_not_project_away_slow_random_variation():
    cfg={'content_dim':8,'emotion_dim':8,'style_dim':8,'dit_dim':12,'dit_depth':1,'heads':3,'dropout':0.}
    model=AudioResidualFlow(cfg,stride=4)
    # Zero velocity makes integration retain its supplied noise exactly. The
    # earlier Q projection would erase this constant slow random component.
    for p in model.parameters():p.data.zero_()
    valid=torch.ones(1,8,dtype=torch.bool);q={'valid':valid,'h0':torch.zeros(1,8,8)}
    noise=torch.ones(1,8,9);state=torch.zeros(1,8,4);local=torch.zeros(1,8,8)
    affect={'global':torch.zeros(1,8),'intensity_value':torch.zeros(1,1)}
    result=model.decode(q,{'code':torch.zeros(1,8)},affect,local,state,noise,2)
    torch.testing.assert_close(result,noise)
    torch.testing.assert_close(project_upper_innovation(result,valid,stride=4),torch.zeros_like(result),atol=1e-6,rtol=0)


def test_queue_pairing_rejects_different_clips_and_keeps_direction(tmp_path):
    from scripts.run_paper_full_queue import compare
    root=tmp_path/'run';art=tmp_path/'artifacts'
    names=('arkit_mbe','arkit_lbe','arkit_fdd_signed','arkit_fdd_absolute','supp_upper9_fdd_absolute')
    for label,score in [('audio',.2),('static',.4)]:
        folder=root/label/'dynamics';folder.mkdir(parents=True)
        report={mode:{'coefficient':{n:{'value':score} for n in names},'temporal':{}}
                for mode in ('full','static_state','oracle_state','reverse_audio')}
        (folder/'benchmark_summary.json').write_text(json.dumps(report),encoding='utf8')
        folder=art/label/'dynamics';folder.mkdir(parents=True)
        records=[{'clip_id':str(i),'sentence':'sentence'+str(i),'scales':[1]*9,
                  'joint_fair_es':{'raw':score,'centered':score}} for i in range(3)]
        (folder/'temporal_full.json').write_text(json.dumps(records),encoding='utf8')
    compare(root,art)
    result=json.loads((root/'paired_intervals.json').read_text())
    assert result['centered']['audio_minus_static']==pytest.approx(-.2)
    assert result['centered']['sentence_cluster_95ci']==pytest.approx([-.2,-.2])
    path=art/'static/dynamics/temporal_full.json';records=json.loads(path.read_text());records[0]['clip_id']='foreign'
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError,match='memberships'):compare(root,art)
