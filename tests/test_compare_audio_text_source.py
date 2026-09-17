import copy

import pytest
import torch

from scripts import audit_audio_text_dynamics as audit
from scripts.compare_audio_text_source import analyze, validate_source_reproduction


def fixture():
    motion=torch.tensor([.1,.4,.2,.3])[None,:,None].expand(4,4,52).clone()
    ref={'q':{'motion':motion,'valid':torch.ones(4,4,dtype=torch.bool),'channel_mask':torch.ones(4,52,dtype=torch.bool),
        'times':torch.arange(4).float()[None].expand(4,-1)/25,'clip_id':['c0','c1','c2','c3'],
        'sentence_id':['s0','s1','s0','s1'],'speaker_id':torch.tensor([0,0,1,1]),'emotion_id':torch.tensor([0,1,0,1])},
        'base':{'b0':torch.zeros_like(motion)},'identity':{'baseline':torch.zeros(4,52)}}
    historical={'motion':{str(seed):{mode:motion.clone()*.9+.01*i for mode in ('full','zero','reverse','oracle')}
        for i,seed in enumerate(audit.SEEDS)}}
    uniform={'noise_seeds':[42,123,2026],'decode_steps':12,'motion':{s:copy.deepcopy(historical['motion'][s]) for s in ('42','123','2026')}}
    arms={arm:{'epoch0':copy.deepcopy(historical),'final':copy.deepcopy(historical)} for arm in audit.ARMS}
    return historical,uniform,arms,ref


def test_three_seed_original_and_eight_seed_zero_bound_exactly():
    h,u,arms,_=fixture();assert validate_source_reproduction(h,u,arms)['historical_all_eight_zero_equal_new_matched_epoch0']
    u['motion']['123']['full'][0,0,41]+=.1
    with pytest.raises(AssertionError):validate_source_reproduction(h,u,arms)


def test_eighth_seed_zero_mismatch_rejected():
    h,u,arms,_=fixture();h['motion']['997']['zero'][0,0,41]+=.1
    with pytest.raises(AssertionError):validate_source_reproduction(h,u,arms)


def test_original_seed_protocol_not_assumed_eight():
    h,u,arms,_=fixture();u['noise_seeds']=audit.SEEDS
    with pytest.raises(ValueError,match='sampling'):validate_source_reproduction(h,u,arms)


def test_comparison_retains_raw_mean_centered_mouth_and_per_person():
    h,_,arms,ref=fixture()
    for rows in arms['text']['final']['motion'].values():rows['full']+=.2
    result=analyze(h,arms,ref,samples=10)
    c=result['comparisons']['text/full_vs_old_full_local']
    assert c['raw_relative']['all']['mouth']['relative_mse_increase']>0
    assert c['mean_relative']['all']['mouth']['relative_mse_increase']>0
    assert abs(c['centered_relative']['all']['mouth']['relative_mse_increase'])<.01
    assert 'speaker_0/nonneutral' in c['centered']
    assert result['domain']['old_full_local']['all']['brows']['noise_seeds']==8
