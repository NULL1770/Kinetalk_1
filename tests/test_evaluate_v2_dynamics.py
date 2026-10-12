import numpy as np
import pytest

from scripts.evaluate_v2_dynamics import (contiguous_runs, event_execution, events_from_tensor,
    match_events, regional_dynamics, probability_scores, seed_diversity,
    expression_direction_error, wasserstein_1d)


def test_regional_statistics_never_bridge_invalid_gap_or_wrong_clock():
    x = np.zeros((7,52), dtype=float)
    x[:3,0] = [0., .1, .2]
    x[4:,0] = [10., 10.1, 10.2]
    valid = np.array([1,1,1,0,1,1,1], bool)
    times = np.arange(7)*.04
    # A second discontinuity in nominal indices must also split the run.
    times[5] += .02
    stats = regional_dynamics(x, valid, times, np.ones(52,bool), [0])
    assert stats['observed_channel_pairs'] == 2  # 2 + 0 adjacent valid pairs
    assert stats['observed_channel_triples'] == 1
    assert stats['mean_abs_acceleration_per_second2'] == pytest.approx(0.)
    assert stats['mean_abs_velocity_per_second'] == pytest.approx((2.5+2.5)/2)
    # The 9.8 coefficient jump across the invalid row is absent.
    assert stats['mean_abs_velocity_per_second'] < 10.
    assert stats['observed_channel_frames'] == 6


def test_event_matching_is_one_to_one_and_tolerance_limited_by_run():
    target = [
        {'time': 1.00, 'run': 0, 'duration_seconds': .12, 'amplitude': .3},
        {'time': 1.40, 'run': 0, 'duration_seconds': .20, 'amplitude': .5},
    ]
    pred = [
        {'time': 1.06, 'run': 0, 'duration_seconds': .16, 'amplitude': .25},
        {'time': 1.07, 'run': 0, 'duration_seconds': .16, 'amplitude': .25},
        {'time': 1.80, 'run': 1, 'duration_seconds': .20, 'amplitude': .5},
    ]
    pairs = match_events(target, pred, .08)
    assert pairs == [(0,0)]
    rows = event_execution([target, [], [], []], [pred, [], [], []], .08)
    assert rows['eyelid_closure']['matched_events'] == 1
    assert rows['eyelid_closure']['realized_events'] == 3


def test_event_parser_censors_boundaries_and_keeps_complete_event_distribution():
    valid = np.ones(8, bool); times = np.arange(8)*.04
    events = np.zeros((8,4,3), float)
    events[0:2,0,0] = 1.  # left boundary, censored
    events[3:5,0,0] = 1.; events[3:5,0,1] = [0.,1.]; events[3:5,0,2] = .2
    events[6:8,0,0] = 1.  # right boundary, censored
    parsed, censored = events_from_tensor(events, valid, times)
    assert len(parsed[0]) == 1 and parsed[0][0]['frame'] == 3
    assert censored[0] == 2
    valid[5] = False
    parsed, censored = events_from_tensor(events, valid, times)
    assert not parsed[0] and censored[0] == 3


def test_contiguous_runs_split_on_gap_and_clock_discontinuity():
    valid = np.array([1,1,1,1,1], bool); times=np.arange(5)*.04;times[3]+=.01
    assert contiguous_runs(valid,times)[0] == [(0,3),(3,4),(4,5)]


def test_unobserved_or_masked_regions_return_zero_and_do_not_read_nan_targets():
    x=np.full((6,52),np.nan);times=np.arange(6)*.04
    none=regional_dynamics(x,np.zeros(6,bool),times,np.ones(52,bool),[0,7])
    assert all(v==0 for v in none.values())
    disabled=regional_dynamics(x,np.ones(6,bool),times,np.zeros(52,bool),[0,7])
    assert all(v==0 for v in disabled.values())
    x[:,0]=.5
    valid=np.array([1,1,0,1,1,1],bool)
    x[2,0]=np.nan
    flat=regional_dynamics(x,valid,times,np.array([True]+[False]*51),[0,7])
    assert flat['within_run_population_std']==0 and flat['mean_abs_velocity_per_second']==0
    assert flat['mean_abs_acceleration_per_second2']==0


def test_maximum_matching_rejects_cross_gap_and_respects_exact_tolerance():
    def event(t,run=0):return {'time':t,'run':run}
    assert match_events([event(1.)],[event(1.080001)],.08)==[]
    assert match_events([event(1.)],[event(1.08)],.08)==[(0,0)]
    assert match_events([event(1.,0)],[event(1.,1)],.08)==[]
    # Both may match the second candidate; ordered maximum matching keeps two.
    assert match_events([event(0.),event(.1)],[event(-.06),event(.05)],.08)==[(0,0),(1,1)]


def test_probability_masks_exclude_nan_and_invalid_duration_targets():
    import torch
    hazard=torch.zeros(1,3,4); duration=torch.zeros(1,3,4,5)
    targets={'onset':torch.zeros(1,3,4,dtype=torch.bool),
             'onset_valid':torch.zeros(1,3,4,dtype=torch.bool),
             'duration':torch.full((1,3,4),-99,dtype=torch.long),
             'duration_valid':torch.zeros(1,3,4,dtype=torch.bool)}
    hazard[:]=float('nan');duration[:]=float('nan')
    report=probability_scores(hazard,duration,targets)
    assert all(v['onset_count']==0 and v['brier_sum']==0 and v['duration_conditional_nll_sum']==0 for v in report.values())
    targets['onset_valid'][0,1,0]=True;targets['onset'][0,1,0]=True;hazard[0,1,0]=0
    targets['duration_valid'][0,1,0]=True;targets['duration'][0,1,0]=2;duration[0,1,0]=0
    report=probability_scores(hazard,duration,targets)['eyelid_closure']
    assert report['onset_count']==1 and report['positive_onsets']==1
    assert report['brier_sum']==pytest.approx(.25)
    assert report['binary_nll_sum']==pytest.approx(np.log(2))
    assert report['duration_conditional_nll_sum']==pytest.approx(np.log(5))


def test_diversity_uses_every_pair_and_direction_is_not_emotional_truth():
    motions=np.zeros((3,5,52));motions[1,:,0]=.3;motions[2,:,0]=.9
    events=np.zeros((3,5,12));valid=np.array([1,1,0,1,1],bool)
    report=seed_diversity(motions,events,valid,np.array([1]+[0]*51,bool),[0])
    assert report['seed_pairs']==3 and report['upper_coefficient_pairwise_mae']==pytest.approx(.6)
    target=np.ones((5,10))*.1;pred=-target
    direction=expression_direction_error(pred,target,np.ones((5,10),bool))
    assert direction['direction_cosine_error']==pytest.approx(2.)
    assert direction['axis_metrics'][0]['nonzero_target_sign_error']==1.
    assert wasserstein_1d([0.,1.],[1.,2.])==pytest.approx(1.)
    assert wasserstein_1d([],[]) is None


def test_small_real_model_sample_diagnostic_has_no_observed_reconstruction():
    import torch
    from kinetalk_v2.unified_model import UnifiedExpressionModel,UnifiedConfig,teacher_targets
    torch.set_num_threads(2);torch.manual_seed(47)
    cfg=UnifiedConfig(hidden=16,decoder_hidden=24,content_dim=8)
    model=UnifiedExpressionModel(cfg,torch.zeros(772),torch.ones(772),torch.full((52,),.1)).eval()
    valid=torch.ones(1,12,dtype=torch.bool);valid[:,5]=False
    times=torch.arange(12,dtype=torch.float64)[None]/25
    base=torch.zeros(1,12,52);target=base.clone();target[:,2:4,0]=.8;target[:,2:4,7]=.8
    channels=torch.tensor([[True]*51+[False]])
    audio=torch.randn(1,12,1540);content=torch.randn(1,12,8)
    ref={'motion':torch.rand(1,12,52),'valid':valid,'channel_mask':channels}
    with torch.no_grad():
        weak=teacher_targets(target,base,valid,channels,times)
        output=model.predict(audio,content,base,valid,ref,times=times,seed=91)
        realized=teacher_targets(output['motion'],base,valid,channels,times)
        state=model.audio_state(audio,valid);style=model.encode_style(ref)
        condition=model.events(state['hidden'],state['state'],style,valid,weak['events'],mode='observed',times=times)
    score=probability_scores(condition['hazard_logits'],condition['duration_logits'],weak)
    assert score['eyelid_closure']['positive_onsets']==1
    y=output['motion'][0].numpy();stats=regional_dynamics(y,valid[0],times[0],channels[0],[0,7])
    assert stats['observed_channel_pairs']==18
    assert all(np.isfinite(v) for v in stats.values())
    schedule,_=events_from_tensor(output['schedule']['events'][0].numpy(),valid[0].numpy(),times[0].numpy())
    actual,_=events_from_tensor(realized['events'][0].numpy(),valid[0].numpy(),times[0].numpy(),duration_valid=realized['duration_valid'][0].numpy())
    assert set(event_execution(schedule,actual))=={'eyelid_closure','brow_emphasis','cheek_nose_emphasis','lip_tension'}

