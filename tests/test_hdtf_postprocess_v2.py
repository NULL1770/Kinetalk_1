import json
from pathlib import Path
import numpy as np
import pytest
from kinetalk_v2.hdtf_postprocess import Policy, aggregate_audit, blink_events, candidate_metrics, process_file, smooth_runs

def channels():
    x=["EyeBlinkLeft","EyeLookDownLeft","EyeLookInLeft","EyeLookOutLeft","EyeLookUpLeft","EyeSquintLeft","EyeWideLeft","EyeBlinkRight","EyeLookDownRight","EyeLookInRight","EyeLookOutRight","EyeLookUpRight","EyeSquintRight","EyeWideRight","JawForward","JawLeft","JawRight","JawOpen"]
    x += [f"Mouth{i}" for i in range(23)] + [f"Brow{i}" for i in range(5)] + [f"Cheek{i}" for i in range(5)] + ["TongueOut"]
    return np.array(x, dtype=object)

def test_runs_never_bridge_invalid_gap_and_short_run_raw():
    x=np.zeros((30,52),np.float32); x[:,17]=np.linspace(0,1,30); valid=np.ones(30,bool); valid[12:14]=False
    out,smoothed,fallback=smooth_runs(x,valid,Policy(5))
    assert np.array_equal(out[12:14],x[12:14]); assert not smoothed[12:14].any(); assert fallback[12:14].all()
    assert smoothed[:12,17].any()

def test_blink_and_jaw_metric_threshold_is_explicit():
    raw=np.zeros((30,52),np.float32); cand=raw.copy(); raw[:,17]=.04; cand[:,17]=.10
    raw[10:15,0]=raw[10:15,7]=.8; cand[10:14,0]=cand[10:14,7]=.75
    m=candidate_metrics(raw,cand,17,(0,7),.05)
    assert m["jaw"]["closure_rate_delta"] < 0
    assert m["blink"]["0"]["peak_retention"] == pytest.approx(.9375)
    assert m["closure_threshold"] == .05

def test_smoothing_guard_falls_back_for_disappearing_blink():
    x=np.zeros((30,52),np.float32); x[13:15,0]=1.; x[13:15,7]=1.; valid=np.ones(30,bool)
    out, smoothed, fallback=smooth_runs(x,valid,Policy(7))
    assert fallback[:,0].any() and fallback[:,7].any()
    np.testing.assert_array_equal(out[:,0],x[:,0])

def test_summary_ignores_nan_and_records_provenance():
    r={"n_frames":10,"fallback_fraction":0.,"smoothed_fraction":1.,"split":"train","metrics":{"mean_abs_velocity_ratio":.8,"mean_abs_acceleration_ratio":.4,"jaw":{"peak_retention":1.,"closure_rate_delta":0.,"duration_ratio":1.},"blink":{"0":{"peak_retention":1.,"duration_ratio":1.},"7":{"peak_retention":1.,"duration_ratio":1.}}}}
    s=aggregate_audit([r],5); assert s["candidate_window"]==5 and s["n_sources"]==1 and s["median_acceleration_ratio"]==.4

def test_jaw_closure_episode_and_clock_break_remain_raw():
    x=np.zeros((30,52),np.float32); x[:,17]=.5; x[13:15,17]=.01
    valid=np.ones(30,bool); times=np.arange(30)/25; times[20:]+=.04
    out, smoothed, fallback=smooth_runs(x,valid,Policy(5),times=times)
    np.testing.assert_array_equal(out[11:17,17],x[11:17,17])
    assert fallback[11:17,17].all()
    metrics=candidate_metrics(x,out,17,(0,7),motion_valid=valid,times=times)
    assert metrics['valid_adjacent_pairs']==28 and metrics['valid_adjacent_triples']==26

def test_local_event_identity_and_named_channel_raw_hash_receipt(tmp_path):
    x=np.zeros((30,52),np.float32);x[5,0]=.6;x[7,0]=1.;x[13:15,17]=.01
    for event in blink_events(x[:,0],x[:,0],Policy(5)):
        assert event['peak_retention']==1 and event['duration_ratio']==1 and event['area_ratio']==1
    # Permute channel names and arrays together, proving name-based guards.
    idx=np.arange(52)[::-1];raw=tmp_path/'source.npz';out=tmp_path/'motion'/'source.npz'
    np.savez(raw,motion=x[:,idx],times=np.arange(30)/25,motion_valid=np.ones(30,bool),
             channel_mask=np.ones(52,bool),channels=np.asarray(channels()[idx],dtype=str))
    from kinetalk_v2.hdtf import sha256
    before=sha256(raw)
    result=process_file(raw,out,Policy(5),source_record={'split':'train','source_group':'a','speaker_group':'p'})
    assert sha256(raw)==before and result['raw_motion_sha256']==before
    assert Path(result['motion_processed_path']).is_file()
    assert process_file(raw,out,Policy(5))['processing_status']=='cached'
    with np.load(out) as data:
        np.testing.assert_array_equal(data['motion_raw'],x[:,idx])
        np.testing.assert_array_equal(data['motion_valid'],np.ones(30,bool))
