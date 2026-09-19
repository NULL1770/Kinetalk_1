import numpy as np
import pytest

from scripts.event_schedule_teacher import fit_teacher, extract_schedule, fit_event_thresholds, extract_event_labels
from scripts.event_schedule_metrics import evaluate_clip, evaluate_arms


def pulse(n=60, start=16, length=24):
    x = np.zeros((n,9)); x[start:start+length,2:5] = .3*np.sin(np.linspace(0,np.pi,length))[:,None]
    return x


def row(x, samples=None):
    return {"target":x, "native_valid":np.ones(len(x),bool), "samples":x[None] if samples is None else samples}


def teacher():
    return fit_teacher([row(pulse())])


def test_complete_pulse_offset_invariance_and_schedule_has_no_amplitude():
    x=pulse(); valid=np.ones(len(x),bool); th=teacher()
    a=extract_schedule(x,valid,th); b=extract_schedule(x+2.5,valid,th); c=extract_schedule(x-1.25,valid,th)
    assert len(a["events"])==1
    np.testing.assert_allclose(a["schedule"],b["schedule"],atol=0,rtol=0)
    np.testing.assert_allclose(a["schedule"],c["schedule"],atol=0,rtol=0)
    np.testing.assert_array_equal(a["known"],b["known"])
    assert a["schedule"].shape==(len(x),12)
    event=a["events"][0]; assert event["group"]=="up"
    assert a["schedule"][event["start"],4]==0
    assert a["schedule"][event["end"]-1,4]==1
    assert a["onset"].sum()==1


def test_constant_high_values_and_single_frame_spike_have_no_events():
    valid=np.ones(60,bool); th=teacher()
    for level in (0.,.5,-.5,2.5):
        z=extract_schedule(np.full((60,9),level),valid,th)
        assert not z["events"] and not z["onset"].any() and not z["schedule"].any()
    x=np.zeros((60,9));x[25,2:5]=1
    z=extract_schedule(x,valid,th)
    assert not z["events"] and not z["onset"].any()


def test_censored_edges_and_gap_do_not_create_complete_events():
    x=pulse(); th=teacher(); valid=np.ones(60,bool);valid[27:32]=False
    z=extract_schedule(x,valid,th)
    assert not z["events"] and not z["onset"].any()
    assert not z["known"][27:32].any()
    assert not z["known"][22:27,0].all()
    rising=np.zeros((40,9));rising[:,2:5]=np.linspace(0,.5,40)[:,None]
    z=extract_schedule(rising,np.ones(40,bool),th)
    assert not z["events"] and not z["onset"].any()
    assert not z["known"][:,0].all()


def test_motion_channel_gap_only_affects_its_group():
    x=pulse(); mask=np.ones_like(x,bool);mask[27:32,2]=False
    z=extract_schedule(x,np.ones(60,bool),teacher(),motion_mask=mask)
    assert not z["events"]
    assert not z["known"][27:32,0].any()
    assert z["known"][27:32,1:].all()


def test_generator_fit_iterable_matches_list_and_ignores_eval():
    rows=[row(pulse()),row(pulse()+1.)]
    assert fit_teacher(iter(rows))==fit_teacher(rows)
    assert all(v>=.02 for v in fit_teacher(rows)["threshold"].values())


def test_exact_draw_and_reversed_shifted_timing():
    x=pulse(start=7,length=20); th=fit_event_thresholds([row(pulse())])
    exact=evaluate_clip(row(x),th)
    reverse=evaluate_clip(row(x,x[::-1][None]),th)
    assert exact["groups"]["up"]["brier"]==0
    assert exact["groups"]["up"]["nll"]<1e-5
    assert exact["groups"]["up"]["duration_error_s"]==0
    assert reverse["groups"]["up"]["brier"]>0


def test_missing_prediction_empty_score_and_strict_arm_members():
    th=fit_event_thresholds([row(pulse())]); bad=row(pulse());del bad["samples"]
    with pytest.raises(ValueError,match="predictions are required"):evaluate_clip(bad,th)
    empty=row(pulse());empty["score_mask"]=np.zeros(60,bool)
    assert evaluate_clip(empty,th)["groups"]["up"]["status"]=="pending_no_known_frames"
    fit={"fit":row(pulse())}
    with pytest.raises(ValueError,match="disjoint"):evaluate_arms(fit,{"audio":fit})
    with pytest.raises(ValueError,match="same clip IDs"):
        evaluate_arms(fit,{"audio":{"test":row(pulse())},"static":{"other":row(pulse())}})
    good=evaluate_arms(fit,{"audio":{"test":row(pulse())},"static":{"test":row(pulse(),np.zeros((1,60,9)))}})
    assert good["protocol"]["fit_eval_ids_disjoint"]
    assert good["arms"]["audio"]["summary"]["macro"]["brier"]==0
