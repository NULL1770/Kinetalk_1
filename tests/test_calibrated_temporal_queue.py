import copy
import json

import pytest

from scripts.run_calibrated_temporal_queue import assess, interval


def records(count=446, value=.2):
    return [{'clip_id':f'clip_{i}', 'sentence':f'sentence_{i%85}', 'scales':[1.]*9,
             'sample_count':3, 'valid_frames':100, 'valid_runs':1,
             'joint_fair_es':{'centered':value}, 'variogram':{'aggregate':value}}
            for i in range(count)]


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def artifacts(tmp_path):
    output=tmp_path/'out';art=tmp_path/'artifacts'
    save(output/'queue_plan.json',{'validation_clip_ids':[row['clip_id'] for row in records()]})
    modes=('full','base','static_state','oracle_state','reverse_audio')
    keys=[f'{seed}/{mode}' for seed in (42,123,2026) for mode in modes]
    for arm in ('audio','static'):
        save(art/arm/'dynamics/temporal_full.json',records(value=.2 if arm=='audio' else .4))
        benchmark={mode:{'prediction_keys':[f'{seed}/{mode}' for seed in (42,123,2026)]} for mode in modes}
        save(output/arm/'dynamics/benchmark_summary.json',benchmark)
        save(output/arm/'dynamics/mouth_protection.json',{key:{'passed':True} for key in keys})
    for mode in ('static_state','reverse_audio'):
        save(art/'audio/dynamics'/f'temporal_{mode}.json',records(value=.3))
    return output,art


def test_interval_direction_and_sentence_cluster_accounting():
    real=records(6,.2);other=records(6,.4)
    result=interval(real,other,['joint_fair_es','centered'])
    assert result['passed']
    assert result['audio_minus_control']==pytest.approx(-.2)
    assert result['sentence_cluster_95ci']==pytest.approx([-.2,-.2])
    assert result['clips']==result['clusters']==6
    assert not interval(other,real,['joint_fair_es','centered'])['passed']
    # Unbiased finite-draw fair ES estimates are not constrained to >= 0.
    assert interval(records(6,-.2),records(6,-.1),['joint_fair_es','centered'])['passed']


@pytest.mark.parametrize('problem',['left_duplicate','right_duplicate','membership','sentence','scale','missing','nan','inf','empty'])
def test_interval_rejects_incomplete_or_nonfinite_pairs(problem):
    left=records(3,.2);right=records(3,.4)
    if problem=='left_duplicate':left.append(copy.deepcopy(left[0]))
    elif problem=='right_duplicate':right.append(copy.deepcopy(right[0]))
    elif problem=='membership':right[0]['clip_id']='foreign'
    elif problem=='sentence':right[0]['sentence']='foreign'
    elif problem=='scale':right[0]['scales'][0]=2.
    elif problem=='missing':right[0]['joint_fair_es']['centered']=None
    elif problem=='nan':right[0]['joint_fair_es']['centered']=float('nan')
    elif problem=='inf':right[0]['joint_fair_es']['centered']=float('inf')
    elif problem=='empty':left=[];right=[]
    with pytest.raises(ValueError):
        interval(left,right,['joint_fair_es','centered'])


def test_acceptance_remains_numeric_only_without_promotion(tmp_path):
    output,art=artifacts(tmp_path)
    result=assess(output,art)
    assert result['quantitative_timing_passed']
    assert result['visual_review_pending']
    assert result['independent_emotion_and_AV_pending']
    assert result['test_loaded'] is result['default_replaced'] is False
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('problem',['empty_mouth','missing_seed','truthy_not_boolean','mouth_failed','incomplete_coverage',
                                    'centered_worse','variogram_worse','draw_count','frame_support','benchmark_seed','unbound_membership'])
def test_acceptance_fails_closed(problem,tmp_path):
    output,art=artifacts(tmp_path)
    mouth_path=output/'audio/dynamics/mouth_protection.json'
    mouth=json.loads(mouth_path.read_text())
    if problem=='empty_mouth':mouth={}
    elif problem=='missing_seed':mouth.pop('2026/full')
    elif problem=='truthy_not_boolean':mouth['42/full']['passed']='true'
    elif problem=='mouth_failed':mouth['42/full']['passed']=False
    save(mouth_path,mouth)
    if problem=='incomplete_coverage':
        for path in art.rglob('temporal_*.json'):
            save(path,json.loads(path.read_text())[:10])
    elif problem in ('centered_worse','variogram_worse'):
        path=art/'audio/dynamics/temporal_full.json';rows=json.loads(path.read_text())
        for row in rows:
            if problem=='centered_worse':row['joint_fair_es']['centered']=.9
            else:row['variogram']['aggregate']=.9
        save(path,rows)
    elif problem in ('draw_count','frame_support'):
        path=art/'static/dynamics/temporal_full.json';rows=json.loads(path.read_text())
        rows[0]['sample_count' if problem=='draw_count' else 'valid_frames']=2 if problem=='draw_count' else 99
        save(path,rows)
    elif problem=='benchmark_seed':
        path=output/'audio/dynamics/benchmark_summary.json';benchmark=json.loads(path.read_text())
        benchmark['full']['prediction_keys']=['42/full']
        save(path,benchmark)
    elif problem=='unbound_membership':
        save(output/'queue_plan.json',{})
    try:
        result=assess(output,art)
    except ValueError:
        return
    assert not result['quantitative_timing_passed']
    json.dumps(result,allow_nan=False)
