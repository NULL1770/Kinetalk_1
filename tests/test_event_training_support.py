import numpy as np

from scripts.audit_event_training_support import analyze_clip, chunk_support, population


def fixture(n=420):
    motion=np.sin(np.arange(n)[:,None]/20)*np.ones((1,9))*.1
    clip=dict(clip_id='a',sentence='s',speaker=1,emotion=2,valid=np.ones(n,bool),
              motion_mask=np.ones((n,9),bool),motion9=motion)
    label=dict(known=np.ones((n,4),bool),schedule=np.zeros((n,12)),events=[])
    return clip,label


def test_exact_tail_and_gap_rules():
    mask=np.ones(209,bool)
    source,segments,_=chunk_support(mask,5)
    receiver,other,dropped=chunk_support(mask,10)
    assert segments==[(0,200),(200,209)] and source.sum()==209
    assert other==[(0,200)] and dropped==[(200,209)] and receiver.sum()==200
    mask[4:8]=False
    assert not chunk_support(mask,5)[0][:8].any()


def test_intersection_discards_other_groups_activity_and_fixed_energy_accounting():
    clip,label=fixture(100)
    label['known'][30:60,3]=False
    label['schedule'][30:60,0]=1
    label['events']=[dict(group_index=0,start=30,end=60)]
    result=analyze_clip(clip,label)
    assert result['frames']['source_lost_at_known4']==30
    assert result['groups']['up']['source_active_frames']==30
    assert result['groups']['up']['receiver_active_frames']==0
    assert result['groups']['up']['source_active_lost_other_group_unknown']==30
    assert result['groups']['up']['source_complete_events']==1
    assert result['groups']['up']['receiver_complete_events']==0
    centered=clip['motion9']-clip['motion9'].mean(0)
    expected=np.square(centered[:,[2,3,4]]).sum()
    assert abs(result['groups']['up']['source_common_center_energy']-expected)<1e-12
    kept=np.ones(100,bool);kept[30:60]=False
    actual=np.square(centered[kept][:,[2,3,4]]).sum()
    assert abs(result['groups']['up']['receiver_common_center_energy']-actual)<1e-12
    summary=population([result])
    assert summary['all']['receiver_clips_retained']==1
    assert summary['by_native_clock_length']['gt_200']['clips']==0


def test_changing_chunk_origin_can_admit_source_dropped_tail():
    clip,label=fixture(203)
    label['known'][:4]=False
    result=analyze_clip(clip,label)
    assert result['frames']['source_kept']==200
    assert result['frames']['receiver_kept']==199
    assert result['frames']['receiver_outside_source']==3
