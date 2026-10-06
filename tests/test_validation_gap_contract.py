import pytest
import torch
from scripts.build_validation_gap_table import curve_clip,check_contract


def test_packed_and_padded_native_curve_contract_preserves_missing_frames():
    valid=torch.tensor([True,False,True,True]);motion=torch.randn(4,52);times=torch.arange(4)/25
    channel=torch.tensor([True]*51+[False])
    target={'motion':motion,'valid':valid,'times':times,'channel_mask':channel}
    packed={'compaction_schema':'native_curve_pack_v1','native_lengths':[4],
            'target':motion,'valid':valid,'times':times,'channel_mask':channel[None],
            'predictions':{'42/full':motion}}
    p=curve_clip(packed,0,4,[0,4]);check_contract(p,target)
    assert p['valid'].tolist()==[True,False,True,True]
    padded={k:packed[k][None] for k in ('target','valid','times')}
    padded.update(channel_mask=channel[None],predictions={'42/full':motion[None]})
    check_contract(curve_clip(padded,0,4,[]),target)
    p['valid']=torch.ones(4,dtype=torch.bool)
    with pytest.raises(AssertionError):check_contract(p,target)
