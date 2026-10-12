import numpy as np
import pytest
import torch
from scripts.audit_v5_style_targets import independent_pairs,direction


def test_style_selection_rejects_shared_reference_sentence_and_ambiguous_performance():
    q=dict(clip_id=['a','b','c','d','e','f'],sentence_id=['x','x','y','y','z','z'],
        emotion_id=torch.tensor([1,1,1,1,2,2]),intensity_id=torch.tensor([2,2,2,2,2,2]),
        speaker_id=torch.tensor([0,1,0,1,0,1]))
    refs={0:dict(clip_id=['r0'],sentence_id=['y']),1:dict(clip_id=['r1'],sentence_id=['other'])}
    pairs,ambiguous=independent_pairs(q,refs,4)
    assert len(pairs)==4 and not ambiguous
    assert all(q['sentence_id'][i]!='y' for i,j in pairs)
    assert set(pairs)=={(0,1),(1,0),(4,5),(5,4)}
    q['clip_id'].append('again');q['sentence_id'].append('x')
    for key,value in [('emotion_id',1),('intensity_id',2),('speaker_id',0)]:
        q[key]=torch.cat((q[key],torch.tensor([value])))
    pairs,ambiguous=independent_pairs(q,refs,2)
    assert len(ambiguous)==1
    assert all(q['sentence_id'][i]=='z' for i,j in pairs)


def test_style_target_direction_uses_only_supported_statistics():
    own=np.zeros((4,52));donor=np.ones((4,52));swapped=np.ones((4,52))*.5
    mask=np.ones(52,dtype=bool);mask[51]=False
    own[:,51]=float('nan');donor[:,51]=float('nan')
    values=direction(own,swapped,donor,mask,np.ones(52))
    assert values['mouth/mean_before']==1.
    assert values['mouth/mean_after']==.25
    assert values['mouth/mean_improved']==1.
