import copy
import numpy as np
import pytest
import torch

from scripts import evaluate_semantic_student_transfer as transfer
from tests.test_visual_semantic_training import inputs
from scripts.train_visual_semantic_pilot import stats
from kinetalk_b0.models.semantic_upper_flow import SemanticUpperFlow


def test_conditions_never_require_query_teacher_or_motion():
    clips,rows=inputs();scales=stats(clips);clip=clips[-1]
    clean={k:clip[k] for k in ('valid','affect_global','identity_code')}
    a=transfer.conditions(clip,rows[-1],'va','new',scales,'cpu')
    b=transfer.conditions(clean,rows[-1],'va','new',scales,'cpu')
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_paired_old_and_new_equal_conditions_have_equal_draws_and_protection():
    clips,rows=inputs();clip=clips[-1];scales=stats(clips)
    torch.manual_seed(772)
    model=SemanticUpperFlow(8,global_dim=4,identity_dim=3,hidden=16,depth=2).eval()
    a=transfer.sample(model,clip,rows[-1],'va','old',scales,'cpu')
    b=transfer.sample(model,clip,rows[-1],'va','new',scales,'cpu')
    np.testing.assert_array_equal(a,b)
    invalid=~clip['valid'].numpy()
    np.testing.assert_array_equal(a[:,invalid],np.broadcast_to(clip['baseline52'][invalid][:,transfer.UPPER],a[:,invalid].shape))
    corrupted=copy.deepcopy(clip);corrupted['motion'][:]=float('nan');corrupted['semantic_valid'][:]=False
    c=transfer.sample(model,corrupted,rows[-1],'va','new',scales,'cpu')
    np.testing.assert_array_equal(a,c)


def test_constant_is_distinct_from_static_and_rejects_bad_condition():
    clips,rows=inputs();scales=stats(clips);row=copy.deepcopy(rows[-1])
    row['va']['constant_audio']=torch.full_like(row['va']['actual'],.2)
    a=transfer.conditions(clips[-1],row,'va','constant',scales,'cpu')[1]
    b=transfer.conditions(clips[-1],row,'va','static',scales,'cpu')[1]
    assert not torch.equal(a,b)
    row['va']['constant_audio'][0,0]=float('nan')
    with pytest.raises(ValueError):transfer.conditions(clips[-1],row,'va','constant',scales,'cpu')
