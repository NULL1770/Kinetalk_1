import torch
from kinetalk_v5.style_calibration import ReferenceCalibration,V5CalibratedStyle
from tests.test_v5_factor_rebuild import setup


def test_person_constant_offsets_are_not_removed_by_within_person_fit():
    torch.manual_seed(9)
    people=torch.arange(20).repeat_interleave(10)
    context=torch.randn(200,3);identity=torch.randn(20,2)
    nuisance=torch.tensor([[.3,.2],[.5,-.2],[-.1,.6]])
    code=identity[people]+context@nuisance
    a=ReferenceCalibration(3,2);a.fit(context,code,people,ridge=.001);a.strength.fill_(1.)
    calibrated=a(code,context)
    for person in people.unique():
        assert float(calibrated[people==person].std(0).max())<1e-4
    shifted=code+torch.randn(20,2)[people]*10.
    b=ReferenceCalibration(3,2);b.fit(context,shifted,people,ridge=.001)
    assert torch.allclose(a.coefficient,b.coefficient,atol=3e-6)


def test_zero_strength_is_exact_and_inference_has_no_person_or_query_labels():
    _,old,b,refs=setup();model=V5CalibratedStyle(old).eval()
    with torch.no_grad():
        assert torch.equal(old.encode_style(refs),model.encode_style(refs))
        before=model.encode_style(refs)
        changed=model.encode_style({**refs,'speaker_id':torch.tensor([100,200]),
                                    'emotion_id':torch.tensor([7,7])})
    assert torch.equal(before,changed)
