import pytest
import torch
from kinetalk_v5.joint_factor_repair import V5JointFactorRepair
from tests.test_v5_affect_trend import trend_setup


def test_joint_reference_endpoints_and_strict_roundtrip():
    old,trend,b,refs=trend_setup();style_ck=old.checkpoint()
    style_ck['model']={k:v.clone() for k,v in style_ck['model'].items()}
    style_ck['model']['reference_context.output.bias']+=.1
    model=V5JointFactorRepair.from_parts(trend.checkpoint(),style_ck,style_weight=0.).eval()
    with torch.no_grad():
        assert torch.equal(model.encode_style(refs),trend.encode_style(refs))
        model.reference_blend_weight.fill_(1.)
        assert torch.equal(model.encode_style(refs),super(V5JointFactorRepair,model).encode_style(refs))
        restored=V5JointFactorRepair.from_checkpoint(model.checkpoint()).eval()
        args=(b['audio_features'],b['h0'],b['b0'],b['valid'],refs)
        assert torch.equal(model.predict(*args)['motion'],restored.predict(*args)['motion'])


def test_joint_parts_require_identical_native_core_and_teacher():
    old,trend,_,_=trend_setup();style_ck=old.checkpoint()
    style_ck['model']={k:v.clone() for k,v in style_ck['model'].items()}
    style_ck['model']['core.student.state.bias']+=.1
    with pytest.raises(ValueError,match='incompatible'):V5JointFactorRepair.from_parts(trend.checkpoint(),style_ck)
