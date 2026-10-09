import torch
from tests.test_reference_response import fixture
from scripts.audit_style_receiver_paths import decode_paths


def test_diagnostic_replay_and_static_bias_range_identity():
    _,m,b,r=fixture();m.eval()
    with torch.no_grad():
        p=m.audio_prior(b['audio_features'],b['valid']);g,u=m.conditions(p,b['valid']);s=m.encode_style(r)['code']
        torch.nn.init.normal_(m.decoder.bias.weight,std=.02)
        y=m.decode(b['b0'],p,s,b['valid']);full=decode_paths(m,b['b0'],g,u,s,b['valid'])
        torch.testing.assert_close(y,full,rtol=0,atol=0)
        off=decode_paths(m,b['b0'],g,u,s,b['valid'],bias=False)
        for j in range(len(y)):
            v=b['valid'][j];diff=(y-off)[j,v]
            torch.testing.assert_close(diff,diff[:1].expand_as(diff),atol=2e-7,rtol=1e-5)
        before={k:v.clone() for k,v in m.state_dict().items()}
        decode_paths(m,b['b0'],g,u,s,b['valid'],modulation=False)
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
