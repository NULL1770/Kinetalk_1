import json
import numpy as np
import torch

from scripts.evaluate_condition_probes import feature


def test_affect_global_intensity_is_fixed_width_and_finite():
    row = {"affect_global": torch.zeros(64), "affect_intensity": torch.tensor([.5])}
    x = feature(row, "affect_global_intensity")
    assert x.shape == (65,) and np.isfinite(x).all() and x[-1] == .5


def test_identity_feature_rejects_wrong_width():
    try:
        feature({"identity_code": torch.zeros(4)}, "identity_code")
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-width identity feature must fail")


def test_protocol_hash_mismatch_is_rejected_before_loading(tmp_path):
    import pytest
    from scripts.evaluate_condition_probes import evaluate
    data=tmp_path/'data.pt';data.write_bytes(b'not a dataset')
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps({'dataset_sha256':'wrong'}))
    with pytest.raises(ValueError,match='hash'):
        evaluate(data,protocol,tmp_path/'output.json','identity_code')
