import numpy as np
import pytest
from scripts.audit_brow_states_v3 import validate_arrays
from scripts.extract_brow_states_v3 import build_clip, HOLD


def arrays():
    raw = np.full((20, 5), .3); mask = np.ones(20, bool)
    eligible = mask.copy(); eligible[:2] = False; eligible[-2:] = False
    return build_clip({"source_id": "x"}, dict(raw5=raw, smooth5=raw.copy(), valid=mask, eligible=eligible))[1]


def test_audit_rejects_eligible_edge_known_label():
    a = arrays(); a["labels"][-2, 0] = HOLD; a["known_mask"][-2, 0] = True
    with pytest.raises(ValueError, match="eligibility"): validate_arrays(a)


def test_audit_rejects_static_phase_and_conflict():
    a = arrays(); a["phase_known"][5, 0] = True
    with pytest.raises(ValueError, match="phase"): validate_arrays(a)
    a = arrays(); a["conflict"][5, 0] = True
    with pytest.raises(ValueError, match="conflict"): validate_arrays(a)


def test_audit_requires_provenance_and_group_axis():
    a = arrays(); del a["censored_mask"]
    with pytest.raises(KeyError): validate_arrays(a)
    a = arrays(); a["trend"] = a["trend"][:, 0]
    with pytest.raises(ValueError, match="T,2"): validate_arrays(a)
