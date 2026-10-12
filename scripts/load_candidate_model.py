"""Strict inference loading for v4 and isolated dual-timescale v5 checkpoints."""
from kinetalk_v4.unified_model_v4 import KineTalkV4Config, KineTalkV4Model
from kinetalk_v5.dual_timescale_model import DualTimescaleKineTalkV5


def load_candidate_model(checkpoint, device="cpu"):
    state = checkpoint["model"]
    if 'student_logvar.weight' in state:
        from kinetalk_v6.factor_paths import FactorPathKineTalkV6
        return FactorPathKineTalkV6.from_checkpoint(checkpoint, device=device)
    if "backbone.feature_mean" in state:
        return DualTimescaleKineTalkV5.from_checkpoint(checkpoint, device=device)
    model = KineTalkV4Model(KineTalkV4Config(**checkpoint["config"]),
        state["feature_mean"], state["feature_std"], state["scales"],
        state["decoder.geometry_jacobian"], state["decoder.geometry_offset"],
        state["decoder.geometry_scale"]).to(device)
    model.load_state_dict(state, strict=True)
    return model
