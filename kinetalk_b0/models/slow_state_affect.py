"""Signed slow expression states and native-frame acoustic conditioning.

The four states are motion proxies (raise, down, squint, wide), not emotional
labels.  Targets require independent neutral references and training-only
channel scales.  This module never estimates either from a query sequence.
No text, VA estimator, target-motion inference input, or window-activity
condition is used.  An explicitly named override is for oracle diagnostics.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .label_guided_affect import TemporalBlock
from .dit import ResidualDiT


STATE_NAMES = ('raise', 'down', 'squint', 'wide')
STATE_GROUPS = ((43, 44, 45), (41, 42), (5, 12), (6, 13))
UPPER_INDICES = (41, 42, 43, 44, 45, 5, 6, 12, 13)
_UPPER_TO_STATE = (1, 1, 0, 0, 0, 2, 3, 2, 3)
UPPER_STATE_GROUPS = ((2, 3, 4), (0, 1), (5, 7), (6, 8))


def _sequence_mask(value, mask):
    if (not torch.is_tensor(value) or value.ndim != 3 or not value.is_floating_point()
            or min(value.shape) < 1):
        raise ValueError('Values must be nonempty floating [B,T,D]')
    if (not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.device != value.device
            or mask.shape not in (value.shape[:2], value.shape)):
        raise ValueError('Mask must be Boolean [B,T] or [B,T,D] on the values device')
    mask = mask[..., None].expand_as(value) if mask.ndim == 2 else mask
    if not torch.isfinite(value[mask]).all():
        raise ValueError('Observed values must be finite')
    return mask


def masked_slow_state(value, mask, *, stride=16):
    """Orthogonally project observed values onto a fixed linear-spline basis.

    Hat basis knots are at 0, stride, ... ceil((T-1)/stride)*stride. Each
    sample/feature uses P = B pinv(B' W B) B' W, with Boolean observation W.
    The pseudoinverse is computed in float64 from masks only. Rank-deficient
    masks and entirely missing features have a well-defined minimum-norm fit.
    Appending padding only appends zero-support columns; earlier knot supports
    do not move. The continuous spline has no step at an internal knot.

    Missing frames/groups remain invalid zero placeholders, including gaps.
    ``bin_state`` now denotes fitted knot coefficients, ``bin_mask`` denotes
    knots with observed basis support, and floating ``bin_count`` is the sum
    of observed hat weights. These retained API names are not bin averages.
    No temporal centering or amplitude clipping occurs.
    """
    if type(stride) is not int or stride < 1:
        raise ValueError('stride must be a positive integer')
    mask = _sequence_mask(value, mask)
    _, frames, _ = value.shape
    clean = torch.where(mask, value, 0.)
    with torch.no_grad():
        frame_clock = torch.arange(frames, device=value.device, dtype=torch.float64)
        last_knot = (frames - 1 + stride - 1) // stride
        knot_clock = torch.arange(last_knot + 1, device=value.device, dtype=torch.float64) * stride
        basis = (1. - (frame_clock[:, None] - knot_clock[None]).abs() / stride).clamp_min(0.)
        weights = mask.transpose(1, 2).to(torch.float64)
        weighted_basis = weights[..., None] * basis
        gram = torch.einsum('bdtk,tl->bdkl', weighted_basis, basis)
        # Fixed tolerance avoids changing numerical rank when padding adds
        # unsupported knots. W and the basis do not receive gradients.
        gram_inverse = torch.linalg.pinv(gram, hermitian=True, rtol=1e-12)
        coefficient_map = gram_inverse @ weighted_basis.transpose(-1, -2)
        count = weighted_basis.sum(-2).transpose(1, 2)
    coefficients = (coefficient_map @ clean.transpose(1, 2).double().unsqueeze(-1)).squeeze(-1)
    expanded = torch.einsum('tk,bdk->btd', basis, coefficients).to(value.dtype)
    return {'state': torch.where(mask, expanded, 0.), 'state_mask': mask,
            'bin_state': coefficients.transpose(1, 2).to(value.dtype),
            'bin_mask': count > 0, 'bin_count': count.to(value.dtype)}


def _scales(scales, reference):
    if (not torch.is_tensor(scales) or scales.shape != (52,) or not scales.is_floating_point()
            or scales.device != reference.device or not torch.isfinite(scales).all()
            or (scales <= 0).any()):
        raise ValueError('Training scales must be finite positive floating [52] on the same device')
    return scales.to(dtype=reference.dtype)


def readout_slow_state(motion, observed, neutral_anchor, scales):
    """Read signed frame states and per-frame/group masks, both [B,T,4].

    Each state averages available normalized channel deviations within its
    group; no available channel means an invalid zero placeholder. Project
    these frame states with ``masked_slow_state`` onto the slow spline targets.
    """
    if (not torch.is_tensor(motion) or motion.ndim != 3 or motion.shape[-1] != 52
            or not motion.is_floating_point() or min(motion.shape[:2]) < 1):
        raise ValueError('motion must be nonempty floating [B,T,52]')
    if (not torch.is_tensor(observed) or observed.shape != motion.shape
            or observed.dtype != torch.bool or observed.device != motion.device):
        raise ValueError('observed must be Boolean [B,T,52] on the motion device')
    if (not torch.is_tensor(neutral_anchor) or neutral_anchor.shape != (len(motion), 52)
            or not neutral_anchor.is_floating_point() or neutral_anchor.device != motion.device):
        raise ValueError('neutral_anchor must be independent floating [B,52] on the motion device')
    channel_scales = _scales(scales, motion)
    anchor = neutral_anchor[:, None].expand_as(motion)
    required = torch.zeros_like(observed)
    required[..., list(UPPER_INDICES)] = observed[..., list(UPPER_INDICES)]
    if (not torch.isfinite(motion[required]).all()
            or not torch.isfinite(anchor[required]).all()):
        raise ValueError('Observed upper-face motion and neutral anchors must be finite')
    normalized = (torch.where(required, motion, 0.) - torch.where(required, anchor, 0.)) / channel_scales
    states, masks = [], []
    for group in STATE_GROUPS:
        count = required[..., list(group)].sum(-1)
        states.append(normalized[..., list(group)].sum(-1) / count.clamp_min(1))
        masks.append(count > 0)
    return torch.stack(states, -1), torch.stack(masks, -1)


def lift_slow_state(state, scales, state_mask=None):
    """Lift four normalized states to a 52D raw delta with only nine nonzeros.

    The fixed group lift preserves sign and physical per-channel scales.
    With all channels observed, reading this lift relative to a zero anchor
    recovers the supplied four states exactly up to floating point error.
    """
    if not torch.is_tensor(state) or state.ndim != 3 or state.shape[-1] != 4:
        raise ValueError('state must be floating [B,T,4]')
    if state_mask is None:
        state_mask = torch.ones_like(state, dtype=torch.bool)
    mask = _sequence_mask(state, state_mask)
    channel_scales = _scales(scales, state)
    clean = torch.where(mask, state, 0.)
    values = clean[..., list(_UPPER_TO_STATE)] * channel_scales[list(UPPER_INDICES)]
    output = state.new_zeros(*state.shape[:2], 52)
    output[..., list(UPPER_INDICES)] = values
    return output


def project_upper_innovation(value, valid, *, stride=16):
    """Apply Q = I - P to normalized upper-face coefficients [B,T,9].

    Channels use UPPER_INDICES order (five brows, then squint/wide eyes).
    P averages each disjoint channel group, projects onto the fixed-clock
    linear-spline basis, then repeats onto the group's channels. With this
    Boolean whole-frame mask, P is an orthogonal projection, so Q is idempotent
    and cannot change the coarse group state. Nine channels must be observed
    whenever valid is true; callers with channel masks must enforce this.

    Apply Q to innovation targets, starting noise, velocities, and ODE states.
    Values are normalized before this operation; do not project raw values
    with unequal physical channel scales.
    """
    if not torch.is_tensor(value) or value.ndim != 3 or value.shape[-1] != 9:
        raise ValueError('Normalized upper innovation must be [B,T,9]')
    if not torch.is_tensor(valid) or valid.ndim != 2:
        raise ValueError('Innovation mask must be whole-frame Boolean [B,T]')
    mask = _sequence_mask(value, valid)
    clean = torch.where(mask, value, 0.)
    grouped = torch.stack([clean[..., list(group)].mean(-1) for group in UPPER_STATE_GROUPS], -1)
    slow = masked_slow_state(grouped, valid, stride=stride)['state']
    return torch.where(mask, clean - slow[..., list(_UPPER_TO_STATE)], 0.)


def compose_upper_face(baseline, upper_face, valid=None):
    """Replace absolute upper-face coefficients; copy all other 43 unchanged.

    ``upper_face`` may be full [B,T,52] or [B,T,9] in UPPER_INDICES order.
    Invalid frames retain baseline values, rather than clearing mouth output.
    This function does not clamp coefficients or hide output-domain errors.
    """
    if (not torch.is_tensor(baseline) or baseline.ndim != 3 or baseline.shape[-1] != 52
            or not baseline.is_floating_point() or min(baseline.shape[:2]) < 1):
        raise ValueError('baseline must be floating [B,T,52]')
    if (not torch.is_tensor(upper_face) or upper_face.shape[:2] != baseline.shape[:2]
            or upper_face.ndim != 3 or upper_face.shape[-1] not in (9, 52)
            or upper_face.dtype != baseline.dtype or upper_face.device != baseline.device):
        raise ValueError('upper_face must match baseline with 9 or 52 floating channels')
    if valid is None:
        valid = torch.ones(baseline.shape[:2], device=baseline.device, dtype=torch.bool)
    if valid.dtype != torch.bool or valid.shape != baseline.shape[:2] or valid.device != baseline.device:
        raise ValueError('valid must be Boolean [B,T] on the baseline device')
    selected = upper_face[..., list(UPPER_INDICES)] if upper_face.shape[-1] == 52 else upper_face
    if not torch.isfinite(selected[valid]).all():
        raise ValueError('Observed generated upper face must be finite')
    output = baseline.clone()
    output[..., list(UPPER_INDICES)] = torch.where(valid[..., None], selected, baseline[..., list(UPPER_INDICES)])
    return output


class SlowStateAffect(nn.Module):
    """Audio global affect, native-rate local features, and four slow states.

    Local and state output heads start at zero. Global features and classifiers
    remain trainable for independent motion-teacher distillation.  The slow
    state is exposed for direct output lifting, not hidden in a global FiLM.
    """

    def __init__(self, feature_mean, feature_std, *, global_dim=64, hidden=128,
                 local_dim=64, num_emotions=8, num_levels=4, stride=16):
        super().__init__()
        if (not torch.is_tensor(feature_mean) or feature_mean.ndim != 1
                or not torch.is_tensor(feature_std) or feature_std.shape != feature_mean.shape
                or not feature_mean.is_floating_point() or not feature_std.is_floating_point()
                or feature_mean.numel() < 1 or not torch.isfinite(feature_mean).all()
                or not torch.isfinite(feature_std).all() or (feature_std <= 0).any()):
            raise ValueError('Finite fitted feature statistics with positive std are required')
        if any(type(v) is not int or v < 1 for v in (global_dim, hidden, local_dim, num_emotions, num_levels, stride)):
            raise ValueError('All model dimensions and stride must be positive integers')
        self.stride = stride
        self.register_buffer('feature_mean', feature_mean.detach().float().clone())
        self.register_buffer('feature_std', feature_std.detach().float().clone())
        self.input = nn.Linear(len(feature_mean), hidden)
        self.blocks = nn.ModuleList(TemporalBlock(hidden, dilation) for dilation in (1, 2, 4, 8))
        self.global_head = nn.Linear(hidden, global_dim)
        self.emotion_classifier = nn.Linear(global_dim, num_emotions)
        self.intensity_classifier = nn.Linear(global_dim, num_levels)
        self.local_head = nn.Linear(hidden, local_dim)
        self.state_head = nn.Linear(hidden, 4)
        for head in (self.local_head, self.state_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features, valid, *, state_override=None, state_override_mask=None):
        if (not torch.is_tensor(features) or features.ndim != 3
                or features.shape[-1] != len(self.feature_mean)
                or not torch.is_tensor(valid) or valid.shape != features.shape[:2]
                or valid.dtype != torch.bool or valid.device != features.device
                or not valid.any(1).all()):
            raise ValueError('Native audio features and nonempty Boolean [B,T] masks must match')
        mask = _sequence_mask(features, valid)
        # Clear missing entries before arithmetic, including for safe gradients.
        clean = torch.where(mask, features, self.feature_mean)
        normalized = (clean - self.feature_mean) / self.feature_std
        hidden = torch.where(valid[..., None], F.silu(self.input(normalized)), 0.)
        for block in self.blocks:
            hidden = block(hidden, valid)
        pooled = hidden.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        global_code = self.global_head(pooled)
        emotion_logits = self.emotion_classifier(global_code)
        intensity_logits = self.intensity_classifier(global_code)
        levels = torch.arange(intensity_logits.shape[-1], device=features.device, dtype=features.dtype)
        local = torch.where(valid[..., None], self.local_head(hidden), 0.)
        prediction = masked_slow_state(self.state_head(hidden), valid, stride=self.stride)
        driving = prediction
        if state_override is not None:
            if (not torch.is_tensor(state_override)
                    or state_override.shape != prediction['state'].shape
                    or state_override.device != features.device
                    or state_override.dtype != features.dtype):
                raise ValueError('Oracle state override must match [B,T,4], dtype and device')
            override_mask = valid[..., None].expand_as(state_override)
            if state_override_mask is not None:
                if (not torch.is_tensor(state_override_mask) or state_override_mask.dtype != torch.bool
                        or state_override_mask.device != features.device
                        or state_override_mask.shape not in (valid.shape, state_override.shape)):
                    raise ValueError('Oracle mask must be matching Boolean [B,T] or [B,T,4]')
                extra_mask = (state_override_mask[..., None].expand_as(state_override)
                              if state_override_mask.ndim == 2 else state_override_mask)
                override_mask = override_mask & extra_mask
            driving = masked_slow_state(state_override, override_mask, stride=self.stride)
        elif state_override_mask is not None:
            raise ValueError('An oracle mask requires an oracle state override')
        return {'global': global_code, 'emotion_logits': emotion_logits,
                'intensity_logits': intensity_logits,
                'intensity_value': (intensity_logits.softmax(-1) * levels).sum(-1, keepdim=True),
                'local': local, 'predicted_state': prediction['state'],
                'predicted_state_mask': prediction['state_mask'],
                'predicted_bin_state': prediction['bin_state'],
                'predicted_bin_mask': prediction['bin_mask'], **driving}


class UpperInnovationFlow(nn.Module):
    """Randomly initialized nine-channel flow constrained to the Q subspace.

    Inputs and returned innovation are normalized by training channel scales;
    this module does not apply the full-face renderer's residual_scale. State
    lifting and the exact non-upper-face bypass happen outside this module.
    ``global_affect`` is a dictionary with global and intensity_value tensors.
    No target motion is consumed by decode. The explicit noise controls seeds.
    """

    def __init__(self, cfg=None, *, stride=16):
        super().__init__()
        if type(stride) is not int or stride < 1:
            raise ValueError('stride must be a positive integer')
        model = {} if cfg is None else cfg.get('model', cfg)
        self.stride = stride
        self.content_dim = int(model.get('content_dim', 128))
        self.emotion_dim = int(model.get('emotion_dim', 64))
        self.style_dim = int(model.get('style_dim', 128))
        self.state_projection = nn.Linear(4, self.emotion_dim)
        self.renderer = ResidualDiT(
            motion_dim=9, content_dim=self.content_dim, emotion_dim=self.emotion_dim,
            style_dim=self.style_dim, dim=int(model.get('dit_dim', 192)),
            depth=int(model.get('dit_depth', 4)), heads=int(model.get('heads', 6)),
            dropout=float(model.get('dropout', 0.)), intensity_dim=1,
            global_dropout=float(model.get('global_condition_dropout', 0.)),
            style_dropout=float(model.get('style_condition_dropout', 0.)))

    def _conditions(self, valid, base_h0, identity_code, global_affect, local, state):
        if (not torch.is_tensor(valid) or valid.ndim != 2 or valid.dtype != torch.bool
                or not valid.any(1).all()):
            raise ValueError('Flow requires nonempty Boolean valid [B,T]')
        batch, frames = valid.shape
        if not isinstance(global_affect, dict) or not {'global', 'intensity_value'} <= global_affect.keys():
            raise ValueError('global_affect must contain global and intensity_value')
        for name, value, width in (('base_h0', base_h0, self.content_dim),
                                    ('local', local, self.emotion_dim), ('state', state, 4)):
            if not torch.is_tensor(value) or value.shape != (batch, frames, width):
                raise ValueError(f'{name} must match [B,T,{width}]')
            _sequence_mask(value, valid)
        global_code, intensity = global_affect['global'], global_affect['intensity_value']
        for name, value, width in (('identity_code', identity_code, self.style_dim),
                                    ('global', global_code, self.emotion_dim), ('intensity_value', intensity, 1)):
            if (not torch.is_tensor(value) or value.shape != (batch, width)
                    or not value.is_floating_point() or value.device != valid.device
                    or not torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite matching [B,{width}]')
        content = torch.where(valid[..., None], base_h0, 0.)
        safe_local = torch.where(valid[..., None], local, 0.)
        safe_state = torch.where(valid[..., None], state, 0.)
        condition = torch.where(valid[..., None], safe_local + self.state_projection(safe_state), 0.)
        return content, identity_code, global_code, intensity, condition

    def _velocity(self, x, time, valid, conditions):
        content, identity_code, global_code, intensity, condition = conditions
        velocity = self.renderer(x, time, content, global_code, intensity, identity_code, valid,
                                 local_emotion=condition, condition_dropout=False)
        return project_upper_innovation(velocity, valid, stride=self.stride)

    def flow_loss(self, target_norm9, valid, base_h0, identity_code, global_affect,
                  local, state, noise, time):
        """Mean squared Q-velocity error with caller-controlled noise and time.

        target_norm9 may be the complete normalized upper face: its slow
        component is removed here. Noise, interpolation and velocity each
        stay in Q. State is an explicit condition, never read from this target.
        """
        conditions = self._conditions(valid, base_h0, identity_code, global_affect, local, state)
        if not torch.is_tensor(noise) or noise.shape != target_norm9.shape:
            raise ValueError('Flow noise must match target_norm9')
        if (not torch.is_tensor(time) or time.shape != (len(valid),)
                or time.device != valid.device or not time.is_floating_point()
                or not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any()):
            raise ValueError('Flow time must be finite [B] in [0,1]')
        target = project_upper_innovation(target_norm9, valid, stride=self.stride)
        start = project_upper_innovation(noise, valid, stride=self.stride)
        interpolation = time[:, None, None]
        x = project_upper_innovation((1. - interpolation) * start + interpolation * target,
                                     valid, stride=self.stride)
        velocity = self._velocity(x, time, valid, conditions)
        error = torch.where(valid[..., None], velocity - (target - start), 0.)
        return error.square().sum() / (valid.sum() * 9).clamp_min(1)

    def decode(self, valid, base_h0, identity_code, global_affect, local, state, noise, *, steps=12):
        """Euler decode in Q; caller supplies noise [B,T,9] for reproducibility."""
        if type(steps) is not int or steps < 1:
            raise ValueError('steps must be a positive integer')
        conditions = self._conditions(valid, base_h0, identity_code, global_affect, local, state)
        x = project_upper_innovation(noise, valid, stride=self.stride)
        for index in range(steps):
            time = x.new_full((len(x),), index / steps)
            velocity = self._velocity(x, time, valid, conditions)
            x = project_upper_innovation(x + velocity / steps, valid, stride=self.stride)
        return x
