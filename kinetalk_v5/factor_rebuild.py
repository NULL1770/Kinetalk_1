"""V5 continuation with explicit factor-learning contracts.

The joint 52D decoder is retained. Its generated residual is a conditional
motion correction, never a ground-truth 'pure emotion' residual. A reference
is explained in its own content/affect context; only a global personal code
is transferred. The privileged teacher is trained by actual reconstruction
in the deployed decoder coordinates, not by motion-minus-neutral labels.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from kinetalk_b0.models.expression_response import TemporalEncoder, clean, masked_pool
from kinetalk_v4.unified_model_v4 import KineTalkV4Config, KineTalkV4Model
from kinetalk_v4.acoustic_geometry import geometry_audio_context
from kinetalk_v2.unified_model import adjacency


@dataclass(frozen=True)
class RebuildConfig:
    flow_steps: int = 12
    flow_hidden: int = 128
    teacher_delta_limit: float = 2.0
    # Versioned base distribution. Old checkpoints retain white noise.
    flow_noise_kernel: int = 1
    flow_noise_correlation: float = 1.0


def trajectory_noise(white, valid, times=None, kernel=1):
    """Unit-variance Gaussian base on each contiguous native-25fps run.

    This changes the flow BASE covariance, never the teacher endpoint or the
    decoded motion. Boundary normalization preserves unit marginal variance;
    no coefficient is mixed across invalid frames or native timestamp gaps.
    """
    if kernel < 1 or kernel % 2 != 1:
        raise ValueError('flow noise kernel must be a positive odd integer')
    if white.ndim != 3 or valid.shape != white.shape[:2]:
        raise ValueError('noise expects [B,T,D] and valid [B,T]')
    if times is not None and times.shape != valid.shape:
        raise ValueError('times must match valid')
    if kernel == 1:
        return clean(white, valid)
    edge = (valid[:, 1:] & valid[:, :-1]) if times is None else adjacency(valid, times)
    runs = torch.cat((torch.zeros_like(valid[:, :1], dtype=torch.long),
                      (~edge).long().cumsum(-1)), -1)
    t = white.shape[1]
    index = torch.arange(t, device=white.device)
    half = kernel // 2
    noise = torch.zeros_like(white)
    variance = white.new_zeros(*valid.shape, 1)
    # Raised-cosine weights define the finite-run Gaussian covariance.
    for offset in range(-half, half + 1):
        source = index + offset
        inside = (source >= 0) & (source < t)
        source = source.clamp(0, t - 1)
        use = valid & valid[:, source] & inside[None] & (runs == runs[:, source])
        weight = .5 * (1. + torch.cos(white.new_tensor(torch.pi * offset / (half + 1))))
        contribution = use[..., None].to(white.dtype) * weight
        noise = noise + clean(white[:, source], use) * weight
        variance = variance + contribution.square()
    return clean(noise / variance.clamp_min(1e-12).sqrt(), valid)


def load_core(checkpoint):
    """Strictly load the effective retained-v5 (v4-family) checkpoint."""
    cfg = KineTalkV4Config(**checkpoint['config'])
    s = checkpoint['model']
    core = KineTalkV4Model(cfg, s['feature_mean'], s['feature_std'], s['scales'],
                         s['decoder.geometry_jacobian'], s['decoder.geometry_offset'],
                         s['decoder.geometry_scale'])
    core.load_state_dict(s, strict=True)
    return core


class ObservationTeacher(nn.Module):
    """Privileged correction to an audio prior in the working decoder space.

Y and B0 are aligned to the query's OWN audio. They are concatenated as
observations/context, not subtracted into an emotion target. Zero-start
heads make its initial trajectory exactly the retained audio trajectory.
"""
    def __init__(self, cfg, limit):
        super().__init__()
        self.limit = limit
        self.encoder = TemporalEncoder(156 + cfg.content_dim + cfg.state_dim,
                                       cfg.hidden, 3, attention=1)
        self.state = nn.Linear(cfg.hidden, cfg.state_dim)
        self.emotion = nn.Linear(cfg.hidden, cfg.classes)
        self.intensity = nn.Linear(cfg.hidden, cfg.intensities)
        for head in (self.state, self.emotion, self.intensity):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, motion, base, content, valid, channels, prior, scales):
        obs = valid[..., None] & channels[:, None]
        y = torch.where(obs, motion, 0.) / scales
        c = clean(base / scales, valid)
        mask = channels[:, None].expand_as(y).float()
        hidden = self.encoder(torch.cat((y, c, mask, content.detach(),
                                         prior['state'].detach()), -1), valid)
        pooled = masked_pool(hidden, valid)
        delta = self.limit * torch.tanh(self.state(hidden) / self.limit)
        return {'state': clean(prior['state'].detach() + delta, valid),
                'emotion_logits': prior['emotion_logits'].detach() + self.emotion(pooled),
                'intensity_logits': prior['intensity_logits'].detach() + self.intensity(pooled),
                'hidden': prior['hidden'].detach(), 'observation_hidden': hidden,
                'correction': clean(delta, valid)}


class ReferenceContext(nn.Module):
    """Amortized context correction, paired with cross-reference query losses.

Pooling alone does not remove linguistic information. This module is only
a mechanism; invariance must be measured with unseen sentences/emotions.
"""
    def __init__(self, cfg):
        super().__init__()
        self.encoder = TemporalEncoder(156 + cfg.content_dim + cfg.state_dim +
                                       cfg.classes + 1, cfg.hidden, 3, attention=1)
        self.output = nn.Linear(cfg.hidden, cfg.style_dim)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, motion, valid, channels, content, affect, scales):
        obs = valid[..., None] & channels[:, None]
        y = torch.where(obs, motion, 0.) / scales
        centered = clean(y - masked_pool(y, valid)[:, None], valid)
        probs = affect['emotion_logits'].softmax(-1)
        levels = torch.linspace(.5, 1.5, affect['intensity_logits'].shape[-1],
                                device=y.device)
        intensity = (affect['intensity_logits'].softmax(-1) * levels).sum(-1)
        x = torch.cat((y, centered, channels[:, None].expand_as(y).float(),
                       content.detach(), affect['state'].detach(),
                       probs[:, None].expand(-1, y.shape[1], -1),
                       intensity[:, None, None].expand(-1, y.shape[1], 1)), -1)
        return self.output(masked_pool(self.encoder(x, valid), valid))


class PersonalResponseField(nn.Module):
    """Low-rank, content-conditioned personal response after v5 decoding.

    The output basis is zero at initialization, so v5 is an exact control.
    A basis depends on the query content/state, while the small
style code selects the person's response direction; the reference sequence is
never copied frame by frame.
"""
    def __init__(self, cfg, rank=8):
        super().__init__(); self.rank = int(rank)
        self.basis = nn.Sequential(nn.Linear(cfg.content_dim + cfg.state_dim, cfg.hidden),
                                   nn.SiLU(), nn.Linear(cfg.hidden, 52 * self.rank))
        self.code = nn.Linear(cfg.style_dim, self.rank)
        nn.init.zeros_(self.basis[-1].weight); nn.init.zeros_(self.basis[-1].bias)
        nn.init.normal_(self.code.weight, std=.01); nn.init.zeros_(self.code.bias)

    def forward(self, content, state, style, valid, scales):
        b, t, _ = content.shape
        basis = self.basis(torch.cat((content.detach(), state.detach()), -1)).reshape(b, t, 52, self.rank)
        code = self.code(style).tanh()
        response = torch.einsum('btdr,br->btd', basis, code)
        return clean(torch.tanh(response) * scales * .5, valid)


class StateFlow(nn.Module):
    """Conditional flow of compact teacher states; no reference/style input.

The endpoint is a standardized teacher-minus-audio STATE correction, not
an ARKit residual or a raw unmatched sequence subtraction. Existing event
hazard/duration/amplitude sampling remains active after state sampling.
"""
    def __init__(self, cfg, hidden, noise_kernel=1, noise_correlation=1.0):
        super().__init__()
        self.condition = nn.Linear(cfg.hidden + cfg.classes + 1, hidden)
        self.encoder = TemporalEncoder(cfg.state_dim + hidden + 4, hidden, 2, attention=1)
        self.output = nn.Linear(hidden, cfg.state_dim)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.register_buffer('scale', torch.ones(cfg.state_dim) * .1)
        # The continuation is exactly v5 until the student has learned a
        # trajectory distribution. The pilot opens this gate only after the
        # matched flow stage, so initialization cannot silently change output.
        self.register_buffer('sample_gain', torch.tensor(0.0))
        if noise_kernel < 1 or noise_kernel % 2 != 1:
            raise ValueError('flow noise kernel must be a positive odd integer')
        self.noise_kernel = int(noise_kernel)
        if not 0. <= noise_correlation <= 1.:
            raise ValueError('noise_correlation must be in [0,1]')
        self.noise_correlation = float(noise_correlation)

    def base_noise(self, like, valid, *, times=None, generator=None):
        if self.noise_kernel != 1 and self.noise_correlation != 1.:
            return self.mixed_base_noise(like, valid, times=times, generator=generator,
                                        colored_weight=self.noise_correlation)
        white = torch.randn(like.shape, device=like.device, dtype=like.dtype,
                            generator=generator)
        return trajectory_noise(white, valid, times, self.noise_kernel)

    def mixed_base_noise(self, like, valid, *, times=None, generator=None, colored_weight=1.0):
        """Independent Gaussian mixture, shared by training and sampling."""
        if not 0. <= colored_weight <= 1.:
            raise ValueError('colored_weight must be in [0,1]')
        colored_white = torch.randn(like.shape, device=like.device, dtype=like.dtype,
                                    generator=generator)
        white = torch.randn(like.shape, device=like.device, dtype=like.dtype,
                            generator=generator)
        colored = trajectory_noise(colored_white, valid, times, self.noise_kernel)
        return clean(colored_weight ** .5 * colored + (1. - colored_weight) ** .5 *
                     torch.where(valid[..., None], white, 0.), valid)

    def forward(self, x, tau, hidden, emotion, intensity, valid):
        if tau.ndim != 3 or tau.shape[1:] != (1, 1):
            raise ValueError('tau must be [B,1,1]')
        t = x.shape[1]
        c = self.condition(torch.cat((hidden, emotion[:, None].expand(-1, t, -1),
                                      intensity[:, None, None].expand(-1, t, 1)), -1))
        clock = torch.cat((tau, tau.sin(), tau.cos(), tau.square()), -1).expand(-1, t, -1)
        h = self.encoder(torch.cat((x, c, clock), -1), valid)
        return clean(self.output(h), valid)

    def loss(self, target, prior, hidden, emotion, intensity, valid, *, times=None):
        end = clean((target.detach() - prior.detach()) / self.scale, valid)
        start = self.base_noise(end, valid, times=times)
        tau = torch.rand(len(end), 1, 1, device=end.device) * .9 + .05
        x = (1. - tau) * start + tau * end
        velocity = self(x, tau, hidden.detach(), emotion.detach(), intensity.detach(), valid)
        use = valid[..., None].expand_as(velocity)
        return (velocity - (end - start)).square()[use].mean()

    @torch.no_grad()
    def sample(self, prior, hidden, emotion, intensity, valid, seed, steps, *, times=None):
        if steps < 2:
            raise ValueError('at least two integration steps required')
        rng = torch.Generator(device=prior.device).manual_seed(int(seed))
        x = self.base_noise(prior, valid, times=times, generator=rng)
        for i in range(steps):
            tau = torch.full((len(x), 1, 1), i / steps, device=x.device)
            x = clean(x + self(x, tau, hidden, emotion, intensity, valid) / steps, valid)
        return clean(prior + self.sample_gain * x * self.scale, valid)


class V5FactorRebuild(nn.Module):
    def __init__(self, core, config=RebuildConfig()):
        super().__init__()
        self.core = core
        self.cfg = core.cfg
        self.rebuild_config = config
        self.teacher = ObservationTeacher(core.cfg, config.teacher_delta_limit)
        self.reference_context = ReferenceContext(core.cfg)
        self.flow = StateFlow(core.cfg, config.flow_hidden, config.flow_noise_kernel,
                              config.flow_noise_correlation)
        self.style_response = PersonalResponseField(core.cfg)
        # Immutable starting audio context defines stable teacher coordinates
        # and the reference's own affect context while the student is fitted.
        import copy
        self.anchor_student = copy.deepcopy(core.student).eval().requires_grad_(False)
        self.style_emotion = nn.Linear(self.cfg.style_dim, self.cfg.classes)
        self.style_content = nn.Linear(self.cfg.style_dim, self.cfg.content_dim)
        self.affect_style = nn.Linear(self.cfg.state_dim, self.cfg.style_dim)

    @classmethod
    def from_v5(cls, checkpoint, config=RebuildConfig()):
        return cls(load_core(checkpoint), config)

    @classmethod
    def from_checkpoint(cls, ck):
        core = load_core({'config': ck['config'], 'model':
                          {k[5:]: v for k, v in ck['model'].items() if k.startswith('core.')}})
        model = cls(core, RebuildConfig(**ck['rebuild_config']))
        model.load_state_dict(ck['model'], strict=True)
        return model

    def checkpoint(self):
        return {'config': asdict(self.cfg), 'rebuild_config': asdict(self.rebuild_config),
                'model': self.state_dict(), 'schema': 'v5_factor_rebuild_v1'}

    def affect_audio(self, audio, valid, anchor=False):
        x = audio[..., 768:] if audio.shape[-1] == 1540 else audio
        if x.shape[-1] != self.cfg.audio_dim:
            raise ValueError('expected raw 772D affect or raw 1540D audio')
        x = clean((x - self.core.feature_mean) / self.core.feature_std, valid)
        return (self.anchor_student if anchor else self.core.student)(x, valid)

    def encode_style(self, refs, contextual=True):
        if refs['motion'].ndim != 3:
            raise ValueError('deployment takes one reference [B,T,52]')
        old = self.core.encode_style(refs)
        if not contextual or not self.cfg.use_style:
            return old
        if 'h0' not in refs or 'audio_features' not in refs:
            raise ValueError('contextual style requires reference OWN audio and h0')
        if 'h0_available' in refs and not refs['h0_available'].all():
            raise ValueError('reference h0 must be genuine, not zero missing context')
        affect = self.affect_audio(refs['audio_features'], refs['valid'], anchor=True)
        correction = self.reference_context(refs['motion'], refs['valid'], refs['channel_mask'],
                                              refs['h0'], affect, self.core.scales)
        return old + correction

    def visual_teacher(self, batch):
        prior = self.affect_audio(batch['audio_features'], batch['valid'], anchor=True)
        return self.teacher(batch['motion'], batch['b0'], batch['h0'], batch['valid'],
                            batch['channel_mask'], prior, self.core.scales)

    def decode(self, batch, prediction, style, *, events=None, seed=47, mode='sample',
               emotion=None, intensity=None):
        valid = batch['valid']
        conditioned = self.core.condition(prediction, valid, emotion=emotion, intensity=intensity)
        rng = torch.Generator(device=style.device).manual_seed(int(seed))
        schedule = self.core.events(prediction['hidden'], conditioned['state'], style, valid,
                                    events, generator=rng, mode='observed' if events is not None else mode,
                                    times=batch.get('times'))
        context = geometry_audio_context(self.cfg.geometry_audio_mode, batch['audio_features'],
                                         batch['h0'], valid, self.core.feature_mean, self.core.feature_std)
        out = self.core.decode(batch['h0'], batch['b0'], valid, conditioned, style,
                               schedule['events'], geometry_audio=context)
        response = self.style_response(batch['h0'], conditioned['state'], style, valid,
                                       self.core.scales)
        out['motion'] = clean(out['motion'] + response, valid)
        out['style_response'] = response
        return {**out, 'conditioned': conditioned, 'schedule': schedule, 'style': style,
                'prediction': prediction}

    def predict(self, audio, content, base, valid, refs, *, times=None, seed=47,
                mode='sample', sample_flow=True, contextual_style=True, emotion=None, intensity=None):
        """No query motion or teacher; single independent enrollment only."""
        style = self.encode_style(refs, contextual=contextual_style)
        prediction = self.affect_audio(audio, valid)
        if sample_flow:
            cond = self.core.condition(prediction, valid, emotion=emotion, intensity=intensity)
            state = self.flow.sample(prediction['state'], prediction['hidden'], cond['emotion'],
                                     cond['intensity'], valid, seed + 100000, self.rebuild_config.flow_steps,
                                     times=times)
            prediction = {**prediction, 'state': state}
        b = {'audio_features': audio, 'h0': content, 'b0': base, 'valid': valid, 'times': times}
        return self.decode(b, prediction, style, seed=seed, mode=mode, emotion=emotion, intensity=intensity)
