"""KineTalk v3: dual-rate articulatory residual + privileged affect routing.

This module is intentionally isolated from Phase53 and kinetalk_v2.  B0 remains
the frozen coarse speech prior.  A low-rank content-conditioned articulatory
residual handles fast mouth/jaw correction, while a slower affect path and a
duration-aware event student handle expression.  The motion teacher is
privileged during training and is absent from ``predict``.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
from kinetalk_b0.models.expression_response import (
    TemporalEncoder, clean, masked_pool, stride_pool, native_interpolate
)
from kinetalk_v2.unified_model import (
    teacher_targets, ConditionalEventStudent, FeasibleResidualComposer,
    relation_style_loss, anchor_matrix,
)


@dataclass(frozen=True)
class UnifiedV3Config:
    audio_dim: int = 772
    content_dim: int = 128
    hidden: int = 128
    decoder_hidden: int = 192
    state_dim: int = 10
    style_dim: int = 64
    event_groups: int = 4
    duration_frames: tuple = (2, 4, 8, 16, 32)
    classes: int = 8
    intensities: int = 4
    articulatory_rank: int = 16
    slow_stride: int = 5


class PrivilegedAffectTeacher(nn.Module):
    """Teacher with explicit global affect, continuous state and reliability."""
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.encoder = TemporalEncoder(52 * 3, cfg.hidden, 3, attention=1)
        self.state = nn.Linear(cfg.hidden, cfg.state_dim)
        self.emotion = nn.Linear(cfg.hidden, cfg.classes)
        self.intensity = nn.Linear(cfg.hidden, cfg.intensities)
        self.reliability = nn.Linear(cfg.hidden, cfg.state_dim)

    def forward(self, motion, base, valid, channels, scales):
        obs = valid[..., None] & channels[:, None]
        x = torch.where(obs, motion - base.detach(), 0.) / scales
        h = self.encoder(torch.cat((x, torch.where(obs, motion, 0.) / scales,
                                    channels[:, None].expand_as(x).float()), -1), valid)
        rel = torch.sigmoid(self.reliability(h)).clamp(.1, .99)
        return {
            'hidden': h, 'state': clean(self.state(h), valid),
            'emotion_logits': self.emotion(masked_pool(h, valid)),
            'intensity_logits': self.intensity(masked_pool(h, valid)),
            'reliability': clean(rel, valid),
        }


class AudioAffectStudent(nn.Module):
    """Text-free affect student; input is emotion2vec768 + prosody4."""
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.encoder = TemporalEncoder(cfg.audio_dim, cfg.hidden, 4, attention=1)
        self.state = nn.Linear(cfg.hidden, cfg.state_dim)
        self.slow_state = nn.Linear(cfg.hidden, cfg.state_dim)
        self.emotion = nn.Linear(cfg.hidden, cfg.classes)
        self.intensity = nn.Linear(cfg.hidden, cfg.intensities)

    def forward(self, audio, valid):
        h = self.encoder(audio, valid)
        slow, slow_mask = stride_pool(h, valid, self.cfg.slow_stride)
        slow_state = self.slow_state(slow)
        state_slow = native_interpolate(slow_state, slow_mask, valid, self.cfg.slow_stride)
        fast = self.state(h)
        # The slow branch carries the expression envelope; the fast branch only
        # supplies bounded local correction and retains native audio timing.
        state = clean(state_slow + .25 * fast, valid)
        return {
            'hidden': h, 'state': state, 'state_slow': state_slow,
            'emotion_logits': self.emotion(masked_pool(h, valid)),
            'intensity_logits': self.intensity(masked_pool(h, valid)),
        }


class ArticulatoryResidual(nn.Module):
    """Low-rank content residual restricted to mouth/jaw ARKit directions."""
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.encoder = TemporalEncoder(cfg.content_dim + 52, cfg.hidden, 3, attention=1)
        self.coeff = nn.Linear(cfg.hidden, cfg.articulatory_rank)
        basis = torch.zeros(cfg.articulatory_rank, 52)
        # jaw and mouth channels: jawForward..mouthUpperUpRight.
        allowed = list(range(14, 41))
        basis[:, allowed] = torch.randn(cfg.articulatory_rank, len(allowed)) * .01
        self.basis = nn.Parameter(basis)
        mask = torch.zeros(52); mask[allowed] = 1.
        self.register_buffer('art_mask', mask)
        nn.init.zeros_(self.coeff.bias)

    def forward(self, content, base, valid, scales):
        x = torch.cat((content, base / scales), -1)
        h = self.encoder(x, valid)
        coeff = torch.tanh(self.coeff(h))
        residual = torch.einsum('btr,rc->btc', coeff, self.basis) * scales
        return clean(residual * self.art_mask, valid)


class DualRateExpressionDecoder(nn.Module):
    """Expression decoder with a soft regional gate and legal factor interactions."""
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        dim = cfg.content_dim + 52 + cfg.state_dim + cfg.classes + cfg.style_dim + cfg.event_groups * 3
        self.encoder = TemporalEncoder(dim, cfg.decoder_hidden, 4, attention=1)
        self.output = nn.Linear(cfg.decoder_hidden, 52)
        # State/style affect all face regions; mouth is attenuated, not removed.
        gate = torch.ones(52); gate[14:41] = .35
        self.register_buffer('expression_gate', gate)
        nn.init.normal_(self.output.weight, std=.004); nn.init.zeros_(self.output.bias)

    def forward(self, content, base, state, emotion, style, events, valid, scales):
        t = base.shape[1]
        z = torch.cat((content, base / scales, state,
                       emotion[:, None].expand(-1, t, -1),
                       style[:, None].expand(-1, t, -1), events), -1)
        h = self.encoder(z, valid)
        out = torch.tanh(self.output(h)) * scales * self.expression_gate
        return clean(out, valid)


class UnifiedExpressionV3Model(nn.Module):
    def __init__(self, cfg, feature_mean, feature_std, scales):
        super().__init__(); self.cfg = cfg
        if cfg.state_dim != 10 or cfg.event_groups != 4:
            raise ValueError('V3 anchored schema requires 10 states / 4 events')
        self.register_buffer('feature_mean', feature_mean.float())
        self.register_buffer('feature_std', feature_std.float().clamp_min(1e-3))
        self.register_buffer('scales', scales.float().clamp_min(.02))
        self.register_buffer('guidance_weights', torch.ones(cfg.state_dim))
        self.teacher = PrivilegedAffectTeacher(cfg)
        self.student = AudioAffectStudent(cfg)
        # Keep the established single-reference style encoder contract.
        from kinetalk_v2.unified_model import SingleReferenceStyle
        self.style = SingleReferenceStyle(cfg)
        self.events = ConditionalEventStudent(cfg)
        self.articulatory = ArticulatoryResidual(cfg)
        self.expression = DualRateExpressionDecoder(cfg)
        self.emotion_to_state = nn.Linear(cfg.classes, cfg.state_dim, bias=False)
        self.composer = FeasibleResidualComposer()
        # Explicit affect control must reach the continuous state from the
        # first deployment smoke run; training can subsequently calibrate it.
        nn.init.normal_(self.emotion_to_state.weight, std=.02)

    def encode_style(self, refs):
        return self.style(refs['motion'], refs['valid'], refs['channel_mask'], self.scales)

    def audio_state(self, audio, valid):
        if audio.shape[-1] == 1540: audio = audio[..., 768:]
        if audio.shape[-1] != self.cfg.audio_dim:
            raise ValueError('V3 student audio width differs')
        audio = clean((audio - self.feature_mean) / self.feature_std, valid)
        return self.student(audio, valid)

    def _condition_state(self, student, emotion):
        if emotion.ndim == 2:
            add = self.emotion_to_state(emotion)[:, None]
        else:
            add = self.emotion_to_state(emotion)
        return clean(student['state'] + add, student['state'].new_ones(student['state'].shape[:2], dtype=torch.bool))

    def decode(self, content, base, valid, state, emotion, style, events, constraints=None):
        art = self.articulatory(content, base, valid, self.scales)
        expr = self.expression(content, base, state, emotion, style, events, valid, self.scales)
        residual = art + expr
        proposal = base + residual
        motion = clean(self.composer(proposal, constraints), valid)
        return {'motion': motion, 'proposal': clean(proposal, valid), 'residual': residual,
                'articulatory_residual': art, 'expression_residual': expr}

    def training_forward(self, batch, refs, targets, *, teacher_state=False, constraints=None):
        style = self.encode_style(refs)
        student = self.audio_state(batch['audio_features'], batch['valid'])
        teacher = self.teacher(batch['motion'], batch['b0'], batch['valid'], batch['channel_mask'], self.scales)
        student_probs = student['emotion_logits'].softmax(-1)
        state = teacher['state'] if teacher_state else student['state']
        probs = teacher['emotion_logits'].softmax(-1) if teacher_state else student_probs
        schedule = self.events(student['hidden'], state, style, batch['valid'], targets['events'], mode='observed', times=batch['times'])
        decoded = self.decode(batch['h0'], batch['b0'], batch['valid'], state, probs, style, targets['events'], constraints)
        return {**decoded, 'student': student, 'teacher': teacher, 'event_student': schedule, 'style': style}

    def predict(self, audio, content, base, valid, refs, *, times=None, seed=47, mode='sample', emotion=None, intensity=None, constraints=None):
        style = self.encode_style(refs)
        student = self.audio_state(audio, valid)
        probs = student['emotion_logits'].softmax(-1) if emotion is None else emotion
        if intensity is not None:
            # Intensity is an explicit control of the continuous state and event
            # distribution, not merely a decoder class replacement.
            scale = intensity.reshape(-1, 1, 1).to(student['state']).clamp(.25, 2.)
            student = {**student, 'state': student['state'] * scale}
        conditioned = self._condition_state(student, probs)
        generator = torch.Generator(device=audio.device).manual_seed(seed)
        schedule = self.events(student['hidden'], conditioned, style, valid, generator=generator,
                               mode=mode, times=times)
        decoded = self.decode(content, base, valid, conditioned, probs, style, schedule['events'], constraints)
        return {**decoded, 'student': {**student, 'state': conditioned}, 'style': style, 'schedule': schedule}

    def checkpoint_config(self): return asdict(self.cfg)


def masked_mean(value, mask):
    return torch.where(mask, value, 0.).sum() / mask.sum().clamp_min(1)


def semantic_loss(logits, labels):
    use = labels.ge(0) & labels.lt(logits.shape[-1])
    return F.cross_entropy(logits[use], labels[use]) if use.any() else logits.sum() * 0.


def v3_supervision_losses(model, output, targets, batch, *, teacher_warmup=False):
    valid = batch['valid']; student = output['student']; teacher = output['teacher']; event = output['event_student']
    quality = targets['quality'].bool() & valid[..., None]
    # Learned teacher reliability is floored and detached for routing; it cannot
    # evade its own state supervision by predicting zero confidence.
    routing = (targets['quality'] * teacher['reliability'].detach() * model.guidance_weights).clamp_min(0.)
    teacher_state = masked_mean((teacher['state'] - targets['state']).square(), quality)
    state_distill = ((student['state'] - teacher['state'].detach()).square() * routing).sum() / routing.sum().clamp_min(1)
    state_direct = ((student['state'] - targets['state']).square() * routing).sum() / routing.sum().clamp_min(1)
    emotion_distill = F.kl_div(F.log_softmax(student['emotion_logits'], -1), teacher['emotion_logits'].softmax(-1), reduction='batchmean')
    intensity_distill = F.kl_div(F.log_softmax(student['intensity_logits'], -1), teacher['intensity_logits'].softmax(-1), reduction='batchmean')
    onset = F.binary_cross_entropy_with_logits(event['hazard_logits'], targets['onset'].float(), reduction='none')
    hazard = masked_mean(onset, targets['onset_valid'])
    duration = F.cross_entropy(event['duration_logits'].reshape(-1, event['duration_logits'].shape[-1]), targets['duration'].flatten(), reduction='none').reshape_as(targets['duration'])
    duration = masked_mean(duration, targets['duration_valid'])
    params = event['amplitude_params']; logamp = targets['amplitude'].clamp_min(.005).log(); logstd = params[..., 1].clamp(-4, 0)
    amplitude = masked_mean(.5 * ((logamp - params[..., 0]) / logstd.exp()).square() + logstd, targets['duration_valid'])
    emotion = semantic_loss(student['emotion_logits'], batch['emotion_id'])
    teacher_emotion = semantic_loss(teacher['emotion_logits'], batch['emotion_id'])
    intensity_labels = torch.where(batch['intensity_valid'], batch['intensity_id'], torch.full_like(batch['intensity_id'], -1))
    intensity = semantic_loss(student['intensity_logits'], intensity_labels)
    losses = {'teacher_state': teacher_state, 'state_distill': state_distill, 'state_direct': state_direct,
              'emotion_distill': emotion_distill, 'intensity_distill': intensity_distill,
              'hazard': hazard, 'duration': duration, 'amplitude': amplitude,
              'emotion': emotion, 'teacher_emotion': teacher_emotion, 'intensity': intensity}
    losses['total'] = (teacher_state + teacher_emotion + .2 * intensity_distill if teacher_warmup else
                       .2 * teacher_state + .7 * state_distill + .5 * state_direct + .5 * emotion_distill +
                       .2 * intensity_distill + hazard + .2 * duration + .02 * amplitude + emotion + .2 * intensity)
    return losses


__all__ = ['UnifiedV3Config', 'UnifiedExpressionV3Model', 'teacher_targets',
           'v3_supervision_losses', 'relation_style_loss', 'anchor_matrix']
