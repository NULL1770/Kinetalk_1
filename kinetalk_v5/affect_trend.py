"""Audio-conditioned mean-state distillation on an immutable v5 decoder.

This is a mean/innovation split, not a claim that audio uniquely determines
the observed performance, nor that decoder-native states are pure emotion.
The initial mean prediction equals the saved v5 repair exactly.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from kinetalk_b0.models.expression_response import clean, masked_pool
from kinetalk_v5.factor_rebuild import RebuildConfig, V5FactorRebuild, load_core


@dataclass(frozen=True)
class TrendConfig:
    hidden: int = 128
    correction_limit: float = 1.0


class AudioTrendCorrection(nn.Module):
    """Infer a bounded conditional mean correction from audio features only."""
    def __init__(self, cfg, config):
        super().__init__()
        self.limit = config.correction_limit
        self.network = nn.Sequential(nn.Linear(2 * cfg.hidden + cfg.state_dim + cfg.classes + 1,
                                             config.hidden), nn.SiLU(),
                                     nn.Linear(config.hidden, config.hidden), nn.SiLU(),
                                     nn.Linear(config.hidden, cfg.state_dim))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)

    def forward(self, prediction, valid):
        hidden = prediction['hidden'].detach()
        global_hidden = masked_pool(hidden, valid)[:, None].expand_as(hidden)
        emotion = prediction['emotion_logits'].detach().softmax(-1)
        logits = prediction['intensity_logits'].detach()
        levels = torch.linspace(.5, 1.5, logits.shape[-1], device=hidden.device, dtype=hidden.dtype)
        intensity = (logits.softmax(-1) * levels).sum(-1)
        x = torch.cat((hidden, global_hidden, prediction['state'].detach(),
                       emotion[:, None].expand(-1, hidden.shape[1], -1),
                       intensity[:, None, None].expand(-1, hidden.shape[1], 1)), -1)
        return clean(self.limit * torch.tanh(self.network(x) / self.limit), valid)


class V5AffectTrend(V5FactorRebuild):
    def __init__(self, core, rebuild_config=RebuildConfig(), trend_config=TrendConfig()):
        super().__init__(core, rebuild_config)
        self.trend_config = trend_config
        self.audio_trend = AudioTrendCorrection(core.cfg, trend_config)

    @classmethod
    def from_rebuild(cls, ck, trend_config=TrendConfig()):
        """Explicit zero-start extension; unrelated missing keys are errors."""
        prior = V5FactorRebuild.from_checkpoint(ck)
        model = cls(prior.core, RebuildConfig(**ck['rebuild_config']), trend_config)
        missing, unexpected = model.load_state_dict(ck['model'], strict=False)
        expected = {'audio_trend.' + k for k in model.audio_trend.state_dict()}
        if set(missing) != expected or unexpected:
            raise RuntimeError('unrecognized rebuild migration keys')
        return model

    @classmethod
    def from_checkpoint(cls, ck):
        if ck.get('schema') != 'v5_affect_trend_v1':
            raise ValueError('expected a v5_affect_trend_v1 checkpoint')
        core = load_core({'config': ck['config'], 'model':
                          {k[5:]: v for k, v in ck['model'].items() if k.startswith('core.')}})
        model = cls(core, RebuildConfig(**ck['rebuild_config']), TrendConfig(**ck['trend_config']))
        model.load_state_dict(ck['model'], strict=True)
        return model

    def checkpoint(self):
        return {**super().checkpoint(), 'schema': 'v5_affect_trend_v1',
                'trend_config': asdict(self.trend_config)}

    def affect_audio(self, audio, valid, anchor=False):
        prior = super().affect_audio(audio, valid, anchor=anchor)
        if anchor:
            return prior
        correction = self.audio_trend(prior, valid)
        return {**prior, 'state': clean(prior['state'] + correction, valid),
                'trend_correction': correction}
