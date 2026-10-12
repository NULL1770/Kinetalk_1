"""Audio-conditioned rig-coordinate correction, isolated from event history.

The native acoustic arm receives HuBERT768 + prosody4, whereas a capacity-matched
coarse control receives only frozen h0 information + the same prosody. A zero
output head preserves the complete source model at initialization. This adds
capacity in measured geometry, not a physiological articulatory subspace.
"""
import math
import torch
from torch import nn
from kinetalk_b0.models.expression_response import TemporalEncoder, clean


class AcousticGeometryReceiver(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.encoder = TemporalEncoder(772 + cfg.state_dim + cfg.style_dim + cfg.classes + 1,
                                       cfg.hidden, 2, attention=0)
        self.output = nn.Linear(cfg.hidden, cfg.geometry_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, state, emotion, intensity, style, valid):
        if features is None or features.shape != (*state.shape[:2], 772):
            raise ValueError('Native geometry receiver requires aligned 772D audio context')
        acoustic = torch.cat((self.norm(clean(features[..., :768], valid)),
                              clean(features[..., 768:], valid)), -1)
        t=state.shape[1]
        condition=torch.cat((state, style[:,None].expand(-1,t,-1),
            emotion[:,None].expand(-1,t,-1), intensity[:,None,None].expand(-1,t,1)), -1)
        hidden=self.encoder(torch.cat((acoustic,condition),-1),valid)
        return clean(self.output(hidden),valid)


def geometry_audio_context(mode, audio, content, valid, mean, std):
    if mode == 'none':
        return None
    if audio.ndim != 3 or audio.shape[:2] != valid.shape:
        raise ValueError('Geometry audio must share the native frame mask')
    if mode == 'hubert':
        if audio.shape[-1] != 1540:
            raise ValueError('HuBERT receiver needs the full 1540D cache')
        source=audio[...,:768]
    elif mode == 'coarse':
        if content.shape[:2] != valid.shape or content.shape[-1] < 1:
            raise ValueError('Coarse receiver needs aligned frozen h0')
        # A fixed tiling adds no information or trainable parameters. Both
        # control and acoustic arms have the exact same receiver architecture.
        source=content.repeat(1,1,math.ceil(768/content.shape[-1]))[...,:768]
    else:
        raise ValueError('Unknown geometry audio context')
    if audio.shape[-1] not in (772,1540):
        raise ValueError('Geometry prosody needs the known feature schema')
    prosody=(audio[...,-4:]-mean[-4:])/std[-4:]
    return clean(torch.cat((source,prosody),-1),valid)


def load_zero_geometry_extension(model, source):
    current=model.state_dict()
    missing=set(current)-set(source)
    allowed={name for name in current if name.startswith('decoder.geometry_audio.')}
    if set(source)-set(current) or missing not in (set(),allowed):
        raise ValueError('Checkpoint mismatch exceeds the declared geometry receiver')
    if missing and (not allowed or torch.count_nonzero(model.decoder.geometry_audio.output.weight)
                    or torch.count_nonzero(model.decoder.geometry_audio.output.bias)):
        raise ValueError('Geometry receiver must start exactly at zero')
    loaded=model.load_state_dict(source,strict=not missing)
    if set(loaded.missing_keys)!=missing or loaded.unexpected_keys:
        raise ValueError('Unexpected geometry receiver checkpoint mismatch')
    return bool(missing)
