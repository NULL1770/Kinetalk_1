"""Native-clock expression response with an audio prior and training posterior.

No full-motion flow, teacher prototypes, speaker embedding table, h0 shortcut,
or anatomical mouth mask. B0 is an external, frozen neutral prediction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ResponseConfig:
    hidden: int = 128
    decoder_hidden: int = 192
    global_dim: int = 32
    local_dim: int = 16
    style_dim: int = 64
    stride: int = 2
    classes: int = 8
    intensities: int = 4
    logvar_min: float = -6.
    logvar_max: float = 2.
    # Diagnostic ablation: fix only p's covariance to I. q's covariance,
    # sampling and reconstruction stay unchanged; no learned p variance head.
    prior_variance: str = 'learned'
    # A deterministic decoder transform; KL remains over the original Gaussians.
    center_local: bool = False
    reference_encoder: str = 'temporal'
    style_modulation: str = 'joint'
    response_head: str = 'residual'


def clean(x, mask):
    return torch.where(mask[..., None], x, 0.)


def check_sequence(x, mask):
    if x.ndim != 3 or mask.shape != x.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('Expected sequence[B,T,D] with bool mask[B,T]')
    if not mask.any(1).all() or not torch.isfinite(x[mask]).all():
        raise ValueError('Nonempty finite observed sequence required')


def masked_pool(x, mask):
    return clean(x, mask).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation=1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.out = nn.Linear(width, width)

    def forward(self, x, mask):
        h = clean(self.norm(x), mask).transpose(1, 2)
        h = clean(F.silu(self.conv(h).transpose(1, 2)), mask)
        return clean(x + self.out(h), mask)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden, depth, attention=2):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(TemporalBlock(hidden, 2**i) for i in range(depth))
        self.attention = nn.ModuleList(nn.TransformerEncoderLayer(
            hidden, 4, 2*hidden, dropout=0., activation='gelu', batch_first=True,
            norm_first=True) for _ in range(attention))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, mask):
        check_sequence(x, mask)
        h = clean(self.input(clean(x, mask)), mask)
        for block in self.blocks:
            h = block(h, mask)
        # Native index, never the compacted valid index or padded batch length.
        t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
        freq = torch.exp(torch.arange(0, h.shape[-1], 2, device=x.device,
                                      dtype=x.dtype) * (-math.log(10000.)/h.shape[-1]))
        pos = torch.stack(((t*freq).sin(), (t*freq).cos()), -1).flatten(-2)
        if self.attention:
            h = clean(h + pos[None], mask)
        for layer in self.attention:
            h = clean(layer(h, src_key_padding_mask=~mask), mask)
        return clean(self.norm(h), mask)


def stride_pool(x, mask, stride):
    pad = (-x.shape[1]) % stride
    z = F.pad(clean(x, mask), (0, 0, 0, pad))
    m = F.pad(mask, (0, pad)).reshape(len(x), -1, stride)
    z = z.reshape(len(x), -1, stride, x.shape[-1])
    return z.sum(2)/m.sum(2)[..., None].clamp_min(1), m.any(2)


def native_interpolate(tokens, token_mask, valid, stride):
    """Interpolate fixed token centers without rescaling the clip's duration."""
    t = torch.arange(valid.shape[1], device=tokens.device, dtype=tokens.dtype)
    position = (t - (stride-1)/2) / stride
    left = position.floor().long()
    fraction = (position-left).clamp(0, 1)[None, :, None]
    lo, hi = left.clamp(0, tokens.shape[1]-1), (left+1).clamp(0, tokens.shape[1]-1)
    wl = (1-fraction)*token_mask[:, lo, None]
    wh = fraction*token_mask[:, hi, None]
    z = (clean(tokens, token_mask)[:, lo]*wl + clean(tokens, token_mask)[:, hi]*wh)
    return clean(z/(wl+wh).clamp_min(1e-8), valid)


class GaussianEncoder(nn.Module):
    def __init__(self, cfg, input_dim, depth, *, fixed_variance=False):
        super().__init__()
        self.cfg = cfg
        self.encoder = TemporalEncoder(input_dim, cfg.hidden, depth)
        self.fixed_variance = fixed_variance
        self.global_head = nn.Linear(cfg.hidden, 2*cfg.global_dim)
        self.local_head = nn.Linear(cfg.hidden, 2*cfg.local_dim)
        if fixed_variance:
            # Initialize identically to the learned-covariance control, then
            # discard variance rows without drawing RNG or keeping parameters.
            for head, width in ((self.global_head,cfg.global_dim),(self.local_head,cfg.local_dim)):
                head.weight = nn.Parameter(head.weight[:width].detach().clone())
                head.bias = nn.Parameter(head.bias[:width].detach().clone())
                head.out_features = width

    def forward(self, x, valid):
        h = self.encoder(x, valid)
        tokens, mask = stride_pool(h, valid, self.cfg.stride)
        global_raw = self.global_head(masked_pool(h, valid))
        local_raw = self.local_head(tokens)
        if self.fixed_variance:
            gm, um = global_raw, local_raw
            gv, uv = torch.zeros_like(gm), torch.zeros_like(um)
        else:
            gm, gv = global_raw.chunk(2, -1)
            um, uv = local_raw.chunk(2, -1)
        return {'g_mean': gm, 'g_logvar': gv.clamp(self.cfg.logvar_min, self.cfg.logvar_max),
                'u_mean': clean(um, mask), 'u_logvar': clean(uv.clamp(
                    self.cfg.logvar_min, self.cfg.logvar_max), mask), 'u_mask': mask}


class ReferenceStyle(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = TemporalEncoder(52*3, cfg.hidden, 3, attention=0)
        self.project = nn.Linear(cfg.hidden, cfg.style_dim)
        self.aggregate = nn.Sequential(nn.Linear(cfg.style_dim, cfg.style_dim), nn.SiLU(),
                                       nn.Linear(cfg.style_dim, cfg.style_dim))

    def forward(self, motion, base, valid, channels, scales):
        if motion.ndim != 4 or base.shape != motion.shape or valid.shape != motion.shape[:3]:
            raise ValueError('References must be independent [B,R,T,52] sequences')
        b, r, t, _ = motion.shape
        if channels.shape != (b, r, 52) or channels.dtype != torch.bool:
            raise ValueError('Reference channel support must be bool[B,R,52]')
        mask = valid & channels.any(-1)[..., None]
        ref_valid = mask.any(-1)
        if not ref_valid.any(1).all():
            raise ValueError('Every query requires an observed reference')
        obs = mask[..., None] & channels[:, :, None]
        m = torch.where(obs, motion, 0.) / scales
        c = torch.where(obs, base.detach(), 0.) / scales
        if not torch.isfinite(m).all() or not torch.isfinite(c).all():
            raise ValueError('Nonfinite observed reference')
        features = torch.cat((m-c, c, channels[:, :, None].expand(-1,-1,t,-1)), -1)
        # Empty slots never enter attention/pooling; absent refs receive no weight.
        ix = ref_valid.flatten().nonzero(as_tuple=True)[0]
        h = self.encoder(features.flatten(0,1)[ix], mask.flatten(0,1)[ix])
        codes = self.project(masked_pool(h, mask.flatten(0,1)[ix]))
        per_ref = codes.new_zeros(b*r, codes.shape[-1]).index_copy(0, ix, codes).reshape(b,r,-1)
        code = self.aggregate(per_ref.sum(1)/ref_valid.sum(1,keepdim=True))
        return {'code': code, 'per_reference': per_ref, 'reference_valid': ref_valid}


def reference_statistics(motion, base, valid, channels, scales):
    """Equal-reference native moments, without reference temporal order.

    Means describe posture while centered moments describe response to B0.
    These statistics can still depend on phonetic coverage and fitting error;
    they are not guaranteed causal identity coordinates.
    """
    if motion.ndim != 4 or base.shape != motion.shape or valid.shape != motion.shape[:3]:
        raise ValueError('References must be [B,R,T,52] with matching observations')
    b,r,t,c=motion.shape
    if c!=52 or channels.shape!=(b,r,c) or channels.dtype!=torch.bool or valid.dtype!=torch.bool:
        raise ValueError('Invalid reference channel/frame mask')
    obs=valid[...,None]&channels[:,:,None]
    if not obs.any((1,2,3)).all():raise ValueError('Every query requires observed references')
    m=torch.where(obs,motion,0.)/scales
    x=torch.where(obs,base.detach(),0.)/scales
    if not torch.isfinite(m).all() or not torch.isfinite(x).all():
        raise ValueError('Nonfinite observed reference')
    count=obs.sum(2);support=count>0
    mm=m.sum(2)/count.clamp_min(1);xm=x.sum(2)/count.clamp_min(1)
    mc=torch.where(obs,m-mm[:,:,None],0.);xc=torch.where(obs,x-xm[:,:,None],0.)
    mv=mc.square().sum(2)/count.clamp_min(1)
    xv=xc.square().sum(2)/count.clamp_min(1)
    cov=(mc*xc).sum(2)/count.clamp_min(1)
    # Zero variance stays zero; this is descriptor extraction, not a motion gradient.
    mr=mv.sqrt();xr=xv.sqrt();cor=cov/(mr*xr).clamp_min(1e-6)
    def pool(z):
        return torch.where(support,z,0.).sum(1)/support.sum(1).clamp_min(1)
    present=support.any(1).to(m.dtype)
    posture=torch.cat((pool(mm),pool(xm),present),-1)
    response=torch.cat((pool(mr),pool(xr),pool(cor),present),-1)
    return posture,response,support.any(-1)


class StatisticalReferenceStyle(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        half=cfg.style_dim//2
        self.posture=nn.Sequential(nn.Linear(156,cfg.hidden),nn.SiLU(),nn.Linear(cfg.hidden,half))
        self.response=nn.Sequential(nn.Linear(208,cfg.hidden),nn.SiLU(),nn.Linear(cfg.hidden,half))

    def forward(self,motion,base,valid,channels,scales):
        posture,response,present=reference_statistics(motion,base,valid,channels,scales)
        return {'code':torch.cat((self.posture(posture),self.response(response)),-1),
                'reference_valid':present}


class ResponseDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input = nn.Linear(52, cfg.decoder_hidden)
        self.blocks = nn.ModuleList(TemporalBlock(cfg.decoder_hidden, 2**i) for i in range(4))
        self.partitioned = cfg.reference_encoder == 'statistics'
        self.factorized = cfg.style_modulation == 'factorized'
        self.affect_width = cfg.global_dim+cfg.local_dim
        self.style_width = cfg.style_dim//2 if self.partitioned else cfg.style_dim
        cond_dim = cfg.global_dim+cfg.local_dim+self.style_width
        self.modulations = nn.ModuleList(nn.Linear(cond_dim, cfg.decoder_hidden*2) for _ in range(4))
        self.output = nn.Linear(cfg.decoder_hidden, 52)
        self.bias = nn.Linear(self.style_width, 52)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.bias.bias)

    def forward(self, base, g, u, style, valid, scales, posture_grad_mask=None):
        base = clean(base.detach(), valid)
        h = clean(self.input(base/scales), valid)
        posture,response=style.chunk(2,-1) if self.partitioned else (style,style)
        condition = torch.cat((g[:,None].expand(-1,len(u[0]),-1),u,
                               response[:,None].expand(-1,len(u[0]),-1)), -1)
        for block, mod in zip(self.blocks, self.modulations):
            if self.factorized:
                # Reference magnitude cannot saturate the affect nonlinearity.
                affect = F.linear(condition[...,:self.affect_width],mod.weight[:,:self.affect_width],mod.bias)
                reference = F.linear(condition[...,self.affect_width:],mod.weight[:,self.affect_width:])
                gain,shift=affect.chunk(2,-1)
                style_gain,style_shift=reference.chunk(2,-1)
                h=h*(1+.1*gain.tanh())+.1*shift
                h=clean(h*(1+.1*style_gain.tanh())+.1*style_shift.tanh(),valid)
            else:
                gain, shift = mod(condition).chunk(2,-1)
                h = clean(h*(1+.1*gain.tanh())+.1*shift, valid)
            h = block(h, valid)
        offset=self.bias(posture)
        if posture_grad_mask is not None:
            if posture_grad_mask.dtype!=torch.bool or posture_grad_mask.shape!=(base.shape[0],):
                raise ValueError('Posture gradient mask must be bool[B]')
            # The reference offset learns neutral tendencies, not the actor's
            # average emotion. Forward values are identical for every label.
            offset=torch.where(posture_grad_mask[:,None],offset,offset.detach())
        return clean(base + (self.output(h)+offset[:,None])*scales, valid)


class NativeAffineDecoder(nn.Module):
    """Keep B0 on its native clock; conditions control a positive global gain.

    The additive expression network never receives B0 or content features.
    This constrains the content path, not the complete output: additive
    expression can still change closures and must be checked empirically.
    """
    def __init__(self, cfg):
        super().__init__()
        width = cfg.style_dim // 2
        self.log_gain_limit = math.log(4.)
        self.gain = nn.Sequential(nn.LayerNorm(cfg.global_dim + width),
            nn.Linear(cfg.global_dim + width, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, 52))
        self.expression = TemporalEncoder(cfg.global_dim + cfg.local_dim + width,
                                         cfg.hidden, 3, attention=0)
        self.output = nn.Linear(cfg.hidden, 52)
        self.bias = nn.Linear(width, 52)
        nn.init.zeros_(self.gain[-1].weight)
        nn.init.zeros_(self.gain[-1].bias)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.bias.bias)

    def components(self, g, u, style, valid):
        posture, response = style.chunk(2, -1)
        raw = self.gain(torch.cat((g, response), -1))
        gain = (self.log_gain_limit * (raw / self.log_gain_limit).tanh()).exp()
        conditions = clean(torch.cat((g[:, None].expand(-1, u.shape[1], -1),
                            clean(u, valid), response[:, None].expand(-1, u.shape[1], -1)), -1), valid)
        expression = clean(self.output(self.expression(conditions, valid)), valid)
        return gain, expression, self.bias(posture)

    def forward(self, base, g, u, style, valid, scales, posture_grad_mask=None):
        gain, expression, offset = self.components(g, u, style, valid)
        if posture_grad_mask is not None:
            if posture_grad_mask.dtype != torch.bool or posture_grad_mask.shape != (base.shape[0],):
                raise ValueError('Posture gradient mask must be bool[B]')
            offset = torch.where(posture_grad_mask[:, None], offset, offset.detach())
        return clean(clean(base.detach(), valid) * gain[:, None] +
                     (expression + offset[:, None]) * scales, valid)


class ExpressionResponse(nn.Module):
    def __init__(self, cfg: ResponseConfig, feature_mean, feature_std, scales):
        super().__init__()
        if cfg.stride < 1 or cfg.hidden % 4 or cfg.logvar_min >= cfg.logvar_max:
            raise ValueError('Invalid response architecture')
        if cfg.reference_encoder not in ('temporal','statistics') or cfg.style_dim%2:
            raise ValueError('Invalid reference encoder configuration')
        if cfg.style_modulation not in ('joint','factorized'):
            raise ValueError('Invalid style modulation')
        if cfg.style_modulation=='factorized' and cfg.reference_encoder!='statistics':
            raise ValueError('Factorized modulation requires statistical references')
        if cfg.response_head not in ('residual', 'native_affine'):
            raise ValueError('Invalid response head')
        if cfg.response_head == 'native_affine' and cfg.reference_encoder != 'statistics':
            raise ValueError('Native affine response requires statistical references')
        if cfg.prior_variance not in ('learned', 'unit'):
            raise ValueError('prior_variance must be learned or unit')
        if cfg.prior_variance=='unit' and not cfg.logvar_min<=0<=cfg.logvar_max:
            raise ValueError('Unit prior variance requires logvar bounds containing zero')
        self.cfg = cfg
        mean, std = torch.as_tensor(feature_mean).float(), torch.as_tensor(feature_std).float()
        if mean.shape != (772,) or std.shape != (772,) or (std <= 0).any():
            raise ValueError('TRAIN affect/prosody normalization must have exactly 772 dimensions')
        scales = torch.as_tensor(scales).float()
        if scales.shape != (52,) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('Finite positive TRAIN motion scales required')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError('Nonfinite feature normalization')
        self.register_buffer('feature_mean', mean)
        self.register_buffer('feature_std', std)
        self.register_buffer('scales', scales)
        self.prior = GaussianEncoder(cfg, 772, 4, fixed_variance=cfg.prior_variance=='unit')
        self.posterior = GaussianEncoder(cfg, 52*4+cfg.style_dim, 3)
        self.style = StatisticalReferenceStyle(cfg) if cfg.reference_encoder=='statistics' else ReferenceStyle(cfg)
        self.decoder = NativeAffineDecoder(cfg) if cfg.response_head == 'native_affine' else ResponseDecoder(cfg)
        self.emotion_head = nn.Linear(cfg.global_dim, cfg.classes)
        self.intensity_head = nn.Linear(cfg.global_dim, cfg.intensities)

    def audio_prior(self, audio, valid):
        if audio.shape[-1] == 1540:
            audio = audio[...,768:]  # Select BEFORE normalization or finite checks.
        if audio.shape[-1] != 772:
            raise ValueError('Only emotion2vec768 + prosody4 enter the prior')
        check_sequence(audio, valid)
        return self.prior(clean((clean(audio, valid)-self.feature_mean)/self.feature_std, valid), valid)

    def encode_style(self, refs):
        return self.style(refs['motion'],refs['b0'],refs['valid'],refs['channel_mask'],self.scales)

    def motion_posterior(self, motion, base, style, valid, channels, times):
        observed = valid[...,None] & channels[:,None]
        residual = torch.where(observed, motion-base.detach(), 0.)/self.scales
        c = clean(base.detach(),valid)/self.scales
        adjacent = valid[:,1:] & valid[:,:-1] & torch.isclose(
            times[:,1:]-times[:,:-1], torch.full_like(times[:,1:],.04), rtol=1e-4, atol=1e-7)
        delta = F.pad(clean(residual[:,1:]-residual[:,:-1],adjacent),(0,0,1,0))
        z = torch.cat((residual,c,delta,channels[:,None].expand_as(c),
                       style.detach()[:,None].expand(-1,motion.shape[1],-1)), -1)
        return self.posterior(z,valid)

    def conditions(self, distribution, valid, sample=False, generator=None):
        def latent(key):
            mu = distribution[key+'_mean']
            if not sample:return mu
            eps = torch.randn(mu.shape,device=mu.device,dtype=mu.dtype,generator=generator)
            return mu + (.5*distribution[key+'_logvar']).exp()*eps
        g = latent('g')
        u = native_interpolate(latent('u'),distribution['u_mask'],valid,self.cfg.stride)
        if self.cfg.center_local:
            u = clean(u-masked_pool(u,valid)[:,None],valid)
        return g,u

    def decode(self, base, distribution, style, valid, sample=False, generator=None,posture_grad_mask=None):
        g,u = self.conditions(distribution,valid,sample,generator)
        return self.decoder(base,g,u,style,valid,self.scales,posture_grad_mask)

    def predict(self, audio, base, valid, refs, *, sample=False, generator=None):
        """Deployment API deliberately has no query-motion/labels input."""
        prior = self.audio_prior(audio,valid)
        style = self.encode_style(refs)['code']
        return self.decode(base,prior,style,valid,sample,generator)

    def checkpoint_config(self):
        return asdict(self.cfg)


def normalized_kl(q,p):
    def term(key):
        qm,pm = q[key+'_mean'],p[key+'_mean']
        qv,pv = q[key+'_logvar'],p[key+'_logvar']
        return .5*(pv-qv+(qv-pv).exp()+(qm-pm).square()*(-pv).exp()-1.)
    if not torch.equal(q['u_mask'],p['u_mask']):raise ValueError('Latent clocks differ')
    g = term('g').mean(-1)
    mask = q['u_mask']
    u = (term('u').mean(-1)*mask).sum(1)/mask.sum(1).clamp_min(1)
    return (g+u).mean()/2


def motion_objective(prediction,target,valid,channels,times,scales,velocity_weight=.5):
    observed = valid[...,None] & channels[:,None]
    delta = torch.where(observed,prediction-target,0.)/scales
    position = delta.square().sum()/observed.sum().clamp_min(1)
    adjacent = valid[:,1:] & valid[:,:-1] & torch.isclose(
        times[:,1:]-times[:,:-1],torch.full_like(times[:,1:],.04),rtol=1e-4,atol=1e-7)
    adjacent = adjacent[...,None] & channels[:,None]
    error = torch.where(adjacent,delta[:,1:]-delta[:,:-1],0.)
    velocity = error.square().sum()/adjacent.sum().clamp_min(1)
    return position+velocity_weight*velocity,{'position':position,'velocity':velocity}


def semantic_objective(model,distribution,emotion,intensity,intensity_valid,weights=None):
    g=distribution['g_mean']
    e=F.cross_entropy(model.emotion_head(g),emotion,weight=weights)
    iv=intensity_valid & intensity.ge(0) & intensity.lt(model.cfg.intensities)
    i=F.cross_entropy(model.intensity_head(g[iv]),intensity[iv]) if iv.any() else g.sum()*0
    return e+i
