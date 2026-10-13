"""Within-person reference-context regression for a diagnostic native code.

Subtracts a learned reference nuisance effect in latent space, never raw
unmatched motion. It is a calibration ablation, not a novel style definition
or a guarantee that the linear nuisance model suffices.
"""
import torch
from torch import nn
from kinetalk_b0.models.expression_response import masked_pool
from kinetalk_v5.factor_rebuild import V5FactorRebuild


class ReferenceCalibration(nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.register_buffer('feature_mean',torch.zeros(features))
        self.register_buffer('feature_scale',torch.ones(features))
        self.register_buffer('coefficient',torch.zeros(features,width))
        self.register_buffer('strength',torch.tensor(0.))

    @torch.no_grad()
    def fit(self, features, codes, people, ridge=10.):
        """TRAIN references only; person means are removed before fitting."""
        if ridge<=0: raise ValueError('ridge must be positive')
        if len(features)!=len(codes) or len(features)!=len(people):
            raise ValueError('calibration shapes disagree')
        self.feature_mean.copy_(features.mean(0))
        self.feature_scale.copy_(features.std(0,correction=0).clamp_min(.05))
        x=(features-self.feature_mean)/self.feature_scale
        xc=torch.zeros_like(x);yc=torch.zeros_like(codes);use=torch.zeros_like(people,dtype=torch.bool)
        for person in people.unique():
            mask=people.eq(person)
            if int(mask.sum())<2: continue
            xc[mask]=x[mask]-x[mask].mean(0)
            yc[mask]=codes[mask]-codes[mask].mean(0)
            use|=mask
        if not use.any(): raise ValueError('multiple TRAIN references per person required')
        # Double precision solve, frozen buffers thereafter. Missing identity
        # diversity is not inferred or imputed from HDTF emotion labels.
        a,b=xc[use].double(),yc[use].double()
        gram=a.T@a+ridge*torch.eye(a.shape[-1],device=a.device,dtype=a.dtype)
        self.coefficient.copy_(torch.linalg.solve(gram,a.T@b).to(self.coefficient))

    def forward(self, codes, features):
        nuisance=(features-self.feature_mean)/self.feature_scale@self.coefficient
        return codes-self.strength*nuisance


class V5CalibratedStyle(V5FactorRebuild):
    def __init__(self, original):
        super().__init__(original.core,original.rebuild_config)
        self.load_state_dict(original.state_dict(),strict=True)
        self.reference_calibration=ReferenceCalibration(self.cfg.content_dim+self.cfg.classes+1,
                                                       self.cfg.style_dim)

    def reference_features(self, refs):
        affect=self.affect_audio(refs['audio_features'],refs['valid'],anchor=True)
        probs=affect['emotion_logits'].softmax(-1)
        logits=affect['intensity_logits']; levels=torch.linspace(.5,1.5,logits.shape[-1],device=logits.device)
        intensity=(logits.softmax(-1)*levels).sum(-1,keepdim=True)
        return torch.cat((masked_pool(refs['h0'],refs['valid']),probs,intensity),-1)

    def encode_style(self, refs, contextual=True):
        codes=super().encode_style(refs,contextual=contextual)
        if not contextual: return codes
        return self.reference_calibration(codes,self.reference_features(refs))
