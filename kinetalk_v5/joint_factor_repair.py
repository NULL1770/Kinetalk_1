"""Strict assembly of independently tested mean and reference corrections."""
import copy
import torch
from kinetalk_v5.affect_trend import V5AffectTrend,TrendConfig
from kinetalk_v5.factor_rebuild import RebuildConfig,load_core


class V5JointFactorRepair(V5AffectTrend):
    def __init__(self, core, rebuild_config, trend_config, style_weight=.5):
        super().__init__(core,rebuild_config,trend_config)
        self.style_anchor=copy.deepcopy(self.reference_context).eval().requires_grad_(False)
        self.register_buffer('reference_blend_weight',torch.tensor(float(style_weight)))

    @classmethod
    def from_parts(cls, trend_checkpoint, style_checkpoint, style_weight=.5):
        if not 0.<=style_weight<=1.:raise ValueError('style weight must be in [0,1]')
        trend=V5AffectTrend.from_checkpoint(trend_checkpoint)
        # A reference-only repair is compatible only when every other saved
        # tensor has the same value. Never silently combine different cores.
        for key,value in style_checkpoint['model'].items():
            if key.startswith('reference_context.'):continue
            if key not in trend_checkpoint['model'] or not torch.equal(value,trend_checkpoint['model'][key]):
                raise ValueError('incompatible non-reference state: '+key)
        model=cls(trend.core,trend.rebuild_config,trend.trend_config,style_weight)
        missing,unexpected=model.load_state_dict(trend_checkpoint['model'],strict=False)
        expected={'style_anchor.'+k for k in model.style_anchor.state_dict()}|{'reference_blend_weight'}
        if set(missing)!=expected or unexpected:raise RuntimeError('unrecognized joint migration')
        model.style_anchor.load_state_dict(trend.reference_context.state_dict(),strict=True)
        model.reference_context.load_state_dict({k[len('reference_context.'):]:v for k,v in
                   style_checkpoint['model'].items() if k.startswith('reference_context.')},strict=True)
        return model

    @classmethod
    def from_checkpoint(cls, ck):
        if ck.get('schema')!='v5_joint_factor_repair_v1':raise ValueError('joint schema required')
        core=load_core({'config':ck['config'],'model':{k[5:]:v for k,v in ck['model'].items() if k.startswith('core.')}})
        model=cls(core,RebuildConfig(**ck['rebuild_config']),TrendConfig(**ck['trend_config']))
        model.load_state_dict(ck['model'],strict=True)
        return model

    def checkpoint(self):
        return {**super().checkpoint(),'schema':'v5_joint_factor_repair_v1'}

    def encode_style(self, refs, contextual=True):
        current=super().encode_style(refs,contextual=contextual)
        if not contextual or not self.cfg.use_style:return current
        affect=self.affect_audio(refs['audio_features'],refs['valid'],anchor=True)
        anchor=self.style_anchor(refs['motion'],refs['valid'],refs['channel_mask'],refs['h0'],affect,self.core.scales)
        baseline=self.core.encode_style(refs)+anchor
        return baseline+self.reference_blend_weight*(current-baseline)
