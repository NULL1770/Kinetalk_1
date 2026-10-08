"""Fit-only frozen native-coordinate probes; no motion loss or validation fitting.

The hidden probe is exactly a possible recalibration of the existing mean heads.
Audio/B0 probes and input-zeroing diagnose association/sensitivity, not causality.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import (ExpressionResponse, ResponseConfig,
    clean, masked_pool, stride_pool, native_interpolate)
from scripts.train_expression_response import (configure, load_runtime, cache_base,
    development_fold, reference_batch, state_digest, sha, write)


class ClipRidge:
    """Weighted normal equations: every clip has weight one regardless of length."""
    def __init__(self, dim, targets, device='cpu', intercept=False):
        self.dim, self.intercept = dim, intercept
        n = dim + int(intercept)
        self.xx = torch.zeros(n, n, device=device, dtype=torch.float64)
        self.xy = torch.zeros(n, targets, device=device, dtype=torch.float64)
        self.yy = torch.zeros(targets, device=device, dtype=torch.float64)
        self.ys = self.yy.clone()
        self.count = 0

    def add(self, x, y, mask):
        if x.shape[:2] != mask.shape or y.shape[:2] != mask.shape or mask.dtype != torch.bool:
            raise ValueError('Native probe clocks/masks differ')
        if not mask.any(1).all():
            raise ValueError('Empty probe clip')
        x, y = clean(x, mask).double(), clean(y, mask).double()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Nonfinite observed probe features')
        if self.intercept:
            x = torch.cat((x, mask[..., None].to(x)), -1)
        # sqrt clip weights make flattened matrix multiplication equivalent.
        w = mask.to(x).div(mask.sum(1, keepdim=True)).sqrt()[..., None]
        a, b = (x*w).flatten(0, 1), (y*w).flatten(0, 1)
        self.xx += a.T @ a
        self.xy += a.T @ b
        self.yy += b.square().sum(0)
        self.ys += (y*w.square()).sum((0, 1))
        self.count += len(x)

    def solve(self, strength=1e-3):
        if self.count < 1 or strength <= 0:
            raise ValueError('Positive regularizer and nonempty FIT required')
        xx, xy = self.xx/self.count, self.xy/self.count
        scale = xx.diag().clamp_min(1e-8).sqrt()
        if self.intercept:
            scale[-1] = 1
        z = xx/(scale[:, None]*scale[None])
        penalty = torch.eye(len(scale), device=z.device, dtype=z.dtype)*strength
        if self.intercept:
            penalty[-1, -1] = 0
        coefficient = torch.linalg.solve(z+penalty, xy/scale[:, None])/scale[:, None]
        normal = self.yy/self.count
        if self.intercept:
            normal = normal-(self.ys/self.count).square()
        return coefficient.float(), normal.clamp_min(1e-8).float()

    @staticmethod
    def predict(x, coefficient, intercept=False):
        if intercept:
            return x @ coefficient[:-1] + coefficient[-1]
        return x @ coefficient


def encode_inputs(model, b, style, zero=''):
    """Same posterior input construction; intervention leaves other blocks intact."""
    observed = b['valid'][..., None] & b['channel_mask'][:, None]
    residual = torch.where(observed, b['motion']-b['b0'], 0.)/model.scales
    base = clean(b['b0'], b['valid'])/model.scales
    adjacent = b['valid'][:, 1:] & b['valid'][:, :-1] & torch.isclose(
        b['times'][:, 1:]-b['times'][:, :-1], torch.full_like(b['times'][:, 1:], .04),
        rtol=1e-4, atol=1e-7)
    delta = F.pad(clean(residual[:, 1:]-residual[:, :-1], adjacent), (0, 0, 1, 0))
    if zero == 'base':
        base = torch.zeros_like(base)
    elif zero == 'residual':
        residual, delta = torch.zeros_like(residual), torch.zeros_like(delta)
    elif zero == 'delta':
        delta = torch.zeros_like(delta)
    elif zero:
        raise ValueError('Unknown sensitivity intervention')
    return torch.cat((residual, base, delta, b['channel_mask'][:, None].expand_as(base),
                      style[:, None].expand(-1, base.shape[1], -1)), -1)


def frame_features(model, b, h):
    tokens, mask = stride_pool(h, b['valid'], model.cfg.stride)
    hidden = native_interpolate(tokens, mask, b['valid'], model.cfg.stride)
    audio = (clean(b['audio_features'][..., 768:], b['valid'])-model.feature_mean)/model.feature_std
    base = b['b0']/model.scales
    if model.cfg.center_local:
        return {k:clean(v-masked_pool(v, b['valid'])[:, None], b['valid'])
                for k, v in [('hidden', hidden), ('audio', audio), ('base', base)]}
    return {k:clean(v, b['valid']) for k, v in [('hidden', hidden), ('audio', audio), ('base', base)]}


def score(pred, target, mask, normalizer):
    return (torch.where(mask[..., None], (pred-target).square(), 0.).sum(1)
            /mask.sum(1)[:, None].clamp_min(1)/normalizer).mean(-1).cpu().tolist()


@torch.no_grad()
def run(a):
    configure(47)
    device = torch.device(a.device)
    out = Path(a.output)
    if (out/'report.json').exists():
        raise FileExistsError('Completed diagnostic already exists')
    out.mkdir(parents=True, exist_ok=True)
    binding = json.loads(Path(a.binding).read_text())
    if 'parent_checkpoint' in binding:
        assert sha(a.checkpoint) == binding['parent_checkpoint']['sha256']
    source_root = Path(__file__).resolve().parents[1]
    for name, digest in binding.get('source_files', {}).items():
        assert sha(source_root/name) == digest, name
    data, base = load_runtime(binding, device)
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    cfg = ResponseConfig(**ck['config'])
    if not cfg.center_local:
        raise ValueError('This audit is preregistered for centered Phase43-A')
    model = ExpressionResponse(cfg, ck['model']['feature_mean'], ck['model']['feature_std'], ck['model']['scales']).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval().requires_grad_(False)
    before, frozen_base = state_digest(model), state_digest(base.stage1)
    fold = development_fold(data['splits']['train'], data['fit_sids'])
    if a.smoke:
        fold = {k:v[:min(16, len(v))] if k in ('train','speaker_dev','sentence_dev') else v for k,v in fold.items()}
    ids = [i for role in ('train','speaker_dev','sentence_dev') for i in fold[role]]
    cache_base(data, base, device, {'train': ids, 'validation': []})
    train = data['splits']['train']
    capture = {}
    hook = model.prior.encoder.register_forward_hook(lambda m, inp, h: capture.update(h=h))
    probes = {k:ClipRidge(dim, cfg.local_dim, device) for k,dim in [('hidden',cfg.hidden),('audio',772),('base',52)]}
    global_probe = ClipRidge(cfg.hidden, cfg.global_dim, device, intercept=True)
    weights, normalizer = {}, {}
    rows = {}
    for role in ('train','speaker_dev','sentence_dev'):
        rows[role] = []
        # Two passes over FIT: first solve all probes, then independent scoring.
        passes = ('fit','score') if role == 'train' else ('score',)
        for phase in passes:
            for sub in torch.tensor(fold[role]).split(a.batch_size):
                b = train.batch(sub, device)
                s = model.encode_style(reference_batch(data,b,device))['code']
                p = model.audio_prior(b['audio_features'],b['valid'])
                h = capture['h']
                q = model.motion_posterior(b['motion'],b['b0'],s,b['valid'],b['channel_mask'],b['times'])
                qg, qu = model.conditions(q,b['valid'])
                pg, pu = model.conditions(p,b['valid'])
                features = frame_features(model,b,h)
                pooled = masked_pool(h,b['valid'])[:,None]
                # The probe must correspond to the existing deployed mean heads,
                # including native interpolation, gaps and temporal centering.
                torch.testing.assert_close(features['hidden'] @ model.prior.local_head.weight[:cfg.local_dim].T,
                                           pu, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(pooled[:,0] @ model.prior.global_head.weight[:cfg.global_dim].T
                                           + model.prior.global_head.bias[:cfg.global_dim], pg,
                                           rtol=1e-4, atol=1e-5)
                one = torch.ones(len(sub),1,dtype=torch.bool,device=device)
                if phase == 'fit':
                    for k,probe in probes.items(): probe.add(features[k],qu,b['valid'])
                    global_probe.add(pooled,qg[:,None],one)
                    continue
                values = {'student_u':score(pu,qu,b['valid'],normalizer['hidden']),
                          'student_g':score(pg[:,None],qg[:,None],one,normalizer['global'])}
                for k in probes:
                    values[k+'_u'] = score(ClipRidge.predict(features[k],weights[k]),qu,b['valid'],normalizer[k])
                values['hidden_g'] = score(ClipRidge.predict(pooled,weights['global'],True),qg[:,None],one,normalizer['global'])
                # Sensitivity on held-out clips only; avoids another full FIT q pass.
                if role != 'train':
                    for zero in ('base','residual','delta'):
                        changed = model.posterior(encode_inputs(model,b,s,zero),b['valid'])
                        zg, zu = model.conditions(changed,b['valid'])
                        values['zero_'+zero+'_u'] = score(zu,qu,b['valid'],normalizer['hidden'])
                        values['zero_'+zero+'_g'] = score(zg[:,None],qg[:,None],one,normalizer['global'])
                for j,i in enumerate(sub.tolist()):
                    rows[role].append({'index':i,'clip_id':train['clip_id'][i],
                                      'emotion_id':int(b['emotion_id'][j]),
                                      **{k:v[j] for k,v in values.items()}})
            if phase == 'fit':
                for k,probe in probes.items(): weights[k],normalizer[k] = probe.solve(a.ridge)
                weights['global'],normalizer['global'] = global_probe.solve(a.ridge)
            write(out/'state.json',{'status':'running','role':role,'phase':phase,'clips':len(fold[role]),'test_loaded':False})
            print(json.dumps({'role':role,'phase':phase,'clips':len(fold[role])}),flush=True)
    hook.remove()
    assert state_digest(model) == before and state_digest(base.stage1) == frozen_base
    classes = data['config']['data']['emotion_classes']
    average = lambda records: {k:float(np.mean([r[k] for r in records])) for k in records[0] if k not in ('index','clip_id','emotion_id')}
    report = {'schema':'frozen_native_teacher_audit_v1','checkpoint_sha256':sha(a.checkpoint),
              'binding_sha256':sha(a.binding),'source_sha256':sha(__file__),
              'data_manifest_sha256':binding['data_manifest_sha256'],
              'ridge':a.ridge,'clip_weight':'one per clip, mean over valid frames',
              'existing_mean_head_mapping_verified':True,
              'frozen_state_exact':True,'test_loaded':False,'external_validation_used':False,
              'fit_clips':len(fold['train']),'smoke':a.smoke,'results':{}}
    for role,records in rows.items():
        report['results'][role] = {'n':len(records),'normalized_latent_mse':average(records),
            'by_emotion':{name:{'n':len(group),'normalized_latent_mse':average(group)}
                for c,name in enumerate(classes) if (group:=[r for r in records if r['emotion_id']==c])}}
    report['limits'] = ['Linear input probes differ in capacity and cannot prove content leakage.',
        'Teacher depends on query GT; ridge only fits TRAIN teacher targets, not deployment motion.',
        'Zero-input sensitivity may be off-manifold; it is not causal independence.',
        'Latent error in effective coordinates is not final motion quality.']
    np.savez(out/'mean_head_fit.npz',**{k:v.cpu().numpy() for k,v in weights.items()},
             **{k+'_normalizer':v.cpu().numpy() for k,v in normalizer.items()})
    write(out/'fold.json',fold); write(out/'per_clip.json',rows); write(out/'report.json',report)
    write(out/'state.json',{'status':'complete','report_sha256':sha(out/'report.json'),
                          'fit_sha256':sha(out/'mean_head_fit.npz'),'test_loaded':False})
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--binding',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--ridge',type=float,default=1e-3)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    try:run(a)
    except Exception as exc:
        write(Path(a.output)/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
