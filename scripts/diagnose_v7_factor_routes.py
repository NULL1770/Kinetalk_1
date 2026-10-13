"""Read-only audit of saved v7 latent routes and properly cross-person swaps.

Loads fit/held_sentence/unseen_speaker only. No fitting, test data, or model
mutation. The checkpoint's exact model source can be imported from an archive.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_probe import MotionEmotionProbe, motion_features


def read_data(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: torch.from_numpy(z[k]) for k in z.files}


class AuditProbe:
    def __init__(self, item, device):
        p = torch.load(item['path'], map_location='cpu', weights_only=False)
        self.net = MotionEmotionProbe(p['feature_dim'], p['hidden'], 8).to(device).eval()
        self.net.load_state_dict(p['model'])
        self.mask = p.get('feature_mask', torch.ones(306, dtype=torch.bool)).to(device)

    def predict(self, y, valid):
        f = torch.stack([motion_features(x[:, :51].clamp(0, 1), v) for x, v in zip(y, valid)])
        return self.net(f[:, self.mask]).argmax(-1)


def cross_person_donors(sids, emotions):
    donors = []
    for i, sid in enumerate(sids):
        different = (sids != sid).nonzero().flatten()
        if not len(different):
            raise ValueError('Cross-person style audit needs multiple speakers')
        matched = different[emotions[different] == emotions[i]]
        donors.append(int((matched if len(matched) else different)[0]))
    result = torch.tensor(donors)
    assert bool((sids[result] != sids).all())
    return result


def summarize(y, target, valid):
    selected = y[valid]
    delta = (y - target)[valid]
    return {'mae': float(delta.abs().mean()), 'mouth_mae': float(delta[:, 14:40].abs().mean()),
            'mean': float(selected.mean()), 'std': float(selected.std()),
            'negative_fraction': float((selected < 0).float().mean()),
            'above_one_fraction': float((selected > 1).float().mean())}


@torch.no_grad()
def audit(model, data, probes, device):
    n = len(data['motion'])
    sids, labels = data['speaker_id'], data['emotion_id']
    donors = cross_person_donors(sids, labels)
    codes = []
    other_codes = []
    for i in range(0, n, 8):
        refs = {k[4:]: v[i:i+8].to(device) for k, v in data.items() if k.startswith('ref_')}
        codes.append(model.encode_style({k: v[:, :1] for k, v in refs.items()})['mean'])
        other_codes.append(model.encode_style({k: v[:, 1:2] for k, v in refs.items()})['mean'])
    s = torch.cat(codes); other_s = torch.cat(other_codes)
    known = labels.ge(0)
    all_outputs = {}
    latents = {}; internals = {}
    for i in range(0, n, 8):
        sl = slice(i, min(i+8, n))
        b = {k: v[sl].to(device) for k, v in data.items() if not k.startswith('ref_')}
        refs = {k[4:]: v[sl, :1].to(device) for k, v in data.items() if k.startswith('ref_')}
        pack = model(b, refs)
        c = pack['content']; e = pack['teacher']['trajectory']; v = b['valid'].bool()
        no_e = e.clone(); no_e[..., :model.cfg.emotion_dim] = 0
        no_n = e.clone(); no_n[..., model.cfg.emotion_dim:] = 0
        only_global = no_n.clone()
        vv = v[..., None].float()
        g = (e[..., :model.cfg.emotion_dim]*vv).sum(1)/vv.sum(1).clamp_min(1)
        only_global[..., :model.cfg.emotion_dim] = g[:, None]
        local = e[..., :model.cfg.emotion_dim] - g[:, None]
        variants = {
            'GT': b['motion'], 'teacher': pack['teacher_out']['motion'],
            'mean': pack['student_out']['motion'],
            'population': pack['teacher_out']['population'],
            'style_wrong': model.decode(c, e, s[donors[sl].to(device)], v)['motion'],
            'style_other_ref': model.decode(c, e, other_s[sl], v)['motion'],
            'no_E': model.decode(c, no_e, s[sl], v)['motion'],
            'no_N': model.decode(c, no_n, s[sl], v)['motion'],
            'global_E_only': model.decode(c, only_global, s[sl], v)['motion'],
        }
        for name, y in variants.items(): all_outputs.setdefault(name, []).append(y)
        for name, value in {'C':c, 'E_global':g, 'E_local':local[v],
                            'N':e[...,model.cfg.emotion_dim:][v],
                            'response':pack['teacher_out']['response'][v]}.items():
            latents.setdefault(name, []).append(value.reshape(-1).cpu())
        for name, logits in {'teacher_GT':pack['teacher']['emotion_logits'],
                              'student':pack['student']['emotion_logits'],
                              'teacher_generated':model.teacher(variants['teacher'],v)['emotion_logits']}.items():
            internals.setdefault(name, []).append(logits.argmax(-1).cpu())
    y_gt = data['motion'].to(device); valid = data['valid'].bool().to(device)
    result = {'rows':n, 'donor_speaker_disjoint':True,
              'style_same_ref_distance':float((s-other_s).square().mean(-1).sqrt().mean()),
              'style_different_person_distance':float((s-s[donors.to(device)]).square().mean(-1).sqrt().mean()),
              'style_code_rms':float(s.square().mean().sqrt()),
              'latent_rms':{k:float(torch.cat(v).square().mean().sqrt()) for k,v in latents.items()},
              'internal_accuracy':{k:float((torch.cat(v)[known]==labels[known]).float().mean()) for k,v in internals.items()},
              'outputs':{}}
    result['style_separation_ratio'] = result['style_same_ref_distance']/max(result['style_different_person_distance'],1e-8)
    for name, chunks in all_outputs.items():
        y = torch.cat(chunks)
        item = summarize(y, y_gt, valid)
        item['teacher_delta'] = float((y-torch.cat(all_outputs['teacher']))[valid].abs().mean())
        item['probes'] = []
        for p in probes:
            pred = p.predict(y, valid).cpu()[known]; truth = labels[known]
            confusion = torch.bincount(truth*8+pred,minlength=64).reshape(8,8)
            item['probes'].append({'accuracy':float((pred==truth).float().mean()),
                                    'predicted_counts':torch.bincount(pred,minlength=8).tolist(),
                                    'confusion':confusion.tolist()})
        result['outputs'][name] = item
    return result


def main(a):
    spec = importlib.util.spec_from_file_location('audit_v7_model', a.model_source)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = module.V7DisentangledModel(module.V7Config(**saved['config'])).to(a.device).eval()
    model.load_state_dict(saved['model'])
    binding = json.loads(Path(a.binding).read_text())
    probes = [AuditProbe(item, a.device) for item in binding['probes']]
    result = {'checkpoint':a.checkpoint,'source':a.model_source,'read_only':True,'test_loaded':False,
              'roles':{role:audit(model,read_data(Path(a.data)/(role+'.npz')),probes,a.device)
                       for role in ['held_sentence','unseen_speaker']}}
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({r:{'style_ratio':s['style_separation_ratio'],'internal':s['internal_accuracy'],
                         'emotion':{k:[p['accuracy'] for p in v['probes']] for k,v in s['outputs'].items()}}
                      for r,s in result['roles'].items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--model-source',required=True);p.add_argument('--data',required=True)
    p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');main(p.parse_args())
