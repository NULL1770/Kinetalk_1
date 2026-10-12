"""Frozen diagnostics, including explicitly labelled motion/label oracles.

No fitting, checkpoint writes, or sealed/test reads. A fixed balanced validation
subset and fixed display clips are evaluated under every declared condition.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v4.unified_model_v4 import KineTalkV4Config, KineTalkV4Model, affine_geometry
from kinetalk_b0.emotion_probe import MotionEmotionProbe, motion_features, classification_metrics
from scripts.train_expression_response import configure, load_runtime, sha, state_digest, write
from scripts.train_unified_v4 import select_balanced, build_base_cache, attach_base, motion_targets
from scripts.evaluate_expression_response import metrics, aggregate, finite


@torch.no_grad()
def run(args):
    configure(args.seed)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Choose a fresh output')
    out.mkdir(parents=True)
    binding = json.loads(Path(args.binding).read_text())
    device = torch.device(args.device)
    data, base = load_runtime(binding, device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    s = saved['model']
    cfg = KineTalkV4Config(**saved['config'])
    model = KineTalkV4Model(cfg, s['feature_mean'], s['feature_std'], s['scales'],
                          s['decoder.geometry_jacobian'], s['decoder.geometry_offset'],
                          s['decoder.geometry_scale']).to(device)
    model.load_state_dict(s); model.eval()
    before, b0_before = state_digest(model), state_digest(base.stage1)
    q = data['splits']['validation']
    ids = select_balanced(q, range(len(q['_lengths'])), args.limit, args.seed)
    display = set(binding['display_clips'].values())
    ids = sorted(set(ids) | {i for i, cid in enumerate(q['clip_id']) if cid in display})
    write(out / 'selection.json', {'seed': args.seed, 'indices': ids,
          'clip_ids': [q['clip_id'][i] for i in ids], 'test_loaded': False})
    cache = build_base_cache(q, base, ids, device)
    rig = dict(np.load(binding['rig'], allow_pickle=False))
    names = data['config']['data']['emotion_classes']
    support = torch.tensor([True] * 51 + [False])
    labels = q['emotion_id'][ids].cpu()
    scores, features, state_rows = {}, {}, []
    student_labels, teacher_labels = [], []
    (out / 'render_inputs').mkdir()
    for start in range(0, len(ids), 8):
        use = ids[start:start + 8]
        b = attach_base(q.batch(torch.tensor(use), device), use, cache)
        enrolled = [data['refs'][int(q['speaker_id'][i])] for i in use]
        if any(q['clip_id'][i] in r['clip_id'] for i, r in zip(use, enrolled)):
            raise ValueError('Query/reference overlap')
        refs = {k: torch.stack([r[k][0] for r in enrolled]).to(device)
                for k in ('motion', 'valid', 'channel_mask')}
        target = motion_targets(b)
        style = model.encode_style(refs)
        student = model.audio_state(b['audio_features'], b['valid'])
        teacher = model.motion_teacher(b)
        student_labels.extend(student['emotion_logits'].argmax(-1).cpu().tolist())
        teacher_labels.extend(teacher['emotion_logits'].argmax(-1).cpu().tolist())
        emotion = F.one_hot(b['emotion_id'], cfg.classes).float()
        levels = torch.linspace(.5, 1.5, cfg.intensities, device=device)
        strength = levels[b['intensity_id'].clamp(0, cfg.intensities - 1)]
        auto_strength = model.condition(student, b['valid'])['intensity']
        strength = torch.where(b['intensity_valid'], strength, auto_strength)
        conditions = {
            'audio': model.condition(student, b['valid']),
            'audio_label_oracle': model.condition(student, b['valid'], emotion=emotion, intensity=strength),
            'teacher_motion_oracle': model.condition(teacher, b['valid']),
            'teacher_label_oracle': model.condition(teacher, b['valid'], emotion=emotion, intensity=strength),
        }
        preds = {'B0': b['b0']}
        for tag, cond in conditions.items():
            for mode in ('sample', 'modal'):
                generator = torch.Generator(device=device).manual_seed(args.seed + start)
                schedule = model.events(student['hidden'], cond['state'], style, b['valid'],
                    generator=generator, mode=mode, times=b['times'])
                decoded = model.decode(b['h0'], b['b0'], b['valid'], cond, style, schedule['events'])
                preds[tag + '_' + mode] = decoded['motion']
                if tag == 'audio':
                    preds['audio_' + mode + '_no_projection'] = decoded['proposal']
            if tag in ('audio', 'teacher_label_oracle'):
                zero = torch.zeros_like(target['events'])
                preds[tag + '_zero_event'] = model.decode(b['h0'], b['b0'], b['valid'],
                    cond, style, zero)['motion']
                preds[tag + '_observed_event_oracle'] = model.decode(b['h0'], b['b0'],
                    b['valid'], cond, style, target['events'])['motion']
        for j, i in enumerate(use):
            obs = b['valid'][j, :, None] & b['channel_mask'][j, None]
            quality = target['quality'][j].bool() & b['valid'][j, :, None]
            state_row = {'clip_id': q['clip_id'][i], 'emotion': names[int(q['emotion_id'][i])],
                         'student_state_mse': float((student['state'][j] - target['state'][j]).square()[quality].mean()),
                         'teacher_state_mse': float((teacher['state'][j] - target['state'][j]).square()[quality].mean())}
            state_rows.append(state_row)
            for name, prediction in preds.items():
                p = prediction[j].clamp(0, 1)
                row = metrics(p, b, j, rig)
                row.update(clip_id=q['clip_id'][i], emotion=names[int(q['emotion_id'][i])])
                g = affine_geometry(p, model.decoder.geometry_jacobian, model.decoder.geometry_offset)
                gt_g = affine_geometry(b['motion'][j], model.decoder.geometry_jacobian, model.decoder.geometry_offset)
                dependencies = model.decoder.geometry_jacobian.abs().gt(1e-7)
                valid_geom = (b['channel_mask'][j, None] | ~dependencies).all(-1)
                for k, coordinate in enumerate(('corner_width', 'corner_height', 'aperture', 'corner_depth')):
                    mask = b['valid'][j] & valid_geom[k]
                    row['geometry/' + coordinate + '_mae'] = float((g[mask, k] - gt_g[mask, k]).abs().mean()) if mask.any() else None
                scores.setdefault(name, []).append(row)
                features.setdefault(name, []).append(motion_features(p.cpu()[:, support], b['valid'][j].cpu()))
            if q['clip_id'][i] in display:
                length = int(q['_lengths'][i])
                modes = ['GT'] + list(preds)
                np.savez_compressed(out / 'render_inputs' / (q['clip_id'][i] + '.npz'),
                    motions=torch.stack([b['motion'][j]] + [v[j] for v in preds.values()]).cpu().numpy()[:, :length],
                    mode_names=np.asarray(modes), times=b['times'][j, :length].cpu().numpy(),
                    valid=b['valid'][j, :length].cpu().numpy(), channels=rig['channels'],
                    channel_mask=b['channel_mask'][j].cpu().numpy(), clip_id=q['clip_id'][i], noise_seed=args.seed + start)
        write(out / 'state.json', {'status': 'scoring', 'clips': start + len(use), 'total': len(ids)})
        print(json.dumps({'clips': start + len(use), 'total': len(ids)}), flush=True)
    probes = []
    for spec in binding['probes']:
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('Probe hash differs')
        ck = torch.load(spec['path'], map_location='cpu', weights_only=False)
        net = MotionEmotionProbe(ck['feature_dim'], ck['hidden'], len(names)).eval()
        net.load_state_dict(ck['model'])
        probes.append((net, ck.get('feature_mask', torch.ones(ck['feature_dim'], dtype=torch.bool))))
    results = {}
    for name, rows in scores.items():
        f = torch.stack(features[name])
        numeric_rows = [{k: v for k, v in row.items() if k not in ('clip_id', 'emotion')} for row in rows]
        results[name] = {
            'metrics': aggregate(numeric_rows),
            'by_emotion': {emotion: aggregate([r for r, raw in zip(numeric_rows, rows) if raw['emotion'] == emotion]) for emotion in names if any(raw['emotion'] == emotion for raw in rows)},
            'probes': [classification_metrics(labels, net(f[:, mask]).argmax(-1), names) for net, mask in probes],
        }
    if state_digest(model) != before or state_digest(base.stage1) != b0_before:
        raise RuntimeError('Frozen diagnostics modified weights')
    report = dict(schema='v4_frozen_diagnostics_v1', clips=len(ids), checkpoint_sha256=sha(args.checkpoint),
                  binding_sha256=sha(args.binding), source_sha256=sha(__file__), test_loaded=False,
                  selection_policy='fixed balanced by speaker/emotion plus predeclared display clips',
                  seed=args.seed, results=results,
                  student_audio_classification=classification_metrics(labels, torch.tensor(student_labels), names),
                  teacher_motion_classification_oracle=classification_metrics(labels, torch.tensor(teacher_labels), names),
                  state_readout=state_rows,
                  limits=['All *_oracle conditions use query labels or motion and cannot be deployment rows.',
                          'No-projection inference is not a trained architectural ablation.',
                          'Shared-parser observed events and state anchors are weak labels.',
                          'Subset results are diagnostic; full development gates still required.'])
    write(out / 'report.json', finite(report))
    write(out / 'per_clip.json', finite(scores))
    write(out / 'complete.json', {'status': 'complete', 'clips': len(ids), 'report_sha256': sha(out / 'report.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binding', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--limit', type=int, default=128)
    p.add_argument('--seed', type=int, default=47); p.add_argument('--device', default='cuda')
    run(p.parse_args())
