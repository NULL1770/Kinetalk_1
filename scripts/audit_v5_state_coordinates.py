"""Frozen audit of raw versus decoder-conditioned affect coordinates.

Query motion is used only to construct diagnostic targets and the privileged
teacher oracle. No generation, fitting, or sealed-test target access occurs.
The coordinates describe ARKit residual motion, not psychological affect.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.load_candidate_model import load_candidate_model
from scripts.train_expression_response import configure, load_runtime, sha, state_digest, write
from scripts.train_unified_v4 import select_balanced, build_base_cache, attach_base, motion_targets
from scripts.evaluate_v4_dynamics import STATE_NAMES


def decomposition(prediction, target, quality):
    mask = quality.bool()
    count = mask.sum(0)
    p = torch.where(mask, prediction, 0.).sum(0) / count.clamp_min(1)
    q = torch.where(mask, target, 0.).sum(0) / count.clamp_min(1)
    bias = (p - q).square()
    error = torch.where(mask, (prediction - p - target + q).square(), 0.).sum(0) / count.clamp_min(1)
    variance = torch.where(mask, (target - q).square(), 0.).sum(0) / count.clamp_min(1)
    return dict(bias_mse=bias.cpu().tolist(), centered_mse=error.cpu().tolist(),
                target_centered_variance=variance.cpu().tolist(), valid_count=count.cpu().tolist())


@torch.no_grad()
def run(args):
    configure(47)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Fresh diagnostic output required')
    out.mkdir(parents=True)
    device = torch.device(args.device)
    binding = json.loads(Path(args.binding).read_text())
    data, base = load_runtime(binding, device)
    q = data['splits']['validation']
    ids = select_balanced(q, list(range(len(q['_lengths']))), args.limit, 47)
    cache = build_base_cache(q, base, ids, device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = load_candidate_model(saved, device).eval().requires_grad_(False)
    before = state_digest(model), state_digest(base.stage1)
    rows = []
    for start in range(0, len(ids), 8):
        use = ids[start:start+8]
        batch = attach_base(q.batch(torch.tensor(use), device), use, cache)
        target = motion_targets(batch)
        if hasattr(model.cfg, 'lowpass_kernel'):
            student = model.audio_state(batch['audio_features'], batch['valid'], times=batch['times'])
        else:
            student = model.audio_state(batch['audio_features'], batch['valid'])
        teacher = model.motion_teacher(batch)
        conditioned = {name:model.condition(value, batch['valid'])
                       for name,value in [('student',student),('teacher_oracle',teacher)]}
        for j,i in enumerate(use):
            mask = target['quality'][j].bool() & batch['valid'][j,:,None]
            row = dict(clip_id=q['clip_id'][i], emotion=int(q['emotion_id'][i]), coordinates={})
            for name,value in [('student',student),('teacher_oracle',teacher)]:
                for kind,prediction in [('raw',value['state']),('conditioned',conditioned[name]['state'])]:
                    row['coordinates'][name+'_'+kind] = decomposition(prediction[j], target['state'][j], mask)
                row[name+'_predicted_strength'] = float(conditioned[name]['intensity'][j])
            rows.append(row)
        print(json.dumps(dict(clips=len(rows), total=len(ids))), flush=True)
    summary = {}
    for name in rows[0]['coordinates']:
        summary[name] = {}
        for key in ('bias_mse', 'centered_mse', 'target_centered_variance'):
            values = np.array([r['coordinates'][name][key] for r in rows])
            supported = np.array([r['coordinates'][name]['valid_count'] for r in rows]) > 0
            count = supported.sum(0)
            summary[name][key] = ((values*supported).sum(0)/count.clip(1)).tolist()
        summary[name]['macro_mse'] = float(np.mean(np.array(summary[name]['bias_mse']) + summary[name]['centered_mse']))
    if before != (state_digest(model), state_digest(base.stage1)):
        raise RuntimeError('Frozen diagnostic modified model')
    report = dict(schema='v5_state_coordinate_diagnostic_v1', clips=len(ids), indices=ids,
        coordinate_names=STATE_NAMES, checkpoint_sha256=sha(args.checkpoint),
        binding_sha256=sha(args.binding), source_sha256=sha(__file__), test_loaded=False,
        training_protocol=saved.get('protocol', {}), summary=summary, records=rows,
        limits=['Teacher rows use query motion and are privileged diagnostic oracles.',
                'Raw states and conditioned decoder inputs are compared with the same motion-residual coordinates.',
                'Bias/centered MSE decomposition is per clip; it is not dynamic emotion accuracy.',
                'No architectural change or contribution claim is established by this audit.'])
    write(out/'report.json', report)
    write(out/'complete.json', dict(status='complete', checkpoint_sha256=sha(args.checkpoint), clips=len(ids)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True); parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, default=32)
    run(parser.parse_args())
