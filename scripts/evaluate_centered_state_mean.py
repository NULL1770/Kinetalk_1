"""Development follow-up: existing audio state supplies only expression mean.

No retraining, fitted gains, query targets, or per-clip branch selection.
The fixed four-state mean is lifted around independent neutral references.
Compare aligned and white-flow dynamics with the same new static prediction.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_temporal_repair as r
from scripts.audit_temporal_repair import summarize, metadata_equal
from scripts.train_formal_predictable_projection import save_json, save_checkpoint
from kinetalk_b0.models.slow_state_affect import lift_slow_state
from kinetalk_b0.models.mean_preserving_upper import compose_mean_preserving_upper


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'audio', 'targets', 'enrollment', 'native-root', 'trained-run', 'centered-run', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Fresh state-mean output required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    data = r.load_training_inputs(a.source_run, a.audio, a.targets, a.enrollment, a.native_root)
    ck, bindings, steps, stride = r.matching_checkpoints(a.trained_run, data)
    system = data['system'].to(a.device).eval().requires_grad_(False)
    system.load_state_dict(ck['teacher']['system'])
    audio = r._make_audio(ck['audio']['audio'], stride, a.device)
    r.cache_current_base(system, data, a.device, a.batch_size)
    identities = r.identity_cache(system, data, a.device)
    path = a.centered_run/'white/curves.pt'
    complete = json.loads((a.centered_run/'white/complete.json').read_text())
    recipe = json.loads((a.centered_run/'white/provenance.json').read_text())['recipe']
    if r.sha(path) != complete['curves_sha256'] or recipe['source_bindings'] != bindings:
        raise ValueError('Centered branch binding differs')
    curves = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    q = data['splits']['validation']
    if q['clip_id'] != curves['clip_id']:
        raise ValueError('Development clip order differs')
    for key in ('valid', 'times', 'channel_mask'):
        if not torch.equal(q[key], curves[key]):
            raise ValueError('Native metadata mismatch: '+key)
    if not torch.equal(q['motion'], curves['target']):
        raise ValueError('Target mismatch')
    cc = list(r.UPPER_INDICES)
    if not (q['anchor_valid'][:, cc] & q['channel_mask'][:, cc]).all():
        raise ValueError('Observed independent anchors required')
    upper_means = []
    for ix in torch.arange(len(q['valid'])).split(a.batch_size):
        b = r.subset(q, ix, a.device)
        predicted = audio(b['audio_features'], b['valid'])['state']
        mean_state = torch.where(b['valid'][..., None], predicted, 0.).sum(1, keepdim=True)/b['valid'].sum(1)[:, None, None]
        mean = b['anchors'][:, None]+lift_slow_state(mean_state, data['target_scales'].to(a.device))
        upper_means.append(mean[:, 0, cc].cpu())
    upper_means = torch.cat(upper_means)
    a.output.mkdir(parents=True)
    summary = {}
    for mode, source in [('state_aligned', 'aligned_centered'), ('state_white', 'full')]:
        outputs, readouts = {}, {}
        for seed in r.SEEDS:
            baseline = curves['predictions'][f'{seed}/base'].clone()
            baseline[..., cc] = upper_means[:, None]
            pred = compose_mean_preserving_upper(baseline, curves['predictions'][f'{seed}/{source}'][..., cc], q['valid'])
            if not torch.equal(pred[..., list(r.NOT_UPPER)], curves['predictions'][f'{seed}/base'][..., list(r.NOT_UPPER)]):
                raise RuntimeError('Nonupper changed')
            logits = []
            for ix in torch.arange(len(q['valid'])).split(a.batch_size):
                b = r.subset(q, ix, a.device); ident = r.batch_identity(identities, b)
                residual = pred[ix].to(a.device)-curves['b0'][ix].to(a.device)-ident['baseline'][:, None]
                teacher = system.encode_motion(torch.where(r.obs(b), residual, 0.), b['valid'])
                logits.extend((teacher['emotion_logits'].argmax(-1) == b['emotion_id']).cpu().tolist())
            outputs[f'{seed}/full'] = pred
            readouts[str(seed)] = sum(logits)/len(logits)
        saved = {**curves, 'schema': 'fixed_audio_state_mean_v1', 'predictions': outputs}
        summary[mode] = {'metrics': summarize(saved, q['emotion_id']), 'generated_emotion_accuracy_nonindependent': readouts}
        save_checkpoint(a.output/(mode+'_curves.pt'), saved)
    report = {'schema': 'fixed_audio_state_mean_v1', 'source_bindings': bindings, 'source_curve_sha256': r.sha(path),
        'development_selected_design': True, 'test_loaded': False, 'trained_new_parameters': False,
        'static_definition': 'Independent neutral anchor plus lift of valid-window mean frozen audio four-state prediction',
        'no_gain_or_timing_fit': True, 'nonupper_exact': True, 'results': summary, 'source_sha256': r.sha(__file__)}
    save_json(a.output/'evaluation.json', report)
    save_json(a.output/'complete.json', {key: r.sha(a.output/(key+'_curves.pt')) for key in summary})
    print('STATE_MEAN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
