"""Frozen reference-style swap audit and render-input export.

The query audio/B0/prior is fixed. Only the independent neutral reference
speaker and reference slot are changed. This is a diagnostic: it measures
whether the existing style path is active and stable; it does not train a
speaker classifier or claim facial-identity transfer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.static_expression import StaticExpressionCorrection
from scripts.train_expression_response import configure, load_runtime, cache_base, sha, state_digest
from scripts.evaluate_expression_response import metrics
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES
from scripts.refine_expression_prior import restore_model


def write(path: Path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def reference_for(data, speaker_id: int, device, slot: int | None = None):
    ref = data['refs'][int(speaker_id)]
    count = len(ref['valid'])
    if slot is None:
        indices = list(range(count))
    else:
        if slot < 0 or slot >= count:
            raise ValueError(f'Invalid reference slot {slot} for speaker {speaker_id}')
        indices = [slot]
    values = {k: ref[k][indices].to(device) for k in ('motion', 'b0', 'valid', 'channel_mask')}
    width = int((values['valid'] * torch.arange(1, values['valid'].shape[-1] + 1, device=device)).amax())
    for key in ('motion', 'b0', 'valid'):
        values[key] = values[key][:, :width]
    return {k: v.unsqueeze(0) for k, v in values.items()}


def expand_reference(ref, batch_size: int):
    return {k: v.expand(batch_size, *v.shape[1:]) for k, v in ref.items()}


def latent_apply(model, correction, prediction, prior, style, valid):
    features = torch.cat((prior['g_mean'].detach(), style.detach()), -1)
    return correction(prediction, features, valid, model.scales)


def cosine(a, b):
    return float(F.cosine_similarity(a, b, dim=-1).mean().detach().cpu())


@torch.no_grad()
def run(args):
    configure(47)
    device = torch.device(args.device)
    binding = json.loads(Path(args.binding).read_text(encoding='utf-8-sig'))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Fresh audit output required')
    output.mkdir(parents=True)
    parent, parent_ck = restore_model(binding, device)
    parent.eval().requires_grad_(False)
    parent_digest = state_digest(parent)
    if args.correction:
        correction_ck = torch.load(args.correction, map_location=device, weights_only=False)
        assert correction_ck['mode'] == 'latent'
        assert correction_ck['parent_checkpoint_sha256'] == binding['parent_checkpoint']['sha256']
        correction = StaticExpressionCorrection('latent', **correction_ck['state']).to(device).eval()
        correction.requires_grad_(False)
    else:
        correction = None
    data, base = load_runtime(binding, device)
    q = data['splits']['validation']
    cache_base(data, base, device, {'train': [], 'validation': list(range(len(q['valid'])))})
    rig_path = Path(binding['rig'])
    assert sha(rig_path) == binding['rig_sha256']
    with np.load(rig_path, allow_pickle=False) as z:
        rig = {k: z[k].copy() for k in z.files}
    names = binding['display_clips']
    query_indices = []
    for emotion, clip_id in names.items():
        if clip_id not in q['clip_id']:
            raise ValueError(f'Display clip missing from validation: {clip_id}')
        query_indices.append((emotion, q['clip_id'].index(clip_id)))
    speaker_ids = sorted(set(int(x) for x in q['speaker_id']))
    if len(speaker_ids) < 3:
        raise ValueError('Style audit requires at least three validation speakers')
    source_sid = int(q['speaker_id'][query_indices[0][1]])
    donors = [sid for sid in speaker_ids if sid != source_sid]
    if len(donors) < 2:
        raise ValueError('Style audit requires two cross-speaker donors')
    donor_a, donor_b = donors[:2]
    all_ids = [source_sid, donor_a, donor_b]
    report = {
        'schema': 'phase48_reference_style_swap_audit_v1',
        'test_loaded': False, 'training_performed': False,
        'parent_checkpoint_sha256': binding['parent_checkpoint']['sha256'],
        'correction_sha256': sha(args.correction) if args.correction else None,
        'query_speaker_id': source_sid, 'donor_speaker_ids': [donor_a, donor_b],
        'reference_policy': 'two independent neutral enrollment clips; A/B are separate one-reference views',
        'fixed_query': 'same validation audio, sentence, emotion, intensity and frozen B0/prior for every reference',
        'content_alignment_claim': 'none; cross-speaker GT is used only for descriptive matched statistics where available',
        'results': [], 'style_code': [], 'render_inputs': [],
    }
    arrays = {}
    for emotion, index in query_indices:
        b = q.batch(torch.tensor([index]), device)
        p = parent.audio_prior(b['audio_features'], b['valid'])
        target = b['motion'][0].detach().cpu()
        valid = b['valid'][0].detach().cpu()
        times = b['times'][0].detach().cpu()
        channel_mask = b['channel_mask'][0].detach().cpu()
        refs = {}
        for label, sid, slot in (
            ('own_A', source_sid, 0), ('own_B', source_sid, 1),
            ('donor_A', donor_a, 0), ('donor_B', donor_b, 0)):
            ref = expand_reference(reference_for(data, sid, device, slot), 1)
            style = parent.encode_style(ref)['code']
            prediction = parent.decode(b['b0'], p, style, b['valid'])[0]
            if correction is not None:
                prediction = latent_apply(parent, correction, prediction[None], p, style, b['valid'])[0]
            refs[label] = {'style': style.detach().cpu(), 'motion': prediction.detach().cpu()}
            row_metrics = metrics(prediction.clamp(0, 1), b, 0, rig)
            report['results'].append({
                'emotion': emotion, 'clip_id': names[emotion], 'reference': label,
                'reference_speaker_id': sid, 'reference_slot': slot,
                'mbe': row_metrics.get('arkit_mbe'), 'lbe': row_metrics.get('arkit_lbe'),
                'lip_mean_mm': row_metrics.get('lve_mean_mm_mean'),
                'jaw_range': row_metrics.get('jawOpen/pred_q90_q10'),
                'jaw_correlation': row_metrics.get('jawOpen/centered_correlation'),
                'mouth_mean_abs': float(prediction[:, 14:41].abs().mean().cpu()),
                'brow_mean_abs': float(prediction[:, 41:46].abs().mean().cpu()),
            })
        own_delta = float((refs['own_A']['motion'][valid] - refs['own_B']['motion'][valid]).abs().mean())
        donor_delta = float((refs['donor_A']['motion'][valid] - refs['donor_B']['motion'][valid]).abs().mean())
        cross_a = float((refs['own_A']['motion'][valid] - refs['donor_A']['motion'][valid]).abs().mean())
        cross_b = float((refs['own_A']['motion'][valid, 14:41] - refs['donor_A']['motion'][valid, 14:41]).abs().mean())
        report['style_code'].append({
            'emotion': emotion, 'clip_id': names[emotion],
            'own_A_B_cosine': cosine(refs['own_A']['style'], refs['own_B']['style']),
            'own_donor_A_cosine': cosine(refs['own_A']['style'], refs['donor_A']['style']),
            'own_donor_B_cosine': cosine(refs['own_A']['style'], refs['donor_B']['style']),
            'same_speaker_output_mae': own_delta,
            'cross_speaker_output_mae': cross_a,
            'cross_speaker_mouth_mae': cross_b,
            'donor_A_B_output_mae': donor_delta,
        })
        length = int(valid.sum())
        source = {'GT': target[:length], 'own_A': refs['own_A']['motion'][:length],
                  'own_B': refs['own_B']['motion'][:length], 'donor_A': refs['donor_A']['motion'][:length],
                  'donor_B': refs['donor_B']['motion'][:length]}
        # Save one input for this correction mode. The same native clock and
        # audio are used for every reference; no retiming/filling is performed.
        mode = 'latent' if correction is not None else 'parent'
        render_path = output / 'render_inputs' / mode / f'{emotion}.npz'
        render_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(render_path, channels=np.asarray(ARKIT_NAMES), times=times[:length].numpy(),
                            valid=np.ones(length, dtype=bool), channel_mask=channel_mask.numpy(),
                            motions=np.stack(list(source.values())), mode_names=np.asarray(list(source)),
                            clip_id=np.asarray(names[emotion]), source_speaker_id=np.asarray(source_sid),
                            donor_speaker_ids=np.asarray([donor_a, donor_b]), native_frame_count=np.asarray(length))
        report['render_inputs'].append({'mode': mode, 'emotion': emotion, 'path': str(render_path),
            'clip_id': names[emotion], 'frames': length, 'native_clock': True,
            'query_gt_used_for_deployment': False})
        arrays[f'{emotion}_{mode}'] = np.stack(list(source.values()))
    assert state_digest(parent) == parent_digest
    report['aggregate'] = {
        'same_speaker_output_mae': float(np.mean([x['same_speaker_output_mae'] for x in report['style_code'] ])),
        'cross_speaker_output_mae': float(np.mean([x['cross_speaker_output_mae'] for x in report['style_code'] ])),
        'cross_speaker_mouth_mae': float(np.mean([x['cross_speaker_mouth_mae'] for x in report['style_code'] ])),
        'same_speaker_code_cosine': float(np.mean([x['own_A_B_cosine'] for x in report['style_code'] ])),
        'cross_speaker_code_cosine': float(np.mean([x['own_donor_A_cosine'] for x in report['style_code'] ])),
    }
    np.savez_compressed(output / 'swap_curves.npz', **arrays)
    write(output / 'report.json', report)
    write(output / 'state.json', {'status': 'complete', 'report_sha256': sha(output / 'report.json'),
                                  'test_loaded': False, 'training_performed': False})
    print(json.dumps({'status': 'complete', 'output': str(output), 'mode': 'latent' if correction else 'parent'}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binding', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--correction')
    p.add_argument('--device', default='cuda')
    run(p.parse_args())
