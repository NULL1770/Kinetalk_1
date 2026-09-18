"""Read-only exact-window diagnostics for a completed clocked residual run.

No optimizer, fitting, checkpoint/scale selection, generation, or OLA is used.
Query motion supplies scoring targets only. This separates failure already in
the trained window distribution from additional full-rollout composition loss.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_clocked_residual_prior as r
from kinetalk_b0.models.clocked_motion_prior import ClockedMotionPrior, TemporalResidualPrior, categorical_energy_score


SCHEMA = 'clocked_residual_exact_window_diagnostic_v1'
SCALES = (.25, .5, 1.)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def reconstruct(run, device):
    """Load frozen saved buffers and assert static/base equality before scoring."""
    run = Path(run)
    status, protocol = read(run/'status.json'), read(run/'protocol.json')
    if (status.get('schema') != r.SCHEMA or protocol.get('schema') != r.SCHEMA
            or status.get('status') != 'complete' or status.get('smoke') is not False
            or status.get('test_loaded') is not False):
        raise ValueError('Completed formal residual run required')
    root = Path(__file__).resolve().parents[1]
    for relative, digest in protocol['code_sha256'].items():
        if r.c.old.sha(root/relative) != digest:
            raise ValueError('Live diagnostic dependency differs from completed source: '+relative)
    static = torch.load(run/'static_final.pt', map_location='cpu', weights_only=False)
    residual = torch.load(run/'residual_final.pt', map_location='cpu', weights_only=False)
    if (static.get('schema') != r.SCHEMA or residual.get('schema') != r.SCHEMA
            or static.get('epochs') != 30 or residual.get('epochs') != 30
            or static.get('order_sha256') != residual.get('order_sha256')):
        raise ValueError('Saved completed stages/order differ')
    state, combined = static['state'], residual['state']
    for key, value in state.items():
        if not torch.equal(value, combined['base.'+key]):
            raise ValueError('Residual checkpoint changed static baseline: '+key)
    stats = [state[key] for key in ('feature_mean', 'feature_std', 'global_mean', 'global_std')]
    base = ClockedMotionPrior(*stats, horizon=r.c.HORIZON, hidden=state['input.weight'].shape[0],
                               k=state['output.weight'].shape[0])
    base.load_state_dict(state, strict=True)
    model = TemporalResidualPrior(base)
    model.load_state_dict(combined, strict=True)
    model.to(device).eval().requires_grad_(False)
    dictionary = torch.load(run/'dictionary.pt', map_location='cpu', weights_only=False)
    if (dictionary.get('coordinate_system') != 'logit raw minus observed training clip logit mean'
            or dictionary.get('horizon') != r.c.HORIZON or dictionary.get('hop') != r.c.HOP
            or len(dictionary['shapes']) != model.base.k):
        raise ValueError('Saved dictionary coordinate/clock differs')
    binding = {'files_sha256': {name: r.c.old.sha(run/name) for name in
        ('static_final.pt', 'residual_final.pt', 'dictionary.pt', 'protocol.json', 'status.json')},
        'base_unchanged_exact': True, 'normalization_source': 'saved checkpoint buffers, no refit',
        'weights_frozen': True, 'epochs': 30, 'source_code_sha256': protocol['code_sha256']}
    return model, dictionary, protocol, binding


def bootstrap(rows, arm, metric, baseline='static'):
    """Paired sentence resampling, reporting both clip- and sentence-equal gain."""
    groups = {}
    for row in rows:
        groups.setdefault(row['sentence'], []).append(row['scores'][baseline][metric]-row['scores'][arm][metric])
    keys = sorted(groups)
    values = [value for key in keys for value in groups[key]]
    result = {'gain_clip_equal': float(np.mean(values)),
              'gain_sentence_equal': float(np.mean([np.mean(groups[key]) for key in keys])),
              'sentences': len(keys), 'clips': len(rows), 'ci95_clip_equal': None, 'ci95_sentence_equal': None}
    if len(keys) >= 2:
        generator = np.random.default_rng(r.c.SEED)
        clip_boot, sentence_boot = [], []
        for _ in range(2000):
            selected = generator.integers(len(keys), size=len(keys))
            clip_boot.append(np.mean([value for j in selected for value in groups[keys[j]]]))
            sentence_boot.append(np.mean([np.mean(groups[keys[j]]) for j in selected]))
        result['ci95_clip_equal'] = np.quantile(clip_boot, [.025, .975]).tolist()
        result['ci95_sentence_equal'] = np.quantile(sentence_boot, [.025, .975]).tolist()
    return result


@torch.no_grad()
def exact_windows(clips, ids, model, dictionary, device):
    """Score only full teacher windows, keeping original static/global inputs."""
    transformed, clipping = r.coordinate_clips(clips, ids)
    windows = r.c.shapes.extract_windows(transformed, ids, horizon=r.c.HORIZON, hop=r.c.HOP)
    by_clip = {}
    for window in windows:
        by_clip.setdefault(window['clip_id'], []).append(window)
    pairwise = torch.tensor(r.c.shapes.pairwise_distances(dictionary['shapes'], dictionary['scales']),
                            dtype=torch.float64, device=device)
    rows, unsupported = [], []
    for index in ids:
        clip = clips[index]
        items = by_clip.get(clip['clip_id'], [])
        if not items:
            unsupported.append(clip['clip_id'])
            continue
        starts = [item['start'] for item in items]
        # Reverse the full observed run before slicing, as in rollout controls.
        reversed_audio = r.c.intervention_audio(clip, 'reverse')
        static_audio = r.c.static_acoustics(clip['features'], clip['valid'])
        feature_windows = {name: torch.stack([source[start:start+r.c.HORIZON] for start in starts])
                           for name, source in [('real', clip['features']), ('reverse', reversed_audio), ('static', static_audio)]}
        mask = torch.ones(len(items), r.c.HORIZON, dtype=torch.bool, device=device)
        global_ = torch.as_tensor(clip['global'], dtype=torch.float32, device=device)[None].expand(len(items), -1)
        targets = np.stack([item['future'] for item in items])
        distances = torch.tensor(r.c.shapes.target_distances(targets, dictionary['shapes'], dictionary['scales']),
                                  dtype=torch.float64, device=device)
        nearest = distances.argmin(-1)
        scores = {'static': {'energy_sum': 0., 'nearest_nll_sum': 0., 'nearest_mass_sum': 0.}}
        for intervention in ('real', 'reverse'):
            for scale in SCALES:
                scores[f'{intervention}_{scale:g}'] = {'energy_sum': 0., 'nearest_nll_sum': 0., 'nearest_mass_sum': 0.}
        null_error = 0.
        for start in range(0, len(items), 128):
            end = min(start+128, len(items)); valid = mask[start:end]
            static = feature_windows['static'][start:end].to(device)
            g = global_[start:end]
            baseline = model.base(static, valid, g)
            null_error = max(null_error, float(model.residual_logits(static, valid, g, static).abs().max()))
            logits = {'static': baseline}
            for intervention in ('real', 'reverse'):
                audio = feature_windows[intervention][start:end].to(device)
                adjustment = model.residual_logits(audio, valid, g, static)
                for scale in SCALES:
                    logits[f'{intervention}_{scale:g}'] = baseline+scale*adjustment
            for name, value in logits.items():
                log_probability = value.double().log_softmax(-1)
                probability = log_probability.exp()
                score = categorical_energy_score(probability, distances[start:end], pairwise)
                ids_tensor = torch.arange(end-start, device=device)
                chosen_log_probability = log_probability[ids_tensor, nearest[start:end]]
                scores[name]['energy_sum'] += float(score.sum())
                scores[name]['nearest_nll_sum'] += float((-chosen_log_probability).sum())
                scores[name]['nearest_mass_sum'] += float(chosen_log_probability.exp().sum())
        row = {key: clip[key] for key in ('clip_id', 'speaker', 'sentence', 'emotion')}
        row.update(windows=len(items), starts=starts, static_null_max_abs=null_error,
                   scores={name: {'energy_score': value['energy_sum']/len(items),
                                  'nearest_token_nll': value['nearest_nll_sum']/len(items),
                                  'nearest_token_mass': value['nearest_mass_sum']/len(items)}
                           for name, value in scores.items()})
        rows.append(row)
    if not rows:
        raise ValueError('No complete diagnostic windows')
    arms = list(rows[0]['scores'])
    summary = {}
    for arm in arms:
        summary[arm] = {}
        for metric in ('energy_score', 'nearest_token_nll', 'nearest_token_mass'):
            grouped = {}
            for row in rows:
                grouped.setdefault(row['sentence'], []).append(row['scores'][arm][metric])
            summary[arm][metric] = {'clip_equal': float(np.mean([row['scores'][arm][metric] for row in rows])),
                                   'sentence_equal': float(np.mean([np.mean(value) for value in grouped.values()])),
                                   'window_equal': float(sum(row['windows']*row['scores'][arm][metric] for row in rows)/sum(row['windows'] for row in rows))}
    comparisons = {arm: {metric: bootstrap(rows, arm, metric) for metric in ('energy_score', 'nearest_token_nll')}
                   for arm in arms if arm != 'static'}
    timing = {f'{scale:g}': bootstrap(rows, f'real_{scale:g}', 'energy_score', baseline=f'reverse_{scale:g}') for scale in SCALES}
    return {'rows': rows, 'summary': summary, 'static_minus_arm_gains': comparisons,
            'reverse_minus_real_energy_gains': timing, 'unsupported_clip_ids': unsupported,
            'clips': len(rows), 'windows': sum(row['windows'] for row in rows),
            'sentences': len({row['sentence'] for row in rows}), 'source_clipping': clipping,
            'static_null_max_abs': max(row['static_null_max_abs'] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio', 'targets', 'native-root', 'native-manifest', 'delta-dir', 'audio-checkpoint', 'run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic directory required')
    args.output.mkdir(parents=True); started = time.monotonic()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    model, dictionary, protocol, binding = reconstruct(args.run, args.device)
    loading = copy.copy(args); loading.smoke = False
    clips, original, lineage = r.c.old.load_clips(loading)
    if lineage != protocol['source']:
        raise ValueError('Live dataset lineage differs from completed run')
    split = r.c.previous.split_inner(clips, original)
    for cell in ('fit', 'calibration', 'confirmation'):
        actual = [{key: clips[i][key] for key in ('clip_id', 'sentence', 'speaker', 'emotion')} for i in split[cell]]
        if actual != protocol['split'][cell]:
            raise ValueError('Diagnostic membership differs: '+cell)
    diagnostic_ids = {'fit_first64': sorted(split['fit'], key=lambda i: clips[i]['clip_id'])[:64],
                      'calibration': split['calibration']}
    selected = sorted(set(i for ids in diagnostic_ids.values() for i in ids))
    target = torch.load(args.targets, map_location='cpu', weights_only=False, mmap=True)['splits']['train']
    anchors = {cid: row[r.c.previous.CC].numpy() for cid, row in zip(target['clip_id'], target['anchors'])}
    audio = r.c.load_frozen_audio(args.audio_checkpoint, args.device)
    source = r.c.assert_source_binding(audio, lineage)
    if source != protocol['frozen_audio']:
        raise ValueError('Frozen audio source differs')
    contexts = r.c.encode_clips(audio, [clips[i] for i in selected], batch_size=16, device=args.device)
    for i, context in zip(selected, contexts):
        clips[i]['global'] = np.r_[context['global'].numpy(), context['intensity'].numpy()]
        clips[i]['anchor_upper'] = anchors[clips[i]['clip_id']]
    del audio, contexts, target
    summaries = {}
    for cell, ids in diagnostic_ids.items():
        report = exact_windows(clips, ids, model, dictionary, args.device)
        r.c.old.save_json(args.output/(cell+'.json'), report)
        summaries[cell] = {key: value for key, value in report.items() if key not in ('rows', 'source_clipping')}
        print('DIAGNOSTIC', cell, json.dumps(summaries[cell]), flush=True)
    manifest = {'schema': SCHEMA, 'seconds': time.monotonic()-started, 'binding': binding,
                'source': lineage, 'script_sha256': r.c.old.sha(__file__),
                'selected_clip_ids': {cell: [clips[i]['clip_id'] for i in ids] for cell, ids in diagnostic_ids.items()},
                'fit_selection': 'First 64 lexicographic clip IDs of original fit, metadata only',
                'normalization_refitted': False, 'dictionary_refitted': False, 'training_started': False,
                'checkpoint_selected': False, 'scale_selected': False, 'test_loaded': False,
                'confirmation_encoded': False, 'confirmation_scored': False,
                'loader_scope': 'Unchanged native loader loads historical train sources including confirmation; only selected fit/calibration are encoded/transformed/scored.',
                'target_scope': 'Observed GT logit clip mean and motion used only to form exact training-coordinate scoring targets, never model inputs.',
                'nearest_token_nll_scope': 'Auxiliary nearest-dictionary teacher-token proxy, not continuous trajectory likelihood; ES is primary.',
                'interpretation': 'If exact window generalization fails, OLA alone cannot explain the failure. Window gains do not prove complete-rollout or visual quality.',
                'scores': summaries, 'reports_sha256': {cell: r.c.old.sha(args.output/(cell+'.json')) for cell in diagnostic_ids}}
    r.c.old.save_json(args.output/'summary.json', manifest)
    r.c.old.save_json(args.output/'complete.json', {'status': 'complete', 'summary_sha256': r.c.old.sha(args.output/'summary.json'),
                                                  'seconds': manifest['seconds'], 'training_started': False})


if __name__ == '__main__':
    main()
