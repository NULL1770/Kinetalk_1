"""Frozen validation-only global/scalar intervention; never a deployable score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_probe import MotionEmotionProbe, classification_metrics, motion_features
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from scripts.audit_matched_motion_curves import MC, RC, REGIONS, means, region_values
from scripts.compare_projection_ablation import macro_f1
from scripts.diagnose_flow_sampling import clip_metrics, sha
from scripts.packed_trainval_cache import load_packed
from scripts.phase1_condition_diagnostic import _identity_cache
from scripts.train_full_staged import audio_affect, base_forward, batch_identity, teacher_affect


MODES = ('audio', 'teacher_scalar', 'label_scalar', 'teacher_global', 'teacher_both')


def factorial_conditions(audio, teacher, labels, intensity_valid):
    """Replace only renderer inputs; retain audio timing and readout logits.

    Labels lacking a valid ordinal annotation keep the audio scalar. All
    GT-assisted branches are diagnostic counterfactuals, never inference.
    """
    if labels.shape != intensity_valid.shape or labels.shape != audio['intensity_value'].shape[:1]:
        raise ValueError('Ordinal labels and valid mask must match batch')
    if intensity_valid.dtype != torch.bool:
        raise ValueError('Intensity annotation mask must be Boolean')
    annotated = torch.where(intensity_valid[:, None],
                            labels[:, None].to(audio['intensity_value']), audio['intensity_value'])
    return {
        'audio': audio,
        'teacher_scalar': {**audio, 'intensity_value': teacher['intensity_value']},
        'label_scalar': {**audio, 'intensity_value': annotated},
        'teacher_global': {**audio, 'global': teacher['global']},
        'teacher_both': {**audio, 'global': teacher['global'],
                         'intensity_value': teacher['intensity_value']},
    }


def paired_intervals(labels, predictions, metrics, regions, classes, draws=2000):
    """Fixed paired clip resampling; uncertainty only, no threshold tuning."""
    rng = np.random.default_rng(2718)
    metric_draws, f1_draws, region_draws = {}, {}, {}
    for mode in MODES[1:]:
        metric_draws[mode], f1_draws[mode], region_draws[mode] = [], [], []
    for _ in range(draws):
        ix = rng.integers(0, len(labels), len(labels))
        base_f1 = [macro_f1(labels[ix], p[ix], classes) for p in predictions['audio']]
        for mode in MODES[1:]:
            metric_draws[mode].append((metrics[mode] - metrics['audio'])[ix].mean(0))
            delta = (regions[mode] - regions['audio'])[ix]
            region_draws[mode].append(np.nanmean(delta, axis=0))
            f1_draws[mode].append([macro_f1(labels[ix], p[ix], classes) - b
                                  for p, b in zip(predictions[mode], base_f1)])
    # Interaction of the two teacher interventions is descriptive geometry,
    # not a sum of independent nonlinear F1 effects.
    interaction = metrics['teacher_both'] - metrics['teacher_global'] - metrics['teacher_scalar'] + metrics['audio']
    return {
        'policy': f'{draws} paired clip bootstraps seed2718, clip_all, validation only',
        'delta_direction': 'intervention minus audio',
        'branches': {mode: {
            'metrics_95ci': dict(zip(MC, np.quantile(metric_draws[mode], [.025, .975], axis=0).T.tolist())),
            'probe_f1_95ci': np.quantile(f1_draws[mode], [.025, .975], axis=0).T.tolist(),
            'regions_95ci': {name: dict(zip(RC, np.quantile(region_draws[mode], [.025, .975], axis=0)[:, j].T.tolist()))
                             for j, name in enumerate(REGIONS)},
        } for mode in MODES[1:]},
        'teacher_geometry_interaction_mean': dict(zip(MC, interaction.mean(0).tolist())),
    }


@torch.no_grad()
def run(a):
    if a.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(a.device)
    if sha(a.checkpoint) != a.checkpoint_sha256:
        raise ValueError('Checkpoint checksum mismatch')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    provenance = json.loads((a.run_root / 'provenance.json').read_text())
    recipe = provenance['recipe']
    assert ck['stage'] == 'audio' and ck['recipe_sha256'] == provenance['recipe_sha256']
    assert not recipe['test_loaded'] and not recipe['args'].get('independent_probe_weight', 0)
    data = load_packed(a.data, materialize=False, with_refs=True)
    manifest = data['provenance']['manifest_sha256']
    assert set(data['splits']) == {'train', 'validation'} and not data['provenance']['test_loaded']
    assert manifest == ck['data_manifest_sha256']
    system = NeutralAffectSystem(ck['config']).to(device).eval()
    system.load_state_dict(ck['system'], strict=True)
    system.requires_grad_(False)
    audio = SlowStateAffect(ck['feature_stats']['mean'], ck['feature_stats']['std'],
                            stride=recipe['args']['stride']).to(device).eval()
    audio.load_state_dict(ck['audio'], strict=True)
    audio.requires_grad_(False)
    identities = _identity_cache(system, data, device)
    names = data['config']['data']['emotion_classes']
    probes, probe_meta, support = [], [], None
    for path in a.probe:
        pc = torch.load(path, map_location='cpu', weights_only=False)
        assert pc['train_manifest_sha256'] == manifest and pc['classes'] == names
        assert not pc.get('test_used_for_selection') and not pc.get('generator_outputs_used_for_fitting')
        observed = pc['channel_support'].bool()
        if support is not None:
            assert torch.equal(support, observed)
        support = observed
        probe = MotionEmotionProbe(pc['feature_dim'], pc['hidden'], len(names)).eval()
        probe.load_state_dict(pc['model'], strict=True)
        probes.append((probe, pc.get('feature_mask', torch.ones(6 * int(support.sum()), dtype=torch.bool))))
        probe_meta.append({'sha256': sha(path), 'kind': pc['kind'], 'path': str(path)})
    q = data['splits']['validation']
    assert q['channel_mask'][:, support].all()
    ids = torch.arange(len(q['_lengths']))
    if a.limit_per_class:
        selected = []
        for cls in range(len(names)):
            eligible = (q['emotion_id'] == cls).nonzero().flatten()
            selected.extend(eligible[torch.linspace(0, len(eligible)-1,
                                                     min(len(eligible), a.limit_per_class)).long()].tolist())
        ids = torch.tensor(sorted(selected))
    stores = {mode: {policy: {'metrics': [], 'regions': [], 'features': [], 'delta_rms': []}
                    for policy in ('raw', 'clip_all')} for mode in ('GT', *MODES)}
    groups = {key: [] for key in ('labels', 'intensity', 'intensity_valid', 'speaker', 'clip_id')}
    scalars = []
    noise_gen = torch.Generator().manual_seed(42)
    for batch_no, ix in enumerate(ids.split(a.batch_size)):
        b = q.batch(ix, device)
        base = base_forward(system, b['content'], b['valid'])
        b.update(base)
        ident = batch_identity(identities, b)
        ao = audio_affect(audio, b['audio_features'], b['valid'])
        teacher = teacher_affect(system, b, ident)
        conditions = factorial_conditions(ao, teacher, b['intensity_id'], b['intensity_valid'])
        noise = torch.randn(b['motion'].shape, generator=noise_gen).to(device)
        predictions = {'GT': b['motion'].cpu()}
        for mode, condition in conditions.items():
            predictions[mode] = system.generate(b['content'], b['valid'], ident, condition,
                                                initial_noise=noise, steps=12, base=base)['raw_motion'].cpu()
        for output, key in [('labels', 'emotion_id'), ('intensity', 'intensity_id'),
                            ('intensity_valid', 'intensity_valid'), ('speaker', 'speaker_id')]:
            groups[output].extend(b[key].cpu().tolist())
        groups['clip_id'].extend(b['clip_id'])
        scalars.extend(torch.cat([ao['intensity_value'], teacher['intensity_value']], -1).cpu().tolist())
        for j in range(len(ix)):
            valid, channel = b['valid'][j].cpu(), b['channel_mask'][j].cpu()
            target, times = b['motion'][j].cpu(), b['times'][j].cpu()
            for mode, generated in predictions.items():
                for policy in ('raw', 'clip_all'):
                    pred = generated[j] if policy == 'raw' else generated[j].clamp(0, 1)
                    ref = predictions['audio'][j] if policy == 'raw' else predictions['audio'][j].clamp(0, 1)
                    store = stores[mode][policy]
                    store['metrics'].append(clip_metrics(pred, target, valid, channel, times))
                    store['regions'].append([region_values(pred, target, valid, channel, r) for r in REGIONS.values()])
                    store['features'].append(motion_features(pred[:, support], valid).numpy())
                    store['delta_rms'].append([float((pred-ref)[valid][:, [i for i in r if channel[i]]].square().mean().sqrt())
                                               for r in REGIONS.values()])
        if batch_no % 10 == 0:
            print(json.dumps({'event': 'batch', 'clips': len(groups['labels']), 'total': len(ids)}), flush=True)
    arrays = {k: np.asarray(v) for k, v in groups.items()}
    arrays['scalar_values_audio_teacher'] = np.asarray(scalars)
    labels = arrays['labels']
    report = {'schema': 'frozen_intensity_global_factorial_v1', 'test_loaded': False,
              'training_performed': False, 'promotion_performed': False,
              'checkpoint_sha256': a.checkpoint_sha256, 'data_manifest_sha256': manifest,
              'diagnostic_source_sha256': sha(Path(__file__)), 'recipe_sha256': ck['recipe_sha256'],
              'clips': len(ids), 'classes': names, 'probes': probe_meta, 'batch_size': a.batch_size,
              'noise_seed': 42, 'steps': 12, 'sampler': 'Euler', 'metric_columns': MC,
              'region_columns': RC, 'regions': REGIONS,
              'fixed': 'audio u_a/content/B0/h0/identity/noise; only specified global/scalar renderer inputs change',
              'gt_informed_modes': list(MODES[1:]),
              'warning': 'Teacher uses GT motion; exact label scalar is an annotated ordinal level, not measured channel amplitude. No branch is a deployable repair.',
              'results': {}}
    all_preds, all_metrics, all_regions = {}, {}, {}
    for mode, policies in stores.items():
        report['results'][mode] = {}
        for policy, store in policies.items():
            arr = {k: np.asarray(v) for k, v in store.items()}
            arrays.update({f'{mode}__{policy}__{k}': v for k, v in arr.items()})
            features = torch.from_numpy(arr['features'])
            preds = [p(features[:, fm]).argmax(-1).numpy() for p, fm in probes]
            arrays[f'{mode}__{policy}__probe_predictions'] = np.stack(preds)
            row = {'metrics': means(arr['metrics'], MC),
                   'regions': {name: means(arr['regions'][:, j], RC) for j, name in enumerate(REGIONS)},
                   'probes': [classification_metrics(torch.from_numpy(labels), torch.from_numpy(p), names) for p in preds],
                   'delta_rms_from_audio': dict(zip(REGIONS, arr['delta_rms'].mean(0).tolist())),
                   'groups': {}}
            for key in ('labels', 'intensity', 'speaker'):
                row['groups'][key] = {}
                for value in np.unique(arrays[key]):
                    selected = arrays[key] == value
                    row['groups'][key][str(value)] = {
                        'clips': int(selected.sum()), 'metrics': means(arr['metrics'][selected], MC),
                        'regions': {name: means(arr['regions'][selected, j], RC) for j, name in enumerate(REGIONS)},
                        'probes': [classification_metrics(torch.from_numpy(labels[selected]), torch.from_numpy(p[selected]), names)
                                   for p in preds]}
            report['results'][mode][policy] = row
            if policy == 'clip_all' and mode != 'GT':
                all_preds[mode], all_metrics[mode], all_regions[mode] = preds, arr['metrics'], arr['regions']
    report['paired_comparison'] = paired_intervals(labels, all_preds, all_metrics, all_regions, len(names))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'per_clip.npz', **arrays)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'event': 'complete', 'output': str(a.output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', type=Path, nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit-per-class', type=int, default=0)
    run(parser.parse_args())
