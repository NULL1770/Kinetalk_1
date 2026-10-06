"""Frozen, GT-informed global-only localization with Phase21 replay gate.

Imports the bound Phase21 snapshot, never edits or trains it. Complete padded
noise is drawn before native batch trimming, exactly as staged evaluate(full).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

DRAWS = (42, 123, 2026)
POLICIES = ('raw', 'clip_all')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def native_padded_noise(valid, draw):
    return torch.randn(len(valid), valid.shape[1], 52,
                       generator=torch.Generator().manual_seed(draw))


def replace_global(audio, teacher):
    if audio['global'].shape != teacher['global'].shape:
        raise ValueError('Teacher and audio global shapes differ')
    # All other objects, including scalar, temporal state and logits, are shared.
    return {**audio, 'global': teacher['global']}


def canonical_audio_evaluation_flags(system, audio):
    """Keep the original final evaluator's dispatch flags under no_grad.

    Disabling every requires_grad flag changed the numerical execution path
    despite no_grad. This preserves the original stage flags without creating
    an optimizer, gradients, or parameter updates.
    """
    if torch.is_grad_enabled():
        raise RuntimeError('Frozen diagnostic requires no_grad')
    system.requires_grad_(False)
    audio.requires_grad_(True)
    system.renderer.requires_grad_(True)
    assert all(p.grad is None for m in (system, audio) for p in m.parameters())


def assert_reference_batch(b, saved, ix):
    width = b['valid'].shape[1]
    assert b['clip_id'] == [saved['clip_id'][int(i)] for i in ix]
    for key, reference in (('motion', 'target'), ('valid', 'valid'),
                           ('times', 'times'), ('channel_mask', 'channel_mask'),
                           ('b0', 'b0')):
        target = saved[reference][ix]
        if key != 'channel_mask':
            target = target[:, :width]
        torch.testing.assert_close(b[key].cpu(), target, rtol=0, atol=0, equal_nan=True)


def report_row(metrics, regions, features, labels, groups, probes, names, MC, RC, REGIONS,
               classification_metrics, means):
    predictions, by_draw = [], {}
    for draw, fs, ms in zip(DRAWS, features, metrics):
        f = torch.as_tensor(fs)
        ps = [m(f[:, mask]).argmax(-1).numpy() for m, mask in probes]
        predictions.append(ps)
        by_draw[str(draw)] = {'metrics': means(ms, MC), 'probes': [
            classification_metrics(torch.from_numpy(labels), torch.from_numpy(p), names) for p in ps]}
    predictions = np.asarray(predictions)
    row = {'metrics': means(metrics, MC),
           'regions': {name: means(regions[:, :, j], RC) for j, name in enumerate(REGIONS)},
           'by_draw': by_draw,
           'probe_mean_f1': [float(np.mean([v['probes'][j]['macro_f1'] for v in by_draw.values()]))
                             for j in range(len(probes))], 'groups': {}}
    for group, values in groups.items():
        row['groups'][group] = {str(k): {
            'n': int((values == k).sum()), 'metrics': means(metrics[:, values == k], MC),
            'regions': {name: means(regions[:, values == k, j], RC) for j, name in enumerate(REGIONS)},
            'probe_classification_by_draw': [[classification_metrics(
                torch.from_numpy(labels[values == k]), torch.from_numpy(p[values == k]), names)
                for p in draw_ps] for draw_ps in predictions]}
            for k in np.unique(values)}
    return row, predictions


def paired_intervals(arrays, groups, classes, macro_f1, MC, RC, REGIONS):
    # The same sampled speakers/clips are reused for every branch and probe.
    rng = np.random.default_rng(20261005)
    speakers = np.unique(groups['speaker'])
    labels = arrays['labels']
    metric = arrays['teacher_global__clip_all__metrics'].mean(0) - arrays['audio__clip_all__metrics'].mean(0)
    region = arrays['teacher_global__clip_all__regions'].mean(0) - arrays['audio__clip_all__regions'].mean(0)
    audio = arrays['audio__clip_all__probe_predictions']
    oracle = arrays['teacher_global__clip_all__probe_predictions']
    ms, rs, fs = [], [], []
    for _ in range(2000):
        chosen = rng.choice(speakers, len(speakers), replace=True)
        ix = np.concatenate([np.flatnonzero(groups['speaker'] == s) for s in chosen])
        ms.append(metric[ix].mean(0))
        rs.append(np.nanmean(region[ix], axis=0))
        fs.append([float(np.mean([
            macro_f1(labels[ix], oracle[d, p, ix], classes) -
            macro_f1(labels[ix], audio[d, p, ix], classes) for d in range(3)]))
            for p in range(audio.shape[1])])
    return {'policy': '2000 paired speaker-cluster bootstraps seed20261005; draws averaged, no selection',
            'direction': 'teacher_global minus audio',
            'metrics_ci95': dict(zip(MC, np.quantile(ms, [.025, .975], axis=0).T.tolist())),
            'region_ci95': {name: dict(zip(RC, np.nanquantile(rs, [.025, .975], axis=0)[:, j].T.tolist()))
                            for j, name in enumerate(REGIONS)},
            'probe_f1_ci95': np.quantile(fs, [.025, .975], axis=0).T.tolist()}


@torch.no_grad()
def run(a):
    assert not a.output.exists(), 'Use a fresh diagnostic directory; never overwrite failures'
    a.output.mkdir(parents=True)
    state = a.output / 'state.json'
    def update(status, **kw):
        state.write_text(json.dumps(dict(status=status, updated_at=time.time(),
                                        training_performed=False, test_loaded=False, **kw), indent=2))
    update('binding')
    try:
        root = a.phase21_root
        binding = json.loads(a.binding.read_text())
        assert binding['seeds'] == [47, 48, 49] and a.seed in binding['seeds']
        assert json.loads((root / 'postprocess_state.json').read_text())['status'] == 'complete'
        sys.path.insert(0, str(root / 'code'))
        from kinetalk_b0.emotion_probe import MotionEmotionProbe, classification_metrics, motion_features
        from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
        from kinetalk_b0.models.slow_state_affect import SlowStateAffect
        from scripts.audit_matched_motion_curves import MC, RC, REGIONS, means, region_values
        from scripts.compare_projection_ablation import macro_f1
        from scripts.diagnose_flow_sampling import clip_metrics
        from scripts.packed_trainval_cache import load_packed
        from scripts.train_full_staged import audio_affect, batch_identity, cache_current_base, identity_cache, subset, teacher_affect

        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        device = torch.device(a.device)
        pair = root / f'seed{a.seed}'
        cp = pair / 'checkpoints/native_b0'
        bound = binding['inputs'][str(a.seed)]
        paths = {'final': cp / 'audio/final.pt', 'curves': cp / 'audio/curves.pt',
                 'audit': pair / 'native_audit/per_clip_audit.npz',
                 'report': pair / 'native_audit/report.json'}
        for key, path in paths.items():
            assert sha(path) == bound[key + '_sha256'], key
        ck = torch.load(paths['final'], map_location='cpu', weights_only=False)
        recipe = json.loads((cp / 'provenance.json').read_text())['recipe']
        assert ck['stage'] == 'audio' and ck['recipe_sha256'] == bound['recipe_sha256']
        assert not recipe['test_loaded'] and not recipe['independent_probe']['enabled']
        assert recipe['source_sha256'] == binding['training_source_sha256']
        assert all(sha(root / 'code' / name) == digest for name, digest in recipe['source_sha256'].items())
        assert recipe['args']['batch_size'] == 16 and recipe['args']['decode_steps'] == 12
        assert recipe['flow_source_noise']['mode'] == 'standard'
        assert recipe['args']['ablation'] == 'none'
        saved = torch.load(paths['curves'], map_location='cpu', weights_only=False)
        prior = json.loads(paths['report'].read_text())
        cache = np.load(paths['audit'], allow_pickle=False)
        data = load_packed(a.data, materialize=False, with_refs=True)
        assert not data['provenance']['test_loaded'] and set(data['splits']) == {'train', 'validation'}
        assert data['provenance']['manifest_sha256'] == ck['data_manifest_sha256'] == binding['data_manifest_sha256']
        # Rebuild only validation and independent enrollment, without TRAIN extraction.
        data['splits'] = {'validation': data['splits']['validation']}
        q = data['splits']['validation']
        ids = torch.arange(len(q['valid']))
        assert len(ids) == 1367 and list(saved['noise_seeds']) == list(DRAWS)
        assert list(saved['predictions']) == [f'{d}/full' for d in DRAWS]
        assert q['clip_id'] == saved['clip_id'] == cache['clip_id'].tolist()
        system = NeutralAffectSystem(ck['config']).to(device).eval()
        system.load_state_dict(ck['system'], strict=True)
        system.requires_grad_(False)
        audio = SlowStateAffect(ck['feature_stats']['mean'], ck['feature_stats']['std'],
                               stride=recipe['args']['stride']).to(device).eval()
        audio.load_state_dict(ck['audio'], strict=True)
        audio.requires_grad_(False)
        initial_state = {key: value.cpu().clone() for key, value in system.state_dict().items()}
        initial_audio = {key: value.cpu().clone() for key, value in audio.state_dict().items()}
        cache_current_base(system, data, device, batch_size=32)
        identities = identity_cache(system, data, device)
        canonical_audio_evaluation_flags(system, audio)

        probes, support, names = [], None, data['config']['data']['emotion_classes']
        assert len(binding['probes']) == 4
        for spec in binding['probes']:
            path = root / 'analysis_probes' / spec['name']
            assert sha(path) == spec['sha256']
            pc = torch.load(path, map_location='cpu', weights_only=False)
            assert pc['train_manifest_sha256'] == ck['data_manifest_sha256'] and pc['classes'] == names
            assert not pc.get('test_used_for_selection') and not pc.get('generator_outputs_used_for_fitting')
            if support is not None:
                assert torch.equal(support, pc['channel_support'].bool())
            support = pc['channel_support'].bool()
            m = MotionEmotionProbe(pc['feature_dim'], pc['hidden'], len(names)).eval()
            m.load_state_dict(pc['model'], strict=True)
            probes.append((m, pc.get('feature_mask', torch.ones(6 * int(support.sum()), dtype=torch.bool))))
        assert q['channel_mask'][:, support].all()

        update('replaying_audio')
        replay, coordinates = {}, {k: [] for k in ('audio_global', 'teacher_global', 'audio_logits', 'teacher_logits')}
        for draw in DRAWS:
            noise = native_padded_noise(q['valid'], draw)
            bit_exact, max_diff = True, 0.
            for batch_no, ix in enumerate(ids.split(16)):
                b = subset(q, ix, device)
                assert_reference_batch(b, saved, ix)
                ident = batch_identity(identities, b)
                teacher = teacher_affect(system, b, ident)
                ao = audio_affect(audio, b['audio_features'], b['valid'])
                pred = system.generate(b['content'], b['valid'], ident, ao,
                    initial_noise=noise[ix, :b['valid'].shape[1]].to(device), steps=12,
                    base={'b0': b['b0'], 'h0': b['h0']})['motion'].cpu()
                expected = saved['predictions'][f'{draw}/full'][ix, :pred.shape[1]]
                diff = float((pred - expected).abs().max())
                bit_exact = bit_exact and torch.equal(pred, expected)
                max_diff = max(max_diff, diff)
                torch.testing.assert_close(pred, expected, atol=2e-6, rtol=0)
                if draw == 42:
                    for key, value in (('audio_global', ao['global']), ('teacher_global', teacher['global']),
                                       ('audio_logits', ao['emotion_logits']), ('teacher_logits', teacher['emotion_logits'])):
                        coordinates[key].append(value.cpu().numpy())
                if batch_no % 20 == 0:
                    print(json.dumps(dict(event='replay', draw=draw, clips=int(ix[-1])+1, max_diff=max_diff)), flush=True)
            replay[str(draw)] = {'bit_exact': bit_exact, 'max_abs_difference': max_diff, 'tolerance_abs': 2e-6}
        (a.output / 'replay.json').write_text(json.dumps(replay, indent=2))
        assert all(v['max_abs_difference'] <= 2e-6 for v in replay.values())

        # All three baseline draws must pass before the first oracle decode.
        update('teacher_global', replay=replay)
        stores = {policy: {'metrics': [], 'regions': [], 'features': [], 'delta_rms': []} for policy in POLICIES}
        labels = q['emotion_id'].numpy()
        assert np.array_equal(labels, cache['labels'])
        groups = {k: q[key].numpy() for k, key in (('emotion', 'emotion_id'), ('intensity', 'intensity_id'), ('speaker', 'speaker_id'))}
        for draw in DRAWS:
            noise = native_padded_noise(q['valid'], draw)
            current = {policy: {k: [] for k in stores[policy]} for policy in POLICIES}
            for batch_no, ix in enumerate(ids.split(16)):
                b = subset(q, ix, device)
                ident = batch_identity(identities, b)
                teacher = teacher_affect(system, b, ident)
                ao = audio_affect(audio, b['audio_features'], b['valid'])
                condition = replace_global(ao, teacher)
                assert all(condition[k] is v for k, v in ao.items() if k != 'global')
                pred = system.generate(b['content'], b['valid'], ident, condition,
                    initial_noise=noise[ix, :b['valid'].shape[1]].to(device), steps=12,
                    base={'b0': b['b0'], 'h0': b['h0']})['motion'].cpu()
                for j, i in enumerate(ix.tolist()):
                    target, valid, channel, times = (saved[k][i, :pred.shape[1]] if k != 'channel_mask' else saved[k][i]
                                                    for k in ('target', 'valid', 'channel_mask', 'times'))
                    for policy in POLICIES:
                        p = pred[j] if policy == 'raw' else pred[j].clamp(0, 1)
                        ref = saved['predictions'][f'{draw}/full'][i, :pred.shape[1]]
                        if policy == 'clip_all':
                            ref = ref.clamp(0, 1)
                        out = current[policy]
                        out['metrics'].append(clip_metrics(p, target, valid, channel, times))
                        out['regions'].append([region_values(p, target, valid, channel, r) for r in REGIONS.values()])
                        out['features'].append(motion_features(p[:, support], valid).numpy())
                        out['delta_rms'].append([float((p-ref)[valid][:, [k for k in r if channel[k]]].square().mean().sqrt())
                                                 for r in REGIONS.values()])
                if batch_no % 20 == 0:
                    print(json.dumps(dict(event='teacher_global', draw=draw, clips=int(ix[-1])+1)), flush=True)
            for policy in POLICIES:
                for key in stores[policy]:
                    stores[policy][key].append(current[policy][key])

        arrays = {'clip_id': np.asarray(saved['clip_id']), 'labels': labels, **groups,
                  **{k: np.concatenate(v) for k, v in coordinates.items()}}
        report = {'schema': 'phase22_frozen_global_native_protocol_v1', 'seed': a.seed,
                  'test_loaded': False, 'training_performed': False, 'promotion_performed': False,
                  'clips': 1367, 'draws': list(DRAWS), 'batch_size': 16, 'decode_steps': 12,
                  'noise_policy': 'whole-validation padded CPU randn then native batch slice; staged evaluate(full)',
                  'checkpoint_sha256': bound['final_sha256'], 'binding_sha256': sha(a.binding),
                  'diagnostic_source_sha256': sha(__file__), 'data_manifest_sha256': ck['data_manifest_sha256'],
                  'probes': binding['probes'], 'classes': names, 'replay': replay,
                  'fixed': 'audio u_a/intensity_value/logits/content/B0/h0/identity/noise; only global replaced',
                  'inference_context': 'no_grad with original audio-stage requires_grad flags; no optimizer/backward/updates',
                  'warning': 'teacher_global consumes true query motion; oracle localization only, not deployable performance',
                  'gt_probes': prior['gt_probes'], 'results': {}}
        for policy in POLICIES:
            for mode in ('audio', 'teacher_global'):
                prefix = mode + '__' + policy
                if mode == 'audio':
                    values = {k: cache['native_b0__' + policy + '__' + k] for k in ('metrics', 'regions', 'features')}
                else:
                    values = {k: np.asarray(v) for k, v in stores[policy].items()}
                arrays.update({prefix + '__' + k: v for k, v in values.items()})
                row, ps = report_row(values['metrics'], values['regions'], values['features'], labels,
                                     groups, probes, names, MC, RC, REGIONS, classification_metrics, means)
                arrays[prefix + '__probe_predictions'] = ps
                if mode == 'audio':
                    np.testing.assert_allclose(row['probe_mean_f1'], prior['results']['native_b0__' + policy]['probe_mean_f1'], atol=1e-12, rtol=0)
                else:
                    row['delta_rms_from_audio'] = dict(zip(REGIONS, values['delta_rms'].mean((0, 1)).tolist()))
                report['results'][prefix] = row
        report['global_head_classification_not_generated_f1'] = {k: classification_metrics(
            torch.from_numpy(labels), torch.from_numpy(arrays[k + '_logits'].argmax(-1)), names)
            for k in ('audio', 'teacher')}
        report['paired_intervals'] = paired_intervals(arrays, groups, len(names), macro_f1, MC, RC, REGIONS)
        assert all(torch.equal(v.cpu(), initial_state[k]) for k, v in system.state_dict().items())
        assert all(torch.equal(v.cpu(), initial_audio[k]) for k, v in audio.state_dict().items())
        report['all_model_states_unchanged'] = True
        np.savez_compressed(a.output / 'per_clip.npz', **arrays)
        (a.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
        update('complete', report_sha256=sha(a.output / 'report.json'), arrays_sha256=sha(a.output / 'per_clip.npz'))
        print(json.dumps({'event': 'complete', 'seed': a.seed, 'results': {
            k: {'metrics': v['metrics'], 'probe_mean_f1': v['probe_mean_f1']} for k, v in report['results'].items()}}), flush=True)
    except Exception as e:
        update('failed', error=str(e), error_type=type(e).__name__)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase21-root', type=Path, required=True)
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(47, 48, 49), required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    run(p.parse_args())
