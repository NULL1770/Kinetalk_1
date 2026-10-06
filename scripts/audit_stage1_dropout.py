"""Matched B0-only neutral audit; no generation, fitting, probes or test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from scripts.audit_matched_motion_curves import RC, MC, region_values, means
from scripts.diagnose_flow_sampling import clip_metrics, sha
from scripts.packed_trainval_cache import load_packed
from scripts.train_full_staged import base_forward, subset

WARM_SHA = 'e17659536a6fcaaaec2d9c22f99403fe4d9f1c8690c3cd182583894545f5ab45'


def paired_ci(delta, speaker):
    """Fixed cluster bootstrap, preserving all clips of a sampled speaker."""
    ids = np.unique(speaker)
    rng = np.random.default_rng(20261005)
    values = []
    for _ in range(2000):
        selected = rng.choice(ids, len(ids), replace=True)
        v = np.concatenate([delta[speaker == sid] for sid in selected])
        values.append(np.nanmean(v, axis=0))
    bounds = np.nanquantile(values, [.025, .975], axis=0)
    return {name: {'delta': float(np.nanmean(delta[:, j])),
                   'speaker_bootstrap_ci95': bounds[:, j].tolist()}
            for j, name in enumerate(RC)}


@torch.no_grad()
def run(a):
    if a.output.exists():
        raise FileExistsError('Fresh B0 audit output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    paths = {mode: a.pair_root / 'checkpoints' / mode for mode in ('control', 'dropout')}
    recipes = {mode: json.loads((path / 'provenance.json').read_text())['recipe']
               for mode, path in paths.items()}
    c, e = recipes['control'], recipes['dropout']
    differences = {key: [c['args'].get(key), e['args'].get(key)]
                   for key in c['args'].keys() | e['args'].keys()
                   if c['args'].get(key) != e['args'].get(key)}
    assert differences == {'stage1_train_dropout': [0., .1]}, differences
    for field in ('source_sha256', 'data_provenance', 'stage_checkpoint_sha256',
                  'safe_dtw_manifest_sha256', 'safe_dtw_artifacts_sha256',
                  'articulation_scope', 'motion_support', 'residual_support'):
        assert c[field] == e[field], field
    for recipe in recipes.values():
        assert not recipe['test_loaded'] and not recipe['independent_probe']['enabled']
        assert recipe['stage_checkpoint_sha256'] == WARM_SHA
        assert recipe['stages'] == ['articulation']
        assert recipe['articulation_scope']['effective_articulation_clip_count'] == 3298
        assert recipe['args']['epochs'] == 4 and recipe['args']['batch_size'] == 16
    assert sha(a.warm) == WARM_SHA
    warm = torch.load(a.warm, map_location='cpu', weights_only=False)
    data = load_packed(a.data, with_refs=False)
    assert set(data['splits']) == {'train', 'validation'}
    assert not data['provenance']['test_loaded']
    manifest = data['provenance']['manifest_sha256']
    report = {'schema': 'stage1_dropout_neutral_only_v1', 'test_loaded': False,
              'generation_performed': False, 'probes_used': False,
              'downstream_adaptation_required': True, 'sole_argument_difference': differences,
              'source_sha256': c['source_sha256'], 'warm_sha256': WARM_SHA,
              'data_manifest_sha256': manifest, 'results': {}, 'comparisons': {}}
    arrays = {}
    for mode in ('reference', 'control', 'dropout'):
        ck = warm if mode == 'reference' else torch.load(
            paths[mode] / 'articulation/final.pt', map_location='cpu', weights_only=False)
        assert ck['data_manifest_sha256'] == manifest
        if mode != 'reference':
            complete = json.loads((paths[mode] / 'articulation/complete.json').read_text())
            assert complete['completed_epochs'] == 4
            assert complete['final_sha256'] == sha(paths[mode] / 'articulation/final.pt')
            changed = 0
            for key, value in ck['system'].items():
                if key.startswith('stage1.'):
                    changed += not torch.equal(value, warm['system'][key])
                else:
                    torch.testing.assert_close(value, warm['system'][key], rtol=0, atol=0)
            assert changed > 0
            for key, value in ck['audio'].items():
                torch.testing.assert_close(value, warm['audio'][key], rtol=0, atol=0)
            report.setdefault('stage1_changed_tensors', {})[mode] = changed
        model = NeutralAffectSystem(ck['config']).to(a.device).eval()
        model.load_state_dict(ck['system'], strict=True)
        report['results'][mode] = {}
        for role, split in data['splits'].items():
            ix = (split['emotion_id'] == 0).nonzero(as_tuple=True)[0]
            assert len(ix) == (715 if role == 'train' else 80)
            raw, clipped, mc, jaw_raw, jaw_clip = [], [], [], [], []
            motions, targets, masks, times, channels = [], [], [], [], []
            for batch in ix.split(a.batch_size):
                b = subset(split, batch, a.device,
                           keys=('content', 'motion', 'valid', 'channel_mask', 'times'))
                pred = base_forward(model, b['content'], b['valid'])['b0'].cpu()
                for j, clip_index in enumerate(batch.tolist()):
                    n = int(split['_lengths'][clip_index])
                    p = pred[j, :n]
                    t, v, ch, tm = [b[key][j, :n].cpu() if key != 'channel_mask'
                                    else b[key][j].cpu()
                                    for key in ('motion', 'valid', 'channel_mask', 'times')]
                    pc = p.clamp(0, 1)
                    raw.append(region_values(p, t, v, ch, list(range(14, 41))))
                    clipped.append(region_values(pc, t, v, ch, list(range(14, 41))))
                    jaw_raw.append(region_values(p, t, v, ch, [17]))
                    jaw_clip.append(region_values(pc, t, v, ch, [17]))
                    mc.append(clip_metrics(pc, t, v, ch, tm))
                    if role == 'validation':
                        motions.append(p.numpy())
                        if mode == 'reference':
                            targets.append(t.numpy()); masks.append(v.numpy())
                            times.append(tm.numpy()); channels.append(ch.numpy())
            speaker = split['speaker_id'][ix].numpy()
            for name, rows in [('mouth_raw', raw), ('mouth_clip', clipped),
                               ('jaw_raw', jaw_raw), ('jaw_clip', jaw_clip)]:
                rows = np.asarray(rows)
                arrays[f'{mode}__{role}__{name}'] = rows
                report['results'][mode].setdefault(role, {})[name] = {
                    'clips': len(ix), **means(rows, RC), 'speaker': {
                        str(sid): means(rows[speaker == sid], RC) for sid in np.unique(speaker)}}
            arrays[f'{role}__speaker'] = speaker
            arrays[f'{role}__clip_id'] = np.asarray([split['clip_id'][i] for i in ix.tolist()])
            report['results'][mode][role]['clipped_coefficient_metrics'] = means(np.asarray(mc), MC)
            if motions:
                lens = np.asarray([len(p) for p in motions])
                arrays[f'{mode}__validation__motion'] = np.concatenate(motions)
                arrays['validation__offsets'] = np.r_[0, lens.cumsum()]
                if mode == 'reference':
                    arrays['validation__target'] = np.concatenate(targets)
                    arrays['validation__valid'] = np.concatenate(masks)
                    arrays['validation__times'] = np.concatenate(times)
                    arrays['validation__channel_mask'] = np.stack(channels)
        del model
        torch.cuda.empty_cache() if torch.device(a.device).type == 'cuda' else None
    for role in ('train', 'validation'):
        report['comparisons'][role] = {}
        for name in ('mouth_raw', 'mouth_clip', 'jaw_raw', 'jaw_clip'):
            delta = arrays[f'dropout__{role}__{name}'] - arrays[f'control__{role}__{name}']
            report['comparisons'][role][name] = paired_ci(delta, arrays[f'{role}__speaker'])
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output / 'per_clip_neutral.npz', **arrays)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'complete': str(a.output), 'validation': report['comparisons']['validation']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-root', type=Path, required=True)
    p.add_argument('--warm', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    run(p.parse_args())
