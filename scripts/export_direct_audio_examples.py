"""CPU-only export of fixed direct-audio final-epoch traces for visual review.

Uses the pre-existing metadata-only nine-clip lock. No models, inference,
epoch selection, temporal alignment, coefficient gain or outcome selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_renderer_capacity_examples as common
from scripts.plot_projection_schedule_examples import make_lock, validate_query_metadata
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_predictable_renderer import sha

MODES = ('GT', 'B0', 'source', 'direct flow', 'direct dynamics', 'zero')
SEEDS = [42, 123, 2026, 7, 19, 73, 211, 997]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def bound_curves(run, stem, cache_hash, *, direct_arm=None):
    recipe = read(run/'provenance.json')['recipe']
    digest = canonical_hash(recipe)
    if read(run/'provenance.json')['recipe_sha256'] != digest:
        raise ValueError('Recipe hash differs')
    path = run/(stem+'_curves.pt')
    sidecar = run/(stem+'_curves.provenance.json')
    checkpoint = run/(stem+'.pt')
    expected = {'schema': 'projection_schedule_curves_provenance_v1',
        'curve_sha256': sha(path), 'checkpoint_sha256': sha(checkpoint),
        'recipe_sha256': digest, 'cache_sha256': cache_hash}
    if read(sidecar) != expected:
        raise ValueError('Curve/checkpoint/recipe/cache binding differs')
    if recipe['input_sha256']['cache'] != cache_hash:
        raise ValueError('Recipe cache differs')
    summary = read(run/'summary.json')
    if direct_arm:
        if (recipe.get('schema') != 'direct_audio_dynamics_v1'
                or recipe.get('arm') != direct_arm
                or recipe.get('epochs') != 8
                or recipe.get('eval_noise_seeds') != SEEDS
                or recipe.get('eval_modes') != ['full', 'zero', 'reverse']
                or summary.get('completed_epochs') != 8
                or summary.get('optimizer_steps') != 1160
                or summary.get('protected_unchanged') is not True
                or summary.get('checkpoint_selection_performed') is not False
                or summary.get('recipe_sha256') != digest
                or summary.get('curve_provenance', {}).get('sha256') != sha(sidecar)):
            raise ValueError('Direct fixed-final-epoch contract differs')
        for record in (recipe, summary):
            if record.get('test_loaded') is not False or record.get('default_replaced') is not False:
                raise ValueError('Forbidden test/default exposure')
    curves = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if curves.get('noise_seeds') != SEEDS or curves.get('decode_steps') != 12:
        raise ValueError('Saved curve sampling protocol differs')
    return curves['motion']['42'], {'path': str(path), **expected,
        'sidecar_sha256': sha(sidecar), 'summary_sha256': sha(run/'summary.json')}, recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh output directory required')
    lock = read(args.selection)
    metadata = Path(lock['metadata_dir'])
    if lock != make_lock(metadata) or len(lock['clips']) != 9:
        raise ValueError('Existing metadata-only lock differs')
    cache_path = metadata/'renderer_cache.pt'
    cache_hash = sha(cache_path)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False, mmap=True)
    if (cache.get('schema') != 'predictable_renderer_cache_v1'
            or cache['provenance']['internal_split_lock_sha256'] != lock['internal_split_lock_sha256']
            or cache['provenance']['manifest_hashes']['validation'] != lock['source_sha256']['validation_manifest']):
        raise ValueError('Cache metadata lineage differs')
    split = cache['splits']['validation']
    query = split['q']
    validate_query_metadata(query, lock)
    source, source_binding, source_recipe = bound_curves(
        args.root/'audio_flow_v1/audio_local', 'epoch000', cache_hash)
    flow, flow_binding, flow_recipe = bound_curves(
        args.root/'direct_audio_dynamics_v1/flow', 'final_epoch008', cache_hash, direct_arm='flow')
    dynamics, dynamics_binding, dynamics_recipe = bound_curves(
        args.root/'direct_audio_dynamics_v1/dynamics', 'final_epoch008', cache_hash, direct_arm='dynamics')
    if flow_recipe['input_sha256'] != dynamics_recipe['input_sha256']:
        raise ValueError('Paired direct inputs differ')
    if flow_recipe['input_sha256']['split_lock'] != lock['source_sha256']['split_lock']:
        raise ValueError('Direct input split differs from metadata lock')
    if any(value.shape != query['motion'].shape for values in (source, flow, dynamics) for value in values.values()):
        raise ValueError('Curve/query shape differs')
    fields = {'GT': query['motion'],
        'B0': split['base']['b0'] + split['identity']['baseline'][:, None],
        'source': source['full'], 'direct flow': flow['full'],
        'direct dynamics': dynamics['full'], 'zero': dynamics['zero']}
    # Rule fixed before inspecting any outcome: first locked clip for each speaker.
    picks = common.video_picks(lock['clips'])
    rows = [json.loads(line) for line in (metadata/'validation.jsonl').read_text().splitlines() if line.strip()]
    audio = {pick['clip_id']: common.bind_native_audio(
        query, pick, rows[pick['index']], args.native_root) for pick in picks}
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.selection, args.output/'selection.json')
    common.DISPLAY_MODES = MODES
    common.COLORS = {'GT': '#111111', 'B0': '#8c8c8c', 'source': '#b66b10',
        'direct flow': '#2166ac', 'direct dynamics': '#18834b', 'zero': '#7651a8'}
    for centered in (False, True):
        common.plot_nine(split, lock['clips'], fields,
            args.output/('centered_seed42.png' if centered else 'raw_seed42.png'), centered=centered)
    videos = args.output/'video_npz'
    videos.mkdir()
    exported = []
    for pick in picks:
        index = pick['index']
        values = torch.stack([fields[mode][index] for mode in MODES]).float()
        times, valid, channel_mask, motions = common.native_arrays(query, index, values)
        path = videos/(pick['speaker']+'_first_locked.npz')
        binding = audio[pick['clip_id']]
        np.savez_compressed(path, channels=np.asarray(ARKIT_NAMES), times=times, valid=valid,
            channel_mask=channel_mask, mode_names=np.asarray(MODES), motions=motions,
            clip_id=np.asarray(pick['clip_id']), noise_seed=np.asarray(42),
            audio_path=np.asarray(binding['audio_path']), audio_sha256=np.asarray(binding['audio_sha256']),
            audio_offset_seconds=np.asarray(binding['audio_offset_s']))
        report = inspect_input(path, 25)[-1]
        exported.append({'path': str(path.relative_to(args.output)), 'sha256': sha(path),
            'clip': pick, 'audio_binding': binding, 'display_report': report})
    provenance = {'schema': 'direct_audio_visual_export_v1', 'epoch': 8, 'noise_seed': 42,
        'selection_path': str(args.selection), 'selection_sha256': sha(args.selection),
        'selection_uses_metadata_only': True, 'outcome_based_selection': False,
        'video_selection_rule': 'first locked clip per speaker, original lock order',
        'mode_names': MODES, 'zero_definition': 'direct dynamics epoch8 seed42 zero-local',
        'source_definition': 'source shared audio_flow epoch000 seed42 full-local',
        'cache_path': str(cache_path), 'cache_sha256': cache_hash,
        'bindings': {'source': source_binding, 'flow': flow_binding, 'dynamics': dynamics_binding},
        'videos': exported, 'script_sha256': sha(__file__),
        'limits': 'Internal validation405 only; B0/global have historical exposure; display review is not generalization or perceptual user study.'}
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps({'output': str(args.output), 'videos': [x['path'] for x in exported]}))


if __name__ == '__main__':
    main()
