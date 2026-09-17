"""CPU-only fixed-lock display export for output displacement/std ablation.

No checkpoint model is loaded, no inference or outcome-based selection runs.
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
from scripts.export_direct_audio_examples import bound_curves, read, SEEDS
from scripts.plot_projection_schedule_examples import make_lock, validate_query_metadata
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input
from scripts.train_predictable_renderer import sha

MODES = ('GT', 'source', 'direct L1', 'output audio', 'output own zero', 'trained zero')


def output_curves(run, cache_hash, arm):
    curves, binding, recipe = bound_curves(run, 'final_epoch008', cache_hash)
    summary = read(run/'summary.json')
    inventory = read(run/'output_hashes.json')
    if (recipe.get('schema') != 'output_motion_dynamics_v1'
            or recipe.get('arm') != arm or summary.get('arm') != arm
            or recipe.get('epochs') != 8 or recipe.get('decode_steps') != 12
            or recipe.get('eval_noise_seeds') != SEEDS
            or recipe.get('eval_modes') != ['full', 'zero', 'reverse']
            or summary.get('completed_epochs') != 8 or summary.get('optimizer_steps') != 1160
            or summary.get('protected_unchanged') is not True
            or summary.get('checkpoint_selection_performed') is not False
            or summary.get('curve_provenance', {}).get('sha256') != binding['sidecar_sha256']
            or summary.get('recipe_sha256') != binding['recipe_sha256']):
        raise ValueError('Output-motion fixed-final-epoch contract differs')
    for record in (recipe, summary):
        if record.get('test_loaded') is not False or record.get('default_replaced') is not False:
            raise ValueError('Forbidden test/default exposure')
    if arm == 'zero' and summary.get('encoder_unchanged') is not True:
        raise ValueError('Matched zero encoder changed')
    for name in ('provenance.json', 'summary.json', 'final_epoch008.pt',
                 'final_epoch008_curves.pt', 'final_epoch008_curves.provenance.json'):
        if inventory.get(name) != sha(run/name):
            raise ValueError('Output-motion inventory differs: ' + name)
    return curves, binding, recipe, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh output required')
    lock = read(args.selection)
    metadata = Path(lock['metadata_dir'])
    if lock != make_lock(metadata) or len(lock['clips']) != 9:
        raise ValueError('Existing metadata-only nine-lock differs')
    picks = common.video_picks(lock['clips'])
    cache_path = metadata/'renderer_cache.pt'
    cache_hash = sha(cache_path)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False, mmap=True)
    if (cache.get('schema') != 'predictable_renderer_cache_v1'
            or cache['provenance']['internal_split_lock_sha256'] != lock['internal_split_lock_sha256']
            or cache['provenance']['manifest_hashes']['validation'] != lock['source_sha256']['validation_manifest']):
        raise ValueError('Cache metadata lineage differs')
    split = cache['splits']['validation']; query = split['q']
    validate_query_metadata(query, lock)
    source, source_binding, _ = bound_curves(args.root/'audio_flow_v1/audio_local', 'epoch000', cache_hash)
    direct, direct_binding, _ = bound_curves(args.root/'direct_audio_dynamics_v1/dynamics',
        'final_epoch008', cache_hash, direct_arm='dynamics')
    audio, audio_binding, audio_recipe, audio_summary = output_curves(args.root/'output_motion_dynamics_v1/audio', cache_hash, 'audio')
    zero, zero_binding, zero_recipe, zero_summary = output_curves(args.root/'output_motion_dynamics_v1/zero', cache_hash, 'zero')
    for key in ('input_sha256', 'initial_system_sha256', 'initial_encoder_sha256',
                'protected_sha256', 'motion_scales_sha256', 'loss_weights', 'groups'):
        if audio_recipe[key] != zero_recipe[key]:
            raise ValueError('Paired output-motion recipe differs: ' + key)
    for key in ('minibatch_sha256', 'noise_time_sha256', 'choice_draw_sha256'):
        if audio_summary[key] != zero_summary[key]:
            raise ValueError('Paired training draws differ')
    if audio_recipe['input_sha256']['split_lock'] != lock['source_sha256']['split_lock']:
        raise ValueError('Output-motion split differs from lock')
    if any(value.shape != query['motion'].shape for values in (source, direct, audio, zero) for value in values.values()):
        raise ValueError('Curve/query shape differs')
    fields = {'GT': query['motion'], 'source': source['full'], 'direct L1': direct['full'],
        'output audio': audio['full'], 'output own zero': audio['zero'], 'trained zero': zero['zero']}
    rows = [json.loads(line) for line in (metadata/'validation.jsonl').read_text().splitlines() if line.strip()]
    waveforms = {pick['clip_id']: common.bind_native_audio(query, pick, rows[pick['index']], args.native_root) for pick in picks}
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.selection, args.output/'selection.json')
    common.DISPLAY_MODES = MODES
    common.COLORS = {'GT': '#111111', 'source': '#b66b10', 'direct L1': '#18834b',
        'output audio': '#2166ac', 'output own zero': '#7651a8', 'trained zero': '#8c8c8c'}
    for centered in (False, True):
        common.plot_nine(split, lock['clips'], fields,
            args.output/('centered_seed42.png' if centered else 'raw_seed42.png'), centered=centered)
    video_dir = args.output/'video_npz'; video_dir.mkdir()
    exported = []
    for pick in picks:
        index = pick['index']; values = torch.stack([fields[mode][index] for mode in MODES]).float()
        times, valid, channel_mask, motions = common.native_arrays(query, index, values)
        path = video_dir/(pick['speaker']+'_first_locked.npz'); waveform = waveforms[pick['clip_id']]
        np.savez_compressed(path, channels=np.asarray(ARKIT_NAMES), times=times, valid=valid,
            channel_mask=channel_mask, mode_names=np.asarray(MODES), motions=motions,
            clip_id=np.asarray(pick['clip_id']), noise_seed=np.asarray(42),
            audio_path=np.asarray(waveform['audio_path']), audio_sha256=np.asarray(waveform['audio_sha256']),
            audio_offset_seconds=np.asarray(waveform['audio_offset_s']))
        exported.append({'path': str(path.relative_to(args.output)), 'sha256': sha(path),
            'clip': pick, 'audio_binding': waveform, 'display_report': inspect_input(path, 25)[-1]})
    provenance = {'schema': 'output_motion_visual_export_v1', 'epoch': 8, 'noise_seed': 42,
        'selection_path': str(args.selection), 'selection_sha256': sha(args.selection),
        'selection_uses_metadata_only': True, 'outcome_based_selection': False,
        'video_selection_rule': 'first locked clip per speaker, original lock order', 'mode_names': MODES,
        'mode_definitions': {'GT': 'validation q.motion', 'source': 'shared source epoch000 seed42/full',
            'direct L1': 'direct_audio_dynamics_v1/dynamics epoch8 seed42/full',
            'output audio': 'output_motion_dynamics_v1/audio epoch8 seed42/full',
            'output own zero': 'same audio-trained model epoch8 seed42/zero',
            'trained zero': 'matched zero-trained renderer epoch8 seed42/zero'},
        'cache_path': str(cache_path), 'cache_sha256': cache_hash,
        'bindings': {'source': source_binding, 'direct_L1': direct_binding, 'output_audio': audio_binding, 'output_zero': zero_binding},
        'videos': exported, 'script_sha256': sha(__file__),
        'limits': 'Fixed internal diagnostic only; not generalization, identity, lip-sync or perceptual study.'}
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps({'output': str(args.output), 'videos': [x['path'] for x in exported]}))


if __name__ == '__main__':
    main()
