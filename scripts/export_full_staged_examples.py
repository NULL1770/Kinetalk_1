"""Export the existing nine-clip lock from saved full-staged training curves.

No native datasets, model instantiation, inference, sample selection, amplitude
fitting or rendering. Optional checkpoint state tensors are read only to audit
the frozen cross-stage path. Existing locked audio copies may be reused.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input

SCHEMA = 'full_staged_visual_export_v1'
RUN_SCHEMA = 'full_staged_spline_innovation_v1'
MODES = ('GT', 'B0', 'stage4 base', 'final dynamic', 'static state', 'oracle state')
MODE_DEFINITIONS = {
    'GT': 'Saved internal-development target motion, diagnostic reference.',
    'B0': 'Retrained articulation B0 only; no identity offset or emotional residual.',
    'stage4 base': 'Completed audio-stage full-face model, seed42; audio and independent neutral references only.',
    'final dynamic': 'Completed dynamics-stage output, seed42; audio-predicted slow state and projected random innovation.',
    'static state': 'Same dynamics model and noise with predicted slow state fixed to its valid-frame mean; local audio remains temporal.',
    'oracle state': 'Same dynamics model and noise with target-motion slow state; target-conditioned diagnostic, not deployable audio-only output.',
}
PREDICTION_KEYS = ('42/base', '42/full', '42/static_state', '42/oracle_state')
NOT_UPPER = tuple(i for i in range(52) if i not in UPPER_INDICES)
# Compatibility limits for independently evaluated FP32/TF32 outputs, adopted
# after the run exposed cross-stage numerical differences. These do not assert
# bitwise determinism and cannot replace exact frozen-weight verification.
CROSS_STAGE_MAX_ABS = 5e-4
CROSS_STAGE_RMS = 5e-5


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(2**20), b''): digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')


def local_child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Previous visual asset leaves its explicit directory')
    return path


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not torch.is_tensor(value): raise ValueError('Expected tensor-only frozen state')
        cpu = value.detach().contiguous().cpu()
        digest.update(name.encode('utf8')); digest.update(str(cpu.dtype).encode('ascii'))
        digest.update(str(tuple(cpu.shape)).encode('ascii')); digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def audit_frozen_cross_stage(run, audio_curves, dynamic_curves):
    """Require exact frozen state before accepting small output discrepancies.

    Checkpoint state dictionaries are inspected; no model is instantiated.
    Both file hashes must match each completed stage's immutable manifest.
    Limits are absolute coefficient maximum and global RMS, including padding.
    """
    snapshots, bindings = {}, {}
    for stage in ('audio', 'dynamics'):
        complete = read(run / stage / 'complete.json')
        path = run / stage / 'final.pt'
        if not path.is_file() or complete.get('final_sha256') != sha(path):
            raise ValueError('Frozen-state audit requires hash-bound final checkpoint: ' + stage)
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if (saved.get('schema') != RUN_SCHEMA or saved.get('stage') != stage
                or saved.get('completed_epochs') != complete.get('completed_epochs')
                or saved.get('inference_only') is not True):
            raise ValueError('Frozen-state checkpoint protocol differs: ' + stage)
        snapshots[stage] = saved
        bindings[stage] = {'path': str(path), 'sha256': sha(path)}
    a, d = snapshots['audio'], snapshots['dynamics']
    if not a.get('recipe_sha256') or a['recipe_sha256'] != d.get('recipe_sha256') or a.get('config') != d.get('config'):
        raise ValueError('Frozen-stage checkpoint recipe/config differs')
    module_hashes = {}
    for module in ('system', 'audio'):
        aa, dd = a.get(module, {}), d.get(module, {})
        if not aa or set(aa) != set(dd):
            raise ValueError('Frozen module state keys differ: ' + module)
        for name, value in aa.items():
            other = dd[name]
            if (not torch.is_tensor(value) or not torch.is_tensor(other)
                    or value.dtype != other.dtype or not torch.equal(value, other)):
                raise ValueError('Frozen module tensor changed: ' + module + '.' + name)
        module_hashes[module] = state_digest(aa)
    x, y = audio_curves['predictions']['42/full'], dynamic_curves['predictions']['42/base']
    if x.shape != y.shape or x.dtype != y.dtype or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Cross-stage outputs require matching finite tensors')
    delta = (x.double() - y.double()).abs()
    valid = dynamic_curves['valid'][..., None].expand_as(x)
    observed = delta[valid]
    maximum, rms = float(delta.max()), float(delta.square().mean().sqrt())
    if maximum > CROSS_STAGE_MAX_ABS or rms > CROSS_STAGE_RMS:
        raise ValueError(f'Frozen cross-stage output exceeds numerical limits: max={maximum:.9g}, rms={rms:.9g}')
    return {'schema': 'full_staged_frozen_cross_stage_audit_v1',
        'frozen_modules_equal_exactly': True, 'frozen_module_sha256': module_hashes,
        'checkpoint_bindings': bindings, 'recipe_sha256': a['recipe_sha256'],
        'cross_stage_output_equal_exactly': bool(torch.equal(x, y)),
        'max_absolute_difference': maximum, 'rms_difference': rms,
        'observed_rms_difference': float(observed.square().mean().sqrt()),
        'observed_mean_absolute_difference': float(observed.mean()),
        'observed_changed_values': int((observed != 0).sum()), 'observed_values': observed.numel(),
        'limits': {'max_absolute_difference': CROSS_STAGE_MAX_ABS, 'rms_difference': CROSS_STAGE_RMS},
        'limit_origin': 'Explicit compatibility policy added after completed-run export found non-bitwise cross-stage outputs; not a preregistered metric.',
        'interpretation': 'Identical frozen weights and bounded numerical differences; operation-level cause is not established by this audit.',
        'nonupper_within_dynamics': 'Still required to match the saved same-stage base bitwise; this tolerance does not weaken that assertion.'}


def load_saved(run):
    path = run / 'dynamics/curves.pt'
    complete = read(run / 'dynamics/complete.json')
    if complete.get('stage') != 'dynamics' or complete.get('curves_sha256') != sha(path):
        raise ValueError('Completed dynamics curve hash/stage differs')
    curves = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if curves.get('schema') != RUN_SCHEMA or curves.get('stage') != 'dynamics':
        raise ValueError('Expected full-staged final dynamics curves')
    ids = curves.get('clip_id', [])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Curve clip IDs must be nonempty and unique')
    target, valid, times, channels = (curves[k] for k in ('target', 'valid', 'times', 'channel_mask'))
    if (target.ndim != 3 or target.shape[-1] != 52 or target.shape[0] != len(ids)
            or valid.shape != target.shape[:2] or valid.dtype != torch.bool
            or times.shape != valid.shape or channels.shape != (len(ids), 52)
            or channels.dtype != torch.bool or not valid.any(1).all()
            or not torch.isfinite(times).all()):
        raise ValueError('Invalid saved native dimensions, mask or time')
    if not torch.allclose(times[:, 1:] - times[:, :-1], torch.full_like(times[:, 1:], .04), atol=1e-7, rtol=1e-5):
        raise ValueError('Saved curves must retain the native 25fps clock')
    predictions = curves.get('predictions', {})
    if not set(PREDICTION_KEYS) <= predictions.keys():
        raise ValueError('Missing seed42 base/full/static/oracle saved curves')
    fields = [target, curves['b0'], *(predictions[k] for k in PREDICTION_KEYS)]
    for value in fields:
        if value.shape != target.shape or not torch.isfinite(value[valid]).all():
            raise ValueError('Saved motion shape or observed values differ')
    for key in PREDICTION_KEYS[1:]:
        if not torch.equal(predictions[key][..., list(NOT_UPPER)], predictions['42/base'][..., list(NOT_UPPER)]):
            raise ValueError('Non-upper channel protection differs in saved ' + key)
    # Optional separate audio-stage curves provide an independent base binding.
    audio_path = run / 'audio/curves.pt'
    audio_binding = None
    if audio_path.exists():
        audio_complete = read(run / 'audio/complete.json')
        if audio_complete.get('curves_sha256') != sha(audio_path):
            raise ValueError('Completed audio curve hash differs')
        audio = torch.load(audio_path, map_location='cpu', weights_only=False, mmap=True)
        if audio.get('schema') != RUN_SCHEMA or audio.get('stage') != 'audio' or audio.get('clip_id') != ids:
            raise ValueError('Audio/dynamics saved curve IDs or schema differ')
        for key in ('target', 'valid', 'times', 'channel_mask', 'b0'):
            if not torch.equal(audio[key], curves[key]):
                raise ValueError('Audio/dynamics native binding differs: ' + key)
        audit = audit_frozen_cross_stage(run, audio, curves)
        audio_binding = {'path': str(audio_path), 'sha256': sha(audio_path), 'frozen_audit': audit}
    return curves, fields, {'dynamics': {'path': str(path), 'sha256': sha(path),
                            'completed_epochs': complete.get('completed_epochs')}, 'audio': audio_binding}


def bind_selection(previous, curves):
    provenance = read(previous / 'provenance.json')
    picks = provenance.get('nine_plot_clips', [])
    if (provenance.get('selection_uses_metadata_only') is not True
            or provenance.get('outcome_based_selection') is not False
            or len(picks) != 9 or len({p['clip_id'] for p in picks}) != 9):
        raise ValueError('Expected existing metadata-only nine-clip lock')
    mapping = {cid: i for i, cid in enumerate(curves['clip_id'])}
    if any(p['clip_id'] not in mapping for p in picks):
        raise ValueError('A locked clip is missing from saved final curves')
    reference_path = previous / 'nine_clip_curves.npz'
    inventory_path = previous / 'export_hashes.json'
    if inventory_path.exists() and read(inventory_path).get('nine_clip_curves.npz') != sha(reference_path):
        raise ValueError('Previous locked NPZ inventory differs')
    with np.load(reference_path, allow_pickle=False) as old:
        if old['channels'].tolist() != ARKIT_NAMES or old['clip_id'].tolist() != [p['clip_id'] for p in picks]:
            raise ValueError('Previous locked clip/channel order differs')
        gt_index = old['mode_names'].tolist().index('GT')
        for j, pick in enumerate(picks):
            i = mapping[pick['clip_id']]
            for key in ('times', 'valid', 'channel_mask'):
                if not np.array_equal(curves[key][i].numpy(), old[key][j]):
                    raise ValueError('Locked native clock or mask changed: ' + key)
            observed = old['valid'][j, :, None] & old['channel_mask'][j, None]
            if not np.array_equal(curves['target'][i].numpy()[observed], old['motions'][gt_index, j][observed]):
                raise ValueError('Locked GT coefficients changed')
    return provenance, [{**p, 'old_index': p['index'], 'index': mapping[p['clip_id']]} for p in picks]


def plot_examples(path, picks, motions, times, valid, channels, *, centered=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = (43, 41, 5, 17)
    colors = ('#111111', '#888888', '#bc7518', '#1568af', '#759838', '#b6367b')
    fig, axes = plt.subplots(4, 9, figsize=(25, 11), squeeze=False)
    for col, pick in enumerate(picks):
        for row, ch in enumerate(selected):
            axis = axes[row, col]
            mask = valid[col] & channels[col, ch]
            for k, mode in enumerate(MODES):
                values = motions[k, col, :, ch].astype(np.float64).copy()
                if centered and mask.any(): values -= values[mask].mean()
                values[~mask] = np.nan
                axis.plot(times[col] - times[col, 0], values, label=mode, color=colors[k], linewidth=1.)
            axis.grid(alpha=.18)
            if row == 0: axis.set_title(pick['speaker'].replace('mead_', '') + ' / ' + pick['emotion'], fontsize=8)
            if col == 0: axis.set_ylabel(ARKIT_NAMES[ch] + ('\ncentered coefficient' if centered else '\nraw coefficient'))
            if row == 3: axis.set_xlabel('Seconds from crop start')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc='upper center', bbox_to_anchor=(.5, .96))
    fig.suptitle('Fixed prior nine-clip lock | seed42 | oracle reads target motion | no gain or alignment fitting')
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(path, dpi=130); plt.close(fig)


def export(run, previous_visual, output, *, make_plots=True):
    run, previous, output = map(lambda p: Path(p).resolve(), (run, previous_visual, output))
    if output.exists(): raise FileExistsError('Fresh output directory required')
    curves, fields, bindings = load_saved(run)
    old_provenance, picks = bind_selection(previous, curves)
    indices = [p['index'] for p in picks]
    motions = torch.stack([value[indices] for value in fields]).numpy()
    times, valid, channels = (curves[key][indices].numpy() for key in ('times', 'valid', 'channel_mask'))
    old_videos = {v['clip']['clip_id']: v for v in old_provenance.get('videos', [])}
    selected_videos, seen = [], set()
    for pick in picks:
        if pick['speaker'] not in seen:
            selected_videos.append(pick); seen.add(pick['speaker'])
    audio_bindings = {}
    for pick in selected_videos:
        old = old_videos.get(pick['clip_id'])
        if old and old.get('audio_copy'):
            audio = local_child(previous, old['audio_copy'])
            binding = old.get('audio_binding', {})
            if audio.is_file():
                if sha(audio) != binding.get('audio_sha256'):
                    raise ValueError('Previously locked audio copy hash differs')
                offset = binding.get('audio_offset_s')
                if not isinstance(offset, (float, int)) or not np.isfinite(offset):
                    raise ValueError('Invalid previously locked audio offset')
                audio_bindings[pick['clip_id']] = (audio, binding)
    output.mkdir(parents=True)
    (output / 'video_npz').mkdir(); (output / 'audio').mkdir()
    np.savez_compressed(output / 'nine_clip_curves.npz', channels=np.asarray(ARKIT_NAMES),
        clip_id=np.asarray([p['clip_id'] for p in picks]), mode_names=np.asarray(MODES),
        times=times, valid=valid, channel_mask=channels, motions=motions, noise_seed=np.asarray(42))
    if make_plots:
        for centered in (False, True):
            plot_examples(output / ('centered_seed42.png' if centered else 'raw_seed42.png'), picks,
                          motions, times, valid, channels, centered=centered)
    if bindings['audio'] is not None:
        write(output / 'frozen_cross_stage_audit.json', bindings['audio']['frozen_audit'])
    videos, jobs = [], []
    for pick in selected_videos:
        j = indices.index(pick['index'])
        name = pick['speaker']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid locked speaker path')
        relative = Path('video_npz') / (name + '_first_locked.npz')
        audio_metadata, job_audio = {}, {}
        if pick['clip_id'] in audio_bindings:
            audio, binding = audio_bindings[pick['clip_id']]
            copied = Path('audio') / (name + audio.suffix)
            shutil.copy2(audio, output / copied)
            audio_metadata = {'audio_relative_path': np.asarray(copied.as_posix()),
                              'audio_sha256': np.asarray(binding['audio_sha256']),
                              'audio_offset_seconds': np.asarray(binding['audio_offset_s'])}
            job_audio = {'audio': copied.as_posix(), 'audio_sha256': binding['audio_sha256'],
                         'audio_offset_seconds': binding['audio_offset_s']}
        np.savez_compressed(output / relative, channels=np.asarray(ARKIT_NAMES), mode_names=np.asarray(MODES),
            clip_id=np.asarray(pick['clip_id']), noise_seed=np.asarray(42), times=times[j], valid=valid[j],
            channel_mask=channels[j], motions=motions[:, j], **audio_metadata)
        report = inspect_input(output / relative, 25)[-1]
        videos.append({'path': relative.as_posix(), 'clip': pick, 'sha256': sha(output / relative),
                       'display_report': report, 'rendered': False})
        jobs.append({'input': relative.as_posix(), 'output': name, 'fps': 25, 'columns': 3,
                     'tile_size': 480, 'samples': 32, 'max_frames': 0,
                     'expected_video': name + '/comparison.mp4', **job_audio})
    write(output / 'render_jobs.json', {'schema': 'full_staged_render_jobs_v1', 'rendered': False,
        'paths_relative_to': 'directory containing this JSON', 'driver': 'scripts/render_dynamic_rig_comparison.py',
        'layout': '2 rows x 3 columns, six ordered conditions', 'jobs': jobs})
    provenance = {'schema': SCHEMA, 'noise_seed': 42, 'source_curves': bindings,
        'previous_visual': str(previous), 'previous_provenance_sha256': sha(previous / 'provenance.json'),
        'previous_nine_npz_sha256': sha(previous / 'nine_clip_curves.npz'),
        'mode_names': list(MODES), 'mode_definitions': MODE_DEFINITIONS, 'nine_plot_clips': picks,
        'selection_uses_metadata_only': True, 'outcome_based_selection': False,
        'video_selection_rule': 'first clip per speaker from the unchanged previous nine-clip lock',
        'oracle_is_target_conditioned': True, 'nonupper_matches_stage4_exactly': True,
        'native_clock_copied_exactly': True, 'amplitude_rescaling': False, 'lag_alignment': False,
        'raw_coefficients_clamped': False, 'centered_plot': 'Per-clip per-mode valid coefficient mean removed for visualization only.',
        'videos': videos, 'actual_rendering_performed': False, 'model_loaded': False,
        'checkpoint_tensors_inspected_for_frozen_audit': bindings['audio'] is not None,
        'new_dataset_or_audio_lookup': False, 'test_loaded': False, 'script_sha256': sha(__file__),
        'limits': 'Internal development visualization only. Oracle is not audio-only. B0 is articulation only. '
                  'Identity, perceived emotion, lip synchronization and naturalness require separate review.'}
    write(output / 'provenance.json', provenance)
    links = ''.join('<li><a href="' + html.escape(v['path'], quote=True) + '">' + html.escape(v['clip']['clip_id']) + '</a></li>' for v in videos)
    images = '<img src="raw_seed42.png"><img src="centered_seed42.png">' if make_plots else ''
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Full staged fixed examples</title>'
        '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style><h2>固定九片训练结果</h2>'
        '<p>GT / B0 / stage4 base / final dynamic / static state / oracle state。'
        'oracle 使用真实动作慢状态，不能作为音频推理结果。此页仅导出曲线与待渲染输入，尚未生成视频。</p>'
        '<p><a href="render_jobs.json">待渲染任务</a> · <a href="provenance.json">来源和限制</a></p>'
        '<ul>' + links + '</ul>' + images, encoding='utf8')
    write(output / 'export_hashes.json', {p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'export_hashes.json'})
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'previous-visual', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = export(args.run, args.previous_visual, args.output)
    print(json.dumps({'output': str(args.output), 'clips': len(report['nine_plot_clips']), 'rendered': False}))


if __name__ == '__main__': main()
