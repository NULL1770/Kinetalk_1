"""Reproduce fixed controlled-prior rig videos; never sample or pick outcomes.

Fullface: first prelocked query per person/emotion (16), native clock + audio.
Controls: first prelocked query per person (4), saved seed42 control schedule;
43 other channels are held at the frozen baseline's first valid frame. No GT
trajectory or audio drives the control demo. Existing renders are reused only
after source, settings, display arrays, complete frames and video hash checks.
Partial/mismatched directories fail closed; nothing is deleted or overwritten.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_dynamic_rig_comparison as render_driver


SCHEMA = 'controlled_prior_reproducible_render_v1'
PRIOR_SCHEMA = 'independent_reference_continuous_prior_v1'
EXPORT_SCHEMA = 'controlled_prior_frozen_fullface_diagnostic_v1'
UPPER = [41, 42, 43, 44, 45, 5, 6, 12, 13]
OTHER = [i for i in range(52) if i not in UPPER]
CONTROL_MODES = ['Frozen level control', 'RUN HOLD RELEASE RUN SWAP']
FULLFACE_MODES = ['GT', 'run12 audio baseline', 'bounded medoid prior',
                  'controlled stochastic prior', 'independent reference swap']
DEFAULT_BLEND = Path('D:/实验室项目/新实验/arkit2.blend')
DEFAULT_BLENDER = Path('D:/3d_engine/blender-4.3.0-windows-x64/blender.exe')
FPS, TILE, SAMPLES = 25, 320, 8
sha = render_driver.sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def safe_child(root, relative):
    root = Path(root).resolve(); value = (root/relative).resolve()
    if value == root or root not in value.parents:
        raise ValueError('Input/output must remain inside its declared directory: '+str(relative))
    return value


def fixed_selections(picks):
    fullface, controls, seen_pair, seen_person = [], [], set(), set()
    for row in picks:
        cid = row['clip_id']
        if not isinstance(cid, str) or not cid or '/' in cid or '\\' in cid or cid in ('.', '..'):
            raise ValueError('Safe clip basename required')
        key = (row['speaker'], row['emotion'])
        if key not in seen_pair:
            fullface.append(row); seen_pair.add(key)
        if row['speaker'] not in seen_person:
            controls.append(row); seen_person.add(row['speaker'])
    if len(fullface) != 16 or len(controls) != 4:
        raise ValueError('Expected the fixed4-person/4-emotion metadata selection')
    return fullface, controls


def load_sources(prior, export):
    protocol, status = read(prior/'protocol.json'), read(prior/'status.json')
    selection = read(prior/'query_selection.json'); provenance = read(export/'provenance.json')
    if (protocol.get('schema') != PRIOR_SCHEMA or status.get('schema') != PRIOR_SCHEMA
            or status.get('status') != 'complete' or status.get('smoke') is not False
            or protocol.get('smoke') is not False or protocol.get('extra_expression_reference') is not True
            or protocol.get('audio_timing_claim') is not False or status.get('test_loaded') is not False
            or provenance.get('schema') != EXPORT_SCHEMA or provenance.get('prior_status') != status
            or provenance.get('prior_seed') != 42 or provenance.get('nonupper43_exact') is not True
            or provenance.get('query_teacher_inference') is not False
            or provenance.get('test_loaded') is not False or provenance.get('audio_timing_claim') is not False):
        raise ValueError('Completed formal independent-reference run and bound fullface export required')
    required = {'protocol.json', 'status.json', 'predictions.pt', 'query_selection.json', 'references.json'}
    if set(provenance['source_prior_sha256']) != required:
        raise ValueError('Exporter must bind all five prior inputs')
    for name, digest in provenance['source_prior_sha256'].items():
        if sha(prior/name) != digest:
            raise ValueError('Exporter/prior source mismatch: '+name)
    picks = selection['queries']; ids = [r['clip_id'] for r in picks]
    if (len(ids) != 32 or len(set(ids)) != 32 or ids != protocol['query_clip_ids']
            or [r['clip_id'] for r in provenance['clips']] != ids
            or protocol['seeds'][0] != 42 or len(protocol['seeds']) != 8):
        raise ValueError('Exact32 prelocked queries and fixed first seed42 required')
    exported = {r['clip_id']: r for r in provenance['clips']}
    for pick in picks:
        row = exported[pick['clip_id']]
        if any(pick[k] != row[k] for k in ('speaker', 'emotion', 'sentence')):
            raise ValueError('Fullface metadata differs from prior')
        path = safe_child(export, row['npz'])
        if sha(path) != row['sha256']:
            raise ValueError('Exported input checksum differs: '+pick['clip_id'])
    saved = torch.load(prior/'predictions.pt', map_location='cpu', weights_only=False)
    fullface, controls = fixed_selections(picks)
    if (saved.get('schema') != PRIOR_SCHEMA or set(saved['curves']) != set(ids)
            or set(saved['long']) != {r['clip_id'] for r in controls}):
        raise ValueError('Saved controls must be first metadata query per person')
    return protocol, provenance, saved, exported, fullface, controls


def verify_fullface_input(path, audio, row, saved):
    cid = row['clip_id']; curves = saved['curves'][cid]
    with np.load(path, allow_pickle=False) as data:
        channels = data['channels'].tolist(); modes = data['mode_names'].tolist()
        motions = data['motions']; valid = data['valid']
        if (channels != render_driver.ARKIT_NAMES or modes != FULLFACE_MODES
                or motions.shape != (5, row['native_frames'], 52)
                or not np.array_equal(valid, np.asarray(curves['valid']))
                or data['noise_seed'].item() != 42 or data['clip_id'].item() != cid):
            raise ValueError('Frozen fullface clock/channel/metadata mismatch: '+cid)
        baseline = motions[1]
        for index, arm in enumerate(('medoid', 'process', 'reference_swap'), start=2):
            if (not np.array_equal(motions[index][:, OTHER], baseline[:, OTHER])
                    or not np.array_equal(motions[index][~valid], baseline[~valid])
                    or not np.array_equal(motions[index][:, UPPER][valid], np.asarray(curves['samples'][arm])[0, valid])):
                raise ValueError('Protected channels or seed42 prior differ: '+cid)
    expected_audio = row['native_metadata']['provenance']['audio_sha256']
    if sha(audio) != expected_audio:
        raise ValueError('Native query audio checksum differs: '+cid)


def make_control_input(path, fullface_input, cid, saved):
    """Use frozen baseline only; neither GT nor the query motion enters demo."""
    with np.load(fullface_input, allow_pickle=False) as data:
        channels = data['channels'].copy(); modes = data['mode_names'].tolist()
        valid = data['valid']
        if channels.tolist() != render_driver.ARKIT_NAMES or modes != FULLFACE_MODES or not valid.any():
            raise ValueError('Canonical frozen fullface baseline required')
        first_valid = int(np.flatnonzero(valid)[0])
        baseline = data['motions'][modes.index('run12 audio baseline'), first_valid].copy()
    controlled = np.asarray(saved['long'][cid]['controls'])
    if controlled.shape != (512, 9) or not np.isfinite(controlled).all():
        raise ValueError('Complete saved512-frame seed42 control schedule required')
    motions = np.tile(baseline, (2, 512, 1)); motions[1][:, UPPER] = controlled
    expected = {'channels': channels, 'mode_names': np.asarray(CONTROL_MODES),
        'clip_id': np.asarray(cid+'_EXPLICIT_CONTROL_DEMO'), 'noise_seed': np.asarray(42),
        'times': np.arange(512)/FPS, 'valid': np.ones(512, bool), 'motions': motions}
    if not np.array_equal(motions[1][:, OTHER], np.broadcast_to(baseline[OTHER], (512, 43))):
        raise ValueError('Control construction changed a protected channel')
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(expected) or any(
                    existing[k].dtype != v.dtype or not np.array_equal(existing[k], v) for k, v in expected.items()):
                raise ValueError('Existing control NPZ differs; refusing overwrite: '+str(path))
    else:
        np.savez_compressed(path, **expected)
    return {'baseline_first_valid_index': first_valid, 'nonupper43_exact': True,
            'query_gt_used': False, 'audio_used': False, 'frames': 512,
            'baseline_input_sha256': sha(fullface_input), 'control_input_sha256': sha(path)}


def option(command, flag):
    return command[command.index(flag)+1] if flag in command else None


def verify_render(destination, source, audio, columns, args):
    """A complete status alone cannot authorize skipping stale output."""
    report_path = destination/'display_report.json'
    report = read(report_path)
    channels, times, valid, modes, display, inspected = render_driver.inspect_input(source, FPS)
    worker = Path(render_driver.__file__).with_name('blender_render_dynamic_rig.py')
    expected = {'schema': 'dynamic_rig_display_v1', 'status': 'complete', 'input_sha256': sha(source),
        'blend_sha256': sha(args.blend), 'worker_sha256': sha(worker), 'driver_sha256': sha(render_driver.__file__),
        'audio_sha256': sha(audio) if audio else None, 'frames': len(times), 'rendered_frames': len(times),
        'fps': FPS, 'rendered_modes': modes, 'original_blend_unchanged': True,
        'synthetic_capacity_control': False}
    if any(report.get(k) != value for k, value in expected.items()):
        raise ValueError('Existing render status/source/settings differ: '+str(destination))
    command = report['command']
    for key, value in (('--tile-size', str(TILE)), ('--columns', str(columns)), ('--samples', str(SAMPLES)),
                       ('--fps', str(FPS)), ('--object', 'face.001')):
        if option(command, key) != value:
            raise ValueError('Existing render option differs: '+key)
    if Path(command[0]).resolve() != args.blender.resolve():
        raise ValueError('Existing Blender executable differs')
    video = destination/'comparison.mp4'
    if not video.is_file() or sha(video) != report.get('output_video_sha256'):
        raise ValueError('Completed video checksum missing or mismatched: '+str(destination))
    ff = report['ffmpeg_command']
    if Path(ff[0]).resolve() != Path(args.ffmpeg).resolve() or option(ff, '-t') != f'{len(times)/FPS:.9f}':
        raise ValueError('Existing ffmpeg executable/duration differs')
    with np.load(destination/'display_curves.npz', allow_pickle=False) as data:
        for key, value in {'channels': np.asarray(channels), 'times': times, 'valid': valid,
                           'mode_names': np.asarray(modes), 'motions': display}.items():
            if not np.array_equal(data[key], value):
                raise ValueError('Saved display arrays no longer match input: '+key)
    frame_paths = [destination/'frames'/f'{i:06d}.png' for i in range(1, len(times)+1)]
    if not all(p.is_file() and p.stat().st_size > 0 for p in frame_paths):
        raise ValueError('Completed render has missing/empty native frames')
    preview = destination/'preview.png'
    if not preview.is_file() or sha(preview) != sha(frame_paths[0]):
        raise ValueError('Preview differs from first native render frame')
    audit = read(destination/'rig_audit.json')
    if audit.get('schema') != 'blender_dynamic_rig_v1' or set(audit['mapping']) != set(channels):
        raise ValueError('Complete rig mapping audit required')
    return {'status': 'complete', 'input_sha256': sha(source), 'audio_sha256': sha(audio) if audio else None,
        'output_video_sha256': sha(video), 'display_report_sha256': sha(report_path),
        'display_curves_sha256': sha(destination/'display_curves.npz'), 'preview_sha256': sha(preview),
        'rig_audit_sha256': sha(destination/'rig_audit.json'), 'frames': inspected['frames'],
        'modes': modes, 'video': str(video.resolve())}


def render_or_verify(source, destination, audio, columns, args):
    reused = destination.exists()
    if not reused:
        command = [sys.executable, str(Path(render_driver.__file__).resolve()), '--input', str(source),
            '--output', str(destination), '--blend', str(args.blend), '--blender', str(args.blender),
            '--columns', str(columns), '--tile-size', str(TILE), '--samples', str(SAMPLES),
            '--ffmpeg', str(args.ffmpeg)]
        if audio is not None:
            command += ['--audio', str(audio)]
        log = destination.with_name(destination.name+'.render.log')
        with log.open('w', encoding='utf8') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    # This verifies fresh output too; existing partial/stale directories fail.
    result = verify_render(destination, source, audio, columns, args)
    result['reused_verified_output'] = reused
    return result


def control_index(output, clips):
    body = '<h1>连续运动：显式控制演示</h1><p>固定 seed 42：0–5.12 秒运行，5.12–7.68 秒保持，7.68–10.24 秒释放到均衡，10.24–15.36 秒运行，15.36 秒更换独立参考。保持/释放分别包含 0.32 / 0.64 秒过渡。</p>'
    body += '<p><strong>没有音频，也没有查询 GT 驱动。</strong>两个面板的其余 43 通道均固定在冻结音频基线首个有效帧，左侧完整基线保持，右侧仅替换保存的九维眉眼控制轨迹。这是独立表达参考和显式指令演示，不能代表自然动作事件或口型同步已经学会。</p>'
    for cid in clips:
        safe = html.escape(cid, quote=True)
        body += '<article><h2>'+safe+'</h2><video controls preload="none" poster="'+safe+'/preview.png" src="'+safe+'/comparison.mp4"></video></article>'
    body += '<p>选择与输入输出校验见 <a href="render_selection.json">render_selection.json</a> 和 <a href="render_provenance.json">render_provenance.json</a>。</p>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>显式控制演示</title><style>body{font:16px/1.7 system-ui,sans-serif;margin:32px;background:#edf2f7;color:#263448}main{max-width:1280px;margin:auto}article{padding:20px;background:white;margin:24px 0}video{width:100%;height:auto}h2{overflow-wrap:anywhere;font-size:18px}</style><main>'+body+'</main></html>'
    (output/'index.html').write_text(page, encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-run', type=Path, required=True)
    parser.add_argument('--fullface-export', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--blender', type=Path, default=DEFAULT_BLENDER)
    parser.add_argument('--blend', type=Path, default=DEFAULT_BLEND)
    parser.add_argument('--mode', choices=('fullface', 'controls', 'all'), default='all')
    args = parser.parse_args()
    for name in ('prior_run', 'fullface_export', 'output_root', 'ffmpeg', 'blender', 'blend'):
        setattr(args, name, getattr(args, name).resolve())
    for name in ('ffmpeg', 'blender', 'blend'):
        if not getattr(args, name).is_file():
            parser.error(name+' must name an existing file')
    if (args.output_root == args.prior_run or args.output_root == args.fullface_export
            or args.prior_run in args.output_root.parents or args.fullface_export in args.output_root.parents):
        parser.error('Output root must be separate from source artifact directories')
    protocol, provenance, saved, exported, fullface, controls = load_sources(args.prior_run, args.fullface_export)
    args.output_root.mkdir(parents=True, exist_ok=True)
    selection = {'schema': SCHEMA, 'fullface_rule': 'First prelocked query per person/emotion;16 total',
        'control_rule': 'First prelocked query per person;4 total', 'metadata_only': True,
        'outcome_selection': False, 'seed': 42, 'fullface': [r['clip_id'] for r in fullface],
        'controls': [r['clip_id'] for r in controls]}
    selection_path = args.output_root/'controlled_render_selection.json'
    if selection_path.exists() and read(selection_path) != selection:
        raise ValueError('Existing render selection differs')
    save(selection_path, selection)
    status_path = args.output_root/('controlled_render_status_'+args.mode+'.json')
    status = {'schema': SCHEMA, 'status': 'running', 'mode': args.mode, 'results': []}
    save(status_path, status)
    binding = {'prior_files_sha256': provenance['source_prior_sha256'],
        'export_provenance_sha256': sha(args.fullface_export/'provenance.json'),
        'wrapper_sha256': sha(__file__), 'driver_sha256': sha(render_driver.__file__),
        'blend_sha256': sha(args.blend), 'blender': str(args.blender), 'ffmpeg': str(args.ffmpeg),
        'fps': FPS, 'tile_size': TILE, 'samples': SAMPLES, 'seed': 42,
        'extra_expression_reference': True, 'audio_timing_claim': False,
        'perceptual_naturalness_certified': False}
    try:
        if args.mode in ('fullface', 'all'):
            folder = args.output_root/'fullface'; folder.mkdir(exist_ok=True)
            rows = []
            save(folder/'render_selection.json', {'schema': SCHEMA, 'rule': selection['fullface_rule'], 'clips': selection['fullface']})
            for pick in fullface:
                cid = pick['clip_id']; row = exported[cid]
                source = safe_child(args.fullface_export, row['npz'])
                audio = safe_child(args.fullface_export, 'audio/'+cid+'.wav')
                verify_fullface_input(source, audio, row, saved)
                result = {'clip_id': cid, **render_or_verify(source, folder/cid, audio, 3, args)}
                rows.append(result); status['results'].append({'mode': 'fullface', **result}); save(status_path, status)
                print('VERIFIED_FULLFACE', cid, flush=True)
            save(folder/'render_provenance.json', {'schema': SCHEMA, 'status': 'complete', **binding, 'clips': rows,
                'nonupper43_exact': True, 'audio_used': True, 'columns': 3, 'query_gt_panel_only': True})
        if args.mode in ('controls', 'all'):
            folder = args.output_root/'controls'; folder.mkdir(exist_ok=True)
            rows = []
            save(folder/'render_selection.json', {'schema': SCHEMA, 'rule': selection['control_rule'], 'clips': selection['controls']})
            for pick in controls:
                cid = pick['clip_id']; source = folder/(cid+'.npz')
                construction = make_control_input(source, safe_child(args.fullface_export, exported[cid]['npz']), cid, saved)
                result = {'clip_id': cid, 'construction': construction, **render_or_verify(source, folder/cid, None, 2, args)}
                rows.append(result); status['results'].append({'mode': 'controls', **result}); save(status_path, status)
                print('VERIFIED_CONTROLS', cid, flush=True)
            save(folder/'render_provenance.json', {'schema': SCHEMA, 'status': 'complete', **binding, 'clips': rows,
                'no_audio': True, 'query_gt_used': False, 'nonupper43_exact': True,
                'static_nonupper': 'First valid frozen baseline frame held; no speaking-lip claim',
                'schedule_frames': [128, 192, 256, 384], 'hold_transition_frames': 8,
                'release_transition_frames': 16, 'frames': 512, 'columns': 2})
            control_index(folder, selection['controls'])
        status['status'] = 'complete'; save(status_path, status)
    except Exception as exc:
        status.update(status='failed', error=str(exc)); save(status_path, status)
        raise
    print(json.dumps({'status': 'complete', 'mode': args.mode, 'verified_renders': len(status['results']),
                      'output_root': str(args.output_root)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
