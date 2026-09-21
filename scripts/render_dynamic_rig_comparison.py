"""Validate native ARKit curves and render a matched Blender comparison.

Input NPZ (allow_pickle=False): channels [52] strings, times [T] seconds,
valid [T] bool, mode_names [M] strings, motions [M,T,52]. Alternatively each
mode_names entry may name a [T,52] array. Optional clip_id/noise_seed metadata.
Optional channel_mask [52] or [T,52] bool marks observed coefficients. Missing
channels are disabled in the display only, never filled by model predictions.
This is display only: no model loading, inference, scoring, gain or lag fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


ARKIT_NAMES = [
    'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft',
    'eyeLookUpLeft', 'eyeSquintLeft', 'eyeWideLeft', 'eyeBlinkRight',
    'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight',
    'eyeSquintRight', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawRight',
    'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft',
    'mouthRight', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft',
    'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft',
    'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
    'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft',
    'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
    'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
    'noseSneerLeft', 'noseSneerRight', 'tongueOut',
]
BROWS = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_input(path, fps):
    with np.load(path, allow_pickle=False) as data:
        channels = data['channels'].tolist()
        times = np.asarray(data['times'], dtype=np.float64)
        valid = np.asarray(data['valid'])
        channel_mask = np.asarray(data['channel_mask']) if 'channel_mask' in data else None
        modes = data['mode_names'].tolist()
        if not isinstance(channels, list) or len(channels) != 52 or set(channels) != set(ARKIT_NAMES):
            raise ValueError('channels must contain each standard ARKit52 name exactly once')
        if not isinstance(modes, list) or not modes or len(modes) != len(set(modes)) or not all(isinstance(x, str) for x in modes):
            raise ValueError('mode_names must be distinct nonempty strings')
        values = np.asarray(data['motions'] if 'motions' in data else np.stack([data[x] for x in modes]), dtype=np.float32)
        metadata = {key: data[key].item() for key in ('clip_id', 'noise_seed') if key in data}
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError('times must be a nonempty finite vector')
    if len(times) > 1 and not np.allclose(np.diff(times), 1 / fps, atol=1e-7, rtol=1e-5):
        raise ValueError('Native times must be uniformly spaced at fps; no clock rescaling is performed')
    if valid.dtype != np.bool_ or valid.shape != times.shape or not valid.any():
        raise ValueError('valid must be a bool vector with at least one valid frame')
    if channel_mask is not None and (channel_mask.dtype != np.bool_ or channel_mask.shape not in ((52,), (len(times), 52))):
        raise ValueError('channel_mask must be a bool [52] or [T,52] array in channels order')
    support = np.ones((len(times), 52), dtype=bool) if channel_mask is None else np.broadcast_to(channel_mask, (len(times), 52))
    observed_mask = valid[:, None] & support
    if values.shape != (len(modes), len(times), 52) or not np.isfinite(values[:, observed_mask]).all():
        raise ValueError('motions must be finite [M,T,52] at observed frames')
    # Nearest observed native index; ties go to the earlier observed frame.
    good = np.flatnonzero(valid)
    nearest = good[np.abs(np.arange(len(times))[:, None] - good[None, :]).argmin(axis=1)]
    # An unobserved coefficient is not a prediction with validated semantics.
    # Clear it before display arithmetic so arbitrary values or NaNs from an
    # unsupervised output head cannot animate the rig or pollute the audit.
    filled = np.where(support[nearest][None], values[:, nearest, :], 0.)
    display = np.clip(filled, 0, 1)
    counts = {}
    for i, mode in enumerate(modes):
        observed = values[i][observed_mask]
        unsupported_mask = valid[:, None] & ~support
        unsupported = values[i][unsupported_mask]
        unsupported_finite = np.isfinite(unsupported)
        outside = (observed < 0) | (observed > 1)
        per_channel_count = observed_mask.sum(0)
        per_channel_outside = (((values[i] < 0) | (values[i] > 1)) & observed_mask).sum(0)
        counts[mode] = {
            'observed_value_count': int(observed.size),
            'observed_clamped_count': int(outside.sum()),
            'observed_clamped_fraction': float(outside.mean()) if observed.size else None,
            'raw_observed_min': float(observed.min()) if observed.size else None,
            'raw_observed_max': float(observed.max()) if observed.size else None,
            'clamped_fraction_by_channel': {name: float(per_channel_outside[c] / per_channel_count[c]) if per_channel_count[c] else None
                                            for c, name in enumerate(channels)},
            'unsupported_raw_value_count': int(unsupported.size),
            'unsupported_raw_finite_nonzero_count': int((unsupported[unsupported_finite] != 0).sum()),
            'unsupported_raw_nonfinite_count': int((~unsupported_finite).sum()),
            'unsupported_raw_finite_abs_max': float(np.abs(unsupported[unsupported_finite]).max()) if unsupported_finite.any() else None,
            'brow_display_std': {name: float(display[i, valid, channels.index(name)].std()) for name in BROWS},
        }
    return channels, times, valid, modes, display, {
        'metadata': metadata, 'frames': len(times), 'fps': fps,
        'native_start_seconds': float(times[0]), 'native_last_seconds': float(times[-1]),
        'valid_frames': int(valid.sum()), 'display_filled_frames': int((~valid).sum()),
        'filled_native_indices': np.flatnonzero(~valid).tolist(),
        'display_source_index': nearest.tolist(), 'display_clamp': '[0,1], display only',
        'channel_mask_provided': channel_mask is not None,
        'channel_mask_shape': list(channel_mask.shape) if channel_mask is not None else None,
        'unsupported_channels': [name for c, name in enumerate(channels) if not support[valid, c].any()],
        'partially_supported_channels': [name for c, name in enumerate(channels) if support[valid, c].any() and not support[valid, c].all()],
        'display_channel_policy': 'Zero unobserved channels using channel_mask; legacy inputs without channel_mask assume all channels observed. Raw source arrays and scores are unchanged.',
        'invalid_policy': 'nearest valid native frame; tie earlier; retained timeline; exclude from scoring',
        'mode_statistics': counts,
    }


def smoke_input(path, fps):
    values = np.zeros((6, 3, 52), dtype=np.float32)
    modes = ['Neutral control', 'Raise brows control', 'Lower brows control', 'Left brow control', 'Right brow control', 'Jaw control']
    for name in ('browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'):
        values[1, :, ARKIT_NAMES.index(name)] = 1
    for name in ('browDownLeft', 'browDownRight'):
        values[2, :, ARKIT_NAMES.index(name)] = 1
    values[3, :, ARKIT_NAMES.index('browOuterUpLeft')] = 1
    values[4, :, ARKIT_NAMES.index('browOuterUpRight')] = 1
    values[5, :, ARKIT_NAMES.index('jawOpen')] = .5
    np.savez_compressed(path, channels=np.asarray(ARKIT_NAMES), times=np.arange(3) / fps,
                        valid=np.ones(3, dtype=bool), mode_names=np.asarray(modes), motions=values,
                        clip_id=np.asarray('SYNTHETIC_RIG_CAPACITY_CONTROL_NOT_MODEL_OUTPUT'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blend', type=Path, default=Path('D:/实验室项目/新实验/arkit2.blend'))
    parser.add_argument('--blender', type=Path, default=Path('D:/3d_engine/blender-4.3.0-windows-x64/blender.exe'))
    parser.add_argument('--object', default='face.001')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--tile-size', type=int, default=480)
    parser.add_argument('--columns', type=int, default=3)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--max-frames', type=int, default=0, help='Prefix-only smoke render; 0 renders every frame')
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--audio-offset-seconds', type=float, default=0)
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    parser.add_argument('--smoke', action='store_true', help='Create synthetic rig controls, never model outputs')
    args = parser.parse_args()
    if args.fps <= 0 or args.tile_size < 128 or args.columns < 1 or args.samples < 1 or args.max_frames < 0:
        parser.error('invalid rendering dimensions/fps/sample count')
    if args.output.exists():
        parser.error('output must be a fresh directory; existing results are never overwritten')
    if not args.blend.is_file() or not args.blender.is_file() or not args.ffmpeg:
        parser.error('blend, Blender and ffmpeg are required')
    if bool(args.input) == bool(args.smoke):
        parser.error('use exactly one of --input and --smoke')
    if args.audio and not args.audio.is_file():
        parser.error('audio file does not exist')
    if not np.isfinite(args.audio_offset_seconds):
        parser.error('audio offset must be finite')
    args.output.mkdir(parents=True)
    if args.smoke:
        args.input = args.output / 'synthetic_capacity_input.npz'
        smoke_input(args.input, args.fps)
    channels, times, valid, modes, values, report = inspect_input(args.input, args.fps)
    frames = len(times) if not args.max_frames else min(args.max_frames, len(times))
    prepared = args.output / 'display_curves.npz'
    np.savez_compressed(prepared, channels=np.asarray(channels), times=times[:frames], valid=valid[:frames],
                        mode_names=np.asarray(modes), motions=values[:, :frames])
    before = sha(args.blend)
    worker = Path(__file__).with_name('blender_render_dynamic_rig.py')
    command = [str(args.blender), '--background', '--factory-startup', '--disable-autoexec',
               str(args.blend), '--python', str(worker.resolve()), '--', '--input', str(prepared.resolve()),
               '--output', str(args.output.resolve()), '--object', args.object,
               '--fps', str(args.fps), '--tile-size', str(args.tile_size), '--columns', str(args.columns),
               '--samples', str(args.samples)]
    report.update(schema='dynamic_rig_display_v1', scope='Display only; not coefficient scoring or perceptual validation',
                  synthetic_capacity_control=args.smoke, input_sha256=sha(args.input), blend_sha256=before,
                  worker_sha256=sha(worker), driver_sha256=sha(__file__), command=command,
                  rendered_frames=frames, rendered_modes=modes, shared_clock=True,
                  shared_noise='not generated here; see source metadata', native_coefficients_modified=False)
    try:
        with (args.output / 'blender.log').open('w', encoding='utf8') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        if sha(args.blend) != before:
            raise RuntimeError('Source blend changed during rendering')
        expected = [args.output / 'frames' / f'{i:06d}.png' for i in range(1, frames + 1)]
        if not all(p.is_file() for p in expected):
            raise RuntimeError('Blender exited without complete frame output; inspect blender.log')
        shutil.copy2(expected[0], args.output / 'preview.png')
        ff = [args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(args.fps),
              '-start_number', '1', '-i', str(args.output / 'frames' / '%06d.png')]
        start = float(times[0]) + args.audio_offset_seconds
        if args.audio:
            if start < 0:
                ff += ['-i', str(args.audio), '-filter_complex', f'[1:a]adelay={round(-start * 1000)}:all=1,apad[a]', '-map', '0:v', '-map', '[a]']
            else:
                ff += ['-ss', f'{start:.9f}', '-i', str(args.audio), '-map', '0:v', '-map', '1:a:0', '-af', 'apad']
            ff += ['-c:a', 'aac', '-b:a', '192k']
        ff += ['-t', f'{frames / args.fps:.9f}', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(args.output / 'comparison.mp4')]
        subprocess.run(ff, check=True)
        report.update(status='complete', original_blend_unchanged=True, ffmpeg_command=ff,
                      audio_sha256=sha(args.audio) if args.audio else None,
                      audio_clock='audio time = native motion time + offset',
                      audio_trim_start_seconds=start if args.audio else None,
                      output_video_sha256=sha(args.output / 'comparison.mp4'))
    except Exception as exc:
        report.update(status='failed', error=str(exc), original_blend_unchanged=sha(args.blend) == before)
        raise
    finally:
        (args.output / 'display_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps({'output': str(args.output.resolve()), 'frames': frames, 'modes': modes}, ensure_ascii=True))


if __name__ == '__main__':
    main()
