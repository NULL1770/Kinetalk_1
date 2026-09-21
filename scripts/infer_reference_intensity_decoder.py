"""Run the reference-intensity upper-face adapter on precomputed inputs.

Input NPZ: prior [T,52] float32, audio_features [T,1540], global_code [64],
identity_code [128], independent neutral anchor [52], valid [T] and 25-fps
times [T]. Optional channels/channel_mask/clip_id/noise_seed are retained.
Only these inputs are loaded; query motion, emotion and target arrays are not.
This is not a raw-WAV or complete reference-enrollment inference entry point.

Use only checkpoints produced by this repository's training runner. Loading
uses weights_only=True and validates the schema, statistics and protocol hash;
these checks establish compatibility, not external provenance authentication.
Output modes: original_prior, smooth_prior, full, static. The last is the
student's static-temporal ablation. Other43 channels and inactive frames retain
the supplied prior exactly. Upper means may change; no coefficient clipping or
post-composition smoothing is applied to the learned outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinetalk_b0.models.reference_intensity_decoder import (
    ReferenceIntensityDecoder, ReferenceIntensityStudent, neutral_relative_intensity,
)
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face
from kinetalk_b0.models.temporal_motion_carrier import smooth_motion_carrier
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES

SCHEMA = 'reference_intensity_decoder_v1'
MODE_NAMES = ('original_prior', 'smooth_prior', 'full', 'static')
WINDOW = 5
FPS = 25
NOT_UPPER = tuple(i for i in range(52) if i not in UPPER_INDICES)
MOUTH = tuple(range(14, 41))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def load_models(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('schema') != SCHEMA:
        raise ValueError('Unsupported reference-intensity checkpoint schema')
    protocol = checkpoint.get('protocol')
    if (not isinstance(protocol, dict) or protocol.get('schema') != SCHEMA
            or protocol.get('window') != WINDOW or protocol.get('upper_mean_protected') is not False):
        raise ValueError('Unsupported reference-intensity training protocol')
    if checkpoint.get('protocol_sha256') != _canonical_hash(protocol):
        raise ValueError('Checkpoint protocol hash differs')
    configs = {}
    for key, expected_keys in (('decoder_config', {'global_dim', 'identity_dim', 'hidden'}),
                                ('student_config', {'input_dim', 'global_dim', 'identity_dim', 'hidden'})):
        value = checkpoint.get(key)
        if (not isinstance(value, dict) or set(value) != expected_keys
                or any(type(item) is not int or item < 1 for item in value.values())
                or value['global_dim'] != 64 or value['identity_dim'] != 128):
            raise ValueError(f'Invalid checkpoint configuration: {key}')
        configs[key] = value
    reduced_width = configs['student_config']['input_dim']
    rank = reduced_width - 4
    if rank < 1 or rank > 24 or protocol.get('pca_rank') != rank:
        raise ValueError('PCA rank differs from the training protocol or student width')
    stats = checkpoint.get('stats')
    if not isinstance(stats, dict):
        raise ValueError('Checkpoint needs fitted statistics')
    shapes = {'feature_mean': (1540,), 'feature_std': (1540,), 'projection': (1536, rank),
              'reduced_mean': (reduced_width,), 'reduced_std': (reduced_width,),
              'global_code_mean': (64,), 'global_code_std': (64,),
              'identity_code_mean': (128,), 'identity_code_std': (128,), 'scales9': (9,)}
    for key, shape in shapes.items():
        value = stats.get(key)
        if (not torch.is_tensor(value) or value.shape != shape or value.dtype != torch.float32
                or not torch.isfinite(value).all()
                or ((key.endswith('_std') or key == 'scales9') and (value <= 0).any())):
            raise ValueError(f'Invalid checkpoint statistic: {key}')
    decoder = ReferenceIntensityDecoder(**configs['decoder_config'])
    student = ReferenceIntensityStudent(**configs['student_config'])
    for key, model in (('decoder', decoder), ('student', student)):
        state = checkpoint.get(key)
        if (not isinstance(state, dict) or any(not torch.is_tensor(value)
                or value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values())):
            raise ValueError(f'Invalid or nonfinite {key} weights')
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f'{key} weights differ from configuration') from exc
    return decoder.to(device).eval(), student.to(device).eval(), stats, protocol


def load_input(path):
    required = ('prior', 'audio_features', 'global_code', 'identity_code', 'anchor', 'valid', 'times')
    with np.load(path, allow_pickle=False) as source:
        missing = set(required) - set(source.files)
        if missing:
            raise ValueError('Missing inference inputs: ' + ', '.join(sorted(missing)))
        values = {key: source[key] for key in required}
        for key in ('channels', 'channel_mask', 'clip_id', 'noise_seed'):
            if key in source.files:
                values[key] = source[key]
    prior, features, valid, times = (values[key] for key in ('prior', 'audio_features', 'valid', 'times'))
    if prior.dtype != np.float32 or prior.ndim != 2 or prior.shape[1:] != (52,) or len(prior) < 1:
        raise ValueError('prior must be nonempty float32 [T,52]')
    frames = len(prior)
    if valid.dtype != np.bool_ or valid.shape != (frames,) or not valid.any():
        raise ValueError('valid must be Boolean [T] with an observed frame')
    if features.shape != (frames, 1540) or not np.issubdtype(features.dtype, np.floating):
        raise ValueError('audio_features must be floating [T,1540]')
    if (times.shape != (frames,) or not np.issubdtype(times.dtype, np.floating)
            or not np.isfinite(times).all() or (frames > 1 and not np.allclose(
                np.diff(times.astype(np.float64)), 1 / FPS, atol=1e-7, rtol=1e-5))):
        raise ValueError('times must be finite, uniformly spaced native 25-fps seconds [T]')
    if 'channels' in values and values['channels'].tolist() != ARKIT_NAMES:
        raise ValueError('Input channels must use canonical ARKit52 order')
    upper_valid = valid.copy()
    support = np.ones((frames, 52), dtype=bool)
    if 'channel_mask' in values:
        support = values['channel_mask']
        if support.dtype != np.bool_ or support.shape not in ((52,), (frames, 52)):
            raise ValueError('channel_mask must be Boolean [52] or [T,52]')
        support = np.broadcast_to(support, (frames, 52))
        upper_valid &= support[:, list(UPPER_INDICES)].all(-1)
    if not upper_valid.any():
        raise ValueError('At least one fully observed upper9 frame is required')
    if (not np.isfinite(prior[valid[:, None] & support]).all()
            or not np.isfinite(prior[upper_valid]).all()
            or not np.isfinite(features[upper_valid]).all()):
        raise ValueError('Observed prior and audio features must be finite')
    for key, width in (('global_code', 64), ('identity_code', 128), ('anchor', 52)):
        value = values[key]
        if value.shape != (width,) or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f'{key} must be finite floating [{width}] precomputed deployment condition')
        values[key] = value.astype(np.float32, copy=False)
        if not np.isfinite(values[key]).all():
            raise ValueError(f'{key} cannot be represented in float32')
    for key in ('clip_id', 'noise_seed'):
        if key in values and values[key].shape != ():
            raise ValueError(f'{key} must be scalar metadata')
    if 'clip_id' in values and values['clip_id'].dtype.kind not in ('U', 'S'):
        raise ValueError('clip_id must be string metadata')
    if 'noise_seed' in values and values['noise_seed'].dtype.kind not in ('i', 'u'):
        raise ValueError('noise_seed must be integer metadata')
    values['audio_features'] = features.astype(np.float32, copy=False)
    if not np.isfinite(values['audio_features'][upper_valid]).all():
        raise ValueError('Observed audio features cannot be represented in float32')
    values['upper_valid'] = upper_valid
    return values


def _bit_exact(left, right):
    return bool(torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32)))


def _mean(value, valid):
    return torch.where(valid[..., None], value, 0.).sum(1) / valid.sum(1, keepdim=True)


def prepare_inputs(inputs, stats):
    """Match the training runner's CPU preprocessing and fitted units."""
    valid = torch.from_numpy(inputs['upper_valid'])[None]
    features = torch.from_numpy(inputs['audio_features'])[None]
    clean = torch.where(valid[..., None], features, stats['feature_mean'])
    x = (clean - stats['feature_mean']) / stats['feature_std']
    x = torch.cat((x[..., :1536] @ stats['projection'], x[..., 1536:]), -1)
    reduced = torch.where(valid[..., None], (x - stats['reduced_mean']) / stats['reduced_std'], 0.)
    conditions = {'features': reduced, 'valid': valid, 'anchor': torch.from_numpy(inputs['anchor'])[None]}
    for key in ('global_code', 'identity_code'):
        conditions[key] = (torch.from_numpy(inputs[key])[None] - stats[key + '_mean']) / stats[key + '_std']
    if any(not torch.isfinite(value).all() for value in conditions.values()):
        raise ValueError('Nonfinite prepared inference input')
    return conditions


@torch.inference_mode()
def infer(checkpoint_path, input_path, *, device='cpu'):
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    decoder, student, stats, protocol = load_models(checkpoint_path, device)
    inputs = load_input(input_path)
    prepared = prepare_inputs(inputs, stats)
    b = {key: value.to(device) for key, value in prepared.items()}
    raw = torch.from_numpy(inputs['prior'])[None]
    valid, anchor = prepared['valid'], prepared['anchor']
    outputs = {'original_prior': raw,
               'smooth_prior': smooth_motion_carrier(raw, valid, window=WINDOW)['smoothed']}
    requested = {}
    for mode in ('full', 'static'):
        intensity = student(b['features'], b['valid'], b['global_code'], b['identity_code'], b['anchor'],
                            temporal_mode=mode)['intensity']
        upper = decoder(intensity, b['valid'], b['global_code'], b['identity_code'], b['anchor'])
        if not torch.isfinite(intensity[b['valid']]).all() or not torch.isfinite(upper[b['valid']]).all():
            raise ValueError('Nonfinite model prediction')
        requested[mode] = intensity.cpu()
        # Match the runner: compose with the original prior, never the smooth baseline.
        outputs[mode] = compose_upper_face(raw, upper.cpu(), valid)
    raw_mean = _mean(raw[..., list(UPPER_INDICES)], valid)
    protection, ranges, mean_changes, actual_intensity = {}, {}, {}, []
    for mode in MODE_NAMES:
        motion = outputs[mode]
        upper = motion[..., list(UPPER_INDICES)]
        checks = {'nonupper43_bit_exact': _bit_exact(motion[..., list(NOT_UPPER)], raw[..., list(NOT_UPPER)]),
                  'mouth27_bit_exact': _bit_exact(motion[..., list(MOUTH)], raw[..., list(MOUTH)]),
                  'inactive_frames_bit_exact': _bit_exact(motion[~valid], raw[~valid]),
                  'finite_active_output': bool(torch.isfinite(motion[valid]).all())}
        protection[mode] = {**checks, 'passed': all(checks.values())}
        observed = upper[valid]
        ranges[mode] = {'min': float(observed.min()), 'max': float(observed.max()),
                        'outside_unit_interval_fraction': float(((observed < 0) | (observed > 1)).float().mean())}
        difference = (_mean(upper, valid) - raw_mean)[0]
        mean_changes[mode] = {'max_abs_change_from_original_prior': float(difference.abs().max()),
                              'signed_change_upper9': difference.tolist()}
        actual_intensity.append(neutral_relative_intensity(upper, anchor[:, list(UPPER_INDICES)],
                                                           stats['scales9'], valid, WINDOW)[0])
    if not all(value['passed'] for value in protection.values()):
        raise RuntimeError('Inference violated channel or inactive-frame protection')
    payload = {'channels': np.asarray(ARKIT_NAMES), 'times': inputs['times'], 'valid': inputs['valid'],
               'upper_valid': inputs['upper_valid'], 'mode_names': np.asarray(MODE_NAMES),
               'motions': torch.stack([outputs[mode][0] for mode in MODE_NAMES]).numpy(),
               'predicted_intensity': requested['full'][0].numpy(),
               'static_intensity': requested['static'][0].numpy(),
               'composed_intensities': torch.stack(actual_intensity).numpy(),
               'scales9': stats['scales9'].numpy(), 'anchor': inputs['anchor']}
    for key in ('channel_mask', 'clip_id', 'noise_seed'):
        if key in inputs:
            payload[key] = inputs[key]
    report = {'schema': 'reference_intensity_inference_v1',
              'scope': 'Precomputed frozen prior, audio1540, audio global64, neutral identity128 and independent neutral anchor; not raw-WAV/full-system inference',
              'checkpoint': str(Path(checkpoint_path).resolve()), 'checkpoint_sha256': _sha(checkpoint_path),
              'input': str(Path(input_path).resolve()), 'input_sha256': _sha(input_path),
              'script_sha256': _sha(Path(__file__)), 'protocol': protocol,
              'protocol_sha256': _canonical_hash(protocol), 'device': str(device),
              'condition_provenance': 'Artifact hashes recorded; frozen prior/global/reference conditions are supplied and not independently regenerated or authenticated',
              'frames': len(inputs['valid']), 'valid_frames': int(inputs['valid'].sum()),
              'upper_valid_frames': int(valid.sum()), 'fps': FPS, 'mode_names': list(MODE_NAMES),
              'smooth_prior_window': WINDOW, 'intensity_window': WINDOW,
              'intensity_semantics': 'Five-frame run-local mean of brow5/eye4 mean absolute deviation from independent neutral anchor divided by TRAIN-fitted channel scales; no clip centering',
              'static_semantics': 'Student temporal hidden features replaced by their valid-frame clip mean; same frozen global/reference and pooled audio conditions; constant valid upper output',
              'protection': protection, 'raw_upper_range': ranges, 'upper_mean_change': mean_changes,
              'upper_mean_protected': False, 'upper_mean_scope': 'Diagnostic only; learned outputs deliberately may change the previous upper mean',
              'query_motion_read': False, 'query_emotion_read': False, 'query_target_read': False,
              'range_clipping_applied': False, 'post_composition_smoothing': False,
              'upper_generator_deterministic': True,
              'quality_claim': 'Structural preservation only; temporal prediction quality, emotion, identity and naturalness are not evaluated'}
    return payload, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trusted repository-generated final.pt')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh output .npz path')
    parser.add_argument('--report', type=Path, help='Fresh report path; default output.report.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    report_path = args.report or args.output.with_suffix('.report.json')
    if args.output.suffix.lower() != '.npz':
        raise ValueError('Output must have an .npz extension')
    if args.output.resolve() == report_path.resolve() or args.output.exists() or report_path.exists():
        raise FileExistsError('Distinct fresh output and report paths are required')
    payload, report = infer(args.checkpoint, args.input, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, **payload)
    report.update(output=str(args.output.resolve()), output_sha256=_sha(args.output))
    with report_path.open('x', encoding='utf8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output.resolve()), 'report': str(report_path.resolve()),
                      'frames': report['frames'], 'protection_passed': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
