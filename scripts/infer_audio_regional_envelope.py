"""Apply a trained regional-envelope predictor to a supplied frozen Stage4 prior.

Input NPZ: prior [T,52] float32 in canonical ARKit order, audio_features [T,F],
valid [T] bool, and times [T] native seconds. Optional channels/channel_mask,
clip_id and noise_seed are retained. No query motion, emotion or target array
is loaded. This is a feature-to-motion adapter, not raw-WAV/full-system inference.

Output NPZ is accepted by render_dynamic_rig_comparison.py. Its four modes are
original_prior, prior (the protocol's smooth carrier), full, and static_gain.
No output clipping, temporal realignment or post-composition smoothing occurs.
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

from kinetalk_b0.models.audio_regional_envelope import AudioRegionalEnvelope, compose_prior_with_envelope
from kinetalk_b0.models.regional_intensity_gain import regional_envelope
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES
from kinetalk_b0.models.temporal_motion_carrier import smooth_motion_carrier
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES

WINDOW = 5
GAIN_MIN, GAIN_MAX, DENOMINATOR_FLOOR = .25, 2.5, .02
MEAN_TOLERANCE = 1e-5
MODE_NAMES = ('original_prior', 'prior', 'full', 'static_gain')
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


def _protocol_settings(checkpoint):
    protocol = checkpoint.get('protocol')
    if not isinstance(protocol, dict):
        raise ValueError('Checkpoint must contain its training protocol')
    if (checkpoint.get('protocol_sha256') is not None
            and checkpoint['protocol_sha256'] != _canonical_hash(protocol)):
        raise ValueError('Checkpoint protocol hash differs')
    if (protocol.get('window_frames') != WINDOW or protocol.get('stride') != 1
            or protocol.get('gain_bounds') != [GAIN_MIN, GAIN_MAX]
            or protocol.get('gain_denominator_floor_normalized') != DENOMINATOR_FLOOR):
        raise ValueError('Unsupported envelope/gain protocol; use the native regional-envelope runner')
    smoothing = protocol.get('carrier_smoothing', {})
    if not isinstance(smoothing, dict):
        raise ValueError('Invalid carrier_smoothing protocol')
    nested_window = smoothing.get('window_frames')
    carrier_window = protocol.get('carrier_window', nested_window)
    if (type(carrier_window) is not int or carrier_window not in (1, 3, 5, 7, 9)
            or (nested_window is not None and nested_window != carrier_window)
            or smoothing.get('post_composition_smoothing', False)):
        raise ValueError('Missing, conflicting or unsupported carrier window')
    if 'applied' in smoothing and smoothing['applied'] != (carrier_window > 1):
        raise ValueError('Carrier window and applied flag disagree')
    return protocol, carrier_window


def load_predictor(path, device):
    # The runner stores tensors and basic protocol data only.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must be a mapping')
    protocol, carrier_window = _protocol_settings(checkpoint)
    state, stats = checkpoint.get('model'), checkpoint.get('stats')
    if not isinstance(state, dict) or not isinstance(stats, dict):
        raise ValueError('Checkpoint needs model and stats mappings')
    feature_mean, feature_std = stats.get('feature_mean'), stats.get('feature_std')
    if (not torch.is_tensor(feature_mean) or feature_mean.ndim != 1 or not feature_mean.numel()
            or not torch.is_tensor(feature_std) or feature_std.shape != feature_mean.shape):
        raise ValueError('Malformed checkpoint feature statistics')
    widths = {'feature_mean': feature_mean.numel(), 'feature_std': feature_mean.numel(),
              'channel_scales': 9, 'envelope_scale': 2}
    for key, width in widths.items():
        value = stats.get(key)
        if (not torch.is_tensor(value) or value.shape != (width,) or value.dtype != torch.float32
                or not torch.isfinite(value).all()
                or (key != 'feature_mean' and (value <= 0).any())):
            raise ValueError(f'Invalid checkpoint statistic: {key}')
    for key in ('feature_mean', 'feature_std'):
        if not torch.is_tensor(state.get(key)) or not torch.equal(state[key], stats[key]):
            raise ValueError(f'Model and checkpoint statistics disagree: {key}')
    weight = state.get('input.weight')
    if not torch.is_tensor(weight) or weight.ndim != 2 or weight.shape[1] != feature_mean.numel():
        raise ValueError('Model feature width differs from statistics')
    model = AudioRegionalEnvelope(feature_mean, feature_std, hidden=weight.shape[0], stride=1)
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Nonfinite predictor weights')
    return model.to(device).eval(), stats, protocol, carrier_window


def load_input(path, feature_width):
    with np.load(path, allow_pickle=False) as source:
        # Explicit allowlist: unrelated target/emotion/object arrays are never read.
        required = ('prior', 'audio_features', 'valid', 'times')
        missing = set(required) - set(source.files)
        if missing:
            raise ValueError('Missing inference inputs: ' + ', '.join(sorted(missing)))
        values = {key: source[key] for key in required}
        for key in ('channels', 'channel_mask', 'clip_id', 'noise_seed'):
            if key in source.files:
                values[key] = source[key]
    prior, features, valid, times = (values[key] for key in required)
    if prior.dtype != np.float32 or prior.ndim != 2 or prior.shape[1:] != (52,) or len(prior) < 1:
        raise ValueError('prior must be nonempty float32 [T,52] from the frozen Stage4 output')
    frames = len(prior)
    if valid.dtype != np.bool_ or valid.shape != (frames,) or not valid.any():
        raise ValueError('valid must be Boolean [T] with an observed frame')
    if (features.ndim != 2 or features.shape != (frames, feature_width)
            or not np.issubdtype(features.dtype, np.floating)):
        raise ValueError('audio_features must be floating [T,F] with the checkpoint feature width')
    if (times.shape != (frames,) or not np.issubdtype(times.dtype, np.number)
            or not np.isfinite(times).all()):
        raise ValueError('times must be finite native seconds [T]')
    if frames > 1:
        steps = np.diff(times.astype(np.float64))
        if (steps <= 0).any() or not np.allclose(steps, steps[0], atol=1e-7, rtol=1e-5):
            raise ValueError('times must retain a uniformly spaced increasing native clock')
    if 'channels' in values and values['channels'].tolist() != ARKIT_NAMES:
        raise ValueError('Input channels must use the canonical ARKit52 order; no implicit reordering')
    upper_valid = valid.copy()
    support = np.ones((frames, 52), dtype=bool)
    if 'channel_mask' in values:
        support = values['channel_mask']
        if support.dtype != np.bool_ or support.shape not in ((52,), (frames, 52)):
            raise ValueError('channel_mask must be Boolean [52] or [T,52]')
        support = np.broadcast_to(support, (frames, 52))
        upper_valid &= support[:, list(UPPER_INDICES)].all(-1)
    if not upper_valid.any():
        raise ValueError('Inference needs at least one fully observed upper-face frame')
    if (not np.isfinite(prior[valid[:, None] & support]).all()
            or not np.isfinite(prior[upper_valid]).all() or not np.isfinite(features[upper_valid]).all()):
        raise ValueError('Observed prior and audio features must be finite')
    for key in ('clip_id', 'noise_seed'):
        if key in values and values[key].shape != ():
            raise ValueError(f'{key} must be scalar metadata')
    values['upper_valid'] = upper_valid
    values['audio_features'] = features.astype(np.float32, copy=False)
    return values


def _center(value, valid):
    # Match the runner's numerically stable centering exactly.
    first = valid.long().argmax(1, keepdim=True)[..., None].expand(-1, -1, value.shape[-1])
    relative = torch.where(valid[..., None], value - value.gather(1, first), 0.)
    mean = relative.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)[..., None]
    return torch.where(valid[..., None], relative - mean, 0.)


def _envelope(motion, valid, stats):
    centered = _center(motion[..., list(UPPER_INDICES)], valid)
    return regional_envelope(centered, valid, window=WINDOW,
                             channel_scales=stats['channel_scales']) / stats['envelope_scale']


def _mean(value, valid):
    return torch.where(valid[..., None], value, 0.).sum(1, keepdim=True) / valid.sum(1, keepdim=True)[..., None]


def _bit_exact(left, right):
    return bool(torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32)))


@torch.inference_mode()
def infer(checkpoint_path, input_path, *, device='cpu'):
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model, stats, protocol, carrier_window = load_predictor(checkpoint_path, device)
    inputs = load_input(input_path, stats['feature_mean'].numel())
    raw = torch.from_numpy(inputs['prior'])[None]
    valid = torch.from_numpy(inputs['upper_valid'])[None]
    features = torch.from_numpy(inputs['audio_features'])[None]
    predicted = model(features.to(device), valid.to(device)).cpu()
    if predicted.shape != (*valid.shape, 2) or not torch.isfinite(predicted[valid]).all():
        raise ValueError('Invalid normalized envelope prediction')
    # The training runner performs carrier, envelope and composition on CPU.
    carrier = smooth_motion_carrier(raw, valid, window=carrier_window)['smoothed']
    mean = _mean(carrier[..., list(UPPER_INDICES)], valid).squeeze(1)
    source_envelope = _envelope(carrier, valid, stats)
    gain = (predicted / source_envelope.clamp_min(DENOMINATOR_FLOOR)).clamp(GAIN_MIN, GAIN_MAX)
    gain = torch.where(valid[..., None], gain, torch.ones_like(gain))
    # Static gain is the mean of the bounded full gain, NOT ratio of mean envelopes.
    static_gain = torch.where(valid[..., None], _mean(gain, valid).expand_as(gain), 0.)
    outputs = {'original_prior': raw, 'prior': carrier,
               'full': compose_prior_with_envelope(carrier, gain, mean, valid),
               'static_gain': compose_prior_with_envelope(carrier, static_gain, mean, valid)}
    protection = {}
    raw_mean = _mean(raw[..., list(UPPER_INDICES)], valid)
    for name, motion in outputs.items():
        drift = float((_mean(motion[..., list(UPPER_INDICES)], valid) - raw_mean).abs().max())
        checks = {'nonupper43_bit_exact': _bit_exact(motion[..., list(NOT_UPPER)], raw[..., list(NOT_UPPER)]),
                  'mouth27_bit_exact': _bit_exact(motion[..., list(MOUTH)], raw[..., list(MOUTH)]),
                  'inactive_frames_bit_exact': _bit_exact(motion[~valid], raw[~valid]),
                  'upper_mean_preserved': drift < MEAN_TOLERANCE,
                  'finite_active_output': bool(torch.isfinite(motion[valid]).all())}
        protection[name] = {**checks, 'max_upper_mean_drift': drift, 'passed': all(checks.values())}
    if not all(value['passed'] for value in protection.values()):
        raise RuntimeError('Inference violated raw-prior channel/mean protection')
    actual = torch.stack([_envelope(outputs[name], valid, stats)[0] for name in MODE_NAMES])
    payload = {'channels': np.asarray(ARKIT_NAMES), 'times': inputs['times'], 'valid': inputs['valid'],
               'mode_names': np.asarray(MODE_NAMES), 'motions': torch.stack([outputs[n][0] for n in MODE_NAMES]).numpy(),
               'upper_valid': inputs['upper_valid'], 'predicted_envelope_normalized': predicted[0].numpy(),
               'prior_envelope_normalized': source_envelope[0].numpy(),
               'composed_envelopes_normalized': actual.numpy(), 'gain': gain[0].numpy(),
               'static_gain': static_gain[0].numpy(), 'channel_scales': stats['channel_scales'].numpy(),
               'envelope_scale': stats['envelope_scale'].numpy()}
    for key in ('channel_mask', 'clip_id', 'noise_seed'):
        if key in inputs:
            payload[key] = inputs[key]
    observed_gain = gain[valid]
    report = {'schema': 'regional_envelope_inference_v1',
              'scope': 'Supplied frozen Stage4 prior plus precomputed native audio features; not raw-WAV/full-system inference',
              'checkpoint': str(Path(checkpoint_path).resolve()), 'checkpoint_sha256': _sha(checkpoint_path),
              'input': str(Path(input_path).resolve()), 'input_sha256': _sha(input_path),
              'script_sha256': _sha(Path(__file__)), 'protocol': protocol,
              'protocol_sha256': _canonical_hash(protocol), 'device': str(device),
              'prior_source_verification': 'Input artifact hash recorded; supplied prior is not regenerated or verified against the Stage4 source checkpoint',
              'frames': len(inputs['valid']), 'valid_frames': int(inputs['valid'].sum()),
              'upper_valid_frames': int(valid.sum()), 'mode_names': list(MODE_NAMES),
              'carrier_window': carrier_window, 'envelope_window': WINDOW,
              'envelope_semantics': 'Five-frame box-smoothed RMS of clip-centered brow/eye coefficients, channel-scaled and envelope-scaled; activity, not emotion intensity',
              'gain_semantics': 'Full gain is predicted normalized envelope / max(prior envelope, 0.02), bounded to [0.25,2.5]; static gain is its valid-frame mean',
              'gain': {'min': float(observed_gain.min()), 'max': float(observed_gain.max()),
                       'lower_saturated_fraction': float((observed_gain <= GAIN_MIN + 1e-7).float().mean()),
                       'upper_saturated_fraction': float((observed_gain >= GAIN_MAX - 1e-7).float().mean())},
              'protection': protection, 'mean_tolerance': MEAN_TOLERANCE,
              'protected_mean_scope': 'Per-clip/channel valid-frame mean; not a per-frame expression guarantee',
              'query_motion_read': False, 'query_emotion_read': False, 'query_target_read': False,
              'range_clipping_applied': False, 'post_composition_smoothing': False,
              'quality_claim': 'Structural preservation only; animation quality and lip synchronization are not evaluated'}
    return payload, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh output .npz path')
    parser.add_argument('--report', type=Path, help='Fresh JSON path; default: output.report.json')
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
