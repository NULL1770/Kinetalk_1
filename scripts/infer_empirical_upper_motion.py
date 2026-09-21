"""Apply a TRAIN empirical upper-motion bank to precomputed deployment input.

The deterministic center is recomputed by infer_reference_intensity_decoder.
Inputs are a trusted repository bank.pt, its hash-bound reference final.pt and
the same deployment NPZ. This is not raw-WAV/full-system inference. Query GT,
emotion labels and target arrays are never read. --speaker and --sentence are
optional explicit exclusion metadata; unknown metadata cannot be excluded.

Output compares original_prior, smooth_prior, deterministic, empirical and
unconditional using the same declared seed (default 42) and selected bank
temperature. Native donor crops are never warped or concatenated. Missing
eligible length raises, even if the evaluation runner allowed a documented
deterministic fallback. This is a nonparametric distribution baseline, not a
new audio timing model, and does not replace the default system.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinetalk_b0.models.empirical_upper_motion import sample_empirical_motion
from kinetalk_b0.models.slow_state_affect import compose_upper_face
from scripts import infer_reference_intensity_decoder as reference

MODE_NAMES = ('original_prior', 'smooth_prior', 'deterministic', 'empirical', 'unconditional')


def load_bank(path, reference_checkpoint):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('schema') != 'empirical_upper_motion_v1':
        raise ValueError('Unsupported empirical bank checkpoint schema')
    protocol = checkpoint.get('protocol')
    if (not isinstance(protocol, dict) or protocol.get('schema') != 'empirical_upper_motion_feasibility_v1'
            or protocol.get('window') != 5 or type(protocol.get('top_k')) is not int
            or protocol['top_k'] < 1):
        raise ValueError('Unsupported empirical bank protocol')
    if checkpoint.get('protocol_sha256') != reference._canonical_hash(protocol):
        raise ValueError('Empirical bank protocol hash differs')
    if protocol.get('reference_final_sha256') != reference._sha(reference_checkpoint):
        raise ValueError('Empirical bank is bound to a different reference final checkpoint')
    temperature = checkpoint.get('temperature')
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('Bank temperature must be finite and nonnegative')
    if not isinstance(checkpoint.get('bank'), dict):
        raise ValueError('Missing empirical motion bank')
    return checkpoint['bank'], float(temperature), protocol


def _known_metadata(value, name):
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f'{name} must be a nonempty exact metadata string or None')
    return value


@torch.inference_mode()
def infer(bank_path, checkpoint_path, input_path, *, speaker=None, sentence=None, seed=42, device='cpu'):
    speaker = _known_metadata(speaker, 'speaker')
    sentence = _known_metadata(sentence, 'sentence')
    if type(seed) is not int or seed < 0 or seed >= 2**63:
        raise ValueError('seed must be an integer in [0,2**63)')
    bank, temperature, protocol = load_bank(bank_path, checkpoint_path)
    reference_payload, reference_report = reference.infer(checkpoint_path, input_path, device=device)
    if protocol.get('source_sha256') != reference_report['protocol'].get('source_sha256'):
        raise ValueError('Bank and reference checkpoint frozen-source provenance differs')
    inputs = reference.load_input(input_path)
    original = torch.from_numpy(inputs['prior'])[None]
    valid = torch.from_numpy(inputs['upper_valid'])[None]
    modes = dict(zip(reference_payload['mode_names'].tolist(), torch.from_numpy(reference_payload['motions'])))
    deterministic = modes['full'][None]
    center = deterministic[..., list(reference.UPPER_INDICES)]
    global_code = torch.from_numpy(inputs['global_code'])[None]
    identity = torch.from_numpy(inputs['identity_code'])[None]
    outputs = {'original_prior': original, 'smooth_prior': modes['smooth_prior'][None],
               'deterministic': deterministic}
    donor_records = {}
    for name, mode in (('empirical', 'conditional'), ('unconditional', 'unconditional')):
        samples, metadata = sample_empirical_motion(bank, center, valid, global_code, identity,
            [speaker], [sentence], seeds=[seed], temperature=temperature, top_k=protocol['top_k'], mode=mode)
        outputs[name] = compose_upper_face(original, samples[0], valid)
        records = metadata[0][0]
        for record in records:
            if ((speaker is not None and record['donor_speaker'] == speaker)
                    or (sentence is not None and record['donor_sentence'] == sentence)):
                raise RuntimeError('Sampler violated explicit donor exclusion')
        donor_records[name] = records
    protection, ranges, mean_changes = {}, {}, {}
    center_mean = reference._mean(center, valid)
    for name, motion in outputs.items():
        checks = {'nonupper43_bit_exact': reference._bit_exact(motion[..., list(reference.NOT_UPPER)],
                                                               original[..., list(reference.NOT_UPPER)]),
                  'mouth27_bit_exact': reference._bit_exact(motion[..., list(reference.MOUTH)],
                                                           original[..., list(reference.MOUTH)]),
                  'inactive_frames_bit_exact': reference._bit_exact(motion[~valid], original[~valid]),
                  'finite_active_output': bool(torch.isfinite(motion[valid]).all())}
        protection[name] = {**checks, 'passed': all(checks.values())}
        upper = motion[..., list(reference.UPPER_INDICES)]
        observed = upper[valid]
        ranges[name] = {'min': float(observed.min()), 'max': float(observed.max()),
                        'outside_unit_interval_fraction': float(((observed < 0) | (observed > 1)).float().mean())}
        change = (reference._mean(upper, valid)-center_mean)[0]
        mean_changes[name] = {'signed_change_upper9_from_deterministic': change.tolist(),
                              'max_abs_change_from_deterministic': float(change.abs().max())}
    if not all(value['passed'] for value in protection.values()):
        raise RuntimeError('Empirical inference violated frozen-channel or invalid-frame protection')
    payload = {'channels': reference_payload['channels'], 'times': inputs['times'], 'valid': inputs['valid'],
               'upper_valid': inputs['upper_valid'], 'mode_names': np.asarray(MODE_NAMES),
               'motions': torch.stack([outputs[name][0] for name in MODE_NAMES]).numpy(),
               'empirical_seed': np.asarray(seed, dtype=np.int64), 'temperature': np.asarray(temperature)}
    for key in ('channel_mask', 'clip_id', 'noise_seed'):
        if key in inputs:
            payload[key] = inputs[key]
    unavailable = [name for name, value in (('speaker', speaker), ('sentence', sentence)) if value is None]
    report = {'schema': 'empirical_upper_motion_inference_v1',
              'scope': 'Precomputed deployment features and independent references; empirical TRAIN-motion baseline, not raw-WAV/full-system inference',
              'bank_checkpoint': str(Path(bank_path).resolve()), 'bank_sha256': reference._sha(bank_path),
              'reference_checkpoint': str(Path(checkpoint_path).resolve()),
              'reference_final_sha256': reference_report['checkpoint_sha256'],
              'bank_protocol': protocol, 'bank_protocol_sha256': reference._canonical_hash(protocol),
              'reference_protocol_sha256': reference_report['protocol_sha256'],
              'input': str(Path(input_path).resolve()), 'input_sha256': reference_report['input_sha256'],
              'script_sha256': reference._sha(Path(__file__)), 'device': str(device),
              'frames': len(inputs['valid']), 'upper_valid_frames': int(valid.sum()),
              'mode_names': list(MODE_NAMES), 'seed': seed, 'temperature': temperature,
              'top_k': protocol['top_k'], 'speaker': speaker, 'sentence': sentence,
              'exclusion': {'rule': 'same known speaker OR same known sentence excluded',
                            'unknown_metadata': unavailable,
                            'limitation': 'Unknown metadata cannot enforce the corresponding donor exclusion; identifiers are matched exactly and are not inferred from clip_id',
                            'known_metadata_exclusion_verified': True},
              'donors': donor_records, 'protection': protection, 'raw_upper_range': ranges,
              'upper_mean_change': mean_changes, 'upper_mean_protected': False,
              'sampling': 'Same declared seed for conditional and unconditional draws; uniform eligible run then uniform native crop; no best-of-K, polarity flips, stitching or time warp',
              'missing_donor_policy': 'raise; no deterministic fallback at deployment',
              'headroom_bound': 'Sign-specific tanh using available [0,1] headroom; no post-hoc clamp',
              'query_motion_read': False, 'query_emotion_read': False, 'query_target_read': False,
              'range_clipping_applied': False, 'post_composition_smoothing': False,
              'default_replaced': False, 'innovation_claim': False, 'audio_timing_success_claim': False,
              'quality_evaluated': False, 'reference_inference': reference_report}
    return payload, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True, help='Trusted repository-generated empirical bank.pt')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Hash-bound reference final.pt')
    parser.add_argument('--input', type=Path, required=True, help='Precomputed deployment _input.npz')
    parser.add_argument('--output', type=Path, required=True, help='Fresh renderer .npz output')
    parser.add_argument('--report', type=Path, help='Fresh JSON path; default output.report.json')
    parser.add_argument('--speaker', help='Exact known speaker identifier, otherwise omitted')
    parser.add_argument('--sentence', help='Exact known sentence identifier, otherwise omitted')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    report_path = args.report or args.output.with_suffix('.report.json')
    if args.output.suffix.lower() != '.npz':
        raise ValueError('Output must have an .npz extension')
    if args.output.resolve() == report_path.resolve() or args.output.exists() or report_path.exists():
        raise FileExistsError('Distinct fresh output and report paths are required')
    payload, report = infer(args.bank, args.checkpoint, args.input, speaker=args.speaker,
                            sentence=args.sentence, seed=args.seed, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, **payload)
    report.update(output=str(args.output.resolve()), output_sha256=reference._sha(args.output))
    with report_path.open('x', encoding='utf8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output.resolve()), 'report': str(report_path.resolve()),
                      'seed': args.seed, 'temperature': report['temperature'], 'protection_passed': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
