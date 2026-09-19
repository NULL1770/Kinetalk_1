"""Independent bounded audit of a completed multi-scale audio residual run.

This does not train, select a checkpoint, or regenerate the full development
set. It checks hashes, reconstructs fit support and random draws independently,
tests the trained exact-null contract, and reproduces the first two fixed
metadata-selected examples for both trained arms at seed 42.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from kinetalk_b0.models.continuous_upper_motion import ContinuousLatentFlow, ContinuousUpperAE
from kinetalk_b0.models.prior_audio_multiscale import MultiScalePriorAudioResidual
from scripts import evaluate_continuous_motion_latent as native
from scripts import train_continuous_motion_latent as common
from scripts.probe_motion_condition_predictability import split_train_pool

SCHEMA = 'multiscale_prior_audio_independent_audit_v1'
RUN_SCHEMA = 'multiscale_prior_audio_v2'
ARMS = ('audio', 'matched_static')
PAIR_KEYS = ('initial_sha256', 'order_sha256', 'donor_sha256', 'step', 'donor_counts')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def support_from_masks(clips):
    """Rebuild joint/min5/max200 counts and batch-order metadata only.

    Audio, motion values and event labels are not read. No AE encoding or
    1540-D segment duplication is needed for the coverage/order audit.
    """
    reports, segments = [], []
    for clip in clips:
        require(clip['split'] == 'train', 'Fit support contains a nontrain clip')
        valid = native._array(clip['valid'])
        observed = native._array(clip['motion_mask'])
        require(valid.dtype == np.bool_ and valid.ndim == 1
                and observed.dtype == np.bool_ and observed.shape == (len(valid), 9),
                'Invalid joint support masks')
        joint = valid & observed.all(1)
        row = {'clip_id': clip['clip_id'], 'native_frames': len(joint),
               'native_valid_frames': int(valid.sum()), 'joint_observed_frames': int(joint.sum()),
               'runs': 0, 'kept_frames': 0, 'segments': 0, 'dropped_short_frames': 0,
               'dropped_short_segments': 0}
        for start, stop in native.contiguous_runs(joint):
            row['runs'] += 1
            for left in range(start, stop, 200):
                right = min(left + 200, stop)
                if right - left < 5:
                    row['dropped_short_frames'] += right - left
                    row['dropped_short_segments'] += 1
                    continue
                meta = {**clip.get('metadata', {}), 'clip_id': clip['clip_id'],
                        'split': 'train', 'sentence': clip['sentence'], 'start': left,
                        'end': right, 'run_start': start, 'run_end': stop,
                        'support': 'joint observed, training/scoring only'}
                segments.append({'metadata': meta})
                row['kept_frames'] += right-left
                row['segments'] += 1
        row['unobserved_native_valid_frames'] = row['native_valid_frames']-row['joint_observed_frames']
        reports.append(row)
    keys = ('native_frames', 'native_valid_frames', 'joint_observed_frames', 'runs', 'kept_frames',
            'segments', 'dropped_short_frames', 'dropped_short_segments', 'unobserved_native_valid_frames')
    coverage = {'role': 'train', 'max_frames': 200, 'min_frames': 5, 'clips': reports,
                'totals': {k: sum(r[k] for r in reports) for k in keys},
                'excluded_clip_ids': [r['clip_id'] for r in reports if not r['segments']],
                'retained_clip_ids': sorted({r['metadata']['clip_id'] for r in segments})}
    return segments, coverage


def replay_draws(clips, segments, seed, batch_size, steps):
    """Independently replay clip/segment and audio donor RNG, without tensors."""
    groups = {}
    for i, item in enumerate(segments):
        groups.setdefault(item['metadata']['clip_id'], []).append(i)
    groups = [groups[cid] for cid in sorted(groups)]
    require(bool(groups), 'No retained fitting segments')
    donors, matched = {}, {}
    for clip in clips:
        eligible = [d for d in clips if d['sentence'] != clip['sentence'] and d['emotion'] == clip['emotion']]
        same = [d for d in eligible if d['speaker'] == clip['speaker']]
        donors[clip['clip_id']] = sorted(d['clip_id'] for d in (same or eligible))
        matched[clip['clip_id']] = bool(same)
    rng = np.random.default_rng(seed+1)
    order_chain = donor_chain = '0'*64
    paired = same_count = total = 0
    for step in range(1, steps+1):
        indices = []
        for _ in range(batch_size):
            group = groups[int(rng.integers(len(groups)))]
            indices.append(group[int(rng.integers(len(group)))])
        order_chain = hashlib.sha256(bytes.fromhex(order_chain)+np.asarray(indices, dtype='<i8').tobytes()).hexdigest()
        donor_rng = np.random.default_rng((seed+3)*1000003+step)
        draw = []
        for index in indices:
            cid = segments[index]['metadata']['clip_id']
            options = donors[cid]
            draw.append(options[int(donor_rng.integers(len(options)))] if options else None)
            paired += int(bool(options)); same_count += int(bool(options) and matched[cid]); total += 1
        donor_chain = hashlib.sha256(bytes.fromhex(donor_chain)+json.dumps(draw).encode()).hexdigest()
    return {'order_sha256': order_chain, 'donor_sha256': donor_chain,
            'step': steps, 'donor_counts': [paired, same_count, total]}


def verify_hashes(protocol, run, dataset, source, root):
    require(common._file_sha(dataset) == protocol['dataset_sha256'], 'Dataset hash differs from run')
    checked = {'dataset': protocol['dataset_sha256'], 'source_files': {}, 'code_files': {}}
    for name, digest in protocol['source_files'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe source-relative path')
        require(common._file_sha(source/path) == digest, 'Frozen source hash differs: '+name)
        checked['source_files'][name] = digest
    for relative, digest in protocol['source_sha256'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe code-relative path')
        require(common._file_sha(root/path) == digest, 'Live audited code differs: '+relative)
        require(common._file_sha(run/'source'/path) == digest, 'Saved code snapshot differs: '+relative)
        checked['code_files'][relative] = digest
    return checked


def load_adapter(prior, config, checkpoint, protocol_hash, initial_hash, step):
    require(checkpoint['schema'] == RUN_SCHEMA, 'Adapter checkpoint schema differs')
    require(checkpoint['binding'] == {'protocol_sha256': protocol_hash}, 'Adapter protocol binding differs')
    require(checkpoint['initial_sha256'] == initial_hash and checkpoint['step'] == step,
            'Adapter initialization or endpoint differs')
    require(checkpoint['config'] == config, 'Adapter config differs from protocol')
    options = copy.deepcopy(config)
    require(options.pop('prior') == prior.config, 'Adapter source prior config differs')
    model = MultiScalePriorAudioResidual(copy.deepcopy(prior), **options)
    state = checkpoint['state']
    require(not any(k.startswith('prior.') for k in state), 'Lightweight adapter contains prior weights')
    missing, extra = model.load_state_dict(state, strict=False)
    require(set(missing) == {k for k in model.state_dict() if k.startswith('prior.')} and not extra,
            'Adapter checkpoint contains missing or unexpected residual weights')
    model.eval()
    require(all(not p.requires_grad for p in model.prior.parameters()), 'Prior parameters are trainable')
    return model


@torch.no_grad()
def verify_null(model, clip, stats, device):
    """Bounded first native run, using only audio/context and explicit noise."""
    start, stop = native.contiguous_runs(native._array(clip['valid']))[0]
    stop = min(stop, start+25)
    blocks, frames = native._audio_blocks(clip, start, stop, stats, model.block_size, device, 'real')
    valid = frames.any(-1)
    context = torch.as_tensor(clip['context'], dtype=torch.float32, device=device)
    context = ((context-native._vector(stats, 'context_mean', len(context), device)) /
               native._vector(stats, 'context_scale', len(context), device))[None]
    generator = torch.Generator(device=device).manual_seed(42)
    noise = torch.randn(1, valid.shape[1], model.latent_dim, generator=generator, device=device)
    time_value = noise.new_tensor([.37])
    zero = torch.zeros_like(blocks)
    options = dict(audio_frame_valid=frames)
    expected = model.prior.velocity(noise, time_value, valid, context, blocks, False, **options)
    require(torch.equal(model.velocity(noise, time_value, valid, context, blocks, False, **options), expected),
            'Disabled local audio differs from frozen prior')
    require(torch.equal(model.velocity(noise, time_value, valid, context, zero, True, **options), expected),
            'Zero local audio differs from frozen prior after training')
    require(torch.count_nonzero(model.residual(noise, time_value, valid, context, zero, **options)) == 0,
            'Null residual is not exactly zero')
    reference = model.prior.sample(valid, context, zero, noise, steps=3, use_audio=False, **options)
    require(torch.equal(model.sample(valid, context, zero, noise, steps=3, **options), reference),
            'Zero-local sampling differs from frozen source prior')
    residual = model.residual(noise, time_value, valid, context, blocks, **options)
    require(torch.isfinite(residual).all() and residual.abs().max() <= model.max_delta,
            'Trained residual is nonfinite or exceeds configured velocity bound')
    return {'velocity_exact_disabled': True, 'velocity_exact_null': True, 'null_residual_exact_zero': True,
            'sample3_exact_null': True, 'max_absolute_real_residual': float(residual.abs().max()),
            'bound': model.max_delta, 'frames': stop-start,
            'zero_meaning': 'zero normalized local features; dataset-mean acoustic input, not silence'}


@torch.no_grad()
def verify_example(model, ae, clip, stats, point, arm, device, steps):
    cid = clip['clip_id']; path = point/arm/'npz'/(cid+'.npz')
    with np.load(path, allow_pickle=False) as archive:
        saved = {k: archive[k].copy() for k in archive.files}
    require(str(saved['clip_id'].item()) == cid and int(saved['noise_seed']) == 42
            and str(saved['mode_names'][2]).endswith(' 42'), 'Saved example first draw is not seed42')
    require(np.array_equal(saved['valid'], native._array(clip['valid']))
            and np.array_equal(saved['times'], native._array(clip['times'])), 'Saved example native clock differs')
    require(saved['channels'].tolist() == list(native.ARKIT_NAMES), 'Saved example channel order differs')
    samples, records = native.generate_clip(model, ae, clip, stats, device, use_audio=True,
        intervention='static' if arm == 'matched_static' else 'real', seeds=(42,), steps=steps)
    full = native.compose_full(samples, native._array(clip['baseline52']))[0]
    require(saved['motions'][2].shape == full.shape and np.isfinite(full).all(), 'Invalid regenerated fullface')
    difference = float(np.max(np.abs(full-saved['motions'][2])))
    require(np.allclose(full, saved['motions'][2], rtol=0., atol=1e-6),
            f'Regenerated {arm}/{cid} differs from saved raw NPZ: {difference}')
    require(np.array_equal(full[:, native.OTHER], native._array(clip['baseline52'])[:, native.OTHER]),
            'Regeneration changed protected other43 channels')
    return {'clip_id': cid, 'arm': arm, 'seed': 42, 'native_frames': len(full),
            'raw_max_absolute_difference': difference, 'raw_allclose_atol_1e6': True,
            'other43_exact': True, 'source_npz_sha256': common._file_sha(path),
            'noise_records': records}


def run(args):
    started = time.monotonic(); run_dir = args.run
    protocol, status = read(run_dir/'protocol.json'), read(run_dir/'status.json')
    require(protocol['schema'] == status['schema'] == RUN_SCHEMA, 'Run schema differs')
    require(protocol['smoke'] is False and status['smoke'] is False and status['state'] == 'complete',
            'Audit requires a completed formal run')
    step = status['last_endpoint']; point = run_dir/f'point_{step}'
    require(step in protocol['budgets'], 'Completed endpoint is not declared in protocol')
    source = Path(protocol['source_run']); root = Path(__file__).resolve().parents[1]
    hashes = verify_hashes(protocol, run_dir, args.dataset, source, root)
    source_protocol = read(source/'protocol.json')
    require(source_protocol['dataset_sha256'] == hashes['dataset'], 'Source dataset binding differs')
    payload = torch.load(args.dataset, map_location='cpu', weights_only=False, mmap=True)
    require(payload['schema'] == 'continuous_motion_dataset_v1', 'Dataset schema differs')
    fit, validation = split_train_pool(payload['clips'], source_protocol)
    validation = [{**c, 'split': 'inner_validation'} for c in validation]
    require([c['clip_id'] for c in fit] == protocol['fit_ids']
            and [c['clip_id'] for c in validation] == protocol['validation_ids'], 'Run split differs')
    segments, coverage = support_from_masks(fit)
    require(coverage == protocol['coverage'], 'Independently reconstructed training support differs')
    require(len(fit) == len(coverage['retained_clip_ids']) == 613 and len(validation) == 206
            and coverage['totals']['kept_frames'] == 66272, 'Expected full 613/66272 fitting support and 206 development clips')
    stats = common._load(source/'fit_stats.pt')
    require(set(stats['train_clip_ids']) == {c['clip_id'] for c in fit}, 'Fit statistics membership differs')
    ae_ck, prior_ck = [common._load(source/name) for name in ('ae_final.pt', 'prior_final.pt')]
    source_hash = common._value_sha(source_protocol)
    require(all(ck['binding']['protocol_sha256'] == source_hash for ck in (ae_ck, prior_ck)),
            'Source checkpoint protocol binding differs')
    require(prior_ck['binding']['ae_sha256'] == common._value_sha(ae_ck['state'])
            and prior_ck['binding']['stats_sha256'] == common._value_sha(stats), 'Source AE/statistics binding differs')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    ae = ContinuousUpperAE(**ae_ck['config']).to(device)
    ae.load_state_dict(ae_ck['state']); ae.eval().requires_grad_(False)
    prior = ContinuousLatentFlow(**prior_ck['config']).to(device)
    prior.load_state_dict(prior_ck['state']); prior.eval().requires_grad_(False)
    torch.manual_seed(protocol['seed'])
    options = copy.deepcopy(protocol['model_config']); require(options.pop('prior') == prior.config, 'Prior config differs')
    initial = MultiScalePriorAudioResidual(copy.deepcopy(prior), **options).to(device)
    initial_hash = common._value_sha({k: v for k, v in initial.state_dict().items() if not k.startswith('prior.')})
    del initial
    checkpoints = {arm: common._load(point/(arm+'.pt')) for arm in ARMS}
    declared_pairing = read(point/'pairing.json')
    require(set(declared_pairing) == set(PAIR_KEYS) and all(v is True for v in declared_pairing.values()),
            'Declared paired training contract failed')
    replay = replay_draws(fit, segments, protocol['seed'], protocol['batch_size'], step)
    for key in PAIR_KEYS:
        require(checkpoints['audio'][key] == checkpoints['matched_static'][key], 'Saved paired draws differ: '+key)
    for arm, ck in checkpoints.items():
        require(all(ck[k] == v for k, v in replay.items()), 'Independently replayed draws differ: '+arm)
    frozen_before = {'ae': common._value_sha(ae.state_dict()), 'prior': common._value_sha(prior.state_dict()),
                     'stats': common._value_sha(stats)}
    examples = native._selected(validation)[:2]
    require(len(examples) == 2, 'Two fixed metadata examples required')
    results, null_checks, checkpoint_hashes = [], {}, {}
    for arm in ARMS:
        model = load_adapter(prior, protocol['model_config'], checkpoints[arm],
                             common._file_sha(run_dir/'protocol.json'), initial_hash, step).to(device)
        require(common._value_sha(model.prior.state_dict()) == frozen_before['prior'], 'Restored adapter changed source prior')
        null_checks[arm] = verify_null(model, examples[0], stats, device)
        for clip in examples:
            results.append(verify_example(model, ae, clip, stats, point, arm, device, protocol['flow_steps']))
        require(common._value_sha(model.prior.state_dict()) == frozen_before['prior'], 'Generation changed frozen source prior')
        checkpoint_hashes[arm] = common._file_sha(point/(arm+'.pt'))
        del model
    require(frozen_before == {'ae': common._value_sha(ae.state_dict()), 'prior': common._value_sha(prior.state_dict()),
                             'stats': common._value_sha(stats)}, 'Auditing changed frozen source components')
    result = {'schema': SCHEMA, 'passed': True, 'seconds': time.monotonic()-started,
              'run_protocol_sha256': common._file_sha(run_dir/'protocol.json'),
              'audit_code_sha256': common._file_sha(__file__), 'input_hashes': hashes,
              'checkpoint_sha256': checkpoint_hashes, 'endpoint': step,
              'full_support': {'fit_clips': len(fit), 'kept_frames': coverage['totals']['kept_frames'],
                               'development_clips': len(validation), 'coverage_exact': True},
              'paired_draws_replayed': replay, 'initial_adapter_sha256': initial_hash,
              'frozen_source': {**frozen_before, 'unchanged_exact': True}, 'null_contracts': null_checks,
              'examples': results, 'example_selection': 'native evaluator metadata-only selection, first two; no metric ranking',
              'generation_scope': 'two clips, two trained arms, seed42 only; no full-development regeneration',
              'quality_claim': 'implementation/provenance audit only; does not certify audio timing or naturalness',
              'default_replaced': False}
    common._write(run_dir/'independent_audit.json', result)
    print(json.dumps({'passed': True, 'endpoint': step, 'examples': len(results),
                      'output': str(run_dir/'independent_audit.json')}), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        common._write(args.run/'independent_audit.json', {'schema': SCHEMA, 'passed': False,
                      'error': f'{type(exc).__name__}: {exc}', 'quality_claim': 'failed implementation/provenance audit'})
        raise
