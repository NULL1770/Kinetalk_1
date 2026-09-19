"""Inner-split bounded mouth repair candidate with a matched static control.

This is an offline candidate, never a default-model replacement. Only 27 mouth
coefficients can change. All 25 other coefficients, including invalid frames,
are copied exactly. AV quality and identity quality require separate review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from scripts import probe_motion_condition_predictability as protocol_base
from scripts.arkit_benchmark_report import build_report, score_fullface
from scripts import arkit_benchmark_report, evaluate_arkit_literature_metrics

MOUTH = tuple(range(14, 41))
LIPS = tuple(range(18, 41))
NONMOUTH = tuple(c for c in range(52) if c not in MOUTH)
SEEDS = (42, 123, 2026)
SCHEMA = 'mouth_residual_candidate_v1'
FPS = 25.0


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def native_runs(valid, times, fps=FPS):
    """Audio deployment runs, split on invalid frames or discontinuous clocks."""
    valid, times = array(valid), array(times)
    if valid.ndim != 1 or valid.dtype != bool or times.shape != valid.shape:
        raise ValueError('Boolean valid and matching times required')
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Strictly increasing finite times required')
    linked = np.isclose(np.diff(times), 1 / fps, rtol=1e-4, atol=1e-7)
    runs, start = [], None
    for t, ok in enumerate(valid):
        if start is not None and (not ok or (t > 0 and not linked[t - 1])):
            runs.append((start, t))
            start = None
        if ok and start is None:
            start = t
    if start is not None:
        runs.append((start, len(valid)))
    return runs


def fit_input_statistics(clips):
    """Fit native audio and global-context moments, equal weight per fit clip."""
    if not clips:
        raise ValueError('Nonempty fit clips required')
    first, second, contexts = [], [], []
    ids = []
    for clip in clips:
        if clip['split'] != 'train':
            raise ValueError('Only fit-pool clips can fit statistics')
        valid = array(clip['valid'])
        native_runs(valid, clip['times'])
        x = array(clip['features']).astype(np.float64)
        context = array(clip['context']).astype(np.float64).reshape(-1)
        if x.shape != (len(valid), 1540) or not valid.any() or not np.isfinite(x[valid]).all() or not np.isfinite(context).all():
            raise ValueError('Finite native audio [T,1540] and context required')
        first.append(x[valid].mean(0))
        second.append(np.square(x[valid]).mean(0))
        contexts.append(context)
        ids.append(clip['clip_id'])
    mean = np.mean(first, axis=0)
    context = np.stack(contexts)
    return {'audio_mean': torch.tensor(mean, dtype=torch.float32),
            'audio_scale': torch.tensor(np.sqrt(np.maximum(np.mean(second, axis=0) - mean ** 2, 0)).clip(.01), dtype=torch.float32),
            'context_mean': torch.tensor(context.mean(0), dtype=torch.float32),
            'context_scale': torch.tensor(context.std(0).clip(.01), dtype=torch.float32),
            'fit_ids': ids, 'weighting': 'equal clips, equal native frames within clip',
            'source': 'features and context only; no target or channel mask statistics'}


def prepare_clip(clip, stats, *, supervised):
    """Keep inference features independent of target values/observability."""
    baseline = torch.as_tensor(clip['baseline52']).detach().cpu().float()
    valid = array(clip['valid'])
    runs = native_runs(valid, clip['times'])
    audio = torch.as_tensor(clip['features']).detach().cpu().float()
    context = torch.as_tensor(clip['context']).detach().cpu().float().reshape(-1)
    if baseline.shape != (len(valid), 52) or audio.shape != (len(valid), 1540):
        raise ValueError('Expected baseline[T,52] and features[T,1540]')
    if not torch.isfinite(baseline[valid]).all() or not torch.isfinite(audio[valid]).all() or not torch.isfinite(context).all():
        raise ValueError('Finite native inference inputs required')
    normalized = torch.zeros_like(audio)
    normalized[valid] = (audio[valid] - stats['audio_mean']) / stats['audio_scale']
    output = {'clip_id': clip['clip_id'], 'sentence': clip['sentence'], 'baseline': baseline,
              'audio': normalized, 'context': (context - stats['context_mean']) / stats['context_scale'],
              'runs': runs, 'times': torch.as_tensor(clip['times']).cpu()}
    if supervised:
        raw_mask = array(clip['channel_mask'])
        target = torch.as_tensor(clip['target52']).detach().cpu().float()
        if raw_mask.shape != baseline.shape or raw_mask.dtype != bool or target.shape != baseline.shape:
            raise ValueError('Raw Boolean channel_mask[T,52] and target52 required')
        observed = torch.tensor(raw_mask[:, MOUTH] & valid[:, None])
        y = target[:, MOUTH]
        if not torch.isfinite(y[observed]).all():
            raise ValueError('Nonfinite observed mouth target')
        output.update(target=torch.where(observed, y, 0), observed=observed,
                      observed_count=int(observed.sum()))
    return output


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(width)

    def forward(self, x, valid):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return (x + torch.nn.functional.silu(self.norm(h))) * valid


class MouthResidualTCN(nn.Module):
    def __init__(self, context_dim, width=64, max_delta=.15):
        super().__init__()
        if width < 1 or not 0 < max_delta <= .2:
            raise ValueError('Positive width and 0 < max_delta <= .2 required')
        self.max_delta = float(max_delta)
        self.input = nn.Linear(1540 + 27 + context_dim, width)
        self.blocks = nn.ModuleList(TemporalBlock(width, d) for d in (1, 2, 4, 8))
        self.output = nn.Linear(width, 27)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, audio, mouth, context, lengths):
        valid = (torch.arange(audio.shape[1], device=audio.device)[None] < lengths[:, None])[..., None]
        x = torch.cat((audio, mouth, context[:, None].expand(-1, audio.shape[1], -1)), dim=-1)
        x = torch.nn.functional.silu(self.input(x)) * valid
        for block in self.blocks:
            x = block(x, valid)
        return self.max_delta * torch.tanh(self.output(x)) * valid


def pack_runs(clips, arm, device):
    """No target-derived chunking; static uses full native-run audio means."""
    if arm not in ('audio', 'matched_static', 'audio_reverse'):
        raise ValueError('Unknown intervention')
    rows = [(ci, left, right) for ci, clip in enumerate(clips) for left, right in clip['runs']]
    if not rows:
        raise ValueError('No native audio runs')
    length = max(right - left for _, left, right in rows)
    audio = torch.zeros(len(rows), length, 1540)
    mouth = torch.zeros(len(rows), length, 27)
    contexts, lengths = [], []
    for j, (ci, left, right) in enumerate(rows):
        clip, n = clips[ci], right - left
        values = clip['audio'][left:right]
        if arm == 'matched_static':
            values = values.mean(0, keepdim=True).expand(n, -1)
        elif arm == 'audio_reverse':
            values = values.flip(0)
        audio[j, :n] = values
        mouth[j, :n] = clip['baseline'][left:right, MOUTH]
        contexts.append(clip['context'])
        lengths.append(n)
    return (audio.to(device), mouth.to(device), torch.stack(contexts).to(device),
            torch.tensor(lengths, device=device), rows)


def reconstruction_loss(delta, mouth, rows, clips):
    """Single MSE objective: equal clips, equal observed coefficient values."""
    sums = [delta.sum() * 0 for _ in clips]
    counts = [0 for _ in clips]
    for j, (ci, left, right) in enumerate(rows):
        support = clips[ci]['observed'][left:right].to(delta.device)
        target = clips[ci]['target'][left:right].to(delta.device)
        error = mouth[j, :right - left] + delta[j, :right - left] - target
        sums[ci] = sums[ci] + torch.where(support, error.square(), 0).sum()
        counts[ci] += int(support.sum())
    if not all(counts):
        raise ValueError('Each sampled fitting clip needs observed mouth values')
    return torch.stack([total / count for total, count in zip(sums, counts)]).mean()


@torch.no_grad()
def predict_clip(model, prepared, arm, device):
    """Return full52; no access to target/mask, no clamp or target calibration."""
    result = prepared['baseline'].clone()
    if not prepared['runs']:
        return result
    audio, mouth, context, lengths, rows = pack_runs([prepared], arm, device)
    delta = model(audio, mouth, context, lengths).cpu()
    for j, (_, left, right) in enumerate(rows):
        result[left:right, MOUTH] = result[left:right, MOUTH] + delta[j, :right - left]
    # Bitwise-equivalent copy includes NaN bit patterns outside deployment support.
    if not torch.equal(result[:, NONMOUTH].view(torch.int32), prepared['baseline'][:, NONMOUTH].view(torch.int32)):
        raise RuntimeError('Nonmouth coefficients changed')
    return result


def mouth_diagnostics(prediction, clip):
    x = array(prediction).astype(np.float64)
    y = array(clip['target52']).astype(np.float64)
    m = array(clip['channel_mask']) & array(clip['valid'])[:, None]
    clock = array(clip['times']).astype(np.float64)
    dt = np.diff(clock)
    adjacent = np.isclose(dt, 1 / FPS, rtol=1e-4, atol=1e-7)
    if not np.isfinite(x[m]).all() or not np.isfinite(y[m]).all():
        raise ValueError('Finite observed prediction/target required')
    result = {'clip_id': clip['clip_id'], 'sentence': clip['sentence']}
    for name, region in (('lip23', LIPS), ('mouth27', MOUTH)):
        mask = m[:, region]
        error = np.where(mask, x[:, region] - y[:, region], 0)
        result[name + '_mae'] = float(np.abs(error).sum() / mask.sum()) if mask.any() else None
        pair = mask[1:] & mask[:-1] & adjacent[:, None]
        velocity_error = np.diff(x[:, region], axis=0) - np.diff(y[:, region], axis=0)
        result[name + '_velocity_mae_per_second'] = float(np.abs(np.where(pair, velocity_error / dt[:, None], 0)).sum() / pair.sum()) if pair.any() else None
        pred_std, gt_std = [], []
        for channel in region:
            support = m[:, channel]
            if support.sum() >= 2:
                pred_std.append(x[support, channel].std())
                gt_std.append(y[support, channel].std())
        result[name + '_temporal_std_pred'] = float(np.mean(pred_std)) if pred_std else None
        result[name + '_temporal_std_gt'] = float(np.mean(gt_std)) if gt_std else None
        result[name + '_temporal_std_absolute_gap'] = float(np.abs(np.asarray(pred_std) - gt_std).mean()) if pred_std else None
        result[name + '_raw_oob_fraction'] = float(((x[:, region] < 0) | (x[:, region] > 1))[mask].mean()) if mask.any() else None
    result['seed_diversity_caution'] = 'Training seeds are model repetitions, not stochastic draws; deterministic mouth branch has no within-model multimodality estimate.'
    return result


def summarize_diagnostics(rows):
    keys = [key for key in rows[0] if key not in ('clip_id', 'sentence', 'seed_diversity_caution')]
    return {key: float(np.mean([row[key] for row in rows if row[key] is not None]))
            if any(row[key] is not None for row in rows) else None for key in keys}


def evaluate(model, prepared, raw_clips, arm, device, output):
    rows, diagnostics, predictions = [], [], {}
    if model is not None:
        model.eval()
    for item, raw in zip(prepared, raw_clips):
        pred = item['baseline'] if model is None else predict_clip(model, item, arm, device)
        rows.append(score_fullface(pred.numpy()[None], raw))
        diagnostics.append(mouth_diagnostics(pred, raw))
        # Store candidate trajectories separately; these are inner validation only.
        predictions[item['clip_id']] = pred
    report = build_report(rows, scope='fixed inner sentence validation; historically exposed upstream; not sealed test')
    atomic_json(output / 'arkit_benchmark.json', report)
    atomic_json(output / 'mouth_diagnostics.json', {'summary': summarize_diagnostics(diagnostics), 'per_clip': diagnostics})
    atomic_save(output / 'predictions.pt', {'schema': SCHEMA, 'arm': arm, 'clips': predictions})
    return {'arkit': report['summary'], 'diagnostics': summarize_diagnostics(diagnostics)}


def train_one(fit, *, context_dim, seed, arm, args, output, status):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = MouthResidualTCN(context_dim, args.width, args.max_delta).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    rng = np.random.default_rng(seed + 10000)
    history = []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        indices = rng.integers(0, len(fit), size=args.batch_size)
        clips = [fit[int(i)] for i in indices]
        audio, mouth, context, lengths, rows = pack_runs(clips, arm, args.device)
        loss = reconstruction_loss(model(audio, mouth, context, lengths), mouth, rows, clips)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite mouth candidate loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            item = {'step': step, 'loss': float(loss.detach()), 'elapsed_seconds': time.monotonic() - started}
            history.append(item)
            status('training', seed=seed, arm=arm, **item)
            atomic_json(output / 'history.json', history)
        if step % args.save_every == 0 or step == args.steps:
            atomic_save(output / 'latest.pt', {'schema': SCHEMA, 'state': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'step': step, 'seed': seed, 'arm': arm,
                        'rng_state': rng.bit_generator.state, 'context_dim': context_dim,
                        'width': args.width, 'max_delta': args.max_delta,
                        'torch_rng_state': torch.get_rng_state(), 'default_replaced': False})
    model.eval()
    return model


def run(args):
    if min(args.steps, args.batch_size, args.width, args.log_every, args.save_every) < 1:
        raise ValueError('Positive budgets required')
    if args.output.exists():
        raise FileExistsError('Fresh output directory required; checkpoints are archival, no implicit resume')
    reference = json.loads(args.reference_protocol.read_text(encoding='utf8'))
    if protocol_base.sha(args.dataset) != reference['dataset_sha256']:
        raise ValueError('Dataset hash differs from bound reference protocol')
    data = torch.load(args.dataset, weights_only=False, map_location='cpu', mmap=True)
    if data.get('schema') != protocol_base.DATASET_SCHEMA:
        raise ValueError('Unexpected dataset schema')
    fit_raw, valid_raw = protocol_base.split_train_pool(data['clips'], reference)
    # No outer target, feature, or baseline is indexed. Inherited statistics are discarded.
    del data
    stats = fit_input_statistics(fit_raw)
    fit = [prepare_clip(clip, stats, supervised=True) for clip in fit_raw]
    excluded = [clip['clip_id'] for clip in fit if not clip['observed_count']]
    fit = [clip for clip in fit if clip['observed_count']]
    if not fit:
        raise ValueError('No observed fitting mouth values')
    valid = [prepare_clip(clip, stats, supervised=False) for clip in valid_raw]
    args.output.mkdir(parents=True)
    started = time.time()
    def status(state, **details):
        atomic_json(args.output / 'status.json', {'state': state, 'updated': time.time(),
                    'started': started, 'default_replaced': False, 'lip_sync_certified': False, **details})
    protocol = {'schema': SCHEMA, 'dataset_sha256': reference['dataset_sha256'],
                'reference_protocol_sha256': protocol_base.sha(args.reference_protocol),
                'source_sha256': protocol_base.sha(__file__),
                'dependency_source_sha256': {Path(module.__file__).name: protocol_base.sha(module.__file__)
                    for module in (protocol_base, arkit_benchmark_report, evaluate_arkit_literature_metrics)},
                'fit_ids': [c['clip_id'] for c in fit_raw], 'validation_ids': [c['clip_id'] for c in valid_raw],
                'excluded_no_mouth_observations': excluded, 'seeds': args.seeds,
                'steps_per_arm': args.steps, 'arms': ['audio', 'matched_static'],
                'batch_size': args.batch_size, 'width': args.width, 'learning_rate': args.learning_rate,
                'max_delta': args.max_delta, 'changed_channels': list(MOUTH),
                'exact_preserved_channels': list(NONMOUTH), 'preserved_channel_count': 25,
                'objective': 'one raw masked MSE; equal clips then observed coefficient values',
                'static_control': 'same init/batches/budget; native-run mean audio; dynamic frozen mouth baseline retained in BOTH arms',
                'audio_claim_caution': 'Incremental ordered-audio benefit is relative to a baseline that is already audio-driven.',
                'support': 'native valid plus clock-contiguous runs; target mask used only in training loss/scoring',
                'stats': 'fit only; full native audio/context, independent of target mask',
                'scope': 'inner development only, no historical outer targets indexed, no sealed test',
                'upstream_exposure': 'frozen baseline and feature extractors inherit historical exposure',
                'selection': 'fixed final step, report every seed; no default replacement or best-seed choice',
                'acceptance': 'Coefficient diagnostic only; needs lip-sync render and perceptual evaluation. No LBE-only acceptance.',
                'multimodality': 'Deterministic mouth branch; training seeds are repetitions, not draws; official external metrics remain pending.',
                'device': args.device}
    atomic_json(args.output / 'protocol.json', protocol)
    atomic_save(args.output / 'fit_statistics.pt', stats)
    atomic_json(args.output / 'statistics_manifest.json', {
        'sha256': protocol_base.sha(args.output / 'fit_statistics.pt'),
        'fit_ids': stats['fit_ids'], 'target_statistics_used': False,
        'audio_valid_mask_used': True, 'target_channel_mask_used': False})
    (args.output / 'executed_source.py').write_bytes(Path(__file__).read_bytes())
    result = {'baseline': evaluate(None, valid, valid_raw, 'baseline', args.device, args.output / 'baseline'), 'seeds': {}}
    for seed in args.seeds:
        result['seeds'][str(seed)] = {}
        for arm in ('audio', 'matched_static'):
            path = args.output / f'seed_{seed}' / arm
            model = train_one(fit, context_dim=valid[0]['context'].numel(), seed=seed, arm=arm,
                              args=args, output=path, status=status)
            status('evaluating', seed=seed, arm=arm, step=args.steps)
            result['seeds'][str(seed)][arm] = evaluate(model, valid, valid_raw, arm, args.device, path / 'validation')
            if arm == 'audio':
                result['seeds'][str(seed)]['audio_reverse'] = evaluate(model, valid, valid_raw, 'audio_reverse', args.device, path / 'validation_reverse')
            atomic_json(args.output / 'results.json', result)
            del model
    atomic_json(args.output / 'results.json', result)
    status('completed', elapsed_seconds=time.time() - started, steps_per_arm=args.steps,
           seeds=args.seeds, conclusion='Candidate evaluated; deployment and lip-sync acceptance not performed.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'reference-protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--max-delta', type=float, default=.15)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        run(args)
    except Exception as exc:
        if args.output.exists() and not isinstance(exc, FileExistsError):
            atomic_json(args.output / 'status.json', {'state': 'failed', 'updated': time.time(),
                        'error': f'{type(exc).__name__}: {exc}', 'default_replaced': False})
        raise


if __name__ == '__main__':
    main()
