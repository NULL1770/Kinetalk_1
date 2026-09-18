"""Audio semantic student with frozen ridge level and learned temporal residual.

The visual target is used only when fitting/scoring. Deployment reads acoustic
features and native validity only. A separate pretrained ridge/PCA baseline is
bound to each sentence-OOF fitting partition; neither is refitted on holdout.
The residual has zero *audio-frame-weighted raw* mean before VA clipping and
posterior softmax. Those nonlinear transforms need not preserve output means.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import fit_visual_semantic_student as ridge

SCHEMA = 'temporal_visual_semantic_student_v1'
SEED = 20260919


def audio_input(clip, preprocessing, reverse=False, constant_audio=False):
    """PCA values plus their acoustic clip-centered changes; no visual reads."""
    source = {'features': ridge._numpy(clip['features']).copy(), 'valid': ridge._numpy(clip['valid']).copy()}
    original = ridge.audio_windows(source)
    windows = {**original, 'values': original['values'].copy()}
    if reverse and constant_audio:raise ValueError('choose one acoustic intervention')
    if constant_audio:
        windows['values'][:] = original['static']
    if reverse:
        for run in np.unique(windows['run_ids']):
            indices = np.flatnonzero(windows['run_ids'] == run)
            windows['values'][indices] = windows['values'][indices[::-1]]
    z = ((windows['values']-preprocessing['mean'])/preprocessing['scale']) @ preprocessing['components']
    weights = np.array([right-left for left, right in windows['spans']], np.float64)
    mean = np.average(z, axis=0, weights=weights)
    return {'features': np.concatenate((z, z-mean), axis=1).astype(np.float32),
            'weights': weights.astype(np.float32), 'run_ids': windows['run_ids'].copy(),
            'windows': windows, 'original_windows': original}


def training_target(clip, windows):
    """Center teacher window targets using observed teacher-frame counts only."""
    values, known = ridge._targets(clip, windows)
    observed = ridge._numpy(clip['semantic_valid']) & windows['valid']
    counts = np.array([observed[left:right].sum() for left, right in windows['spans']], np.float64)
    if not known.any():
        raise ValueError('each fitting clip requires an observed semantic window')
    mean = np.average(values[known], axis=0, weights=counts[known])
    centered = np.where(known[:, None], values-mean, 0.)
    return {'target': centered.astype(np.float32), 'known': known, 'counts': counts.astype(np.float32),
            'teacher_mean': mean.copy()}


class _ResidualBlock(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(hidden)
        self.temporal = nn.Linear(hidden*3, hidden)
        self.output = nn.Linear(hidden, hidden)

    def forward(self, value, valid, run_ids):
        h = torch.where(valid[..., None], F.silu(self.norm(value)), 0.)
        d = self.dilation
        if d >= h.shape[1]:
            left, right = torch.zeros_like(h), torch.zeros_like(h)
        else:
            pair = (valid[:, d:] & valid[:, :-d] & (run_ids[:, d:] == run_ids[:, :-d]))[..., None]
            left = F.pad(torch.where(pair, h[:, :-d], 0.), (0, 0, d, 0))
            right = F.pad(torch.where(pair, h[:, d:], 0.), (0, 0, 0, d))
        result = value+self.output(F.silu(self.temporal(torch.cat((left, h, right), -1))))
        return torch.where(valid[..., None], result, 0.)


class TemporalResidualTCN(nn.Module):
    """31-window noncausal local field, followed by acoustic clip centering."""
    def __init__(self, input_dim, hidden=64, paired_small=False):
        super().__init__()
        if type(paired_small) is not bool or (paired_small and input_dim % 2):
            raise ValueError('paired_small requires Boolean setting and paired raw/centered input width')
        self.input_dim = input_dim
        self.paired_small = paired_small
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(_ResidualBlock(hidden, d) for d in (1, 2, 4, 8))
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 10)
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.zeros_(self.output.bias)

    def local_field(self, x, valid, run_ids):
        if (valid.dtype != torch.bool or x.ndim != 3 or x.shape[:2] != valid.shape
                or x.shape[-1] != self.input_dim or run_ids.shape != valid.shape
                or not valid.any(1).all() or not torch.isfinite(x[valid]).all()):
            raise ValueError('matching finite acoustic windows, valid and run IDs required')
        x = torch.where(valid[..., None], x, 0.)
        h = torch.where(valid[..., None], self.input(x), 0.)
        for block in self.blocks: h = block(h, valid, run_ids)
        return torch.where(valid[..., None], self.output(self.norm(h)), 0.)

    def forward(self, x, valid, run_ids, weights):
        if (weights.shape != valid.shape or not torch.isfinite(weights).all()
                or (weights[valid] <= 0).any() or (weights[~valid] != 0).any()):
            raise ValueError('positive native frame counts on valid windows and zero padding required')
        if self.paired_small:
            half = self.input_dim//2
            observed = torch.where(valid[..., None], x, 0.)
            # Algebraically the audio-frame-weighted PCA mean; anchoring on a
            # real first window avoids reduction roundoff on constant input.
            # Consequently a constant raw PCA track produces an exactly
            # identical twin input and bitwise-zero residual, without a gate.
            first = valid.to(torch.int64).argmax(1)
            reference = observed[torch.arange(len(x), device=x.device), first, :half][:, None]
            deviation = torch.where(valid[..., None], observed[..., :half]-reference, 0.)
            mean = reference+(deviation*weights[..., None]).sum(1, keepdim=True)/weights.sum(1)[:, None, None]
            centered = torch.where(valid[..., None], observed[..., :half]-mean, 0.)
            real = torch.cat((observed[..., :half], centered), -1)
            constant = torch.cat((mean.expand_as(centered), torch.zeros_like(centered)), -1)
            constant = torch.where(valid[..., None], constant, 0.)
            # Both shared-weight branches remain in the autograd graph.
            result = self.local_field(real, valid, run_ids)-self.local_field(constant, valid, run_ids)
        else:
            result = self.local_field(x, valid, run_ids)
        mean = (result*weights[..., None]).sum(1, keepdim=True)/weights.sum(1)[:, None, None]
        return torch.where(valid[..., None], result-mean, 0.)


def _batch(records, ids, device):
    length = max(len(records[i]['features']) for i in ids)
    dimension = records[ids[0]]['features'].shape[1]
    x = torch.zeros(len(ids), length, dimension, device=device)
    valid = torch.zeros(len(ids), length, dtype=torch.bool, device=device)
    weights = torch.zeros(len(ids), length, device=device)
    runs = torch.full((len(ids), length), -1, dtype=torch.int64, device=device)
    target = torch.zeros(len(ids), length, 10, device=device)
    target_weights = torch.zeros(len(ids), length, device=device)
    for row, index in enumerate(ids):
        item = records[index]; count = len(item['features'])
        x[row, :count] = torch.as_tensor(item['features'], device=device)
        valid[row, :count] = True
        weights[row, :count] = torch.as_tensor(item['weights'], device=device)
        runs[row, :count] = torch.as_tensor(item['run_ids'], device=device)
        if 'target' in item:
            target[row, :count] = torch.as_tensor(item['target'], device=device)
            target_weights[row, :count] = torch.as_tensor(item['counts'], device=device)
    return x, valid, runs, weights, target, target_weights


def _fit_scale(records):
    # Every clip has equal influence; its observed teacher frames share weight.
    moment = np.stack([(record['target'].astype(float)**2*record['counts'][:, None]).sum(0)
                       /record['counts'].sum() for record in records]).mean(0)
    return np.sqrt(moment).clip(.05).astype(np.float32)


def normalized_residual_loss(prediction, target, counts, scale):
    if counts.ndim != 2 or (counts.sum(1) <= 0).any():
        raise ValueError('every training clip needs observed target support')
    error = ((prediction-target/scale)**2).mean(-1)
    return ((error*counts).sum(1)/counts.sum(1)).mean()


def _membership(model, clips):
    ids = [c['clip_id'] for c in clips]; sentences = sorted({c['sentence'] for c in clips})
    if (len(model['fit_clip_ids']) != len(ids) or set(model['fit_clip_ids']) != set(ids)
            or sorted(model['fit_sentences']) != sentences):
        raise ValueError('baseline fitting clip/sentence membership differs')


def fit_model(clips, baseline_model, *, epochs=120, device='cpu', batch_size=16, paired_small=False):
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('positive epochs and batch size required')
    _membership(baseline_model, clips)
    preprocessing = baseline_model['preprocessing']
    records = []
    for clip in clips:
        item = audio_input(clip, preprocessing)
        item.update(training_target(clip, item['windows']))
        records.append(item)
    scale = _fit_scale(records)
    config = {'input_dim': records[0]['features'].shape[1], 'hidden': 16 if paired_small else 64,
              'paired_small': paired_small}
    # Baseline parameters live in numpy payloads and never enter the optimizer.
    torch.manual_seed(SEED)
    model = TemporalResidualTCN(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    scale_tensor = torch.as_tensor(scale, device=device)
    order_rng = np.random.default_rng(SEED)
    history = []; order_hash = hashlib.sha256(); steps = 0; started = time.time()
    for epoch in range(epochs):
        order = order_rng.permutation(len(records)); order_hash.update(order.tobytes())
        total = 0.; count = 0
        model.train()
        for left in range(0, len(order), batch_size):
            ids = order[left:left+batch_size].tolist()
            x, valid, runs, weights, target, observed = _batch(records, ids, device)
            prediction = model(x, valid, runs, weights)
            loss = normalized_residual_loss(prediction, target, observed, scale_tensor)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite temporal semantic loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            total += float(loss.detach())*len(ids); count += len(ids); steps += 1
        history.append({'epoch': epoch+1, 'normalized_residual_mse': total/count, 'steps': steps})
    return {'schema': SCHEMA, 'config': config, 'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'baseline': copy.deepcopy(baseline_model), 'residual_scale': scale,
            'fit_clip_ids': [c['clip_id'] for c in clips], 'fit_sentences': sorted({c['sentence'] for c in clips}),
            'epochs': epochs, 'steps': steps, 'seed': SEED, 'history': history,
            'order_sha256': order_hash.hexdigest(), 'seconds': time.time()-started,
            'residual_mean': 'audio-native-frame weighted zero in raw VA/log-posterior space before nonlinear decoding'}


@torch.no_grad()
def predict_audio(model, clip, *, device='cpu'):
    """Deployable API: reads only acoustic features and native valid mask."""
    baseline = model['baseline']
    original = ridge.audio_windows(clip)
    static_head = baseline['heads']['static']
    static_raw = ridge._design(original, baseline['preprocessing'], 'static') @ static_head['coefficient']+static_head['intercept']
    net = TemporalResidualTCN(**model['config']).to(device)
    net.load_state_dict(model['state']); net.eval()
    output = {'audio_valid': torch.from_numpy(original['valid'].copy()), 'va': {}, 'posterior': {}}
    for mode in (*ridge.MODES, 'constant_audio'):
        if mode == 'static':
            raw = static_raw.copy(); windows = original
        else:
            record = audio_input(clip, baseline['preprocessing'], reverse=mode == 'reverse',
                                 constant_audio=mode == 'constant_audio')
            x, valid, runs, weights, _, _ = _batch([record], [0], device)
            residual = net(x, valid, runs, weights)[0].cpu().numpy()*model['residual_scale']
            raw = static_raw+residual.astype(np.float64)
            windows = record['windows']
        va = np.clip(raw[:, :2], -1., 1.)
        logits = raw[:, 2:]; posterior = np.exp(logits-logits.max(1, keepdims=True))
        posterior /= posterior.sum(1, keepdims=True)
        for name, values, width in (('va', va, 2), ('posterior', posterior, 8)):
            native = np.zeros((len(original['valid']), width), np.float32)
            for index, (left, right) in enumerate(windows['spans']):native[left:right] = values[index]
            output[name][mode] = torch.from_numpy(native)
    return output


def output_mean_diagnostics(predictions):
    """Report nonlinear mean changes; never correct generated probabilities."""
    clips = []
    for row in predictions:
        valid = ridge._numpy(row['audio_valid'])
        semantic_valid = valid & ridge._numpy(row['semantic_valid'])
        item = {'clip_id': row['clip_id'], 'split': row['split'], 'groups': {}, 'semantic_scoring_support': {}}
        for name in ('va', 'posterior'):
            baseline = ridge._numpy(row[name]['static'])[valid].astype(float).mean(0)
            measures = {}
            for mode in ('actual', 'reverse', 'constant_audio'):
                values = ridge._numpy(row[name][mode])[valid].astype(float)
                delta = values.mean(0)-baseline
                measures[mode] = {'mean_shift_vector': delta.tolist(), 'mean_shift_rms': float(np.sqrt(np.mean(delta**2))),
                                  'temporal_rms': float(np.sqrt(np.mean((values-values.mean(0))**2)))}
            item['groups'][name] = measures
            subset = {}
            if semantic_valid.any():
                static_subset = ridge._numpy(row[name]['static'])[semantic_valid].astype(float).mean(0)
                for mode in ('actual', 'reverse', 'constant_audio'):
                    values = ridge._numpy(row[name][mode])[semantic_valid].astype(float)
                    delta = values.mean(0)-static_subset
                    subset[mode] = {'mean_shift_vector': delta.tolist(), 'mean_shift_rms': float(np.sqrt(np.mean(delta**2))),
                                    'temporal_rms': float(np.sqrt(np.mean((values-values.mean(0))**2)))}
            item['semantic_scoring_support'][name] = subset
        clips.append(item)
    return {'reference': 'frozen ridge static; all native acoustic valid frames, no visual mask',
            'constant_audio': 'same temporal network with every acoustic window set to whole-clip audio mean; not the frozen static baseline',
            'support_caveat': 'raw residual is centered on all acoustic frames; its mean on a visual-observed subset need not be zero even before clipping/softmax',
            'nonlinear_mean_preservation_claim': False, 'clips': clips}


def validate_baseline(data, baseline_predictions, baseline_models):
    clips = data['clips']; rows = baseline_predictions['clips']
    if (not clips or len({c['clip_id'] for c in clips}) != len(clips)
            or [c['clip_id'] for c in clips] != [r['clip_id'] for r in rows]
            or any(c['split'] not in ('train', 'holdout') for c in clips)):
        raise ValueError('baseline and dataset unique ordered membership differs')
    train = [c for c in clips if c['split'] == 'train']; holdout = [c for c in clips if c['split'] == 'holdout']
    sentences = sorted({c['sentence'] for c in train})
    folds = {sentence: index % 3 for index, sentence in enumerate(sentences)}
    if (len(sentences) < 3 or set(sentences) & {c['sentence'] for c in holdout}
            or baseline_predictions['fold_by_sentence'] != folds):
        raise ValueError('baseline sentence split/folds differ')
    _membership(baseline_models['all_train'], train)
    for fold in range(3):_membership(baseline_models['oof'][fold], [c for c in train if folds[c['sentence']] != fold])
    for clip, row in zip(clips, rows):
        expected_fold = folds[clip['sentence']] if clip['split'] == 'train' else None
        if (row['sentence'] != clip['sentence'] or row['split'] != clip['split'] or row['fold'] != expected_fold
                or row['prediction_source'] != ('sentence_oof' if clip['split'] == 'train' else 'all_train')
                or not np.array_equal(ridge._numpy(row['audio_valid']), ridge._numpy(clip['valid']))):
            raise ValueError('baseline OOF prediction metadata differs')
    return train, holdout, folds


def fit_dataset(data, baseline_predictions, baseline_models, *, epochs=120, device='cpu', batch_size=16, paired_small=False):
    train, _, folds = validate_baseline(data, baseline_predictions, baseline_models)
    all_model = fit_model(train, baseline_models['all_train'], epochs=epochs, device=device, batch_size=batch_size, paired_small=paired_small)
    print('TEMPORAL_FIT_COMPLETE all_train', all_model['steps'], round(all_model['seconds'], 2), flush=True)
    fold_models = {}
    for fold in range(3):
        subset = [c for c in train if folds[c['sentence']] != fold]
        fold_models[fold] = fit_model(subset, baseline_models['oof'][fold], epochs=epochs, device=device, batch_size=batch_size, paired_small=paired_small)
        print('TEMPORAL_FIT_COMPLETE oof', fold, fold_models[fold]['steps'], round(fold_models[fold]['seconds'], 2), flush=True)
    predictions = []
    for clip in data['clips']:
        fold = folds[clip['sentence']] if clip['split'] == 'train' else None
        model = all_model if fold is None else fold_models[fold]
        row = predict_audio(model, clip, device=device)
        row.update(clip_id=clip['clip_id'], sentence=clip['sentence'], split=clip['split'], fold=fold,
                   prediction_source='all_train' if fold is None else 'sentence_oof',
                   semantic_valid=torch.from_numpy(ridge._numpy(clip['semantic_valid']).copy()))
        predictions.append(row)
    scores = ridge._scores(predictions, data['clips'])
    scores['output_mean_diagnostics'] = output_mean_diagnostics(predictions)
    constant_rows = []
    for row in predictions:
        temporary = {**row, 'va': {**row['va'], 'actual': row['va']['constant_audio']},
                     'posterior': {**row['posterior'], 'actual': row['posterior']['constant_audio']}}
        constant_rows.append(temporary)
    constant_scores = ridge._scores(constant_rows, data['clips'])
    scores['constant_audio_diagnostic'] = {split: {name: constant_scores[split][name]['actual']
                                         for name in ('va', 'posterior')} for split in ('train', 'holdout')}
    return {'schema': SCHEMA, 'clips': predictions, 'fold_by_sentence': folds}, {
        'all_train': all_model, 'oof': fold_models}, scores


def load_baseline(dataset_path, baseline_path):
    provenance_path = baseline_path.with_name('provenance.json')
    models_path = baseline_path.with_name('models.pt')
    provenance = json.loads(provenance_path.read_text(encoding='utf8'))
    if (provenance.get('schema') != ridge.SCHEMA or provenance['dataset_sha256'] != ridge._sha(dataset_path)
            or provenance['outputs'].get('predictions.pt') != ridge._sha(baseline_path)
            or provenance['outputs'].get('models.pt') != ridge._sha(models_path)):
        raise ValueError('frozen ridge baseline hash or dataset binding mismatch')
    return torch.load(baseline_path, map_location='cpu', weights_only=False), torch.load(models_path, map_location='cpu', weights_only=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'baseline-student', 'output'):parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=120); parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--paired-small', action='store_true', help='Post-hoc16-channel shared-weight real-minus-constant temporal field')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('fresh output required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    baseline_predictions, baseline_models = load_baseline(args.dataset, args.baseline_student)
    data = torch.load(args.dataset, map_location='cpu', weights_only=False)
    predictions, models, scores = fit_dataset(data, baseline_predictions, baseline_models,
                                             epochs=args.epochs, device=args.device, batch_size=args.batch_size,
                                             paired_small=args.paired_small)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, args.output/'predictions.pt'); torch.save(models, args.output/'models.pt')
    (args.output/'report.json').write_text(json.dumps(scores, indent=2, allow_nan=False)+'\n', encoding='utf8')
    provenance = {'schema': SCHEMA, 'dataset_sha256': ridge._sha(args.dataset), 'code_sha256': ridge._sha(__file__),
        'ridge_code_sha256': ridge._sha(ridge.__file__), 'baseline_student_sha256': ridge._sha(args.baseline_student),
        'baseline_models_sha256': ridge._sha(args.baseline_student.with_name('models.pt')),
        'baseline_provenance_sha256': ridge._sha(args.baseline_student.with_name('provenance.json')),
        'epochs': args.epochs, 'batch_size': args.batch_size, 'seed': SEED, 'learning_rate': 3e-4,
        'weight_decay': .01, 'gradient_clip': 1., 'hidden': 16 if args.paired_small else 64,
        'paired_small': args.paired_small, 'dilations': [1, 2, 4, 8],
        'paired_small_interpretation': 'post-hoc exploration jointly reduces capacity and subtracts shared-weight constant-acoustic field; not an isolated single-factor ablation' if args.paired_small else None,
        'local_receptive_field_windows': 31, 'nominal_local_seconds': 6.2,
        'context_is_noncausal': True, 'global_context': 'whole acoustic clip mean centering; no visual availability input',
        'target': 'window VA2 and log(mean posterior8), minus observed-frame-weighted teacher clip mean',
        'input': 'baseline fold-specific PCA values concatenated with audio-frame-weighted clip-centered changes',
        'residual': 'single clip-equal normalized residual MSE, observed-frame weighting, fit-only RMS floor .05',
        'output_mean_caveat': 'zero raw residual mean before clip/softmax does not imply unchanged final VA/posterior mean',
        'static': 'exact old ridge static output; baseline weights and preprocessing frozen',
        'constant_audio': 'same trained TCN fed repeated whole-clip acoustic mean; separate diagnostic for edge/position-induced changes',
        'reverse': 'reverse already-averaged acoustic window values within each valid run; preserve native spans including short last window; keep old static level',
        'centering_weights': 'teacher target mean uses observed visual frame counts; prediction residual mean uses all acoustic native window lengths; masks never enter deployment',
        'deployment_uses_motion_or_visual_targets': False, 'holdout_used_for_selection': False,
        'outputs': {name: ridge._sha(args.output/name) for name in ('predictions.pt', 'models.pt', 'report.json')}}
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({'schema': SCHEMA, 'clips': len(predictions['clips']), 'output': str(args.output)}))


if __name__ == '__main__': main()
