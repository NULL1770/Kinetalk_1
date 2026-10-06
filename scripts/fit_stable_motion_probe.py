"""Fit a separately named train-only motion probe with a preregistered mask.

The mask is determined only from real train feature scales.  This is an
auxiliary robustness readout; it never replaces the original probe protocol
and never reads generated or sealed-test data during fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.packed_trainval_cache import load_packed
from kinetalk_b0.emotion_probe import MotionEmotionProbe, motion_features, classification_metrics


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def collect(split, support):
    rows = []
    for i in range(len(split['_lengths'])):
        b = split.batch(torch.tensor([i]), keys=('motion', 'valid'))
        rows.append(motion_features(b['motion'][0][:, support], b['valid'][0]))
    return torch.stack(rows)


def run(args):
    if args.output.exists():
        raise FileExistsError('Fresh probe output required')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    data = load_packed(args.data, materialize=False)
    if set(data['splits']) != {'train', 'validation'} or data['provenance']['test_loaded']:
        raise ValueError('Only train/validation data allowed')
    train_split, val_split = data['splits']['train'], data['splits']['validation']
    support = train_split['channel_mask'].all(0)
    if not val_split['channel_mask'][:, support].all():
        raise ValueError('Validation support differs from train support')
    train, val = collect(train_split, support), collect(val_split, support)
    print(json.dumps({'event': 'features', 'train': len(train), 'validation': len(val)}), flush=True)
    # The original probe has six statistic groups, followed by channels.
    c = int(support.sum())
    train_scale = train.std(0, correction=0)
    keep = (train_scale.reshape(6, c) > args.threshold).reshape(-1)
    if int(keep.sum()) < 32:
        raise ValueError('Stable feature mask unexpectedly small')
    names = data['config']['data']['emotion_classes']
    model = MotionEmotionProbe(int(keep.sum()), args.hidden, len(names))
    model.fit_normalization(train[:, keep])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    labels = train_split['emotion_id'].clone()
    val_labels = val_split['emotion_id'].clone()
    counts = torch.bincount(labels, minlength=len(names)).float()
    weights = len(labels) / (len(names) * counts.clamp_min(1))
    generator = torch.Generator().manual_seed(args.seed)
    best, best_epoch, best_state, history = -1., None, None, []
    for epoch in range(args.epochs):
        model.train()
        for ids in torch.randperm(len(train), generator=generator).split(args.batch_size):
            loss = F.cross_entropy(model(train[ids][:, keep]), labels[ids], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(val[:, keep]).argmax(-1)
        score = classification_metrics(val_labels, pred, names)
        history.append({'epoch': epoch + 1, **score})
        if score['macro_f1'] > best:
            best, best_epoch = score['macro_f1'], epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError('No best probe state')
    args.output.mkdir(parents=True)
    ck = {
        'kind': 'stable_train_motion_statistics_probe_v1',
        'model': best_state, 'hidden': args.hidden, 'feature_dim': int(keep.sum()),
        'classes': names, 'channel_support': support, 'feature_mask': keep,
        'train_full_feature_scale': train_scale,
        'threshold': args.threshold, 'train_manifest_sha256': data['provenance']['manifest_sha256'],
        'test_used_for_selection': False, 'generator_outputs_used_for_fitting': False,
        'normalization': 'train only', 'selection': 'validation macro-F1; earliest tie',
        'seed': args.seed, 'best_epoch': best_epoch, 'source_sha256': file_sha(Path(__file__)),
    }
    torch.save(ck, args.output / 'probe.pt')
    (args.output / 'report.json').write_text(json.dumps({
        'schema': 'stable_train_motion_probe_v1', 'status': 'complete',
        'best_epoch': best_epoch, 'validation_macro_f1': best,
        'feature_count': int(keep.sum()), 'feature_total': int(keep.numel()),
        'threshold': args.threshold, 'seed': args.seed, 'hidden': args.hidden,
        'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
        'test_loaded': False, 'data_manifest_sha256': data['provenance']['manifest_sha256'],
        'history': history, 'checkpoint_sha256': file_sha(args.output / 'probe.pt'),
        'policy': 'Auxiliary stability readout; original probe protocol remains unchanged.'
    }, indent=2), encoding='utf8')
    print(json.dumps({'event': 'complete', 'output': str(args.output), 'features': int(keep.sum()),
                      'best_epoch': best_epoch, 'validation_macro_f1': best}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=1.0001e-4)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.threshold <= 0 or args.epochs < 1 or args.hidden < 1:
        raise ValueError('Invalid probe configuration')
    run(args)


if __name__ == '__main__':
    main()
