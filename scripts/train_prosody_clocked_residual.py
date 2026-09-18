"""Small prosody-only timing correction on a completed frozen static prior.

The motion dictionary, slow equilibrium, feature statistics and full-acoustic
static prior are reused exactly. Only an eight-bin, four-prosody-channel head
learns; query motion never enters that head or the sampling path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_clocked_residual_prior as r
from kinetalk_b0.models.clocked_motion_prior import ClockedMotionPrior, categorical_energy_score

c = r.c
SCHEMA = 'clocked_prosody_static_residual_v1'
PROSODY = slice(1536, 1540)
SOURCE_FILES = ('static_final.pt', 'residual_final.pt', 'dictionary.pt', 'equilibrium.pt',
                'scales.json', 'training_data.json', 'protocol.json', 'status.json')


class ProsodyStaticResidualPrior(nn.Module):
    """Frozen static logits plus a bounded, exactly static-null prosody head.

The recipient clip's original whole-clip valid acoustic mean is passed as
``static_features``. Under interventions that mean and global state stay fixed.
There is no per-window centering, fitted query normalization or global input
to the correction. Bin boundaries remain on the original H32 clock.
"""

    def __init__(self, base):
        super().__init__()
        if not isinstance(base, ClockedMotionPrior) or base.feature_dim != 1540 or base.horizon != 32:
            raise ValueError('A saved H32, F1540 ClockedMotionPrior base is required')
        self.base = base.eval().requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        self.horizon, self.k = base.horizon, base.k
        self.correction = nn.Sequential(nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, base.k))
        self.correction.to(device=base.feature_mean.device, dtype=base.feature_mean.dtype)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval().requires_grad_(False)
        return self

    def temporal_descriptor(self, features, valid, static_features):
        if (not torch.is_tensor(features) or not features.is_floating_point()
                or features.ndim != 3 or features.shape[0] < 1
                or features.shape[1:] != (32, 1540)
                or features.device != self.base.feature_std.device
                or features.dtype != self.base.feature_std.dtype
                or not torch.is_tensor(static_features) or static_features.shape != features.shape
                or static_features.device != features.device or static_features.dtype != features.dtype
                or not torch.is_tensor(valid) or valid.dtype != torch.bool
                or valid.shape != features.shape[:2] or valid.device != features.device
                or not torch.isfinite(features[..., PROSODY][valid]).all()
                or not torch.isfinite(static_features[..., PROSODY][valid]).all()):
            raise ValueError('Finite observed prosody in matching [B,32,1540] audio and Boolean mask required')
        real = torch.where(valid[..., None], features[..., PROSODY], 0.)
        origin = torch.where(valid[..., None], static_features[..., PROSODY], 0.)
        centered = (real-origin)/self.base.feature_std[PROSODY]
        bins = []
        for index in range(8):
            left, right = index*4, (index+1)*4
            observed = valid[:, left:right]
            bins.append(centered[:, left:right].sum(1)/observed.sum(1, keepdim=True).clamp_min(1))
        descriptor = torch.cat(bins, -1)
        if not torch.isfinite(descriptor).all():
            raise FloatingPointError('Prosody descriptor is nonfinite')
        return descriptor

    def residual_logits(self, features, valid, global_features, static_features):
        # The four-argument signature is shared with the existing evaluator;
        # global_features is deliberately not consumed by the correction.
        descriptor = self.temporal_descriptor(features, valid, static_features)
        return self.correction(descriptor).tanh()-self.correction(torch.zeros_like(descriptor)).tanh()

    def forward(self, features, valid, global_features, static_features, *, scale=1.):
        if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError('Residual scale must be finite numeric in [0,1]')
        self.base.eval().requires_grad_(False)
        with torch.no_grad():
            baseline = self.base(static_features, valid, global_features)
        delta = self.residual_logits(features, valid, global_features, static_features)
        result = baseline+float(scale)*delta
        if not torch.isfinite(result).all():
            raise FloatingPointError('Prosody residual logits are nonfinite')
        return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load_source(source, device):
    """Read a completed formal source, never refitting or training its base."""
    source = Path(source)
    protocol, status = read(source/'protocol.json'), read(source/'status.json')
    if (protocol.get('schema') != r.SCHEMA or status.get('schema') != r.SCHEMA
            or status.get('status') != 'complete' or status.get('smoke') is not False
            or status.get('test_loaded') is not False):
        raise ValueError('Completed formal bounded clocked residual source required')
    root = Path(__file__).resolve().parents[1]
    for relative, digest in protocol['code_sha256'].items():
        if c.old.sha(root/relative) != digest:
            raise ValueError('Source dependency has changed: '+relative)
    static = torch.load(source/'static_final.pt', map_location='cpu', weights_only=False)
    residual = torch.load(source/'residual_final.pt', map_location='cpu', weights_only=False)
    if (static.get('schema') != r.SCHEMA or residual.get('schema') != r.SCHEMA
            or static.get('epochs') != 30 or residual.get('epochs') != 30
            or static.get('order_sha256') != residual.get('order_sha256')):
        raise ValueError('Expected saved static30/residual30 source with matching order')
    state = static['state']
    for key, value in state.items():
        if not torch.equal(value, residual['state']['base.'+key]):
            raise ValueError('Source residual changed its static base: '+key)
    stats = [state[key] for key in ('feature_mean', 'feature_std', 'global_mean', 'global_std')]
    base = ClockedMotionPrior(*stats, horizon=c.HORIZON, hidden=state['input.weight'].shape[0],
                              k=state['output.weight'].shape[0])
    base.load_state_dict(state, strict=True)
    dictionary = torch.load(source/'dictionary.pt', map_location='cpu', weights_only=False)
    if (dictionary.get('coordinate_system') != 'logit raw minus observed training clip logit mean'
            or dictionary.get('horizon') != c.HORIZON or dictionary.get('hop') != c.HOP
            or len(dictionary['shapes']) != base.k or base.k != 128):
        raise ValueError('Saved logit dictionary coordinates/clock/K differ')
    fitted = torch.load(source/'equilibrium.pt', map_location='cpu', weights_only=False)
    scales = np.asarray(read(source/'scales.json'), dtype=np.float64)
    training = read(source/'training_data.json')
    if scales.shape != (9,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Saved raw scales must be nine positive finite values')
    low_cut = float(training['low_activity_train_quantile20'])
    if not np.isfinite(low_cut) or low_cut < 0:
        raise ValueError('Invalid source low-activity threshold')
    binding = {'source_run': str(source.resolve()),
        'files_sha256': {name: c.old.sha(source/name) for name in SOURCE_FILES},
        'base_unchanged_exact': True, 'dictionary_refitted': False,
        'equilibrium_refitted': False, 'normalization_refitted': False,
        'source_code_sha256': protocol['code_sha256']}
    torch.manual_seed(c.SEED)
    model = ProsodyStaticResidualPrior(base.to(device))
    return model, dictionary, fitted, scales, low_cut, protocol, binding


def train_model(model, data, dictionary, output, device, epochs):
    """One proper-score objective; only the 2704-parameter correction learns."""
    before = {key: value.detach().cpu().clone() for key, value in model.base.state_dict().items()}
    tensors = {key: value.to(device) for key, value in data.items()}
    distance = torch.tensor(c.shapes.pairwise_distances(dictionary['shapes'], dictionary['scales']),
                            dtype=torch.float32, device=device)
    parameters = list(model.correction.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.0003, weight_decay=.01)
    rng = np.random.default_rng(c.SEED)
    history, orderhash = [], hashlib.sha256()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(tensors['valid'])); orderhash.update(order.tobytes())
        total, count = 0., 0
        for start in range(0, len(order), 128):
            ix = torch.as_tensor(order[start:start+128], device=device)
            logits = model(tensors['temporal'][ix], tensors['valid'][ix], tensors['global'][ix], tensors['static'][ix])
            scores = categorical_energy_score(logits.softmax(-1), tensors['distance'][ix], distance)
            loss = (scores*tensors['weight'][ix]).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
            total += float(loss.detach())*len(ix); count += len(ix)
        history.append({'epoch': epoch+1, 'energy_score': total/count, 'updates': (len(order)+127)//128})
        c.old.save_json(output/'residual_losses.json', history)
        c.old.save_json(output/'status.json', {'schema': SCHEMA, 'status': 'training',
            'arm': 'prosody_residual', 'epoch': epoch+1, 'epochs': epochs})
        print('EPOCH', 'prosody_residual', epoch+1, total/count, flush=True)
    model.eval()
    if any(not torch.equal(value.detach().cpu(), before[key]) for key, value in model.base.state_dict().items()):
        raise RuntimeError('Frozen static base changed during correction training')
    if any(p.grad is not None or p.requires_grad for p in model.base.parameters()):
        raise RuntimeError('Frozen static base received a training gradient')
    torch.save({'schema': SCHEMA, 'state': model.state_dict(), 'epochs': epochs,
                'order_sha256': orderhash.hexdigest(), 'optimizer': optimizer.state_dict()}, output/'residual_final.pt')
    c.old.save_json(output/'matching.json', {'base_unchanged_exact': True, 'epochs': epochs,
        'windows': len(tensors['valid']), 'order_sha256': orderhash.hexdigest(),
        'trainable_parameters': sum(p.numel() for p in parameters), 'optimizer_base_parameters': 0,
        'correction_global_input': False, 'correction_channels': [1536, 1537, 1538, 1539]})
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio', 'targets', 'native-root', 'native-manifest', 'delta-dir', 'audio-checkpoint', 'source-run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh output required')
    args.output.mkdir(parents=True); started = time.monotonic()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    c.old.save_json(args.output/'status.json', {'schema': SCHEMA, 'status': 'loading'})
    model, dictionary, fitted, scales, low_cut, source_protocol, binding = load_source(args.source_run, args.device)
    loading = copy.copy(args); loading.smoke = False
    clips, original, lineage = c.old.load_clips(loading)
    if lineage != source_protocol['source']:
        raise ValueError('Live dataset lineage differs from frozen source')
    split = c.previous.split_inner(clips, original)
    for cell in ('fit', 'calibration', 'confirmation'):
        actual = [{key: clips[i][key] for key in ('clip_id', 'sentence', 'speaker', 'emotion')} for i in split[cell]]
        if actual != source_protocol['split'][cell]:
            raise ValueError('Source split membership differs: '+cell)
    if not set(dictionary['source_clip_ids']).issubset({clips[i]['clip_id'] for i in split['fit']}):
        raise ValueError('Saved dictionary contains a non-fitting clip')
    if set(fitted['fit_clip_ids']) != {clips[i]['clip_id'] for i in split['fit']}:
        raise ValueError('Saved equilibrium fitting membership differs')
    if args.smoke:
        split = {key: value[:12] for key, value in split.items()}
    ids = sorted({i for cell in ('fit', 'calibration', 'confirmation') for i in split[cell]})
    targets = torch.load(args.targets, map_location='cpu', weights_only=False, mmap=True)['splits']['train']
    anchors = {cid: row[c.previous.CC].numpy() for cid, row in zip(targets['clip_id'], targets['anchors'])}
    frozen = c.load_frozen_audio(args.audio_checkpoint, args.device)
    audio_binding = c.assert_source_binding(frozen, lineage)
    if audio_binding != source_protocol['frozen_audio']:
        raise ValueError('Frozen audio source differs')
    contexts = c.encode_clips(frozen, [clips[i] for i in ids], batch_size=16, device=args.device)
    for i, context in zip(ids, contexts):
        clips[i]['global'] = np.r_[context['global'].numpy(), context['intensity'].numpy()]
        clips[i]['anchor_upper'] = anchors[clips[i]['clip_id']]
    del frozen, contexts, targets
    root = Path(__file__).resolve().parents[1]
    files = sorted(set(source_protocol['code_sha256']) | {'scripts/train_prosody_clocked_residual.py'})
    protocol_path = root/'docs/CLOCKED_PROSODY_PROTOCOL_20260918.md'
    protocol = {'schema': SCHEMA, 'source': lineage, 'source_run_binding': binding,
        'frozen_audio': audio_binding, 'smoke': args.smoke, 'development_only': True,
        'confirmation_scope': 'Consumed development regression, not untouched',
        'split': {cell: [{key: clips[i][key] for key in ('clip_id', 'sentence', 'speaker', 'emotion')}
                         for i in split[cell]] for cell in ('fit', 'calibration', 'confirmation')},
        'code_sha256': {f: c.old.sha(root/f) for f in files}, 'protocol_sha256': c.old.sha(protocol_path),
        'test_loaded': False, 'motion_teacher_inputs': False,
        'prosody_origin': 'Original recipient clip valid acoustic mean, unchanged under interventions',
        'prosody_scale': 'Frozen source feature_std[1536:1540]',
        'epochs': 2 if args.smoke else 30, 'seeds': list(c.old.SEEDS)}
    c.old.save_json(args.output/'protocol.json', protocol)
    for relative in files:
        dest = args.output/'source'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/relative, dest)
    shutil.copyfile(protocol_path, args.output/'source/protocol.md')
    (args.output/'source_run').mkdir()
    for name in SOURCE_FILES:
        shutil.copyfile(args.source_run/name, args.output/'source_run'/name)
        if c.old.sha(args.output/'source_run'/name) != binding['files_sha256'][name]:
            raise RuntimeError('Source changed while snapshotting: '+name)
    for name in ('dictionary.pt', 'equilibrium.pt', 'scales.json', 'static_final.pt'):
        shutil.copyfile(args.output/'source_run'/name, args.output/name)
    transformed, clipping = r.coordinate_clips(clips, split['fit'])
    windows = c.shapes.extract_windows(transformed, split['fit'], horizon=c.HORIZON, hop=c.HOP)
    if not args.smoke and len(windows) != dictionary['source_window_count']:
        raise ValueError('Fitting window count differs from frozen dictionary source')
    if not args.smoke:
        from collections import Counter
        counts = dict(Counter(window['clip_id'] for window in windows))
        if counts != dictionary['source_window_counts']:
            raise ValueError('Fitting per-clip window membership differs from frozen dictionary source')
    c.old.save_json(args.output/'fit_clipping.json', clipping)
    data = c.build_training(clips, windows, dictionary)
    c.old.save_json(args.output/'training_data.json', {'windows': len(windows), 'clips': len(split['fit']),
        'low_activity_train_quantile20': low_cut, 'coordinate': 'logit', 'clip_balanced': True,
        'dictionary_refitted': False, 'equilibrium_refitted': False, 'normalization_refitted': False})
    history = train_model(model, data, dictionary, args.output, args.device, 2 if args.smoke else 30)
    del data, windows, transformed
    choices, baseline = [], None
    for scale in (0., .25, .5, 1.):
        report, _ = r.evaluate(clips, split['calibration'], dictionary, fitted, scales, low_cut,
                                model, args.device, residual_scale=scale)
        c.old.save_json(args.output/('cal_scale_'+str(scale)+'.json'), report)
        if baseline is None:
            baseline = report
        choice = {'scale': scale, 'score': report['summary']['joint_fair_es']['centered'],
                  'eligible': r.eligible(report, baseline)}
        choices.append(choice); print('CAL_SCALE', json.dumps(choice), flush=True)
    chosen = min((row for row in choices if row['eligible']), key=lambda row: row['score'])['scale']
    c.old.save_json(args.output/'selection.json', {'scale': chosen, 'choices': choices,
        'scope': 'Calibration only; smoke uses first12 per original split' if args.smoke else '199 calibration only'})
    for cell in ('calibration', 'confirmation'):
        reports = {}
        for label, intervention, scale in [('static_trained', 'real', 0.), ('temporal', 'real', chosen),
                ('static', 'static', chosen), ('reverse', 'reverse', chosen), ('mismatch', 'mismatch', chosen)]:
            report, curves = r.evaluate(clips, split[cell], dictionary, fitted, scales, low_cut,
                model, args.device, intervention=intervention, residual_scale=scale)
            reports[label] = report
            c.old.save_json(args.output/(cell+'_'+label+'.json'), report)
            torch.save(curves, args.output/(cell+'_'+label+'.pt'))
            print('EVAL', cell, label, report['summary']['joint_fair_es'] if report['summary'] else None, flush=True)
        gate = c.assessment(reports)
        if chosen == 0.:
            gate['timing_passed'] = False; gate['reason'] = 'zero_residual_scale_selected'
        c.old.save_json(args.output/(cell+'_assessment.json'), gate)
        print('ASSESSMENT', cell, json.dumps(gate), flush=True)
    c.old.save_json(args.output/'status.json', {'schema': SCHEMA, 'status': 'complete',
        'seconds': time.monotonic()-started, 'epochs': len(history), 'smoke': args.smoke,
        'selected_scale': chosen, 'timing_passed': gate['timing_passed'], 'quality_passed': gate['quality_passed'],
        'base_unchanged_exact': True, 'development_only': True, 'test_loaded': False,
        'generator_integrated': False, 'default_replaced': False})


if __name__ == '__main__':
    main()
