"""Four deployable stages with explicit teacher/audio temporal conditioning.

The default protocol is articulation → identity → teacher → audio.  The
motion teacher supplies global affect semantics, while the audio encoder
supplies the native-rate temporal condition consumed by the residual flow.
The former upper-face Stage5 remains available only through legacy helpers
used by historical tests/scripts; it is not instantiated or checkpointed by
this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.full_staged_data import load_training_inputs
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, capture_rng, restore_rng, canonical_hash
from scripts.train_predictable_renderer import state_hash
from kinetalk_b0.semantic_losses import style_contrastive
from kinetalk_b0.emotion_probe import MotionEmotionProbe, motion_features
from kinetalk_b0.intensity_losses import build_intensity_triplets, masked_rms, ordinal_intensity_loss
from kinetalk_b0.models.slow_state_affect import (
    SlowStateAffect, UPPER_INDICES, readout_slow_state, masked_slow_state,
    affect_audio_statistics, migrate_affect_audio_state,
    lift_slow_state, compose_upper_face, project_upper_innovation, UpperInnovationFlow,
)
from kinetalk_b0.models.audio_residual_flow import static_audio
from kinetalk_b0.models.model import ARTICULATORY_MOUTH_INDICES, AFFECT_MOUTH_INDICES

SCHEMA = 'full_staged_teacher_audio_temporal_v2'
# The former upper-face Stage5 remains in legacy modules for checkpoint
# compatibility, but it is no longer part of the default experiment protocol.
STAGES = ('articulation', 'identity', 'teacher', 'audio')
MOUTH = tuple(range(14, 41))
ARTICULATORY_MOUTH = tuple(ARTICULATORY_MOUTH_INDICES)
AFFECT_MOUTH = tuple(AFFECT_MOUTH_INDICES)
GROUPS = {'brows': (41,42,43,44,45), 'eyes_expression': (5,6,12,13), 'mouth': MOUTH}
NOT_UPPER = tuple(i for i in range(52) if i not in UPPER_INDICES)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def subset(q, ids, device, *, keys=None):
    if hasattr(q, 'batch'):
        return q.batch(ids, device, keys=keys)
    result = {}
    for key, value in q.items():
        if keys is not None and key not in keys: continue
        if torch.is_tensor(value):
            selected = value[ids]
            result[key] = selected.to(device=device, dtype=torch.float32 if selected.is_floating_point() and key != 'times' else selected.dtype)
        elif isinstance(value, list): result[key] = [value[int(i)] for i in ids]
    return result

def obs(q): return q['valid'][...,None] & q['channel_mask'][:,None]

def base_forward(system, content, valid, *, gradients=False):
    """Match true-length B0 evaluation while explicitly allowing Stage1 gradients."""
    last = (valid * torch.arange(1, valid.shape[1]+1, device=valid.device)).amax(1)
    output = {}
    with torch.set_grad_enabled(gradients):
        for length_tensor in last.unique():
            length = int(length_tensor)
            ids = (last == length).nonzero(as_tuple=True)[0]
            clean = torch.where(valid[ids,:length,None], content[ids,:length], 0.)
            values = system.stage1(clean, valid[ids,:length])
            for key in ('b0','h0'):
                padded = F.pad(values[key], (0,0,0,valid.shape[1]-length))
                if key not in output: output[key] = padded.new_zeros(len(content), valid.shape[1], padded.shape[-1])
                output[key] = output[key].index_copy(0, ids, padded)
    output = {k:torch.where(valid[...,None],v,0.) for k,v in output.items()}
    if hasattr(system, 'motion_support'):
        output['b0'] = torch.where(system.motion_support, output['b0'], 0.)
    return output

def unfreeze(module):
    module.requires_grad_(True)
    return list(module.parameters())

def articulation_update_plan(clip_count, batch_size, epochs, updates=None):
    """Keep full epochs by default, or match isolated B0 studies by updates."""
    if any(type(x) is not int or x < 1 for x in (clip_count, batch_size, epochs)):
        raise ValueError('Positive clip count, batch size and epoch count required')
    batches = math.ceil(clip_count / batch_size)
    if updates is None:
        return epochs, batches, epochs * batches
    if type(updates) is not int or updates < 1:
        raise ValueError('Positive integer articulation update budget required')
    return math.ceil(updates / batches), batches, updates


def articulation_epoch_batches(order, batch_size, completed_updates, update_budget):
    batches = order.split(batch_size)
    if update_budget is None:
        return batches
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError('Completed updates must be a nonnegative integer')
    if type(update_budget) is not int or update_budget < 1:
        raise ValueError('Positive articulation update budget required')
    remaining = update_budget - completed_updates
    if remaining < 1:
        raise ValueError('Articulation update budget already exhausted')
    return batches[:remaining]


def stage1_training_dropout(stage1, probability):
    """Isolated B0 training policy; None preserves legacy mode/probabilities.

    Explicit zero is the matched train-mode control. The caller restores eval
    mode before producing cached B0/h0. No config or state-dict keys change.
    """
    if probability is None:
        return None
    if (isinstance(probability, bool) or not math.isfinite(probability) or
            not 0 <= probability <= .5):
        raise ValueError('Stage1 training dropout must be finite in [0, .5]')
    counts = {'dropout_modules': 0, 'attention_modules': 0}
    for module in stage1.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = float(probability)
            counts['dropout_modules'] += 1
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = float(probability)
            counts['attention_modules'] += 1
    stage1.train()
    return counts


def audio_parameter_groups(system, audio, *, paper_data, freeze_renderer=False):
    """Freeze only receiver weights; preserve gradient paths to audio conditions.

    No no_grad context is used around the renderer. Its training/dropout mode
    remains controlled by the existing stage loop, identically in both arms.
    """
    groups = [{'params': unfreeze(audio), 'lr': 1e-4}]
    if freeze_renderer:
        system.renderer.requires_grad_(False)
        system.renderer.zero_grad(set_to_none=True)
    else:
        groups.append({'params': unfreeze(system.renderer),
                       'lr': 5e-5 if paper_data else 1e-5})
    return groups

def load_warm_system(system, state, *, allow_zero_temporal_adapter=False):
    """Keep strict warm loading except one explicit zero-initialized new weight."""
    if not allow_zero_temporal_adapter:
        return system.load_state_dict(state, strict=True)
    key = 'renderer.motion_temporal.weight'
    expected = system.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if unexpected or missing not in (set(), {key}):
        raise ValueError('Unexpected warm-start state mismatch for temporal adapter')
    if key in missing:
        if system.renderer.motion_temporal is None or torch.count_nonzero(expected[key]):
            raise ValueError('New temporal adapter must start from exactly zero')
        state = {**state, key: expected[key]}
    return system.load_state_dict(state, strict=True)

def mse(x,y,mask):
    if mask.shape != x.shape: mask = mask.expand_as(x)
    if not mask.any(): return x.sum()*0
    return (x[mask]-y[mask]).square().mean()

def huber(x,y,mask):
    if mask.shape != x.shape: mask=mask.expand_as(x)
    return F.smooth_l1_loss(x[mask],y[mask],beta=1.) if mask.any() else x.sum()*0

def flow_vector_mse(prediction, target, observed, channel_std=None):
    """Existing flow error, optionally expressed in fixed TRAIN channel units.

    No vector-field, physical-source, output, support or clock changes.
    """
    if channel_std is None:
        return mse(prediction, target, observed)
    if (prediction.shape != target.shape or prediction.shape != observed.shape or
            observed.dtype != torch.bool or channel_std.shape != prediction.shape[-1:] or
            channel_std.device != prediction.device or not torch.isfinite(channel_std).all() or
            (channel_std <= 0).any() or not observed.any()):
        raise ValueError('Flow units require matching observations and positive fixed channel std')
    error = prediction[observed] - target[observed]
    units = channel_std.view(1, 1, -1).expand_as(prediction)[observed]
    return (error / units).square().mean()


def coordinate_flow_error_std(source_std, *, mode='coordinate'):
    """Choose error units independently of standardized renderer coordinates.

    Physical mode retains the original scalar residual normalization (0.25);
    it does not alter the renderer, its conditions or the Gaussian source.
    Default coordinate mode preserves all existing standardized computations.
    """
    if mode not in ('coordinate', 'physical'):
        raise ValueError('Coordinate flow error mode must be coordinate or physical')
    if (not torch.is_tensor(source_std) or source_std.ndim != 1 or
            not source_std.is_floating_point() or not torch.isfinite(source_std).all() or
            (source_std <= 0).any()):
        raise ValueError('Coordinate flow error requires positive finite TRAIN std')
    return source_std if mode == 'coordinate' else None


def balanced_flow_units(channel_std, support):
    """Normalize TRAIN inverse-variance weights to mean one on fixed support.

    Relative precision weights are preserved; their overall scale does not
    multiply the flow coefficient merely because some channels are tiny.
    """
    if (channel_std.shape != support.shape or support.dtype != torch.bool or
            not support.any() or not torch.isfinite(channel_std).all() or
            (channel_std <= 0).any()):
        raise ValueError('Balanced flow units require positive TRAIN std and fixed support')
    std = channel_std.double()
    mean_precision = std[support.to(std.device)].square().reciprocal().mean()
    return (std * mean_precision.sqrt()).to(channel_std)


def class_balanced_weights(labels, num_classes, *, power=0.5):
    """Return normalized inverse-frequency weights from the fit labels only.

    ``power=0.5`` is intentionally a tempered correction: it gives rare
    emotions more gradient without letting a tiny class dominate the flow
    objective.  Missing classes keep weight zero and are never present in the
    corresponding CE target.
    """
    labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.ndim != 1 or num_classes < 1 or power < 0:
        raise ValueError('Invalid labels/num_classes/power for class weights')
    if labels.numel() and ((labels < 0) | (labels >= num_classes)).any():
        raise ValueError('Emotion label is outside the classifier range')
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float32)
    observed = counts > 0
    weights = torch.zeros_like(counts)
    weights[observed] = counts[observed].reciprocal().pow(float(power))
    if observed.any():
        weights[observed] /= weights[observed].mean().clamp_min(1e-8)
    return weights


def load_flow_source_stats(path, *, checkpoint_sha256, manifest_sha256, support, residual_scale):
    """Validate the fixed TRAIN-only population spread used by both arms."""
    stats = json.loads(path.read_text(encoding='utf8'))
    if (stats.get('schema') != 'frozen_train_residual_source_spread_v1' or
            stats.get('source_fit_split') != 'train' or stats.get('test_loaded') is not False or
            stats.get('validation_motion_used') is not False or stats.get('training_performed') is not False or
            stats.get('checkpoint_sha256') != checkpoint_sha256 or
            stats.get('data_manifest_sha256') != manifest_sha256 or stats.get('normalization') != residual_scale or
            stats.get('residual_support') != support.tolist() or stats.get('train_clips', 0) <= 0 or
            stats.get('nondegeneracy_floor_normalized') != 1e-4):
        raise ValueError('Flow source statistics protocol/checkpoint binding mismatch')
    std = torch.as_tensor(stats['residual_std'], dtype=torch.float64)
    count = torch.as_tensor(stats['count'])
    value = torch.as_tensor(stats['normalized_source_std'], dtype=torch.float64)
    if (std.shape != support.shape or count.shape != support.shape or value.shape != support.shape or
            not torch.isfinite(std).all() or (std < 0).any() or not torch.isfinite(value).all() or
            (value <= 0).any() or not (count[support] > 0).all() or (count[~support] != 0).any()):
        raise ValueError('Invalid flow source statistics observations/spread')
    expected = torch.where(count > 0, (std / residual_scale).clamp_min(1e-4), 1.)
    torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
    return value.float(), {'path': str(path.resolve()), 'sha256': sha(path),
                          'train_clips': stats['train_clips'], 'source_fit_split': 'train',
                          'normalized_source_std': value.tolist(), 'validation_motion_used': False}


def configure_flow_coordinates(system, cfg, path, source_std, support, mode):
    """Refit the head from a shared TRAIN mean field in matched coordinates.

    The caller has already validated the complete TRAIN statistics and warm
    checkpoint binding through load_flow_source_stats. No RNG is consumed.
    """
    if mode not in ('train-centered-scalar', 'train-standardized'):
        raise ValueError('Explicit matched coordinate refit mode required')
    stats = json.loads(path.read_text(encoding='utf8'))
    mean = torch.as_tensor(stats['residual_mean'], dtype=torch.float64)
    if (mean.shape != support.shape or not torch.isfinite(mean).all() or
            (mean[~support] != 0).any()):
        raise ValueError('Invalid TRAIN residual means for fixed support')
    mean = (mean / system.residual_scale).float()
    std = source_std if mode == 'train-standardized' else torch.ones_like(source_std)
    cfg['model']['flow_coordinate_mean'] = mean.tolist()
    cfg['model']['flow_coordinate_std'] = std.tolist()
    system.renderer.set_channel_coordinates(mean, std)
    with torch.no_grad():
        system.renderer.output.weight.zero_()
        system.renderer.output.bias.zero_()
    return {'mode': mode, 'statistics_sha256': sha(path),
            'normalized_mean': mean.tolist(), 'normalized_std': std.tolist(),
            'initialization': 'Shared zero output head: physical initial velocity equals TRAIN mean',
            'scope': 'Input/output coordinate transform and matching flow error units; all channels retained'}


def load_flow_temporal_stats(path, *, checkpoint_sha256, manifest_sha256, support, source_info):
    """Bind TRAIN lag moments to the unchanged diagonal marginal source."""
    stats = json.loads(path.read_text(encoding='utf8'))
    if (stats.get('schema') != 'frozen_train_demeaned_residual_lag_v1' or
            stats.get('source_fit_split') != 'train' or stats.get('test_loaded') is not False or
            stats.get('validation_motion_used') is not False or stats.get('training_performed') is not False or
            stats.get('checkpoint_sha256') != checkpoint_sha256 or
            stats.get('data_manifest_sha256') != manifest_sha256 or
            stats.get('source_stats_sha256') != source_info['sha256'] or
            stats.get('residual_support') != support.tolist() or
            stats.get('train_clips') != source_info['train_clips'] or
            stats.get('nondegeneracy_rho_margin') != 1e-4):
        raise ValueError('Flow temporal statistics protocol/checkpoint/spread binding mismatch')
    result = stats['results']
    count, cross, left, right, rho = [torch.as_tensor(result[k], dtype=torch.float64) for k in
        ('adjacent_count', 'cross_sum', 'left_square_sum', 'right_square_sum', 'source_rho')]
    if (any(x.shape != support.shape or not torch.isfinite(x).all() for x in (count,cross,left,right,rho)) or
            (count < 0).any() or not (count == count.round()).all() or
            (left < 0).any() or (right < 0).any() or (rho.abs() >= 1).any() or
            not (count[support] > 0).all() or (count[~support] != 0).any()):
        raise ValueError('Invalid native-adjacent TRAIN lag moments')
    denominator = (left * right).sqrt()
    expected = torch.where(denominator > 0, cross / denominator.clamp_min(1e-300), 0.).clamp(-1+1e-4, 1-1e-4)
    torch.testing.assert_close(rho, expected, rtol=1e-12, atol=1e-12)
    if (rho[~support] != 0).any():
        raise ValueError('Unobserved source channels require rho=0')
    return rho.float(), {'path': str(path.resolve()), 'sha256': sha(path),
                        'train_clips': stats['train_clips'], 'source_fit_split': 'train',
                        'source_rho': rho.tolist(), 'validation_motion_used': False}


def semantics(a,q, emotion_class_weights=None):
    v = F.cross_entropy(a['emotion_logits'],q['emotion_id'], weight=emotion_class_weights)
    good=q['intensity_valid'] & (q['intensity_id']>=0)
    if good.any(): v=v+F.cross_entropy(a['intensity_logits'][good],q['intensity_id'][good])
    return v

def global_distillation(prediction, target, teacher_logits, labels, *, gate='all'):
    """Keep full-batch normalization when skipping contradictory teacher codes.

    Teacher label agreement is a fixed training reliability rule. This does
    not filter flow/motion or real-label semantic supervision and never uses
    an independent generated-motion probe.
    """
    if gate not in ('all', 'teacher-agreement'):
        raise ValueError('Unknown global distillation gate')
    if gate == 'all':
        return F.mse_loss(prediction, target), prediction.new_tensor(1.)
    keep = teacher_logits.detach().argmax(-1).eq(labels)
    per_clip = (prediction - target.detach()).square().mean(-1)
    return (per_clip * keep.to(per_clip.dtype)).mean(), keep.float().mean()

def articulation_selection(q, scope='neutral'):
    """Return articulation clips and an auditable emotion-count summary.

    The legacy neutral-only stage remains the default.  ``all-emotions``
    changes only this stage's membership; it does not alter the later teacher
    or audio objectives, their emotion cross-entropy masks, or any checkpoint
    loading behavior.  Splits and masks are supplied by the caller's approved
    train manifest, so no validation/test rows can enter here.
    """
    if scope not in ('neutral', 'all-emotions'):
        raise ValueError("articulation scope must be 'neutral' or 'all-emotions'")
    if not isinstance(q, dict) or not torch.is_tensor(q.get('emotion_id')):
        raise ValueError('training split must contain tensor emotion_id')
    labels = q['emotion_id']
    if labels.ndim != 1 or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('emotion_id must be a one-dimensional integer tensor')
    ids = torch.arange(len(labels), dtype=torch.long) if scope == 'all-emotions' else (labels == 0).nonzero(as_tuple=True)[0].long()
    counts = {str(int(label)): int((labels[ids] == label).sum())
              for label in torch.unique(labels[ids], sorted=True)}
    if not len(ids):
        raise ValueError('Articulation scope selected no training clips')
    return ids, {'requested_scope': scope, 'actual_scope': scope,
                 'clip_count': len(ids), 'emotion_counts': counts,
                 'all_train_clips_included': bool(len(ids) == len(labels)),
                 'train_emotion_ids': sorted(int(value) for value in torch.unique(labels, sorted=True))}

def load_safe_native_targets(root, clip_ids):
    """Load audited neutral teachers on each source clip's native motion clock.

    The DTW v3 artifact stores ``neutral_teacher_on_source`` on the source
    motion timeline.  It is the only safe target that can be paired directly
    with the native content used by the packed runner; ``canonical_motion``
    contains the emotional source motion and must never supervise B0.
    """
    if root is None:
        return {}
    root = Path(root)
    manifest = root / 'teacher_manifest_gated.jsonl'
    if not manifest.is_file():
        raise FileNotFoundError(f'Missing safe-DTW manifest: {manifest}')
    targets = {}
    allowed_clips = {str(c) for c in clip_ids}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        clip = str(row['source_clip_id'])
        if clip not in allowed_clips:
            continue
        if row.get('teacher_eligible') is not True:
            continue
        artifact = row.get('teacher_artifact')
        if not artifact:
            raise ValueError(f'Safe-DTW row has no teacher artifact: {clip}')
        pair = Path(artifact)
        if not pair.is_file():
            pair = root / 'pairs' / pair.name
        if not pair.is_file():
            raise FileNotFoundError(f'Missing safe-DTW pair artifact: {pair}')
        with np.load(pair, allow_pickle=False) as z:
            if 'neutral_teacher_on_source' not in z.files or 'native_teacher_mask' not in z.files:
                raise ValueError(f'Unmaterialized native neutral teacher: {pair}')
            teacher = np.asarray(z['neutral_teacher_on_source'], dtype=np.float32)
            mask = np.asarray(z['native_teacher_mask'], dtype=bool)
            times = np.asarray(z['source_motion_times'], dtype=np.float64)
        if teacher.ndim != 2 or teacher.shape[1] < 52 or len(teacher) != len(mask) or len(teacher) != len(times):
            raise ValueError(f'Invalid native neutral teacher shape: {pair}')
        if not np.isfinite(teacher).all() or np.any(~np.isfinite(times)):
            raise ValueError(f'Nonfinite native neutral teacher: {pair}')
        # The mouth-event gate is an explicit quality boundary in the safe
        # release.  When it is false, the neutral DTW teacher remains useful
        # for upper-face channels, but its mouth timing is not reliable.  Keep
        # a frame/channel mask so Stage1 never turns a low-quality mouth pair
        # into a phonetic target.  This is stricter than silently accepting a
        # scalar frame mask for all 52 coefficients.
        channel_mask = np.ones((len(teacher), 52), dtype=bool)
        if row.get('event_local_mask_path') or not bool(row.get('mouth_event_gate', False)):
            event_path = row.get('event_local_mask_path')
            if not event_path:
                raise ValueError(f'Ungated safe-DTW row has no event mask: {clip}')
            event_file = Path(event_path)
            if not event_file.is_file():
                event_file = root / 'pair_masks' / event_file.name
            if not event_file.is_file():
                raise FileNotFoundError(f'Missing mouth event mask: {event_file}')
            with np.load(event_file, allow_pickle=False) as event:
                if 'event_local_mask' not in event.files:
                    raise ValueError(f'Mouth event mask missing event_local_mask: {event_file}')
                event_mask = np.asarray(event['event_local_mask'], dtype=bool)
            if event_mask.shape != (len(teacher),):
                raise ValueError(f'Mouth event mask length differs from native teacher: {clip}')
            channel_mask[:, MOUTH] = event_mask[:, None]
        if clip in targets:
            raise ValueError(f'Duplicate safe-DTW source: {clip}')
        targets[clip] = (torch.from_numpy(teacher[:, :52].copy()), torch.from_numpy(mask.copy()),
                         torch.from_numpy(channel_mask), torch.from_numpy(times.copy()),
                         {'artifact_sha256': sha(pair), 'event_mask_sha256': sha(event_file) if row.get('event_local_mask_path') else None})
    if not targets:
        raise ValueError('Safe-DTW manifest has no targets for the requested split')
    return targets

def safe_target_batch(targets, clip_ids, width, device, fallback_motion=None, fallback_valid=None, native_times=None):
    """Pad native safe-DTW teachers to the current variable-length batch."""
    motion = torch.zeros(len(clip_ids), width, 52, dtype=torch.float32, device=device)
    valid = torch.zeros(len(clip_ids), width, dtype=torch.bool, device=device)
    channel = torch.zeros(len(clip_ids), width, 52, dtype=torch.bool, device=device)
    for j, clip in enumerate(clip_ids):
        if str(clip) in targets:
            teacher, mask, target_channel, target_times, _ = targets[str(clip)]
            n = len(teacher)
            if n > width:
                raise ValueError(f'Safe-DTW teacher exceeds the native batch width: {clip}')
            if native_times is not None and not torch.allclose(
                    native_times[j,:n].to(device='cpu',dtype=torch.float64), target_times,
                    atol=1e-7,rtol=0):
                raise ValueError(f'Safe-DTW teacher time axis differs: {clip}')
            motion[j, :n] = teacher[:n].to(device)
            valid[j, :n] = mask[:n].to(device)
            channel[j, :n] = target_channel[:n].to(device)
        elif fallback_motion is not None and fallback_valid is not None:
            motion[j] = fallback_motion[j, :width]
            valid[j] = fallback_valid[j, :width]
            channel[j] = True
        else:
            raise KeyError(f'Missing safe-DTW target for {clip}')
    return motion, valid, channel

def expression_gradient_view(value):
    """Identical motion/velocity values; only native brow/eye Jacobians remain.

    Used exclusively to compute replacement student gradients. The renderer
    still backpropagates the complete original objective, with all mouth
    channels open. No learned mask, target, forward projection or RNG.
    """
    if value.ndim != 3 or value.shape[-1] != 52:
        raise ValueError('Expression gradient view requires [B,T,52]')
    support = torch.zeros(52, dtype=torch.bool, device=value.device)
    support[list(UPPER_INDICES)] = True
    return torch.where(support, value, value.detach())


def student_gradient_overrides(student_loss, audio):
    """Differentiate the existing loss view only with respect to the student."""
    parameters = [p for p in audio.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(student_loss, parameters, retain_graph=True, allow_unused=True)
    return list(zip(parameters, gradients))


def optimize(loss,opt,params,*,gradient_overrides=None):
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective')
    opt.zero_grad(set_to_none=True); loss.backward()
    if gradient_overrides is not None:
        for parameter, gradient in gradient_overrides:
            parameter.grad = None if gradient is None else gradient.detach()
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    opt.step()
    return float(norm)

def identity_pairs(refs, fit_sids):
    pairs=[]
    for sid in fit_sids:
        n=len(refs[sid]['valid']); half=n//2
        if half<1: raise ValueError('Two independent references required')
        for a in itertools.combinations(range(n),half):
            b=tuple(i for i in range(n) if i not in a)
            if len(a)==len(b) and a>b: continue
            pairs.append((sid,a,b))
    return pairs

@torch.no_grad()
def cache_current_base(system,data,device,batch_size=32):
    """Recompute after B0 changes. No old h0/b0 cache is used."""
    for role,q in data['splits'].items():
        out={'b0':[],'h0':[]}
        native = {'b0':[], 'h0':[]}
        for ids in torch.arange(len(q['valid'])).split(batch_size):
            b=subset(q,ids,device); base=base_forward(system,b['content'],b['valid'])
            for k in out:
                if hasattr(q, 'set_extra'):
                    native[k].extend([base[k][j, :int(q['_lengths'][int(i)])].cpu()
                                      for j, i in enumerate(ids.tolist())])
                else:
                    out[k].append(base[k].cpu())
            if int(ids[0]) % (batch_size * 50) == 0:
                print(json.dumps({'event':'base_cache','split':role,'clips':int(ids[-1])+1,'total':len(q['valid'])}),flush=True)
        if hasattr(q, 'set_extra'):
            for k in out: q.set_extra(k, native[k])
        else:
            q['b0']=torch.cat(out['b0']);q['h0']=torch.cat(out['h0'])
    for sid,q in data['refs'].items():
        b=subset(q,torch.arange(len(q['valid'])),device)
        base=base_forward(system,b['content'],b['valid'])
        q['residual']=torch.where(obs(b),b['motion']-base['b0'],0.).cpu()

def encode_ref(system,data,sid,ids,device):
    q=data['refs'][sid]
    result=system.encode_identity(q['residual'][list(ids)][None].to(device),q['valid'][list(ids)][None].to(device),
        reference_channel_mask=q['channel_mask'][list(ids)][None].to(device))
    result['observed_channels']=q['channel_mask'][list(ids)].all(0)[None].to(device)
    return result

def paper_condition(q, mode):
    """LEGACY helper for historical upper-face intervention scripts."""
    if mode == 'audio': return q
    if mode != 'static': raise ValueError('Unknown condition mode')
    # Both explicit local acoustics and B0's audio-derived per-frame h0 must
    # lose temporal information in the independently trained static control.
    return {**q, 'audio_features':static_audio(q['audio_features'],q['valid']),
            'h0':static_audio(q['h0'],q['valid'])}

def reverse_valid(value,valid):
    output=torch.where(valid[...,None],value,0.).clone()
    for i in range(len(value)):
        ids=valid[i].nonzero(as_tuple=True)[0]
        output[i,ids]=value[i,ids.flip(0)]
    return output

def complete_dynamic_conditions(q,local_audio,mode='audio'):
    """LEGACY helper; not called by the four-stage deployable runner.

    Counterfactuals remove/reverse ALL frame-varying dynamic conditions.

    Global affect/identity and generation noise are held by the caller.
    Static pooling is after the encoder, so padding/TCN edge effects cannot
    recreate a frame-varying local condition. Native clock remains unchanged.
    """
    if mode not in ('audio','static','reverse'):raise ValueError('Unknown dynamic intervention')
    dynamic=local_audio(q['audio_features'],q['valid'])
    b=dict(q)
    if mode=='static':
        b['h0']=static_audio(q['h0'],q['valid'])
        dynamic={**dynamic,'state':static_audio(dynamic['state'],q['valid']),
                 'local':static_audio(dynamic['local'],q['valid'])}
    elif mode=='reverse':
        b['h0']=reverse_valid(q['h0'],q['valid'])
        dynamic={**dynamic,'state':reverse_valid(dynamic['state'],q['valid']),
                 'local':reverse_valid(dynamic['local'],q['valid'])}
    return b,dynamic

@torch.no_grad()
def identity_cache(system,data,device):
    return {sid:{k:v.detach() for k,v in encode_ref(system,data,sid,range(len(q['valid'])),device).items()
                 if k in ('code','baseline')} for sid,q in data['refs'].items()}

def batch_identity(identities,q):
    return {k:torch.cat([identities[int(sid)][k] for sid in q['speaker_id']],0) for k in ('code','baseline')}

def generator_inputs(q, identity, ablation='none'):
    """Direct-generation control retains conditions but removes additive priors."""
    if ablation == 'no_residual':
        return {**q, 'b0':torch.zeros_like(q['b0'])}, {**identity, 'baseline':torch.zeros_like(identity['baseline'])}
    return q, identity

def teacher_affect(system,q,identity):
    residual=torch.where(obs(q),q['motion']-q['b0']-identity['baseline'][:,None],0.)
    return system.encode_motion(residual,q['valid'])


def class_global_means(codes, labels, classes):
    """Train-only deterministic per-emotion mean; never guess missing classes."""
    if (codes.ndim != 2 or labels.shape != (len(codes),) or classes < 1 or
            not torch.isfinite(codes).all() or labels.dtype != torch.long or
            (labels < 0).any() or (labels >= classes).any()):
        raise ValueError('Invalid teacher code/class arrays')
    counts = torch.bincount(labels, minlength=classes)
    if (counts == 0).any():
        raise ValueError('Every declared emotion needs real train prototype observations')
    sums = codes.new_zeros(classes, codes.shape[1]).index_add_(0, labels, codes)
    return sums / counts[:, None].to(codes), counts


@torch.no_grad()
def train_global_prototypes(system, train, identities, device, batch_size, classes):
    codes, labels = [], []
    for ix in torch.arange(len(train['valid'])).split(batch_size):
        b = subset(train, ix, device)
        codes.append(teacher_affect(system, b, batch_identity(identities, b))['global'].cpu())
        labels.append(b['emotion_id'].cpu())
    means, counts = class_global_means(torch.cat(codes), torch.cat(labels), classes)
    return means.to(device), counts


def emotion_intensity_prototypes(codes, emotions, levels, classes, num_levels):
    """TRAIN label lookup: no query-specific residual latent in the target."""
    if (codes.ndim != 2 or emotions.shape != (len(codes),) or levels.shape != emotions.shape or
            emotions.dtype != torch.long or levels.dtype != torch.long or not torch.isfinite(codes).all() or
            (emotions < 0).any() or (emotions >= classes).any() or
            (levels < 0).any() or (levels >= num_levels).any()):
        raise ValueError('Known emotion/intensity labels and finite TRAIN codes required')
    index = emotions*num_levels + levels
    counts = torch.bincount(index, minlength=classes*num_levels)
    sums = codes.new_zeros(classes*num_levels, codes.shape[1]).index_add_(0,index,codes)
    means = sums / counts.clamp_min(1)[:,None].to(codes)
    return means.reshape(classes,num_levels,-1), counts.reshape(classes,num_levels)


@torch.no_grad()
def train_emotion_intensity_prototypes(system, train, identities, device, batch_size, classes, num_levels):
    codes, emotions, levels = [], [], []
    for ix in torch.arange(len(train['valid'])).split(batch_size):
        b = subset(train, ix, device)
        if not b['intensity_valid'].all():
            raise ValueError('Named target trial requires known TRAIN intensity annotations')
        codes.append(teacher_affect(system,b,batch_identity(identities,b))['global'].cpu())
        emotions.append(b['emotion_id'].cpu());levels.append(b['intensity_id'].cpu())
    means, counts = emotion_intensity_prototypes(torch.cat(codes),torch.cat(emotions),torch.cat(levels),classes,num_levels)
    return means.to(device), counts


def native_expression_target(batch, scales, stride):
    """Four native upper-expression proxies relative to independent enrollment."""
    observed = obs(batch) & batch['anchor_valid'][:,None]
    state, mask = readout_slow_state(batch['motion'],observed,batch['anchors'],scales)
    return masked_slow_state(state,mask,stride=stride)

def audio_affect(audio, features, valid, *, global_dropout=0., dropout_generator=None):
    """Call the deployable audio path without reviving legacy state outputs.

    Newer ``SlowStateAffect`` versions accept ``include_legacy_state=False``.
    The fallback keeps this runner compatible with the current checkout while
    ensuring the main path consumes only ``global`` and ``u_a``.
    """
    if global_dropout > 0.:
        # Existing stages deliberately keep child modules in eval mode.
        # Mark only this root call as training for the pooled regularizer;
        # restore it even on failure, without toggling any child module.
        previous_training = audio.training
        audio.training = True
        try:
            return audio(features, valid, include_legacy_state=False,
                         global_dropout=global_dropout, dropout_generator=dropout_generator)
        finally:
            audio.training = previous_training
    try:
        return audio(features, valid, include_legacy_state=False)
    except TypeError as exc:
        if 'include_legacy_state' not in str(exc):
            raise
        return audio(features, valid)

def merge_teacher_audio(teacher, audio_output):
    """Use teacher semantics and audio timing in one renderer condition."""
    merged = dict(audio_output)
    for key in ('global', 'emotion_logits', 'intensity_logits', 'intensity_value'):
        merged[key] = teacher[key]
    temporal = audio_output.get('u_a', audio_output.get('temporal', audio_output.get('local')))
    merged['u_a'] = temporal
    merged['temporal'] = temporal
    # Legacy consumers may inspect ``local``; it aliases the audio condition,
    # never the motion-teacher framewise projection.
    merged['local'] = temporal
    return merged


def generated_emotion_consistency(system, generated, batch, identity, target_affect, *, freeze_critic=True,
                                   emotion_class_weights=None):
    """Constrain the rendered residual to preserve the requested emotion.

    Flow matching alone only constrains a vector field at random interpolation
    points.  The renderer can therefore minimize that objective while drifting
    toward a neutral endpoint.  This helper re-encodes the differentiable
    endpoint produced by ``system.flow`` with the motion-affect teacher and
    applies a label and global-coordinate constraint.  The target affect is
    detached: the teacher/audio branch remains the source of the semantic
    target, while gradients go through the generated endpoint and renderer.
    """
    generated_residual = generated["motion"] - generated["b0"] - identity["baseline"][:, None]
    # During the motion-teacher stage the same encoder supplies both the
    # target and the generated readout.  Freeze its parameters for this
    # auxiliary critic pass while retaining gradients with respect to the
    # generated motion, so the consistency term updates the renderer rather
    # than making the teacher collapse around its own output.
    critic_parameters = list(system.motion_teacher.parameters()) if freeze_critic else []
    critic_states = [parameter.requires_grad for parameter in critic_parameters]
    try:
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        generated_factors = system.encode_motion(generated_residual, batch["valid"])
    finally:
        for parameter, state in zip(critic_parameters, critic_states):
            parameter.requires_grad_(state)
    target_global = target_affect["global"].detach()
    emotion_ce = F.cross_entropy(
        generated_factors["emotion_logits"], batch["emotion_id"],
        weight=emotion_class_weights,
    )
    global_align = 1.0 - F.cosine_similarity(
        generated_factors["global"], target_global, dim=-1
    ).mean()
    intensity_valid = batch["intensity_valid"] & (batch["intensity_id"] >= 0)
    if intensity_valid.any():
        intensity_ce = F.cross_entropy(
            generated_factors["intensity_logits"][intensity_valid],
            batch["intensity_id"][intensity_valid],
        )
    else:
        intensity_ce = emotion_ce.new_zeros(())
    return {
        "emotion_ce": emotion_ce,
        "global": global_align,
        "intensity_ce": intensity_ce,
        "generated": generated_factors,
    }


def independent_probe_consistency(probe, generated_motion, valid, channel_support, labels,
                                  *, class_weights=None):
    """Train-only frozen probe critic for the external motion-F1 metric.

    The probe is fitted on real fit-split motion and never updated here.  Its
    statistics are the same differentiable mean/std/quantile/velocity features
    used by the independent evaluator, so this term cannot silently optimize a
    different proxy.  A per-clip loop is deliberate: clips are variable length
    and this keeps invalid padding out of every statistic.
    """
    if generated_motion.ndim != 3 or valid.shape != generated_motion.shape[:2]:
        raise ValueError('Generated motion and valid mask must be [B,T,C]/[B,T]')
    support = channel_support.to(device=generated_motion.device, dtype=torch.bool)
    if support.shape != (generated_motion.shape[-1],):
        raise ValueError('Probe channel support must match motion dimension')
    features = torch.stack([
        motion_features(generated_motion[index, :, support], valid[index])
        for index in range(generated_motion.shape[0])
    ])
    logits = probe(features)
    return F.cross_entropy(logits, labels.to(device=logits.device, dtype=torch.long), weight=class_weights)


def motion_statistics_consistency(generated_motion, target_motion, valid, channel_support):
    """Match train-target motion statistics without a learned critic.

    This uses the exact mean/std/q10/q90/mean-absolute-velocity/std-velocity
    feature family used by the external probe, but compares generated and GT
    features directly. Invalid padding is excluded per clip.
    """
    support = channel_support.to(device=generated_motion.device, dtype=torch.bool)
    if support.shape != (generated_motion.shape[-1],):
        raise ValueError('Statistics channel support must match motion dimension')
    rows = []
    for index in range(generated_motion.shape[0]):
        pred = motion_features(generated_motion[index, :, support], valid[index])
        target = motion_features(target_motion[index, :, support], valid[index]).detach()
        rows.append(F.smooth_l1_loss(pred, target, beta=1.0))
    return torch.stack(rows).mean() if rows else generated_motion.sum() * 0.0


def content_timing_alignment(generated_motion, base_motion, valid, channel_mask, times,
                             *, channels=ARTICULATORY_MOUTH, max_gap_s=0.12):
    """Keep phonetic mouth timing aligned with frozen B0 while allowing gain changes.

    The loss compares velocity direction only.  It therefore does not force the
    emotional renderer back to B0's amplitude, but discourages changing the
    timing/order of articulatory mouth events.
    """
    if generated_motion.shape != base_motion.shape or generated_motion.ndim != 3:
        raise ValueError('Generated and base motion must have matching [B,T,C] shapes')
    indices = tuple(int(i) for i in channels)
    if not indices or generated_motion.shape[1] < 2:
        return generated_motion.sum() * 0.0
    if times.shape != valid.shape:
        raise ValueError('times and valid must have matching [B,T] shapes')
    dt = torch.diff(times, dim=1).to(generated_motion.dtype)
    pair = valid[:, 1:] & valid[:, :-1] & (dt > 0) & (dt <= max_gap_s)
    pair = pair & channel_mask[:, None, list(indices)].all(-1)
    generated_velocity = torch.diff(generated_motion[..., list(indices)], dim=1) / dt.unsqueeze(-1).clamp_min(1e-4)
    base_velocity = torch.diff(base_motion[..., list(indices)].detach(), dim=1) / dt.unsqueeze(-1).clamp_min(1e-4)
    pair = pair & (base_velocity.norm(dim=-1) > 1e-3)
    if not pair.any():
        return generated_motion.sum() * 0.0
    cosine = F.cosine_similarity(generated_velocity, base_velocity, dim=-1, eps=1e-3)
    return ((1.0 - cosine) * pair.to(cosine.dtype)).sum() / pair.sum().to(cosine.dtype)

def targets(q,scales,stride):
    """LEGACY Stage5 target extraction retained for historical tests."""
    if not (q['channel_mask'][:,list(UPPER_INDICES)] & q['anchor_valid'][:,list(UPPER_INDICES)]).all():
        raise ValueError('Projected flow requires all nine upper channels and independent anchors')
    raw,mask=readout_slow_state(q['motion'],obs(q)&q['anchor_valid'][:,None],q['anchors'],scales)
    state=masked_slow_state(raw,mask,stride=stride)
    normalized=torch.where(q['valid'][...,None],(q['motion'][...,list(UPPER_INDICES)]-q['anchors'][:,None,list(UPPER_INDICES)])/scales[list(UPPER_INDICES)],0.)
    return state,project_upper_innovation(normalized,q['valid'],stride=stride)

class UpperFlow(UpperInnovationFlow):
    """LEGACY Stage5 batch adapter retained for historical tests/scripts."""
    def flow_loss(self,target,q,identity,affect,local,state,noise,flow_time):
        return super().flow_loss(target,q['valid'],q['h0'],identity['code'],affect,local,state,noise,flow_time)

    def decode(self,q,identity,affect,local,state,noise,steps):
        return super().decode(q['valid'],q['h0'],identity['code'],affect,local,state,noise,steps=steps)

def upper_motion(q,scales,state,innovation):
    """LEGACY Stage5 composition helper; unused by the deployable runner."""
    delta=lift_slow_state(state,scales)[...,list(UPPER_INDICES)]
    return q['anchors'][:,None,list(UPPER_INDICES)]+delta+scales[list(UPPER_INDICES)]*innovation

def region_report(pred,q):
    answer={}
    for name,cc in GROUPS.items():
        mask=obs(q)[...,list(cc)];x=pred[...,list(cc)].double();y=q['motion'][...,list(cc)].double()
        count=mask.sum(1,keepdim=True).clamp_min(1)
        xc=torch.where(mask,x-torch.where(mask,x,0.).sum(1,keepdim=True)/count,0.)
        yc=torch.where(mask,y-torch.where(mask,y,0.).sum(1,keepdim=True)/count,0.)
        energy=yc.square().sum();sse=(xc-yc).square().sum()
        pair=mask[:,1:]&mask[:,:-1]
        vals=x[mask]
        answer[name]={'raw_mse':float(mse(x,y,mask)), 'centered_mse':float(sse/mask.sum()),
            'centered_r2':float(1-sse/energy.clamp_min(1e-12)),
            'centered_correlation':float((xc*yc).sum()/(xc.square().sum()*energy).sqrt().clamp_min(1e-12)),
            'rms_ratio':float((xc.square().sum()/energy.clamp_min(1e-12)).sqrt()),
            'frame_displacement_mse':float(mse(x[:,1:]-x[:,:-1],y[:,1:]-y[:,:-1],pair)),
            'outside_fraction':float(((vals<0)|(vals>1)).double().mean())}
    return answer

@torch.no_grad()
def identity_report(system,data,device):
    result={}
    for role,sids in (('fit',data['fit_sids']),('development',data['dev_sids'])):
        a=[];b=[];errors=[]
        for sid in sids:
            n=len(data['refs'][sid]['valid']);cut=max(1,n//2)
            x=encode_ref(system,data,sid,range(cut),device)
            y=encode_ref(system,data,sid,range(cut,n),device)
            a.append(x['code']);b.append(y['code']);errors.append(mse(x['baseline'],y['neutral_mean'],x['observed_channels']&y['observed_channels']))
        similarity=F.normalize(torch.cat(a),dim=-1)@F.normalize(torch.cat(b),dim=-1).T
        result[role]={'speakers':len(sids),'reference_view_retrieval':float((similarity.argmax(-1)==torch.arange(len(sids),device=device)).float().mean()),
            'cross_reference_baseline_mse':float(torch.stack(errors).mean()),
            'scope':'fit reference views trained; development reference views not trained. Coefficient identity, not mesh geometry.'}
    return result

@torch.no_grad()
def evaluate(system, audio, upper=None, local_audio=None, data=None, identities=None,
             stage=None, args=None, *, full=False):
    """Evaluate the four deployable stages.

    ``upper`` and ``local_audio`` are accepted as deprecated positional
    arguments so historical callers keep working; they are deliberately
    ignored by the current protocol.
    """
    q=data['splits']['validation'];limit=len(q['valid']) if full else min(64,len(q['valid']))
    ids=torch.randperm(len(q['valid']),generator=torch.Generator().manual_seed(20260917))[:limit].sort().values
    reference=subset(q,ids,'cpu',keys=('clip_id','motion','valid','times','channel_mask','b0'));all_predictions={};class_correct=0;teacher_correct=0
    seeds=(42,123,2026) if full else (42,)
    for seed in seeds:
        random_noise=torch.randn(len(q['valid']),q['valid'].shape[1],52,generator=torch.Generator().manual_seed(seed))
        predictions=[];gen_emotions=[]
        for ix in ids.split(args.batch_size):
            b=subset(q,ix,args.device);ident=batch_identity(identities,b)
            if stage in ('teacher','audio'):
                b,ident=generator_inputs(b,ident,getattr(args,'ablation','none'))
            n=random_noise[ix,:b['valid'].shape[1]].to(args.device);base={'b0':b['b0'],'h0':b['h0']}
            teacher=None;audio_output=None
            if stage=='articulation':
                pred=base['b0']
            elif stage=='identity':
                pred=torch.where(b['valid'][...,None],base['b0']+ident['baseline'][:,None],0.)
            else:
                teacher=teacher_affect(system,b,ident)
                audio_output=audio_affect(audio,b['audio_features'],b['valid'])
                affect=merge_teacher_audio(teacher,audio_output) if stage=='teacher' and getattr(args,'ablation','none')!='no_teacher' else audio_output
                pred=system.generate(b['content'],b['valid'],ident,affect,initial_noise=n,steps=args.decode_steps,base=base)['motion']
                if seed==42:
                    class_correct+=int((affect['emotion_logits'].argmax(-1)==b['emotion_id']).sum())
                    teacher_correct+=int((teacher['emotion_logits'].argmax(-1)==b['emotion_id']).sum())
                gen_teacher=system.encode_motion(torch.where(obs(b),pred-b['b0']-ident['baseline'][:,None],0.),b['valid'])
                gen_emotions.extend((gen_teacher['emotion_logits'].argmax(-1)==b['emotion_id']).cpu().tolist())
            predictions.append(F.pad(pred.cpu(),(0,0,0,reference['valid'].shape[1]-pred.shape[1])))
        value=torch.cat(predictions);key=f'{seed}/full'
        all_predictions[key]={'motion':value,'metrics':region_report(value,reference),
            'generated_teacher_emotion_accuracy_nonindependent':sum(gen_emotions)/len(gen_emotions) if gen_emotions else None}
    report={'stage':stage,'clips':limit,'noise_seeds':list(seeds),
        'condition_source':{'articulation':'audio_content_B0','identity':'audio_content_B0_plus_neutral_identity',
                            'teacher':'motion_teacher_global_plus_audio_temporal','audio':'audio_student_global_plus_audio_temporal'}[stage],
        'audio_or_teacher_emotion_accuracy':class_correct/limit if stage not in ('articulation','identity') else None,
        'motion_teacher_emotion_accuracy':teacher_correct/limit if stage not in ('articulation','identity') else None,
        'emotion_readout_scope':'Teacher not yet trained; emotion readout unavailable.' if stage in ('articulation','identity') else 'Global emotion readout; temporal condition is audio-native and not a framewise target.',
        'modes':{k:{n:v for n,v in val.items() if n!='motion'} for k,val in all_predictions.items()},
        'test_loaded':False,'default_replaced':False,
        'ablation':getattr(args,'ablation','none'),
        'scope':'paper development; complete native sequences' if getattr(args,'paper_data',None) else 'historical internal development'}
    curves={'schema':SCHEMA,'stage':stage,'clip_id':reference['clip_id'],'target':reference['motion'],'valid':reference['valid'],
        'times':reference['times'],'channel_mask':reference['channel_mask'],'b0':reference['b0'],
        'predictions':{k:v['motion'] for k,v in all_predictions.items()},'noise_seeds':list(seeds)}
    return report,curves

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','audio','targets','enrollment','native-root'):p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--paper-data',type=Path)
    p.add_argument('--ablation',choices=('none','no_teacher','no_residual'),default='none',
                   help='No teacher keeps both generator phases/budgets; direct generation removes additive b0 and identity offset')
    p.add_argument('--start-stage',choices=STAGES,default='articulation')
    p.add_argument('--articulation-scope', choices=('neutral', 'all-emotions'), default='neutral',
                   help='Stage-1 B0 mouth/articulation membership; legacy neutral-only by default')
    p.add_argument('--stage1-train-dropout', type=float, default=None,
                   help='Isolated warm B0 trial: train-mode dropout probability; zero is matched control')
    p.add_argument('--articulation-updates', type=int, default=None,
                   help='Isolated warm B0 trial: fixed optimizer updates instead of fixed epochs')
    p.add_argument('--articulation-target-values', choices=('safe-teacher','native-query'),
                   default='safe-teacher',
                   help='Isolated warm B0 trial: keep safe clip/frame masks, switch only target values')
    p.add_argument('--safe-dtw-root', type=Path, default=None,
                   help='Audited safe_supervision_v1 root containing teacher_manifest_gated.jsonl and pair artifacts')
    p.add_argument('--stage-checkpoint',type=Path)
    p.add_argument('--audio-feature-layout', choices=('affect-prosody', 'legacy-content-affect-prosody'),
                   default='affect-prosody',
                   help='Student uses emotion2vec768+prosody4 only; legacy layout is for historical reproduction')
    p.add_argument('--allow-audio-input-migration', action='store_true',
                   help='Explicitly drop HuBERT columns of a 1540D warm student; downstream adaptation required')
    p.add_argument('--freeze-renderer-audio', action='store_true',
                   help='Isolated warm-start audio-stage ablation: optimize student through a fixed renderer')
    p.add_argument('--audio-gradient-scope', choices=('full', 'expression'), default='full',
                   help='Isolated student gradient trial: native brow/eye flow and generated-semantic derivatives; full renderer/forward retained')
    p.add_argument('--audio-supervision', choices=('flow', 'expression-prosody'), default='flow',
                   help='Named expression/prosody student: native four-state target and TRAIN emotion/intensity prototypes; full renderer conditions detached')
    p.add_argument('--renderer-expression-modulation', action='store_true',
                   help='Isolated named-state receiver: zero-start framewise output-token shift/scale from the four expression states; no new loss or regional output mask')
    p.add_argument('--allow-residual-support-expansion', action='store_true',
                   help='Warm-start from a checkpoint with a strict subset of residual channels; only expansion is allowed and recorded in the recipe')
    p.add_argument('--identity-epochs',type=int,default=None)
    p.add_argument('--end-stage',choices=STAGES,default='audio')
    p.add_argument('--protect-mouth',action='store_true',help='Legacy ablation: protect all mouth channels from the residual flow')
    p.add_argument('--articulatory-mouth-only',action='store_true',
                   help='Legacy ablation: protect the 15 articulatory mouth channels; new runs default to full mouth residual support')
    p.add_argument('--compact',action='store_true')
    p.add_argument('--artifact-dir',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--decode-steps',type=int,default=12)
    p.add_argument('--stride',type=int,default=16);p.add_argument('--seed',type=int,default=47)
    p.add_argument('--emotion-consistency-weight',type=float,default=0.2,
                   help='Weight of generated-endpoint emotion consistency in teacher/audio stages')
    p.add_argument('--emotion-global-weight',type=float,default=1.0,
                   help='Weight of generated global-affect cosine consistency')
    p.add_argument('--emotion-intensity-weight',type=float,default=0.1,
                   help='Weight of generated intensity consistency')
    p.add_argument('--audio-global-dropout', type=float, default=0.,
        help='Training-only pooled-global feature dropout with an independent RNG; default off')
    p.add_argument('--global-distill-target', choices=('clip', 'class-prototype'), default='clip',
                   help='Audio-stage global MSE target: frozen per-clip teacher or real-train class mean')
    p.add_argument('--global-distill-weight', type=float, default=.5,
                   help='Existing coordinate MSE weight; zero tests necessity while retaining the motion teacher')
    p.add_argument('--global-distill-gate', choices=('all', 'teacher-agreement'), default='all',
                   help='Skip only global distill targets whose frozen teacher class disagrees with real TRAIN label')
    p.add_argument('--flow-source-noise', choices=('standard', 'train-residual-std', 'train-residual-ar'), default='standard',
                   help='Isolated TRAIN-fitted Gaussian source at training and rollout')
    p.add_argument('--flow-source-stats', type=Path,
                   help='Bound frozen TRAIN residual spread; use same file in matched control')
    p.add_argument('--flow-vector-units', choices=('scalar', 'train-residual-std', 'train-balanced-std'), default='scalar',
                   help='Existing flow error units; isolated matched diagonal-source trial')
    p.add_argument('--flow-coordinate-system', choices=('scalar', 'train-centered-scalar', 'train-standardized'), default='scalar',
                   help='Isolated TRAIN coordinate refit with shared mean-field head initialization')
    p.add_argument('--flow-coordinate-loss-units', choices=('coordinate', 'physical'), default='coordinate',
                   help='Standardized-coordinate flow error units; physical retains the original common residual scalar')
    p.add_argument('--flow-source-temporal-stats', type=Path,
                   help='Bound TRAIN within-clip lag moments; same evidence file in all temporal trial arms')
    p.add_argument('--emotion-class-balance-power',type=float,default=0.0,
                   help='Tempered inverse-frequency power for emotion CE (0 disables balancing)')
    p.add_argument('--independent-probe', type=Path, action='append', default=None,
                   help='Optional train-fitted frozen motion probe; repeat for an ensemble')
    p.add_argument('--independent-probe-weight', type=float, default=0.0,
                   help='Weight of the frozen independent motion-probe consistency loss')
    p.add_argument('--content-timing-weight',type=float,default=0.0,
                   help='Optional velocity-direction alignment to B0 on articulatory mouth channels')
    p.add_argument('--endpoint-weight',type=float,default=0.0,
                   help='Optional low-weight Huber reconstruction of the differentiable flow endpoint')
    p.add_argument('--bounded-output', action='store_true',
                   help='Explicitly project generated/semantic endpoints to coefficient range [0,1]')
    p.add_argument('--renderer-temporal-adapter', action='store_true',
                   help='Opt-in zero-start learned three-frame motion-token adapter in the residual DiT')
    p.add_argument('--motion-statistics-weight',type=float,default=0.0,
                   help='Optional direct generated/target motion-statistics consistency weight')
    p.add_argument('--ordinal-mouth-weight',type=float,default=0.0,
                   help='Optional target-aware MEAD intensity 1/2/3 mouth-energy ordinal loss')
    p.add_argument('--ordinal-mouth-margin',type=float,default=0.0,
                   help='Nonnegative mouth-energy margin for ordinal supervision')
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    return p

def main():
    args=parser().parse_args()
    if not 0. <= args.audio_global_dropout < 1.:
        raise ValueError('audio-global-dropout must be finite in [0,1)')
    if args.audio_global_dropout > 0. and (
            args.paper_data is None or args.stage_checkpoint is None or
            args.start_stage != 'audio' or args.end_stage != 'audio' or args.ablation != 'none'):
        raise ValueError('Global dropout trial requires isolated warm paper audio stage')
    if args.flow_vector_units != 'scalar' and (
            args.flow_source_noise != 'train-residual-std' or args.flow_source_stats is None):
        raise ValueError('Relative flow units require bound TRAIN std and matched diagonal source')
    if args.flow_coordinate_system != 'scalar' and (
            args.flow_source_noise != 'train-residual-std' or args.flow_source_stats is None or
            args.flow_vector_units != 'scalar' or args.resume):
        raise ValueError('Coordinate refit requires bound diagonal source, scalar units option and a fresh run')
    if args.flow_coordinate_loss_units == 'physical' and (
            args.flow_coordinate_system != 'train-standardized' or args.audio_feature_layout != 'affect-prosody'):
        raise ValueError('Physical error trial requires standardized coordinates and the affect-only student')
    if args.flow_source_noise != 'standard' and args.flow_source_stats is None:
        raise ValueError('TRAIN-scaled flow source requires fixed statistics')
    if args.audio_gradient_scope == 'expression' and (
            args.paper_data is None or args.stage_checkpoint is None or args.resume or
            args.start_stage != 'audio' or args.end_stage != 'audio' or args.ablation != 'none' or
            args.audio_feature_layout != 'affect-prosody' or args.protect_mouth or
            args.articulatory_mouth_only or args.freeze_renderer_audio or
            args.endpoint_weight != 0 or args.content_timing_weight != 0 or
            args.ordinal_mouth_weight != 0 or args.independent_probe_weight != 0 or
            args.motion_statistics_weight != 0):
        raise ValueError('Expression gradient trial requires isolated full-mouth 772D audio stage and original losses')
    if args.renderer_expression_modulation and args.audio_supervision != 'expression-prosody':
        raise ValueError('Expression modulation requires expression-prosody supervision')
    if args.audio_supervision == 'expression-prosody' and (
            args.paper_data is None or args.stage_checkpoint is None or args.resume or
            args.start_stage != 'audio' or args.end_stage != 'audio' or args.ablation != 'none' or
            args.audio_feature_layout != 'affect-prosody' or args.audio_gradient_scope != 'full' or
            args.protect_mouth or args.articulatory_mouth_only or args.freeze_renderer_audio or
            args.global_distill_target != 'clip' or args.global_distill_gate != 'all' or
            args.global_distill_weight != .5 or args.endpoint_weight != 0 or args.content_timing_weight != 0 or
            args.ordinal_mouth_weight != 0 or args.independent_probe_weight != 0 or args.motion_statistics_weight != 0):
        raise ValueError('Named supervision requires isolated full-mouth 772D audio stage and the fixed replacement recipe')
    if (args.flow_source_noise == 'train-residual-ar' and args.flow_source_temporal_stats is None or
            args.flow_source_temporal_stats is not None and args.flow_source_stats is None):
        raise ValueError('Temporal source requires both fixed TRAIN spread and lag statistics')
    if args.flow_source_stats is not None and (
            args.paper_data is None or args.stage_checkpoint is None or args.start_stage != 'audio' or
            args.end_stage != 'audio' or args.ablation != 'none' or args.protect_mouth or
            args.articulatory_mouth_only or args.freeze_renderer_audio or args.renderer_temporal_adapter or
            args.global_distill_weight != .5 or args.global_distill_gate != 'all' or
            args.global_distill_target != 'clip'):
        raise ValueError('Flow source trial requires isolated full-mouth warm audio-stage defaults')
    if args.ablation=='no_residual' and args.protect_mouth:
        raise ValueError('Direct full-face generation requires mouth residual support; omit --protect-mouth')
    if STAGES.index(args.end_stage)<STAGES.index(args.start_stage):raise ValueError('end-stage precedes start-stage')
    if args.articulation_updates is not None and (
            args.articulation_updates < 1 or args.paper_data is None or
            args.start_stage != 'articulation' or args.end_stage != 'articulation' or
            args.stage_checkpoint is None or args.ablation != 'none'):
        raise ValueError('Update budget requires an isolated warm-start paper articulation stage')
    if args.articulation_target_values != 'safe-teacher' and (
            args.safe_dtw_root is None or args.articulation_updates is None or
            args.start_stage != 'articulation' or args.end_stage != 'articulation' or
            args.stage_checkpoint is None or args.ablation != 'none'):
        raise ValueError('Native query target values require isolated warm safe-DTW B0 study')
    if args.stage1_train_dropout is not None:
        if (not math.isfinite(args.stage1_train_dropout) or
                not 0 <= args.stage1_train_dropout <= .5):
            raise ValueError('Stage1 training dropout must be finite in [0, .5]')
        if (args.start_stage != 'articulation' or args.end_stage != 'articulation' or
                args.stage_checkpoint is None or args.ablation != 'none'):
            raise ValueError('Stage1 dropout requires an isolated warm-start articulation stage')
    if args.freeze_renderer_audio and (
            args.start_stage != 'audio' or args.end_stage != 'audio' or
            args.stage_checkpoint is None or args.ablation != 'none'):
        raise ValueError('Frozen renderer requires an isolated warm-start audio stage')
    if args.global_distill_gate != 'all' and (
            args.start_stage != 'audio' or args.end_stage != 'audio' or
            args.stage_checkpoint is None or args.ablation != 'none' or
            args.global_distill_target != 'clip'):
        raise ValueError('Teacher agreement gate requires isolated per-clip warm-start audio distillation')
    if args.global_distill_target != 'clip' and (
            args.start_stage != 'audio' or args.end_stage != 'audio' or
            args.stage_checkpoint is None or args.ablation != 'none'):
        raise ValueError('Class-prototype distillation requires an isolated warm-start audio stage')
    if not math.isfinite(args.global_distill_weight) or args.global_distill_weight < 0:
        raise ValueError('Global distill weight must be finite and nonnegative')
    if args.global_distill_weight != .5 and (
            args.start_stage != 'audio' or args.end_stage != 'audio' or
            args.stage_checkpoint is None or args.ablation != 'none' or
            args.global_distill_target != 'clip' or args.global_distill_gate != 'all'):
        raise ValueError('Coordinate-distillation weight trial requires isolated per-clip warm-start audio stage')
    if not args.smoke and not 1<=args.epochs<=60:raise ValueError('Epoch budget must be 1–60')
    if args.batch_size<1 or args.decode_steps<1 or args.stride<1:raise ValueError('Positive batch/solver/stride required')
    if (args.emotion_consistency_weight < 0 or args.emotion_global_weight < 0 or
            args.emotion_intensity_weight < 0 or args.emotion_class_balance_power < 0 or
            args.independent_probe_weight < 0):
        raise ValueError('Emotion consistency weights must be nonnegative')
    if args.content_timing_weight < 0:
        raise ValueError('Content timing weight must be nonnegative')
    if args.endpoint_weight < 0:
        raise ValueError('Endpoint weight must be nonnegative')
    if args.motion_statistics_weight < 0:
        raise ValueError('Motion statistics weight must be nonnegative')
    if args.ordinal_mouth_weight < 0 or args.ordinal_mouth_margin < 0:
        raise ValueError('Ordinal mouth weight and margin must be nonnegative')
    if args.independent_probe_weight > 0 and not args.independent_probe:
        raise ValueError('independent-probe is required when independent-probe-weight is positive')
    if args.output.exists() and not args.resume:raise FileExistsError('Fresh run required')
    torch.set_num_threads(args.threads);torch.backends.cuda.matmul.allow_tf32=True
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.paper_data:
        from scripts.prepare_paper_full_data import load_paper_data
        print(json.dumps({'event':'data_load_start','paper_data':str(args.paper_data.resolve()),
                          'smoke':bool(args.smoke)}), flush=True)
        data=load_paper_data(args.paper_data,seed=args.seed,smoke=args.smoke)
        print(json.dumps({'event':'data_load_complete','fit_clips':len(data['splits']['train']['valid']),
                          'development_clips':len(data['splits']['validation']['valid']),
                          'smoke':bool(args.smoke)}), flush=True)
    else:
        if any(getattr(args,k) is None for k in ('source_run','audio','targets','enrollment','native_root')):
            raise ValueError('Historical data requires all source paths')
        data=load_training_inputs(args.source_run,args.audio,args.targets,args.enrollment,args.native_root)
    if args.smoke and args.paper_data:
        # Metadata-independent first available clip per person/class. This
        # limits only smoke, never full training membership.
        for role,q in data['splits'].items():
            chosen=[];seen=set()
            for i,(sid,emo) in enumerate(zip(q['speaker_id'],q['emotion_id'])):
                key=(int(sid),int(emo))
                if key not in seen:seen.add(key);chosen.append(i)
            data['splits'][role]=subset(q,torch.tensor(chosen),'cpu')
    system=data['system'].to(args.device).eval();cfg=data['config']
    cfg['model']['renderer_temporal_adapter'] = bool(args.renderer_temporal_adapter)
    system.renderer.set_temporal_adapter(args.renderer_temporal_adapter)
    cfg['model']['output_projection'] = bool(args.bounded_output)
    system.set_output_projection(args.bounded_output)
    # Fixed deployment support is learned from TRAIN metadata only. Query
    # observation masks are used only to censor training targets and scores.
    train_support=data['splits']['train']['channel_mask'].any(0).bool()
    emotion_class_weights = None
    if args.emotion_class_balance_power > 0:
        emotion_class_weights = class_balanced_weights(
            data['splits']['train']['emotion_id'],
            len(cfg['data']['emotion_classes']), power=args.emotion_class_balance_power,
        ).to(args.device)
    independent_probes = []
    independent_probe_meta = None
    probe_payloads = []
    if args.independent_probe:
      for probe_path in args.independent_probe:
        probe_payload = torch.load(probe_path, map_location='cpu', weights_only=False)
        if probe_payload.get('train_manifest_sha256') != data['provenance'].get('manifest_sha256'):
            raise ValueError('Independent probe belongs to another train manifest')
        if probe_payload.get('test_used_for_selection'):
            raise ValueError('Independent probe was selected with sealed/test data')
        independent_probe = MotionEmotionProbe(
            int(probe_payload['feature_dim']), int(probe_payload['hidden']),
            len(probe_payload['classes'])).to(args.device).eval()
        probe_support = torch.as_tensor(probe_payload['channel_support'], dtype=torch.bool)
        if probe_support.shape != (int(cfg['data']['motion_dim']),):
            raise ValueError('Independent probe channel support does not match motion dimension')
        if int(probe_payload['feature_dim']) != int(probe_support.sum()) * 6:
            raise ValueError('Independent probe feature dimension does not match evaluator statistics')
        if list(probe_payload['classes']) != list(cfg['data']['emotion_classes']):
            raise ValueError('Independent probe class order differs from the training protocol')
        independent_probe.load_state_dict(probe_payload['model'], strict=True)
        independent_probe.requires_grad_(False)
        independent_probes.append(independent_probe)
        probe_payloads.append((probe_payload, probe_support))
        independent_probe_meta = independent_probe_meta or []
        independent_probe_meta.append({
            'path': str(probe_path.resolve()),
            'sha256': sha(probe_path),
            'classes': list(probe_payload['classes']),
            'channel_support': int(probe_payload['channel_support'].sum()),
            'train_manifest_sha256': probe_payload['train_manifest_sha256'],
            'test_used_for_selection': bool(probe_payload.get('test_used_for_selection', False)),
        })
    cfg['model']['motion_support']=train_support.tolist()
    system.set_motion_support(train_support.to(args.device))
    # B0 supplies a neutral content-aligned base, while the residual flow may
    # refine every observed coefficient, including articulatory mouth.  This
    # is the deployable default: emotion/intensity must be able to change
    # jawOpen and opening/closing channels.  The two protection switches are
    # explicit legacy ablations, never implicit defaults.
    residual_support=train_support.clone()
    if args.articulatory_mouth_only:
        residual_support[list(ARTICULATORY_MOUTH)]=False
    if args.protect_mouth:
        residual_support[list(MOUTH)]=False
    if args.protect_mouth and args.articulatory_mouth_only:
        raise ValueError('Choose at most one mouth protection mode')
    cfg['model']['residual_support']=residual_support.tolist()
    system.set_residual_support(residual_support.to(args.device))
    # Statistics are fitted on the authorized fit tensor only, not development.
    data['feature_stats'] = affect_audio_statistics(data['feature_stats'], args.audio_feature_layout)
    cfg['model']['audio_feature_layout'] = args.audio_feature_layout
    mean=data['feature_stats']['mean'];std=data['feature_stats']['std']
    temporal_layout = 'expression-prosody' if args.audio_supervision == 'expression-prosody' else 'latent'
    if temporal_layout != 'latent':
        cfg['model']['audio_temporal_layout'] = temporal_layout
        cfg['model']['audio_temporal_stride'] = args.stride
    audio=SlowStateAffect(mean,std,stride=args.stride,temporal_layout=temporal_layout).to(args.device).eval()
    if args.stage_checkpoint:
        saved=torch.load(args.stage_checkpoint,map_location='cpu',weights_only=False)
        if (saved.get('config',{}).get('model',{}).get('audio_temporal_layout') == 'expression-prosody'
                and args.audio_supervision != 'expression-prosody'):
            raise ValueError('Named-condition checkpoint requires matching expression-prosody supervision; use from_checkpoint for inference')
        if saved.get('data_manifest_sha256')!=data['provenance'].get('manifest_sha256'):
            raise ValueError('Stage checkpoint belongs to another data protocol')
        if saved.get('config',{}).get('model',{}).get('renderer_expression_modulation', False):
            if not args.renderer_expression_modulation:
                raise ValueError('Expression-modulated checkpoint requires its matching renderer option')
            system.renderer.set_expression_modulation(True)
        if saved.get('config', {}).get('model', {}).get('flow_source_std') is not None and args.flow_source_noise == 'standard':
            raise ValueError('A scaled-source checkpoint requires its matching explicit source recipe')
        if saved.get('config', {}).get('model', {}).get('flow_source_rho') is not None and args.flow_source_noise != 'train-residual-ar':
            raise ValueError('A temporal-source checkpoint requires its matching explicit source recipe')
        load_warm_system(system, saved['system'],
                         allow_zero_temporal_adapter=args.renderer_temporal_adapter)
        saved_audio = saved['audio']
        old_width, new_width = saved_audio['input.weight'].shape[1], audio.input.in_features
        if old_width != new_width:
            if not args.allow_audio_input_migration:
                raise ValueError('Audio input layout changed; explicit --allow-audio-input-migration is required')
            saved_audio = migrate_affect_audio_state(saved_audio, audio.state_dict())
        audio.load_state_dict(saved_audio, strict=True)
        calibration=saved.get('config',{}).get('model',{}).get('mouth_reference_calibration')
        if calibration is not None:
            cfg['model']['mouth_reference_calibration']=calibration
            system.set_mouth_reference_calibration(calibration)
        saved_support=saved.get('config',{}).get('model',{}).get('motion_support')
        if saved_support is not None and saved_support!=train_support.tolist():raise ValueError('Stage checkpoint train support differs')
        saved_residual=saved.get('config',{}).get('model',{}).get('residual_support')
        if saved_residual is not None and saved_residual!=residual_support.tolist():
            requested = torch.as_tensor(residual_support, dtype=torch.bool)
            previous = torch.as_tensor(saved_residual, dtype=torch.bool)
            expansion = (not bool((previous & ~requested).any())) and bool((requested & ~previous).any())
            if not args.allow_residual_support_expansion or not expansion:
                raise ValueError('Stage checkpoint residual support differs; use matching protection mode or a strict support expansion')
        del saved
    elif args.start_stage!='articulation':raise ValueError('Starting later requires matching stage checkpoint')
    if args.renderer_expression_modulation:
        cfg['model']['renderer_expression_modulation'] = True
        system.renderer.set_expression_modulation(True)
    source_info = None
    temporal_source_info = None
    flow_vector_std = None
    flow_coordinate_info = None
    if args.flow_source_stats is not None:
        source_std, source_info = load_flow_source_stats(args.flow_source_stats,
            checkpoint_sha256=sha(args.stage_checkpoint), manifest_sha256=data['provenance']['manifest_sha256'],
            support=residual_support, residual_scale=system.residual_scale)
        if args.flow_source_noise in ('train-residual-std', 'train-residual-ar'):
            cfg['model']['flow_source_std'] = source_std.tolist()
            system.set_flow_source_std(source_std)
        if args.flow_vector_units != 'scalar':
            flow_vector_std = (balanced_flow_units(source_std, residual_support)
                if args.flow_vector_units == 'train-balanced-std' else source_std).to(args.device)
        if args.flow_coordinate_system != 'scalar':
            flow_coordinate_info = configure_flow_coordinates(system, cfg, args.flow_source_stats,
                source_std, residual_support, args.flow_coordinate_system)
            if args.flow_coordinate_system == 'train-standardized':
                flow_vector_std = coordinate_flow_error_std(source_std.to(args.device), mode=args.flow_coordinate_loss_units)
        if args.flow_source_temporal_stats is not None:
            source_rho, temporal_source_info = load_flow_temporal_stats(args.flow_source_temporal_stats,
                checkpoint_sha256=sha(args.stage_checkpoint), manifest_sha256=data['provenance']['manifest_sha256'],
                support=residual_support, source_info=source_info)
            if args.flow_source_noise == 'train-residual-ar':
                cfg['model']['flow_source_rho'] = source_rho.tolist()
                system.set_flow_source_rho(source_rho)
    input_paths={k:str(getattr(args,k)) for k in ('source_run','audio','targets','enrollment','native_root')}
    if args.safe_dtw_root:
        input_paths['safe_dtw_root'] = str(args.safe_dtw_root.resolve())
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__),root/'scripts/full_staged_data.py',root/'kinetalk_b0/models/slow_state_affect.py',
        root/'kinetalk_b0/models/neutral_affect.py',root/'kinetalk_b0/models/dit.py',root/'kinetalk_b0/models/model.py',
        root/'kinetalk_b0/models/encoders.py',root/'kinetalk_b0/models/label_guided_affect.py',root/'kinetalk_b0/neutral_data.py',
        root/'kinetalk_b0/semantic_losses.py',root/'kinetalk_b0/models/audio_residual_flow.py',
        root/'kinetalk_b0/intensity_losses.py',
        root/'scripts/train_formal_predictable_projection.py']
    if args.paper_data:sources += [root/'scripts/prepare_paper_full_data.py',root/'scripts/paper_generation_report.py',root/'scripts/packed_trainval_cache.py']
    if args.protect_mouth:sources += [root/'scripts/mouth_protection.py',root/'kinetalk_b0/reference_mouth_calibration.py']
    if args.paper_data:input_paths['paper_data']=str(args.paper_data.resolve())
    if args.artifact_dir:input_paths['artifact_dir']=str(args.artifact_dir.resolve())
    articulation_ids, articulation_scope_info = articulation_selection(data['splits']['train'], args.articulation_scope)
    safe_targets = None
    if args.safe_dtw_root:
        train_ids = [str(x) for x in data['splits']['train']['clip_id']]
        safe_targets = load_safe_native_targets(args.safe_dtw_root, train_ids)
        # Neutral clips supervise their exact native motion, not a warped copy.
        neutral_ids = (data['splits']['train']['emotion_id'] == 0).nonzero(as_tuple=True)[0].tolist()
        for i in neutral_ids:
            safe_targets.pop(train_ids[i], None)
        allowed = {str(x) for x in safe_targets}
        safe_ids = [int(i) for i, cid in enumerate(train_ids) if cid in allowed]
        # Native neutral clips are exact identity anchors and do not require
        # DTW.  Union them with the audited cross-emotion teachers so B0 sees
        # every clean neutral utterance plus only approved cross-emotion pairs.
        articulation_ids = torch.tensor(sorted(set(safe_ids).union(neutral_ids)), dtype=torch.long)
        if not len(articulation_ids):
            raise ValueError('No training clips overlap the safe-DTW teacher manifest')
        articulation_scope_info = {**articulation_scope_info,
            'teacher_source': 'safe_dtw_v3_native_neutral_teacher',
            'safe_target_clip_count': len(safe_ids),
            'neutral_identity_clip_count': len(neutral_ids),
            'effective_articulation_clip_count': len(articulation_ids),
            'safe_target_manifest': str((args.safe_dtw_root / 'teacher_manifest_gated.jsonl').resolve())}
    ordinal_triplets = build_intensity_triplets(
        data['splits']['train']['speaker'], data['splits']['train']['sentence_id'],
        data['splits']['train']['emotion_id'], data['splits']['train']['intensity_id'],
        intensity_valid=data['splits']['train']['intensity_valid'],
    ) if args.ordinal_mouth_weight > 0 else torch.empty((0, 3), dtype=torch.long)
    if args.ordinal_mouth_weight > 0 and not len(ordinal_triplets):
        raise ValueError('Ordinal mouth supervision requested but no complete intensity groups exist')
    ordinal_info = {'complete_groups': int(len(ordinal_triplets)), 'levels': [1, 2, 3],
                    'target_region': 'mouth14:41 residual RMS',
                    'target_order_filter': 'higher target energy required per adjacent pair'}
    recipe_args = {
        k: ([str(x) for x in v] if isinstance(v, list) and all(isinstance(x, Path) for x in v) else v)
        for k, v in vars(args).items()
        if k not in ('resume', 'output') and not isinstance(v, Path)
    }
    recipe={'schema':SCHEMA,'args':recipe_args,
        'paths':input_paths,'data_provenance':data['provenance'],'source_sha256':{str(p.relative_to(root)):sha(p) for p in sources},
        'stages':list(STAGES[STAGES.index(args.start_stage):STAGES.index(args.end_stage)+1]),'epochs_per_stage':args.epochs,'stride_frames':args.stride,
        'condition':'teacher global affect + native audio temporal condition + identity; stochastic residual flow handles one-to-many motion',
        'generated_emotion_consistency':{
            'enabled': bool(args.emotion_consistency_weight > 0),
            'weight': float(args.emotion_consistency_weight),
            'global_weight': float(args.emotion_global_weight),
            'intensity_weight': float(args.emotion_intensity_weight),
            'target':'detached teacher/audio affect; generated flow endpoint re-encoded by motion teacher',
        },
        'emotion_class_balance': {
            'power': float(args.emotion_class_balance_power),
            'weights': None if emotion_class_weights is None else [float(x) for x in emotion_class_weights.detach().cpu()],
            'source': 'fit emotion_id counts only',
        },
        'independent_probe': {
            'enabled': bool(independent_probes and args.independent_probe_weight > 0),
            'weight': float(args.independent_probe_weight),
            'metadata': independent_probe_meta,
        },
        'content_timing_weight': float(args.content_timing_weight),
        'renderer_temporal_adapter': {
            'enabled': bool(args.renderer_temporal_adapter), 'kernel_frames': 3,
            'initialization': 'zero; preserves RNG and initial outputs',
            'scope': 'local token interaction before DiT attention; no output filter or new loss',
        },
        'global_distill_target': {
            'mode': 'emotion-intensity-prototype' if temporal_layout != 'latent' else args.global_distill_target,
            'weight': args.global_distill_weight,
            'source': ('frozen teacher TRAIN emotion/intensity cell means' if temporal_layout != 'latent' else
                       'frozen teacher real TRAIN class mean' if args.global_distill_target != 'clip' else 'frozen per-clip teacher'),
            'other_conditions': 'intensity and temporal audio unchanged',
        },
        'global_distill_gate': {
            'mode': args.global_distill_gate,
            'normalization': 'full batch, no upweighting of retained clips',
            'scope': 'TRAIN frozen teacher argmax vs real label; affects only existing global MSE',
        },
        'flow_source_noise': {'mode': args.flow_source_noise, 'statistics': source_info,
                              'temporal_statistics': temporal_source_info,
                              'scope': 'source Gaussian covariance only; targets/output scale/support/loss unchanged'},
        'flow_vector_units': {'mode': args.flow_vector_units,
                              'statistics': source_info if flow_vector_std is not None else None,
                              'scope': 'Existing flow MSE units only; no extra loss or forward/output change'},
        'flow_coordinate_system': flow_coordinate_info,
        'flow_coordinate_loss_units': {
            'mode': args.flow_coordinate_loss_units,
            'effective_error': 'TRAIN channel std' if flow_vector_std is not None else 'common residual scalar',
            'scope': 'Flow error metric only; forward coordinates and all other loss terms unchanged'},
        'endpoint_weight': float(args.endpoint_weight),
        'output_projection': {
            'enabled': bool(args.bounded_output),
            'policy': '[0,1] projection on active residual channels; protects masks/legacy channels',
            'flow_vector_field': 'unchanged; raw endpoints retained',
        },
        'motion_statistics_weight': float(args.motion_statistics_weight),
        'ordinal_mouth': ordinal_info,
        'warm_start':bool(args.stage_checkpoint) or not bool(args.paper_data),'stage_checkpoint_sha256':sha(args.stage_checkpoint) if args.stage_checkpoint else None,
        'allow_residual_support_expansion': bool(args.allow_residual_support_expansion),
        'new_audio_global':f'{audio.input.in_features}D {args.audio_feature_layout} student; global/intensity/u_a share its selected acoustic input',
        'audio_gradient_scope': {
            'mode': args.audio_gradient_scope,
            'channels': list(UPPER_INDICES) if args.audio_gradient_scope == 'expression' else None,
            'scope': 'Student flow/generated-semantic Jacobians only; original full renderer objective, forward and semantic/global targets retained',
            'fully_disentangled': False,
        },
        'audio_supervision': {
            'mode': args.audio_supervision,
            'temporal_layout': temporal_layout,
            'state_target': 'native four signed upper states, independent neutral anchors, validated TRAIN scales and fixed spline' if temporal_layout != 'latent' else None,
            'global_target': 'TRAIN emotion/intensity label prototype' if temporal_layout != 'latent' else args.global_distill_target,
            'state_weight': 1.0 if temporal_layout != 'latent' else 0.,
            'renderer_condition_gradient_to_student': temporal_layout == 'latent',
            'fully_disentangled': False,
        },
        'audio_feature_layout': {
            'mode': args.audio_feature_layout, 'student_dimensions': audio.input.in_features,
            'packed_dimensions': data['feature_stats']['source_feature_width'],
            'hubert_in_student': args.audio_feature_layout == 'legacy-content-affect-prosody',
            'warm_migration_authorized': bool(args.allow_audio_input_migration),
            'scope': 'Student input only; content/B0/h0 remain original; no new loss or mouth exclusion',
        },
        'identity_epoch':'One pass over disjoint complementary reference-view pairs from fit identities only',
        'articulation_epoch':'One shuffled pass over the selected articulation scope',
        'articulation_scope':articulation_scope_info,
        'stage1_training_dropout': {
            'probability': args.stage1_train_dropout,
            'scope': 'Stage1 Dropout and MHA only; train-mode control; eval caches/inference',
            'changed_b0_requires_downstream_adaptation': True,
        },
        'other_epoch':'One shuffled pass over all fit queries',
        'trainable':'Stage-dependent; pretrained acoustic/content extractors remain frozen feature sources',
        'motion_support':train_support.tolist(),'residual_support':residual_support.tolist(),
        'residual_support_mode':('protect_mouth' if args.protect_mouth else
                                  'articulatory_mouth_only' if args.articulatory_mouth_only else
                                  'full_mouth'),
        'articulatory_mouth_indices':list(ARTICULATORY_MOUTH),
        'affect_mouth_indices':list(AFFECT_MOUTH),
        'mouth_reference_calibration':cfg['model'].get('mouth_reference_calibration'),
        'test_loaded':False,'default_replaced':False,'checkpoint_selection':'Fixed final epoch; intermediate validation not used for selection'}
    if args.renderer_expression_modulation:
        recipe['renderer_expression_modulation'] = {
            'enabled': True, 'inputs': 'named ua first four expression states only',
            'parameters': 'Linear4->2*dit_dim, zero weight and bias',
            'scope': 'Framewise shift/scale of complete output tokens; original losses and full mouth support',
        }
    if args.safe_dtw_root:
        recipe['safe_dtw_manifest_sha256'] = sha(args.safe_dtw_root / 'teacher_manifest_gated.jsonl')
        recipe['safe_dtw_artifacts_sha256'] = canonical_hash({cid:v[4] for cid,v in sorted(safe_targets.items())})
        recipe['safe_dtw_mouth_mask'] = 'native_teacher_mask AND event_local_mask AND source_observation_mask; exact native neutral anchors'
    if args.articulation_updates is not None:
        recipe['articulation_update_budget'] = {
            'optimizer_updates': args.articulation_updates,
            'requested_epochs_overridden': args.epochs,
            'scope': 'Isolated warm B0 only; final partial epoch retained; no loss change',
        }
    recipe['articulation_target_values'] = {
        'mode': args.articulation_target_values,
        'scope': 'Same approved clip IDs and safe frame/channel observation mask; only target values switch',
    }
    if args.paper_data:
        recipe.update(
          condition='native full audio + independent neutral identity + teacher global affect + stochastic residual flow',
          dynamic_objective=('full renderer flow with detached named conditions; student native expression-state target and measured prosody'
                             if temporal_layout != 'latent' else
                             'flow matching with audio temporal condition; no framewise target or Stage5'),
          starting_stage=args.start_stage,acoustic_extractors='frozen pretrained content/audio features; trainable audio student in Stage3/4')
    recipe_hash=canonical_hash(recipe);args.output.mkdir(parents=True,exist_ok=True)
    gen=torch.Generator().manual_seed(args.seed)
    global_dropout_gen = (torch.Generator(device='cpu').manual_seed(args.seed + 290000)
                          if args.audio_global_dropout > 0. else None)
    start_stage=STAGES.index(args.start_stage);start_epoch=0;total_steps=0;elapsed_before=0.;resume_payload=None
    if args.resume:
        resume_payload=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        if resume_payload['recipe_sha256']!=recipe_hash:raise ValueError('Resume recipe/source/input binding differs')
        system.load_state_dict(resume_payload['system']);audio.load_state_dict(resume_payload['audio'])
        start_stage=resume_payload['stage_index'];start_epoch=resume_payload['completed_epochs'];total_steps=resume_payload['total_steps'];elapsed_before=resume_payload['elapsed_seconds']
        restore_rng(resume_payload['rng'],gen)
        if global_dropout_gen is not None:
            global_dropout_gen.set_state(resume_payload['audio_global_dropout_rng'])
    else:
        save_json(args.output/'provenance.json',{'recipe':recipe,'recipe_sha256':recipe_hash})
        for source in sources:
            dest=args.output/'source'/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    started=time.monotonic();scales=data['target_scales'].to(args.device)
    epochs=1 if args.smoke else args.epochs
    cache_current_base(system,data,args.device)
    identities=identity_cache(system,data,args.device)
    global_prototypes = None
    prototype_counts = None
    if args.audio_supervision == 'expression-prosody':
        global_prototypes, prototype_counts = train_emotion_intensity_prototypes(
            system, data['splits']['train'], identities, args.device,
            args.batch_size, len(cfg['data']['emotion_classes']), cfg['data']['num_intensity_levels'])
        save_checkpoint(args.output / 'global_distill_prototypes.pt', {
            'means':global_prototypes.cpu(),'counts':prototype_counts,'test_loaded':False,
            'fit_split':'train','recipe_sha256':recipe_hash,
            'data_manifest_sha256':data['provenance'].get('manifest_sha256')})
    if args.global_distill_target == 'class-prototype':
        global_prototypes, prototype_counts = train_global_prototypes(
            system, data['splits']['train'], identities, args.device,
            args.batch_size, len(cfg['data']['emotion_classes']))
        save_checkpoint(args.output / 'global_distill_prototypes.pt', {
            'means': global_prototypes.cpu(), 'counts': prototype_counts,
            'classes': cfg['data']['emotion_classes'], 'test_loaded': False,
            'source': 'frozen real TRAIN motion teacher only', 'recipe_sha256': recipe_hash,
            'data_manifest_sha256': data['provenance'].get('manifest_sha256'),
        })
    pairs=identity_pairs(data['refs'],data['fit_sids'])
    if not args.resume and not args.paper_data:
        save_json(args.output/'identity_initial.json',identity_report(system,data,args.device))
        initial_report,initial_curves=evaluate(system,audio,data=data,identities=identities,stage='teacher',args=args,full=not args.smoke)
        initial_report['scope']='Original checkpoint motion-teacher oracle only; not historical audio baseline'
        save_json(args.output/'initial_teacher_oracle.json',initial_report)
        save_checkpoint(args.output/'initial_teacher_oracle_curves.pt',initial_curves)
    current_stage='setup'
    try:
        for stage_index,stage in enumerate(STAGES):
            if stage_index<start_stage:continue
            if stage_index>STAGES.index(args.end_stage):break
            current_stage=stage;stage_dir=args.output/stage;stage_dir.mkdir(exist_ok=True)
            epochs=1 if args.smoke else (args.identity_epochs if stage=='identity' and args.identity_epochs is not None else args.epochs)
            if epochs<1:raise ValueError('Stage epochs must be positive')
            for module in (system,audio):
                module.requires_grad_(False);module.eval();module.zero_grad(set_to_none=True)
            if stage=='articulation':groups=[{'params':unfreeze(system.stage1),'lr':3e-4 if args.paper_data else 1e-5}]
            elif stage=='identity':groups=[{'params':unfreeze(system.identity_encoder)+unfreeze(system.identity_bias),'lr':1e-4}]
            elif stage=='teacher':groups=[
                {'params':unfreeze(audio),'lr':1e-4},
                {'params':unfreeze(system.renderer),'lr':1e-4 if args.paper_data else 3e-5}]
            elif stage=='audio':groups=audio_parameter_groups(
                system, audio, paper_data=bool(args.paper_data),
                freeze_renderer=args.freeze_renderer_audio)
            else: raise RuntimeError(f'Unknown deployable stage: {stage}')
            if stage == 'articulation':
                dropout_counts = stage1_training_dropout(system.stage1, args.stage1_train_dropout)
                if dropout_counts is not None:
                    print(json.dumps({'event': 'stage1_training_dropout',
                                      'probability': args.stage1_train_dropout,
                                      **dropout_counts}), flush=True)
            if stage=='teacher' and args.ablation!='no_teacher':
                groups.insert(0,{'params':unfreeze(system.motion_teacher),'lr':1e-4})
            parameters=[p for g in groups for p in g['params']]
            optimizer=torch.optim.AdamW(groups,weight_decay=1e-5)
            initial_epoch=start_epoch if stage_index==start_stage else 0
            if resume_payload is not None and stage_index==start_stage:
                optimizer.load_state_dict(resume_payload['optimizer'])
                for state in optimizer.state.values():
                    for key,value in state.items():
                        if torch.is_tensor(value):state[key]=value.to(args.device)
            if stage=='articulation':items=articulation_ids
            elif stage=='identity':items=torch.arange(len(pairs))
            elif args.ordinal_mouth_weight > 0 and stage in ('teacher','audio'):
                # Keep each same-content intensity group together; the
                # ordinal term must not depend on random batch collisions.
                items=torch.arange(len(ordinal_triplets))
            else:items=torch.arange(len(data['splits']['train']['valid']))
            if args.smoke:items=items[:min(len(items),args.batch_size*2)]
            batches=math.ceil(len(items)/args.batch_size)
            if not batches:raise ValueError('Empty training stage: '+stage)
            if stage == 'articulation' and args.articulation_updates is not None:
                epochs, batches, _ = articulation_update_plan(
                    len(items), args.batch_size, epochs, args.articulation_updates)
            trainable={name:sum(p.numel() for p in module.parameters() if p.requires_grad) for name,module in [('system',system),('audio',audio)]}
            if torch.device(args.device).type=='cuda':torch.cuda.reset_peak_memory_stats(args.device)
            print(json.dumps({'event':'stage_start','stage':stage,'epochs':epochs,'samples_per_epoch':len(items),'batches':batches,'trainable':trainable}),flush=True)
            frozen_before={name:state_hash(module.state_dict()) for name,module in [('system',system),('audio',audio)] if not any(p.requires_grad for p in module.parameters())}
            frozen_parameters={name:state_hash({k:v for k,v in module.named_parameters() if not v.requires_grad}) for name,module in [('system',system),('audio',audio)]}
            def checkpoint(epoch):
                payload={'schema':SCHEMA,'recipe_sha256':recipe_hash,'stage_index':stage_index,'stage':stage,'completed_epochs':epoch,'total_steps':total_steps,
                    'system':system.state_dict(),'audio':audio.state_dict(),
                    'optimizer':optimizer.state_dict(),'rng':capture_rng(gen),'elapsed_seconds':elapsed_before+time.monotonic()-started}
                if global_dropout_gen is not None:
                    payload['audio_global_dropout_rng'] = global_dropout_gen.get_state()
                save_checkpoint(args.output/'last.pt',payload)
            if initial_epoch==0:checkpoint(0)
            for epoch in range(initial_epoch,epochs):
                begin=time.monotonic();sums={};order=items[torch.randperm(len(items),generator=gen)]
                epoch_batches = articulation_epoch_batches(
                    order, args.batch_size, total_steps,
                    args.articulation_updates if stage == 'articulation' else None)
                for bi,ix in enumerate(epoch_batches):
                    paired_batch = args.ordinal_mouth_weight > 0 and stage in ('teacher','audio')
                    sample_ix = ordinal_triplets[ix].reshape(-1) if paired_batch else ix
                    if stage=='identity':
                        av=[];bv=[]
                        for index in ix:
                            sid,aa,bb=pairs[int(index)]
                            av.append(encode_ref(system,data,sid,aa,args.device));bv.append(encode_ref(system,data,sid,bb,args.device))
                        ac=torch.cat([a['code'] for a in av]);bc=torch.cat([b['code'] for b in bv])
                        common=torch.cat([a['observed_channels']&b['observed_channels'] for a,b in zip(av,bv)])
                        base_loss=(mse(torch.cat([a['baseline'] for a in av]),torch.cat([b['neutral_mean'] for b in bv]),common)+
                            mse(torch.cat([b['baseline'] for b in bv]),torch.cat([a['neutral_mean'] for a in av]),common))/(2*.25**2)
                        contrast=style_contrastive(ac,bc,torch.tensor([pairs[int(i)][0] for i in ix],device=args.device))
                        loss=base_loss+.05*contrast;values={'baseline':base_loss,'contrast':contrast}
                    else:
                        b=subset(data['splits']['train'],sample_ix,args.device);identity=batch_identity(identities,b)
                        if stage in ('teacher','audio'):
                            b,identity=generator_inputs(b,identity,args.ablation)
                        base={'b0':b['b0'],'h0':b['h0']};mask=obs(b)
                        noise=torch.randn(b['motion'].shape,generator=gen).to(args.device);ft=torch.rand(len(sample_ix),generator=gen).to(args.device)
                        if stage=='articulation':
                            out=base_forward(system,b['content'],b['valid'],gradients=True)['b0']
                            cc=list(system.stage1.art_indices);m=mask[...,cc];scale=scales[cc].clamp_min(.05)
                            if safe_targets is not None:
                                target_motion, target_valid, target_channel = safe_target_batch(
                                    safe_targets, b['clip_id'], b['valid'].shape[1], args.device,
                                    b['motion'], b['valid'], native_times=b['times'])
                                if args.articulation_target_values == 'native-query':
                                    target_motion = b['motion']
                            else:
                                target_motion, target_valid, target_channel = b['motion'], b['valid'], torch.ones_like(b['motion'], dtype=torch.bool)
                            m = m & target_valid[..., None] & target_channel[...,cc]
                            raw=huber(out[...,cc]/scale,target_motion[...,cc]/scale,m)
                            pair=m[:,1:]&m[:,:-1]
                            velocity=huber((out[:,1:,cc]-out[:,:-1,cc])/scale,(target_motion[:,1:,cc]-target_motion[:,:-1,cc])/scale,pair)
                            loss=raw+.1*velocity;values={'raw':raw,'velocity':velocity}
                        elif stage in ('teacher','audio'):
                            if args.ablation=='no_teacher':
                                affect=audio_affect(audio,b['audio_features'],b['valid'])
                                distill=affect['global'].sum()*0
                            elif stage=='teacher':
                                teacher=teacher_affect(system,b,identity)
                                audio_output=audio_affect(audio,b['audio_features'],b['valid'])
                                # Stage 3 uses motion-teacher semantics and
                                # audio-native timing in the same renderer.
                                affect=merge_teacher_audio(teacher,audio_output)
                                distill=affect['global'].sum()*0
                            else:
                                with torch.no_grad():teacher=teacher_affect(system,b,identity)
                                audio_output=audio_affect(audio,b['audio_features'],b['valid'],
                                    global_dropout=args.audio_global_dropout, dropout_generator=global_dropout_gen)
                                affect=audio_output
                                if args.audio_supervision == 'expression-prosody':
                                    if not b['intensity_valid'].all() or not prototype_counts[b['emotion_id'].cpu(),b['intensity_id'].cpu()].gt(0).all():
                                        raise ValueError('Every training target requires an observed emotion/intensity prototype')
                                    distill_target = global_prototypes[b['emotion_id'],b['intensity_id']]
                                else:
                                    distill_target = (teacher['global'] if global_prototypes is None
                                                      else global_prototypes[b['emotion_id']])
                                distill, distill_retained = global_distillation(
                                    affect['global'], distill_target, teacher['emotion_logits'],
                                    b['emotion_id'], gate=args.global_distill_gate)
                            flow_affect = ({k:v.detach() for k,v in affect.items()}
                                           if args.audio_supervision == 'expression-prosody' else affect)
                            out=system.flow(b['motion'],b['content'],b['valid'],identity,flow_affect,noise=noise,time=ft,base=base,observation_mask=b['channel_mask'])
                            flow=flow_vector_mse(out['prediction'],out['velocity_target'],out['observation_mask'],flow_vector_std);semantic=semantics(affect,b,emotion_class_weights)
                            loss=flow+.1*semantic+(args.global_distill_weight*distill if stage=='audio' else 0)
                            values={'flow':flow,'semantic':semantic,'global_distill':distill}
                            if args.audio_supervision == 'expression-prosody':
                                expression_target = native_expression_target(b,scales,args.stride)
                                expression_loss = flow_vector_mse(audio_output['expression_state'],expression_target['state'],expression_target['state_mask'])
                                loss = loss + expression_loss
                                values['expression_state'] = expression_loss
                            student_loss = None
                            if args.audio_gradient_scope == 'expression':
                                student_flow = flow_vector_mse(
                                    expression_gradient_view(out['prediction']), out['velocity_target'],
                                    out['observation_mask'], flow_vector_std)
                                student_loss = student_flow + .1*semantic + args.global_distill_weight*distill
                            if stage == 'audio' and args.ablation != 'no_teacher':
                                values['global_distill_retained'] = distill_retained
                            if args.endpoint_weight > 0:
                                endpoint = huber(
                                    out['predicted_residual'] / system.residual_scale,
                                    out['target_residual'] / system.residual_scale,
                                    out['observation_mask'])
                                loss = loss + args.endpoint_weight * endpoint
                                values['endpoint'] = endpoint
                            if args.motion_statistics_weight > 0:
                                stats_loss = motion_statistics_consistency(
                                    out['motion'], b['motion'], b['valid'],
                                    train_support)
                                loss = loss + args.motion_statistics_weight * stats_loss
                                values['motion_statistics'] = stats_loss
                            if independent_probes and args.independent_probe_weight > 0:
                                probe_losses = [independent_probe_consistency(
                                    probe, out['motion'], b['valid'], support.to(args.device),
                                    b['emotion_id'], class_weights=emotion_class_weights)
                                    for probe, (_, support) in zip(independent_probes, probe_payloads)]
                                probe_loss = torch.stack(probe_losses).mean()
                                loss = loss + args.independent_probe_weight * probe_loss
                                values['independent_probe'] = probe_loss
                            # Flow matching does not by itself require the
                            # sampled endpoint to retain the requested
                            # affect; a neutral endpoint can still obtain a
                            # low vector-field loss.  Re-encode the
                            # differentiable endpoint estimate and supervise
                            # its emotion/global/intensity semantics.  The
                            # target affect is detached so this term cannot
                            # collapse the teacher/audio target.
                            if args.emotion_consistency_weight > 0:
                                consistency=generated_emotion_consistency(
                                    system,out,b,identity,affect,
                                    emotion_class_weights=emotion_class_weights)
                                generated_semantic=(
                                    args.emotion_global_weight*consistency['global']+
                                    consistency['emotion_ce']+
                                    args.emotion_intensity_weight*consistency['intensity_ce'])
                                loss=loss+args.emotion_consistency_weight*generated_semantic
                                if student_loss is not None:
                                    student_consistency = generated_emotion_consistency(
                                        system, {**out, 'motion': expression_gradient_view(out['motion'])},
                                        b, identity, affect, emotion_class_weights=emotion_class_weights)
                                    student_semantic = (
                                        args.emotion_global_weight*student_consistency['global'] +
                                        student_consistency['emotion_ce'] +
                                        args.emotion_intensity_weight*student_consistency['intensity_ce'])
                                    student_loss = student_loss + args.emotion_consistency_weight*student_semantic
                                values.update({
                                    'generated_emotion':consistency['emotion_ce'],
                                    'generated_global':consistency['global'],
                                    'generated_intensity':consistency['intensity_ce'],
                                    'generated_semantic':generated_semantic,
                                })
                            if args.content_timing_weight > 0:
                                timing = content_timing_alignment(
                                    out['motion'], base['b0'], b['valid'], b['channel_mask'], b['times'])
                                loss = loss + args.content_timing_weight * timing
                                values['content_timing'] = timing
                            if args.ordinal_mouth_weight > 0 and paired_batch:
                                mouth_mask = torch.zeros_like(b['channel_mask'])
                                mouth_mask[:, list(MOUTH)] = b['channel_mask'][:, list(MOUTH)]
                                target_residual = b['motion'] - b['b0'] - identity['baseline'][:, None]
                                predicted_residual = out['motion'] - out['b0'] - identity['baseline'][:, None]
                                target_energy = masked_rms(target_residual, b['valid'], mouth_mask).detach()
                                predicted_energy = masked_rms(predicted_residual, b['valid'], mouth_mask)
                                local_triplets = torch.arange(len(sample_ix), device=args.device).reshape(-1, 3)
                                ordinal, ordinal_audit = ordinal_intensity_loss(
                                    predicted_energy, local_triplets,
                                    margin=args.ordinal_mouth_margin,
                                    target_energy=target_energy,
                                )
                                loss = loss + args.ordinal_mouth_weight * ordinal
                                values.update({'ordinal_mouth': ordinal,
                                               'ordinal_candidate_pairs': ordinal.new_tensor(ordinal_audit['candidate_pairs']),
                                               'ordinal_used_pairs': ordinal.new_tensor(ordinal_audit['used_pairs'])})
                            # No single-GT random rollout loss is used here:
                            # flow matching plus global semantic supervision is
                            # the objective, while stochastic sampling handles
                            # the one-to-many motion at inference.
                    if args.audio_gradient_scope == 'expression':
                        gradients = student_gradient_overrides(student_loss, audio)
                        norm=optimize(loss,optimizer,parameters,gradient_overrides=gradients)
                    else:
                        norm=optimize(loss,optimizer,parameters)
                    total_steps+=1
                    for k,v in {'total':loss,**values}.items():sums.setdefault(k,[]).append(float(v.detach()))
                    if bi%25==0:
                        progress={'event':'batch','stage':stage,'epoch':epoch+1,'batch':bi+1,'batches':len(epoch_batches),'loss':float(loss.detach()),'grad_norm':norm}
                        if torch.device(args.device).type=='cuda':
                            progress.update(cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/2**30,
                                            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(args.device)/2**30)
                        save_json(args.output/'status.json',{**progress,'status':'running','completed_epochs':epoch,
                                  'total_steps':total_steps,'elapsed_seconds':elapsed_before+time.monotonic()-started})
                        print(json.dumps(progress),flush=True)
                elapsed=time.monotonic()-begin
                record={'stage':stage,'epoch':epoch+1,'batches':len(epoch_batches),'samples':sum(len(ix) for ix in epoch_batches),'seconds':elapsed,'losses':{k:sum(v)/len(v) for k,v in sums.items()},'total_steps':total_steps}
                checkpoint(epoch+1);save_json(stage_dir/f'epoch{epoch+1:03d}.json',record)
                save_json(args.output/'status.json',{**record,'status':'running','stage_index':stage_index,'stage_count':len(STAGES),'epochs_per_stage':epochs,'elapsed_seconds':elapsed_before+time.monotonic()-started})
                print(json.dumps({'event':'epoch_complete',**record}),flush=True)
                if stage in ('teacher','audio') and ((epoch+1)%4==0 or args.smoke):
                    report,_=evaluate(system,audio,data=data,identities=identities,stage=stage,args=args)
                    save_json(stage_dir/f'dev_epoch{epoch+1:03d}.json',report)
            if stage=='articulation':
                if args.articulation_updates is not None and total_steps != args.articulation_updates:
                    raise RuntimeError('Articulation did not reach the exact registered update budget')
                system.stage1.eval()
                cache_current_base(system,data,args.device)
            if stage in ('articulation','identity'):identities=identity_cache(system,data,args.device)
            for name,want in frozen_before.items():
                if state_hash(dict(system=system,audio=audio)[name].state_dict())!=want:raise RuntimeError('Frozen module drift: '+name)
            for name,module in [('system',system),('audio',audio)]:
                actual=state_hash({k:v for k,v in module.named_parameters() if not v.requires_grad})
                if actual!=frozen_parameters[name]:raise RuntimeError('Frozen parameter subset drift: '+name)
            identity_result=identity_report(system,data,args.device);save_json(stage_dir/'identity.json',identity_result)
            report,curves=evaluate(system,audio,data=data,identities=identities,stage=stage,args=args,full=not args.smoke)
            if args.protect_mouth and stage in ('identity','audio'):
                from scripts.mouth_protection import protection_report
                vq=data['splits']['validation'];lookup={cid:i for i,cid in enumerate(vq['clip_id'])}
                vb=subset(vq,torch.tensor([lookup[cid] for cid in curves['clip_id']]),args.device,
                          keys=('valid','b0','speaker_id','emotion_id'))
                vi=batch_identity(identities,vb)
                protected=torch.where(vb['valid'][...,None],vb['b0'] if stage=='identity' else vb['b0']+vi['baseline'][:,None],0.).cpu()
                gates={key:protection_report(pred,protected,curves['target'],curves['valid'],curves['channel_mask'],vb['emotion_id'].cpu()) for key,pred in curves['predictions'].items()}
                save_json(stage_dir/'mouth_protection.json',gates)
                if not all(g['passed'] for g in gates.values()):
                    save_json(stage_dir/'evaluation.json',report)
                    raise RuntimeError('Mouth protection failed; downstream stages not started')
            report['identity']=identity_result;save_json(stage_dir/'evaluation.json',report)
            if args.paper_data:
                from scripts.paper_generation_report import report_generation
                # ``condition_mode`` was a Stage5-only CLI option. Keep the
                # report adapter's historical field populated without
                # reintroducing that option into the deployable protocol.
                report_args=argparse.Namespace(**vars(args));report_args.condition_mode='audio'
                report_generation(curves,data,stage_dir,stage,report_args)
            if not args.compact or stage == 'audio':
                curve_dir=args.artifact_dir/stage if args.artifact_dir else stage_dir
                curve_dir.mkdir(parents=True,exist_ok=True);save_checkpoint(curve_dir/'curves.pt',curves)
            else:curve_dir=None
            save_checkpoint(stage_dir/'final.pt',{'schema':SCHEMA,'recipe_sha256':recipe_hash,'stage':stage,'completed_epochs':epochs,
                'system':system.state_dict(),'audio':audio.state_dict(),
                'config':cfg,'scales':data['target_scales'],'inference_only':True,
                'feature_stats':data['feature_stats'],'data_manifest_sha256':data['provenance'].get('manifest_sha256'),
                'test_loaded':False})
            save_json(stage_dir/'complete.json',{'stage':stage,'completed_epochs':epochs,'final_sha256':sha(stage_dir/'final.pt'),
                'curves':str(curve_dir/'curves.pt') if curve_dir else None,'curves_sha256':sha(curve_dir/'curves.pt') if curve_dir else None})
            resume_payload=None;start_epoch=0
        save_json(args.output/'summary.json',{'schema':SCHEMA,'status':'complete','stages':recipe['stages'],'epochs_per_stage':epochs,'smoke':args.smoke,
            'total_steps':total_steps,'elapsed_seconds':elapsed_before+time.monotonic()-started,'default_replaced':False,'test_loaded':False})
        save_json(args.output/'status.json',{'status':'complete','elapsed_seconds':elapsed_before+time.monotonic()-started,'summary':'summary.json'})
        print('FULL_STAGED_TRAINING_COMPLETE',flush=True)
    except BaseException as exc:
        failure={'status':'failed','stage':current_stage,'exception':repr(exc),'recover_from':'last.pt','elapsed_seconds':elapsed_before+time.monotonic()-started}
        save_json(args.output/'failure.json',failure)
        save_json(args.output/'status.json',failure)
        raise

if __name__=='__main__':main()
