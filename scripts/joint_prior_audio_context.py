"""Frozen whole-native acoustic context for the joint motion-prior experiment.

The source is the completed run12 audio stage, including its trained global
emotion and intensity heads. Its motion-teacher training is retained in these
unchanged weights; no motion teacher or query-motion tensor runs at inference.
Only global, intensity and native local outputs are evaluated. In particular,
the unused spline-state head is not an inference condition for this experiment.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from kinetalk_b0.models.slow_state_affect import SlowStateAffect


SOURCE_SCHEMA = 'full_staged_spline_innovation_v1'
# Pinned to the already audited completed source, not a newly trained model.
EXPECTED_CHECKPOINT_SHA256 = '73a8f17137772f882bf6d8c65dbf2f6c36b6c42f5724091f7a2e7a42870df42c'
EXPECTED_RECIPE_SHA256 = '3e427c401a78ac87951efc084d8589f2b5ade2149554b77476eddb73eb552c83'
EXPECTED_INPUT_SHA256 = {
    'cache': '89537672239b805876ed15ab24aa1e5abd0fe231cb75cfa7c1e07bf7548efcae',
    'audio': 'f1f03c58e3a2175a72ade9d32bb97831d534919755db5c7d17e6a1a3d49ac7ff',
    'targets': 'a68cb18954f8705d1c2ca124df506c1eb4f9c2e36dea6bbe50bcd3452e5199e0',
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def load_frozen_audio(checkpoint_path, device='cpu'):
    """Load the audited run12 audio model; fail closed on changed lineage.

    Required sidecars are ``audio/complete.json`` next to ``final.pt`` and
    ``run12/provenance.json`` one directory above. All three can be copied to
    another root; path names themselves do not authenticate the source. The
    complete checkpoint and recipe hashes, stage, epochs and input bindings
    must match the historical source used by motion_process30.

    Returns a frozen eval ``SlowStateAffect`` with JSON-serializable
    ``source_binding`` metadata. Call ``assert_source_binding(model, lineage)``
    on the live result of ``train_motion_process.load_clips`` before training.
    Dimensions follow the saved tensors, matching the historical _make_audio.
    """
    path = Path(checkpoint_path)
    complete_path = path.with_name('complete.json')
    provenance_path = path.parent.parent / 'provenance.json'
    complete = json.loads(complete_path.read_text(encoding='utf8'))
    provenance = json.loads(provenance_path.read_text(encoding='utf8'))
    digest = _sha(path)
    if digest != EXPECTED_CHECKPOINT_SHA256 or complete.get('final_sha256') != digest:
        raise ValueError('Audio checkpoint differs from the pinned completed run12 source')
    recipe = provenance.get('recipe')
    if (not isinstance(recipe, dict) or _canonical_hash(recipe) != EXPECTED_RECIPE_SHA256
            or provenance.get('recipe_sha256') != EXPECTED_RECIPE_SHA256):
        raise ValueError('Audio source recipe binding mismatch')
    if (recipe.get('schema') != SOURCE_SCHEMA or recipe.get('epochs_per_stage') != 12
            or recipe.get('stride_frames') != 16 or complete.get('stage') != 'audio'
            or complete.get('completed_epochs') != 12):
        raise ValueError('Completed 12-epoch stride16 audio-stage source required')
    inputs = recipe.get('data_provenance', {}).get('input_sha256', {})
    if any(inputs.get(key) != want for key, want in EXPECTED_INPUT_SHA256.items()):
        raise ValueError('Audio checkpoint and motion-process data sources differ')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if (checkpoint.get('schema') != SOURCE_SCHEMA or checkpoint.get('stage') != 'audio'
            or checkpoint.get('completed_epochs') != 12
            or checkpoint.get('recipe_sha256') != EXPECTED_RECIPE_SHA256
            or checkpoint.get('inference_only') is not True):
        raise ValueError('Audio checkpoint metadata contradicts completed source')
    state = checkpoint.get('audio')
    if (not isinstance(state, Mapping) or not state
            or any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in state.values())):
        raise ValueError('Finite saved audio tensors required')
    # Initialization is overwritten; avoid advancing the caller's training RNG.
    with torch.random.fork_rng(devices=[]):
        model = SlowStateAffect(state['feature_mean'], state['feature_std'],
            global_dim=state['global_head.weight'].shape[0],
            hidden=state['input.weight'].shape[0],
            local_dim=state['local_head.weight'].shape[0],
            num_emotions=state['emotion_classifier.weight'].shape[0],
            num_levels=state['intensity_classifier.weight'].shape[0], stride=16)
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    model.source_binding = {
        'schema': 'joint_prior_frozen_audio_source_v1',
        'checkpoint_sha256': digest, 'recipe_sha256': EXPECTED_RECIPE_SHA256,
        'complete_sha256': _sha(complete_path), 'provenance_sha256': _sha(provenance_path),
        'input_sha256': {key: inputs[key] for key in EXPECTED_INPUT_SHA256},
        'dimensions': {'features': model.feature_mean.numel(),
            'global': model.global_head.out_features, 'intensity': 1,
            'local': model.local_head.out_features},
        'frozen': True, 'full_native_clock': True, 'query_motion_used': False,
        'source_training_scope': 'Historical 2315 fit clips; new inner holdouts were exposed to this frozen source',
    }
    return model


def assert_source_binding(model, lineage):
    """Check newly loaded data against both pinned and actual source inputs.

    ``lineage`` is returned by ``train_motion_process.load_clips`` (not a
    filesystem path). That loader verifies source files, native clocks and
    delta metadata. This check binds those live files to the frozen audio head.
    It does not claim that the new inner holdouts are unseen by the old model.
    """
    binding = getattr(model, 'source_binding', None)
    if (not isinstance(binding, Mapping) or not isinstance(lineage, Mapping)
            or binding.get('checkpoint_sha256') != EXPECTED_CHECKPOINT_SHA256
            or binding.get('recipe_sha256') != EXPECTED_RECIPE_SHA256):
        raise ValueError('Validated frozen audio source and live data lineage required')
    for key, want in EXPECTED_INPUT_SHA256.items():
        if lineage.get(key) != want or binding.get('input_sha256', {}).get(key) != want:
            raise ValueError('Live motion-process '+key+' differs from the audio training source')
    return dict(binding)


@torch.no_grad()
def encode_clips(model, clips, batch_size=16, device='cpu'):
    """Return an ordered list of CPU ``clip_id/global/intensity/local`` dicts.

    Each input needs only ``clip_id``, floating ``features[T,F]`` and Boolean
    ``valid[T]``. No labels, anchors, state, upper coefficients, motion, clip
    mean or teacher boundaries are accessed. Native gaps stay at their original
    indices; padding to multiples of16 never changes the supplied time clock.
    ``global`` is [D], ``intensity`` [1], and ``local`` [T,L]. Invalid local
    positions are zero, and each returned T equals the complete input length.
    """
    if not isinstance(model, SlowStateAffect):
        raise ValueError('A compatible frozen SlowStateAffect is required')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Audio model must be frozen and in eval mode')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive integer batch size required')
    if not isinstance(clips, Sequence) or isinstance(clips, (str, bytes)) or not clips:
        raise ValueError('A nonempty sequence of native acoustic clips is required')
    requested = torch.device(device)
    actual = model.feature_mean.device
    if actual.type != requested.type or (requested.index is not None and requested.index != actual.index):
        raise ValueError('Requested encoding device differs from the frozen model')
    feature_dim = model.feature_mean.numel()
    seen = set()
    # Only inspect the three allowed keys, even if the caller retains GT fields.
    acoustic = []
    for clip in clips:
        cid, features, valid = clip['clip_id'], clip['features'], clip['valid']
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError('Unique nonempty acoustic clip IDs required')
        if (not torch.is_tensor(features) or features.ndim != 2
                or features.shape[-1] != feature_dim or not features.is_floating_point()
                or not torch.is_tensor(valid) or valid.dtype != torch.bool
                or valid.shape != features.shape[:1] or valid.device != features.device
                or not valid.any() or not torch.isfinite(features[valid]).all()):
            raise ValueError('Finite observed native acoustic features and Boolean masks required')
        seen.add(cid)
        acoustic.append((cid, features, valid))
    result = []
    for start in range(0, len(acoustic), batch_size):
        selected = acoustic[start:start+batch_size]
        frames = ((max(len(valid) for _, _, valid in selected)+15)//16)*16
        features = model.feature_mean.new_zeros(len(selected), frames, feature_dim)
        valid = torch.zeros(len(selected), frames, dtype=torch.bool, device=actual)
        for row, (_, source, mask) in enumerate(selected):
            features[row, :len(mask)] = source.to(features)
            valid[row, :len(mask)] = mask.to(actual)
        # Algebraically the global/intensity/local portion of model.forward.
        # Avoid the unused state-head spline pseudoinverse on full-length clips.
        clean = torch.where(valid[..., None], features, model.feature_mean)
        hidden = torch.where(valid[..., None], F.silu(model.input(
            (clean-model.feature_mean)/model.feature_std)), 0.)
        for block in model.blocks:
            hidden = block(hidden, valid)
        pooled = hidden.sum(1)/valid.sum(1, keepdim=True)
        global_code = model.global_head(pooled)
        logits = model.intensity_classifier(global_code)
        levels = torch.arange(logits.shape[-1], device=actual, dtype=features.dtype)
        intensity = (logits.softmax(-1)*levels).sum(-1, keepdim=True)
        local = torch.where(valid[..., None], model.local_head(hidden), 0.)
        if any(not torch.isfinite(value).all() for value in (global_code, intensity, local)):
            raise FloatingPointError('Frozen audio context contains nonfinite outputs')
        for row, (cid, _, mask) in enumerate(selected):
            result.append({'clip_id': cid, 'global': global_code[row].cpu().clone(),
                'intensity': intensity[row].cpu().clone(),
                'local': local[row, :len(mask)].cpu().clone()})
    return result
