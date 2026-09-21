"""Role-isolated data and frozen deployment conditions for an upper decoder.

Only the selected train OR validation query/enrollment shards are opened.
Metadata for all roles is validated; test tensors are never opened. Query
motion is retained as supervision, but never used to construct conditions.
No statistics or teacher-motion targets are fitted/computed by this loader.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import yaml

from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from kinetalk_b0.neutral_data import stack_clips
from scripts.extract_emotion2vec_pilot import sha
from scripts.prepare_paper_full_data import SCHEMA as PREPARED_SCHEMA, pad_clip, validate_manifest
from scripts.train_full_staged import cache_current_base, identity_cache

SCHEMA = 'reference_decoder_role_context_v1'


def _validate_shard(saved, record, row, kind):
    if saved.get('schema') != PREPARED_SCHEMA or saved.get('row') != row:
        raise ValueError('Shard schema or metadata differs from approved manifest')
    frames = int(record['frames'])
    valid, channels = saved.get('valid'), saved.get('channel_mask')
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.shape != (frames,)
            or int(valid.sum()) != record['valid_frames'] or not valid.any()):
        raise ValueError('Shard valid-frame coverage differs')
    if not torch.is_tensor(channels) or channels.dtype != torch.bool or channels.shape != (52,):
        raise ValueError('Shard channel mask must be Boolean [52]')
    widths = {'motion': 52, 'content': 768, 'audio': 83}
    if kind == 'query':
        widths.update(middle=768, prosody=4)
    for key, width in widths.items():
        value = saved.get(key)
        if (not torch.is_tensor(value) or value.shape != (frames, width)
                or not value.is_floating_point()):
            raise ValueError('Invalid shard tensor: ' + key)
        observed = valid[:, None] & channels[None] if key == 'motion' else valid[:, None].expand_as(value)
        if not torch.isfinite(value[observed]).all():
            raise ValueError('Observed nonfinite shard tensor: ' + key)
    times = saved.get('times')
    if (not torch.is_tensor(times) or times.shape != (frames,) or not times.is_floating_point()
            or not torch.isfinite(times).all() or (frames > 1 and not torch.allclose(
                times[1:] - times[:-1], torch.full_like(times[1:], .04), atol=1e-5, rtol=0))):
        raise ValueError('Shard native clock must be finite 25 fps')


def _read_role(data_path, role):
    root = Path(data_path).resolve()
    index_path, manifest_path, config_path = root/'index.json', root/'manifest.json', root/'config.yaml'
    index = json.loads(index_path.read_text(encoding='utf8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    validate_manifest(manifest)
    recipe = index.get('recipe', {})
    if (index.get('schema') != PREPARED_SCHEMA or recipe.get('schema') != PREPARED_SCHEMA
            or recipe.get('manifest_sha256') != manifest['manifest_sha256']
            or index.get('test_loaded') is not False or recipe.get('test_loaded') is not False):
        raise ValueError('Prepared index schema/manifest/test binding differs')
    config_hash = sha(config_path)
    if config_hash != recipe.get('config_sha256'):
        raise ValueError('Prepared configuration hash differs')
    config = yaml.safe_load(config_path.read_text(encoding='utf8'))
    expected_rows = {(r['clip_id'], source, kind): r
                     for source in ('train', 'val') for kind in ('query', 'enrollment')
                     for r in manifest['roles'][source][kind]}
    records = index.get('records', [])
    actual = [(r['clip_id'], r['role'], r['kind']) for r in records]
    if len(set(actual)) != len(actual) or set(actual) != set(expected_rows):
        raise ValueError('Prepared index clip/role coverage differs')
    # Resolve paths for all metadata, but never stat/hash/open nonselected files.
    paths = {}
    for rec in records:
        key = (rec['clip_id'], rec['role'], rec['kind'])
        path = (root/rec['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Shard path escapes prepared directory')
        row = expected_rows[key]
        if rec['frames'] != row['frames'] or rec['valid_frames'] != row['valid_frames']:
            raise ValueError('Index frame metadata differs from manifest')
        paths[key] = path
    if len(set(paths.values())) != len(paths):
        raise ValueError('Distinct role/clip records cannot share one shard path')
    people = sorted({r['speaker'] for source in ('train', 'val') for r in manifest['roles'][source]['query']})
    sids = {person: i for i, person in enumerate(people)}
    source_role = 'train' if role == 'train' else 'val'
    chosen = [r for r in records if r['role'] == source_role]
    if not chosen:
        raise ValueError('Selected role has no prepared shards')
    # Legacy padding shape and speaker numbering depend on metadata only.
    length = max(r['frames'] for r in records)
    parts = {'query': [], 'enrollment': []}; loaded = []
    for rec in chosen:
        key = (rec['clip_id'], rec['role'], rec['kind'])
        path = paths[key]
        if sha(path) != rec['sha256']:
            raise ValueError('Selected shard hash differs: ' + rec['clip_id'])
        saved = torch.load(path, map_location='cpu', weights_only=False)
        _validate_shard(saved, rec, expected_rows[key], rec['kind'])
        person = saved['row']['speaker']
        if person not in sids:
            raise ValueError('Enrollment identity has no query identity')
        parts[rec['kind']].append(pad_clip(saved, length, sids[person]))
        loaded.append({'clip_id': rec['clip_id'], 'kind': rec['kind'], 'sha256': rec['sha256']})
    if not parts['query'] or not parts['enrollment']:
        raise ValueError('Queries and independent enrollment are required')
    refs, anchors = {}, {}
    for person in sorted({x['speaker'] for x in parts['enrollment']}):
        sid = sids[person]
        ref = stack_clips([x for x in parts['enrollment'] if x['speaker'] == person])
        observed = ref['valid'][..., None] & ref['channel_mask'][:, None]
        means = torch.where(observed, ref['motion'], 0.).sum(1) / observed.sum(1).clamp_min(1)
        anchor_valid = observed.sum(1).gt(0).all(0)
        anchor = torch.where(anchor_valid, means.quantile(.5, dim=0), 0.)
        refs[sid] = ref; anchors[sid] = (anchor, anchor_valid)
    q = stack_clips(parts['query'])
    q['anchors'] = torch.stack([anchors[int(s)][0] for s in q['speaker_id']])
    q['anchor_valid'] = torch.stack([anchors[int(s)][1] for s in q['speaker_id']])
    current_sids = sorted(refs)
    provenance = {'schema': SCHEMA, 'prepared_schema': PREPARED_SCHEMA,
                  'manifest_sha256': manifest['manifest_sha256'], 'index_sha256': sha(index_path),
                  'config_sha256': config_hash, 'loaded_role': role, 'loaded_source_role': source_role,
                  'loaded_shards': loaded, 'other_role_shards_opened': False, 'test_loaded': False,
                  'speaker_numbering': 'legacy sorted train+val query speaker metadata',
                  'frame_policy': 'legacy global metadata padding, native frames remain 25fps',
                  'query_motion_role': 'supervision only, no query-motion-derived deployment conditions',
                  'target_scales_fitted': False, 'target_decomposition_computed': False}
    return {'config': config, 'splits': {role: q}, 'refs': refs, 'ref_groups': {},
            'fit_sids': current_sids if role == 'train' else [],
            'dev_sids': current_sids if role == 'validation' else [], 'provenance': provenance}


def _load_frozen_source(data, source_path, device, expected_source_sha256):
    source_path = Path(source_path).resolve()
    source_hash = sha(source_path)
    if expected_source_sha256 is not None and source_hash != expected_source_sha256:
        raise ValueError('Frozen source checkpoint hash differs from requested binding')
    saved = torch.load(source_path, map_location='cpu', weights_only=False)
    if sha(source_path) != source_hash:
        raise ValueError('Frozen source checkpoint changed while loading')
    if saved.get('data_manifest_sha256') != data['provenance']['manifest_sha256']:
        raise ValueError('Frozen source manifest mismatch')
    if saved.get('test_loaded') is not False:
        raise ValueError('Frozen source must declare test_loaded=False')
    if not all(isinstance(saved.get(key), dict) for key in ('system', 'audio', 'config')):
        raise ValueError('Frozen source requires system/audio state dictionaries and config')
    cfg = copy.deepcopy(saved['config'])
    if cfg.get('data') != data['config'].get('data'):
        raise ValueError('Frozen source data configuration differs from prepared configuration')
    for key in ('content_dim', 'emotion_dim', 'style_dim', 'hidden_dim', 'dit_dim', 'dit_depth', 'heads'):
        if cfg['model'].get(key) != data['config']['model'].get(key):
            raise ValueError('Frozen source architecture differs: ' + key)
    if not all(key in cfg['model'] for key in ('motion_support', 'residual_support', 'mouth_reference_calibration')):
        raise ValueError('Frozen source requires fixed support and verified mouth calibration')
    system = NeutralAffectSystem(cfg)
    system.load_state_dict(saved['system'], strict=True)
    system.to(device).requires_grad_(False).eval()
    state = saved['audio']
    mean, std = state.get('feature_mean'), state.get('feature_std')
    if (not torch.is_tensor(mean) or mean.shape != (1540,) or not torch.is_tensor(std)
            or std.shape != mean.shape or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Frozen audio checkpoint must contain valid 1540D feature statistics')
    audio = SlowStateAffect(mean, std, hidden=state['input.weight'].shape[0],
        global_dim=state['global_head.weight'].shape[0], local_dim=state['local_head.weight'].shape[0],
        num_emotions=state['emotion_classifier.weight'].shape[0],
        num_levels=state['intensity_classifier.weight'].shape[0])
    audio.load_state_dict(state, strict=True)
    audio.to(device).requires_grad_(False).eval()
    data.update(system=system, config=cfg, feature_stats={'mean': mean.clone(), 'std': std.clone(),
                'source': 'frozen_source.audio.state_dict', 'statistics_fitted_by_loader': False})
    data['provenance'].update(frozen_source_sha256=source_hash,
        expected_source_sha256=expected_source_sha256, source_hash_verified=expected_source_sha256 is not None,
        feature_statistics_source='source audio feature_mean/feature_std buffers, never selected-role fitting',
        source_stage=saved.get('stage'), frozen_system_and_audio=True)
    return system, audio


@torch.no_grad()
def load_reference_context(data_path, source_path, device, role='train', seed=47,
                           *, batch_size=32, expected_source_sha256=None):
    """Return ``(data, system, audio, identities)`` with one ``data['splits']`` role.

    ``role`` is exactly ``train`` or ``validation``; never test. Pin the source
    digest via ``expected_source_sha256`` for a previously selected checkpoint.
    Query tensors keep the legacy ``subset`` schema. Extra condition fields:
    global_code [B,G], intensity_value [B,1], identity_code [B,S],
    identity_baseline [B,52], frozen_local [B,T,L], b0 [B,T,52], h0 [B,T,H].
    ``anchors``/``anchor_valid`` are raw independent-reference anchors [B,52].
    No target_centered_state, teacher_logits or target_mean is constructed.
    """
    if role not in ('train', 'validation'):
        raise ValueError('role must be train or validation; test tensor access is forbidden')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    data = _read_role(data_path, role)
    torch.manual_seed(seed)
    system, audio = _load_frozen_source(data, source_path, torch.device(device), expected_source_sha256)
    cache_current_base(system, data, device, batch_size=batch_size)
    identities = identity_cache(system, data, device)
    q = data['splits'][role]
    cache = {key: [] for key in ('global_code', 'intensity_value', 'emotion_logits', 'intensity_logits', 'frozen_local')}
    for ids in torch.arange(len(q['valid'])).split(batch_size):
        # Deliberately pass no query motion, query label, target mean or anchor.
        affect = audio(q['audio_features'][ids].to(device), q['valid'][ids].to(device))
        for key, source in (('global_code', 'global'), ('intensity_value', 'intensity_value'),
                            ('emotion_logits', 'emotion_logits'), ('intensity_logits', 'intensity_logits'),
                            ('frozen_local', 'local')):
            cache[key].append(affect[source].cpu())
    q.update({key: torch.cat(values) for key, values in cache.items()})
    q['identity_code'] = torch.cat([identities[int(s)]['code'].cpu() for s in q['speaker_id']])
    q['identity_baseline'] = torch.cat([identities[int(s)]['baseline'].cpu() for s in q['speaker_id']])
    data['provenance']['condition_sources'] = {
        'global_code': 'frozen audio network, query audio only',
        'intensity_value': 'frozen audio network, query audio only',
        'frozen_local': 'frozen audio network, query audio only',
        'b0_h0': 'frozen B0, query content only',
        'identity': 'independent neutral enrollment motion minus enrollment B0',
        'anchors': 'median across independent enrollment raw-motion clip means'}
    return data, system, audio, identities


__all__ = ['SCHEMA', 'load_reference_context']
