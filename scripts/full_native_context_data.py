"""Allowlist-bound native context with old center features and verified deltas.

No normalization is fitted or applied here. Center batches retain the exact
historical query dictionary; full batches expose native raw inputs only, so
96-frame encoded caches cannot silently become full-context conditions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.prepare_label_guided_audio_cache import read_manifest


SCHEMA = 'full_native_audio_delta_v1'
FRAME_KEYS = {'motion', 'content', 'audio', 'audio_features', 'valid', 'motion_valid', 'times'}
# Only explicitly scalar/clip-level fields may pass into the full context.
CLIP_KEYS = ('clip_id', 'sentence_id', 'speaker', 'speaker_id', 'emotion_id',
             'intensity_id', 'intensity_valid', 'dataset_id', 'channel_mask',
             'anchors', 'anchor_valid')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode()+str(tuple(value.shape)).encode()
                          +value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _contained(root, relative):
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path leaves declared root')
    return path


def _row(value, index, count):
    if torch.is_tensor(value):
        if value.ndim == 0 or len(value) != count:
            raise ValueError('Query tensor lacks a clip dimension')
        return value[index].detach().cpu().clone()
    if isinstance(value, (list, tuple)) and len(value) == count:
        return copy.deepcopy(value[index])
    raise ValueError('Query metadata lacks a clip dimension')


def _stack(rows):
    return {key: torch.stack([row[key] for row in rows]) if torch.is_tensor(value)
            else [row[key] for row in rows] for key, value in rows[0].items()}


class NativeContextStore:
    """Lazy native clip loader restricted to one query role's exact allowlist.

    ``batch(..., mode='center')`` is the old96 query, including old encodings.
    ``batch(..., mode='full')`` contains raw full-clip inputs, padded to a
    multiple of16. Callers must recompute full-context encodings explicitly.
    Native invalid frames/padding are zero except that timestamps retain the
    native clock. Audio at old center valid positions is copied bit-exactly.
    """

    def __init__(self, q, native_root, manifest_path, delta_dir, *, role=None):
        self.q = q
        self._cache = {}
        self.native_root = Path(native_root).resolve()
        self.delta_dir = Path(delta_dir).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.clip_ids = list(q.get('clip_id', []))
        self.count = len(self.clip_ids)
        required = ('motion', 'content', 'audio_features', 'valid', 'times',
                    'channel_mask', 'sentence_id', 'speaker_id', 'emotion_id')
        if (not self.count or len(set(self.clip_ids)) != self.count
                or any(key not in q for key in required)):
            raise ValueError('Complete unique center query allowlist required')
        valid = q['valid']
        if (valid.dtype != torch.bool or valid.shape != (self.count, 96)
                or not valid.any(1).all() or q['times'].shape != valid.shape
                or q['audio_features'].shape != (self.count, 96, 1540)
                or q['audio_features'].dtype != torch.float32
                or q['motion'].shape != (self.count, 96, 52)
                or q['content'].shape != (self.count, 96, 768)
                or q['channel_mask'].shape != (self.count, 52)
                or q['channel_mask'].dtype != torch.bool):
            raise ValueError('Expected historical native96 center shape/masks')
        if (not torch.isfinite(q['times']).all()
                or not torch.allclose(q['times'][:, 1:]-q['times'][:, :-1],
                                      torch.full_like(q['times'][:, 1:], .04), atol=1e-7, rtol=0)
                or not torch.isfinite(q['audio_features'][valid]).all()):
            raise ValueError('Invalid center clock or observed audio features')
        self.rows = read_manifest(self.manifest_path)
        index_path = self.delta_dir / 'manifest.json'
        complete_path = self.delta_dir / 'complete.json'
        if not index_path.is_file() or not complete_path.is_file():
            raise ValueError('A complete bound audio delta directory is required')
        self.index = json.loads(index_path.read_text(encoding='utf8'))
        complete = json.loads(complete_path.read_text(encoding='utf8'))
        if (self.index.get('schema') != SCHEMA or complete.get('schema') != SCHEMA
                or self.index.get('status') != 'complete' or complete.get('status') != 'complete'
                or complete.get('manifest_sha256') != sha(index_path)):
            raise ValueError('Audio delta manifest is incomplete or changed')
        self.source = self.index.get('source', {})
        if (self.source.get('manifest_sha256') != sha(self.manifest_path)
                or self.source.get('layers') != [2, 4, 6]
                or not isinstance(self.source.get('audio_sha256'), str)
                or not isinstance(self.source.get('model'), dict)
                or not isinstance(self.source.get('geometry'), dict)):
            raise ValueError('Native manifest or audio extraction lineage differs')
        config = self.index.get('config', {})
        if (config.get('middle_max_atol') != .002 or config.get('middle_rms_atol') != .0001
                or config.get('prosody_max_atol') != .000001 or config.get('rtol') != 0
                or config.get('normalization_recomputed') is not False
                or config.get('storage') != {'middle': 'float16', 'prosody': 'float32'}):
            raise ValueError('Delta overlap policy or frozen normalization contract differs')
        roles = self.index.get('selected_roles', {})
        if set(roles) != {'train', 'validation'}:
            raise ValueError('Only authorized train/validation delta roles accepted')
        selected = roles['train']+roles['validation']
        if (len(set(selected)) != len(selected) or set(self.index.get('clips', {})) != set(selected)
                or complete.get('selected_count') != len(selected)
                or complete.get('completed_count') != len(selected)):
            raise ValueError('Incomplete or conflicting delta allowlist')
        possible = [name for name, ids in roles.items() if set(self.clip_ids) <= set(ids)]
        if role is None:
            if len(possible) != 1:
                raise ValueError('Query allowlist must belong to one recorded role')
            role = possible[0]
        if role not in possible:
            raise ValueError('Query role is not authorized by delta membership')
        self.role = role
        self.entries = []
        for i, cid in enumerate(self.clip_ids):
            row = self.rows.get(cid)
            entry = self.index['clips'][cid]
            if (row is None or row.get('split') != 'train' or entry.get('role') != role
                    or entry.get('status') != 'complete'
                    or str(row.get('sentence')) != str(q['sentence_id'][i])
                    or str(entry.get('sentence_id')) != str(q['sentence_id'][i])
                    or int(row.get('emotion', row.get('emotion_id', -1))) != int(q['emotion_id'][i])
                    or ('speaker' in q and str(row.get('speaker')) != str(q['speaker'][i]))
                    or ('speaker_id' in row and int(row['speaker_id']) != int(q['speaker_id'][i]))):
                raise ValueError('Query/native/delta metadata or role differs: '+cid)
            if (entry.get('native_artifact_sha256') != row['artifact_sha256']
                    or type(entry.get('native_frames')) is not int or entry['native_frames'] < 2
                    or type(entry.get('center_start')) is not int
                    or not 0 <= entry['center_start'] < entry['native_frames']
                    or entry.get('center_length') != 96):
                raise ValueError('Invalid bound native crop geometry: '+cid)
            _contained(self.native_root, row['artifact'])
            _contained(self.delta_dir, entry['file'])
            self.entries.append(entry)
        self.native_lengths = torch.tensor([x['native_frames'] for x in self.entries], dtype=torch.long)
        self.full_lengths = self.native_lengths
        self.center_starts = torch.tensor([x['center_start'] for x in self.entries], dtype=torch.long)
        self.center_lengths = torch.full_like(self.native_lengths, 96)
        self.provenance = {'schema': SCHEMA, 'role': role,
            'native_manifest_sha256': self.source['manifest_sha256'],
            'delta_manifest_sha256': sha(index_path), 'old_audio_sha256': self.source['audio_sha256'],
            'normalization_recomputed': False, 'center_policy': 'exact inherited query values',
            'full_precision': 'native FP16/FP32 arrays promoted to FP32; center motion/audio verified after conversion to old cached dtype',
            'full_encoded_caches_used': [], 'test_loaded': False}

    def __len__(self):
        return self.count

    def _index(self, index):
        if isinstance(index, str):
            try:
                return self.clip_ids.index(index)
            except ValueError as error:
                raise ValueError('Clip is outside query allowlist') from error
        index = int(index)
        if not 0 <= index < self.count:
            raise IndexError('Clip index outside allowlist')
        return index

    def _load(self, index):
        i = self._index(index)
        if i in self._cache:
            return {key: value for key, value in self._cache[i].items()}
        cid, entry = self.clip_ids[i], self.entries[i]
        row = self.rows[cid]
        path = _contained(self.native_root, row['artifact'])
        delta_path = _contained(self.delta_dir, entry['file'])
        if sha(path) != row['artifact_sha256'] or sha(delta_path) != entry['sha256']:
            raise ValueError('Native artifact or audio delta SHA256 differs: '+cid)
        with np.load(path, allow_pickle=False) as archive:
            required = ('motion', 'content', 'audio', 'times', 'mask', 'channel_mask', 'provenance')
            if any(key not in archive for key in required):
                raise ValueError('Incomplete native raw artifact')
            arrays = {key: np.asarray(archive[key]).copy() for key in required[:-1]}
            provenance = json.loads(str(archive['provenance'].item()))
        n, start = entry['native_frames'], entry['center_start']
        times = torch.from_numpy(arrays['times'].astype(np.float64))
        if (not str(provenance.get('schema', '')).startswith('native_affect_style_v4')
                or provenance.get('clock_evidence') != 'embedded_video' or float(provenance.get('fps', 0)) != 25
                or times.shape != (n,) or not torch.isfinite(times).all()
                or not torch.allclose(times[1:]-times[:-1], torch.full_like(times[1:], .04), atol=1e-5, rtol=0)
                or arrays['mask'].shape != (n,) or not np.isin(arrays['mask'], [0, 1]).all()
                or arrays['channel_mask'].shape != (52,) or not np.isin(arrays['channel_mask'], [0, 1]).all()):
            raise ValueError('Invalid native provenance, clock, or masks')
        valid = torch.from_numpy(arrays['mask'].astype(bool))
        channel_mask = torch.from_numpy(arrays['channel_mask'].astype(bool))
        if not valid.any() or not torch.equal(channel_mask, self.q['channel_mask'][i].cpu()):
            raise ValueError('Native observed channels differ from center query')
        full = {}
        for key, width in (('motion', 52), ('content', 768), ('audio', 83)):
            if arrays[key].shape != (n, width):
                raise ValueError('Invalid full native '+key+' shape')
            value = torch.from_numpy(arrays[key].astype(np.float32))
            if not torch.isfinite(value[valid]).all():
                raise ValueError('Nonfinite observed native '+key)
            full[key] = torch.where(valid[:, None], value, 0.)
        count = min(96, n-start)
        oldvalid = self.q['valid'][i].cpu()
        expected = torch.zeros(96, dtype=torch.bool)
        expected[:count] = valid[start:start+count]
        if (not torch.equal(expected, oldvalid)
                or not torch.allclose(times[start:start+count], self.q['times'][i, :count].cpu().double(), atol=1e-7, rtol=0)):
            raise ValueError('Native crop clock/validity differs from old center')
        for key in ('motion', 'content', 'audio'):
            if key not in self.q:
                continue
            old = self.q[key][i, :count].cpu()
            observed = oldvalid[:count]
            if not torch.equal(full[key][start:start+count].to(old.dtype)[observed], old[observed]):
                raise ValueError('Native/cached center storage differs: '+key)
        oldfeatures = self.q['audio_features'][i].cpu()
        if not torch.equal(full['content'][start:start+count][oldvalid[:count]], oldfeatures[:count, :768][oldvalid[:count]]):
            raise ValueError('Native content and old FP32 features differ')
        delta = torch.load(delta_path, map_location='cpu', weights_only=False)
        binding = delta.get('binding', {})
        expected_binding = {'old_audio_sha256': self.source['audio_sha256'],
            'native_manifest_sha256': self.source['manifest_sha256'],
            'native_artifact_sha256': row['artifact_sha256'],
            'wave_sha256': provenance.get('audio_sha256'),
            'model_sha256': canonical_hash({key: value for key, value in self.source['model'].items()
                                           if key != 'model_dir'}),
            'geometry_sha256': canonical_hash(self.source['geometry']), 'layers': [2, 4, 6],
            'center_features_sha256': tensor_sha(oldfeatures),
            'center_valid_sha256': tensor_sha(self.q['valid'][i]),
            'center_times_sha256': tensor_sha(self.q['times'][i])}
        if (delta.get('schema') != SCHEMA or delta.get('clip_id') != cid
                or delta.get('role') != self.role or str(delta.get('sentence_id')) != str(self.q['sentence_id'][i])
                or delta.get('native_frames') != n or delta.get('center_start') != start
                or delta.get('center_length') != 96
                or any(binding.get(key) != value for key, value in expected_binding.items())
                or entry.get('wave_sha256') != provenance.get('audio_sha256')):
            raise ValueError('Delta/query/audio extraction binding differs: '+cid)
        extra = valid.clone()
        extra[start:start+count] = False
        indices = delta.get('native_indices')
        middle, prosody = delta.get('middle'), delta.get('prosody')
        if (not torch.is_tensor(indices) or indices.dtype != torch.long
                or not torch.equal(indices, extra.nonzero(as_tuple=True)[0])
                or entry.get('extra_valid_frames') != len(indices)
                or not torch.is_tensor(middle) or middle.dtype != torch.float16 or middle.shape != (len(indices), 768)
                or not torch.is_tensor(prosody) or prosody.dtype != torch.float32 or prosody.shape != (len(indices), 4)
                or not torch.isfinite(middle).all() or not torch.isfinite(prosody).all()):
            raise ValueError('Delta must cover exactly extra valid native frames')
        check = delta.get('overlap_check', {})
        if (any(not math.isfinite(float(check.get(group, {}).get(metric, float('inf'))))
                for group in ('content', 'middle', 'prosody') for metric in ('max_abs', 'rms', 'changed_fraction'))
                or check.get('middle', {}).get('max_abs', float('inf')) > .002
                or check.get('middle', {}).get('rms', float('inf')) > .0001
                or check.get('prosody', {}).get('max_abs', float('inf')) > .000001
                or check.get('content', {}).get('max_abs', float('inf')) != 0):
            raise ValueError('Delta overlap check failed locked tolerance')
        features = torch.zeros(n, 1540, dtype=torch.float32)
        features[:, :768] = full['content']
        features[indices, 768:1536] = middle.float()
        features[indices, 1536:] = prosody
        center_indices = start+oldvalid[:count].nonzero(as_tuple=True)[0]
        features[center_indices] = oldfeatures[:count][oldvalid[:count]]
        full.update(audio_features=features, valid=valid, motion_valid=valid.clone(),
                    times=times, channel_mask=channel_mask)
        for key in CLIP_KEYS:
            if key in self.q and key != 'channel_mask':
                full[key] = _row(self.q[key], i, self.count)
        full.update(native_lengths=torch.tensor(n), full_lengths=torch.tensor(n),
                    center_starts=torch.tensor(start), center_lengths=torch.tensor(96),
                    metadata={'artifact': str(path), 'artifact_sha256': row['artifact_sha256'],
                              'source_split': 'train', 'query_role': self.role, 'provenance': provenance,
                              'crop_start': start, 'native_frames': n,
                              'delta_sha256': entry['sha256'],
                              'native_motion_precision': str(arrays['motion'].dtype),
                              'native_content_precision': str(arrays['content'].dtype),
                              'compute_precision': 'float32'})
        self._cache[i] = full
        return dict(full)

    def clip(self, index):
        return {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in self._load(index).items()}

    def __getitem__(self, index):
        return self.clip(index)

    def batch(self, ids, mode='full', device='cpu'):
        indices = [self._index(i) for i in ids]
        if not indices or mode not in ('center', 'full'):
            raise ValueError('Nonempty batch and center/full mode required')
        if mode == 'center':
            # Validate each selected source even when only returning old values.
            for i in indices:
                self._load(i)
            result = _stack([{key: _row(value, i, self.count) for key, value in self.q.items()}
                             for i in indices])
            result.update(native_lengths=self.native_lengths[indices], full_lengths=self.native_lengths[indices],
                          center_starts=self.center_starts[indices], center_lengths=self.center_lengths[indices])
        else:
            rows = [self._load(i) for i in indices]
            length = ((max(int(row['native_lengths']) for row in rows)+15)//16)*16
            for row in rows:
                n = int(row['native_lengths'])
                for key in FRAME_KEYS:
                    value = row[key]
                    if key == 'times':
                        pad = value.new_empty(length)
                        pad[:n] = value
                        pad[n:] = value[-1]+torch.arange(1, length-n+1, dtype=value.dtype)/25
                    else:
                        pad = value.new_zeros((length, *value.shape[1:]))
                        pad[:n] = value
                    row[key] = pad
            result = _stack(rows)
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in result.items()}


__all__ = ['NativeContextStore', 'tensor_sha']
