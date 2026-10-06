"""Bind neutral/native B0 lineage and TRAIN-only approved pair coverage.

No optimizer, forward pass, query motion, or sealed split is read. Tensor
hashes establish B0 ownership independently of audio-only CLI defaults.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state, prefix):
    selected = {k: v for k, v in state.items() if k.startswith(prefix)}
    if not selected:
        raise ValueError('Missing state prefix: ' + prefix)
    digest = hashlib.sha256()
    for key, value in sorted(selected.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return {'sha256': digest.hexdigest(), 'tensors': len(selected)}


def approved_train_rows(rows, train_ids, allowed_reference_ids):
    selected, rejected = [], Counter()
    seen = set()
    for row in rows:
        cid = str(row['source_clip_id'])
        if cid not in train_ids:
            rejected['outside_train_source'] += 1
            continue
        if row.get('teacher_eligible') is not True:
            rejected['teacher_ineligible'] += 1
            continue
        if str(row['reference_clip_id']) not in allowed_reference_ids:
            raise ValueError('Approved TRAIN source references an unapproved split: ' + cid)
        if cid in seen:
            raise ValueError('Duplicate approved TRAIN source: ' + cid)
        if not row.get('teacher_artifact'):
            raise ValueError('Missing teacher artifact: ' + cid)
        if row.get('mouth_event_gate') is not True and not row.get('event_local_mask_path'):
            raise ValueError('Ungated mouth requires local observation mask: ' + cid)
        seen.add(cid)
        selected.append(row)
    return selected, dict(rejected)


def emotion_name(row, names):
    value = row['emotion']
    if isinstance(value, bool):
        raise ValueError('Boolean emotion label')
    if isinstance(value, int) or (isinstance(value, str) and value.isdecimal()):
        index = int(value)
        if not 0 <= index < len(names):
            raise ValueError('Unknown emotion index')
        return names[index]
    if value not in names:
        raise ValueError('Unknown emotion name')
    return value


def run(args):
    if args.output.exists():
        raise FileExistsError('Use a fresh report path')
    sys.path.insert(0, str(args.code_root.resolve()))
    from scripts.prepare_paper_full_data import validate_manifest

    marker_path = args.data / 'packed_cache.json'
    marker = json.loads(marker_path.read_text())
    if marker.get('test_loaded') is not False:
        raise ValueError('Packed marker test boundary')
    meta_path = args.data / marker['metadata']
    meta = torch.load(meta_path, map_location='cpu', weights_only=False)
    if meta.get('test_loaded') is not False:
        raise ValueError('Packed metadata test boundary')
    source_manifest = Path(meta['source']) / 'manifest.json'
    manifest = json.loads(source_manifest.read_text())
    validate_manifest(manifest)
    train = manifest['roles']['train']
    train_rows = train['query']
    names = meta['config']['data']['emotion_classes']
    if names[0] != 'neutral':
        raise ValueError('Unexpected neutral label mapping')
    train_ids = {str(r['clip_id']) for r in train_rows}
    approved_reference_ids = train_ids | {str(r['clip_id']) for r in train['enrollment']}
    rows = [json.loads(s) for s in args.safe_manifest.read_text().splitlines() if s.strip()]
    pairs, rejected = approved_train_rows(rows, train_ids, approved_reference_ids)
    row_by_id = {str(r['clip_id']): r for r in train_rows}
    native_neutral_pairs = sum(emotion_name(row_by_id[r['source_clip_id']], names) == 'neutral' for r in pairs)
    pairs = [r for r in pairs if emotion_name(row_by_id[r['source_clip_id']], names) != 'neutral']
    rejected['native_neutral_anchor_overrides_pair'] = native_neutral_pairs
    state_rows = {}
    states = {}
    for name, root in [('neutral', args.neutral), ('neutral_source', args.neutral_source),
                       ('native', args.native)]:
        p = root / 'audio/final.pt'
        recipe = json.loads((root / 'provenance.json').read_text())['recipe']
        if recipe.get('test_loaded') is not False:
            raise ValueError('Checkpoint recipe test boundary')
        before = sha(p)
        checkpoint = torch.load(p, map_location='cpu', weights_only=False)
        if checkpoint['data_manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError('Checkpoint dataset differs')
        if sha(p) != before:
            raise ValueError('Checkpoint modified during read')
        state_rows[name] = {'path': str(p), 'sha256': before,
            'stage1': state_digest(checkpoint['system'], 'stage1.'),
            'articulation_scope': recipe.get('articulation_scope'),
            'parent_sha256': recipe.get('stage_checkpoint_sha256'),
            'safe_manifest_sha256': recipe.get('safe_dtw_manifest_sha256'),
            'source_sha256': recipe['source_sha256']}
        states[name] = {k: v for k, v in checkpoint['system'].items() if k.startswith('stage1.')}
        del checkpoint
    if state_rows['neutral']['stage1'] != state_rows['neutral_source']['stage1']:
        raise ValueError('Default neutral B0 differs from audited neutral source')
    if state_rows['neutral']['safe_manifest_sha256'] != sha(args.safe_manifest):
        raise ValueError('Neutral supervision manifest binding differs')
    neutral, native = states['neutral'], states['native']
    if set(neutral) != set(native):
        raise ValueError('B0 state keys differ')
    changed = [k for k in neutral if not torch.equal(neutral[k], native[k])]
    neutral_query_count = sum(emotion_name(r, names) == 'neutral' for r in train_rows)
    expected_scope = state_rows['neutral']['articulation_scope']
    if (len(pairs) != expected_scope['safe_target_clip_count']
            or neutral_query_count != expected_scope['neutral_identity_clip_count']):
        raise ValueError('TRAIN supervision count differs from original recipe')
    report = {'schema': 'factor_role_input_audit_v1', 'training_performed': False,
        'optimizer_created': False, 'forward_performed': False, 'test_loaded': False,
        'query_motion_read': False, 'checkpoint_files_unchanged': True,
        'script_sha256': sha(__file__), 'packed_metadata_sha256': sha(meta_path),
        'safe_manifest_sha256': sha(args.safe_manifest),
        'data_manifest_sha256': manifest['manifest_sha256'],
        'checkpoint_inputs': state_rows, 'neutral_B0_matches_original_source': True,
        'native_vs_neutral_changed_B0_tensors': changed,
        'scope': 'TRAIN source and reference membership only; no pair motion loaded; no new output quality claim',
        'coverage': {'train_queries': len(train_rows), 'train_speakers': len({r['speaker'] for r in train_rows}),
            'neutral_queries': neutral_query_count, 'approved_pair_sources': len(pairs),
            'effective_sources': len({r['source_clip_id'] for r in pairs} |
                {r['clip_id'] for r in train_rows if emotion_name(r, names) == 'neutral'}),
            'full_mouth_gate_pairs': sum(r.get('mouth_event_gate') is True for r in pairs),
            'local_mouth_mask_pairs': sum(r.get('mouth_event_gate') is not True for r in pairs),
            'emotion_counts': dict(Counter(emotion_name(row_by_id[r['source_clip_id']], names) for r in pairs)),
            'boundary_status': dict(Counter(str(r.get('boundary_status')) for r in pairs)),
            'viseme_status': dict(Counter(str(r.get('viseme_status')) for r in pairs)),
            'rejected': rejected},
        'pair_contract': 'Approved emotion-to-neutral supervision only; no authorization of emotional/emotional or identity swaps'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code-root', 'data', 'safe-manifest', 'neutral', 'neutral-source', 'native', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({'coverage': result['coverage'],
        'B0_changed_tensors': len(result['native_vs_neutral_changed_B0_tensors'])}))
