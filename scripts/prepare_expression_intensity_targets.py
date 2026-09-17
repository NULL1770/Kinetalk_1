"""Prepare expression-intensity supervision from explicitly allowed references.

Only the supplied renderer train/validation cache and native-TRAIN enrollment
manifest are read. Anchors are raw per-reference observed means followed by
the per-channel median across references (midpoint for an even count). Targets
are training/evaluation supervision, never ground-truth inference conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.label_guided_intensity import (
    BROWS, EYES, fit_intensity_scales, regional_intensity, regional_window_activity,
)

SCHEMA = 'expression_intensity_targets_v1'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _query_metadata(cache):
    if cache.get('schema') != 'predictable_renderer_cache_v1' or set(cache.get('splits', {})) != {'train', 'validation'}:
        raise ValueError('Require renderer cache containing only train and validation splits')
    mapping, all_clips, all_sentences = {}, set(), set()
    for name in ('train', 'validation'):
        q = cache['splits'][name]['q']
        motion, valid, cm = q['motion'], q['valid'], q['channel_mask']
        if (motion.ndim != 3 or motion.shape[-1] != 52 or not motion.is_floating_point()
                or valid.shape != motion.shape[:2] or valid.dtype != torch.bool
                or cm.shape != (len(motion), 52) or cm.dtype != torch.bool):
            raise ValueError('Invalid renderer native motion/mask shapes')
        for key in ('clip_id', 'speaker', 'speaker_id', 'sentence_id'):
            if key not in q or len(q[key]) != len(motion):
                raise ValueError('Missing ordered query metadata: ' + key)
        mask = valid[..., None] & cm[:, None]
        if not torch.isfinite(motion[mask]).all():
            raise ValueError('Nonfinite observed query motion')
        for clip, speaker, sid, sentence in zip(q['clip_id'], q['speaker'], q['speaker_id'], q['sentence_id']):
            sid = int(sid)
            if not isinstance(clip, str) or not clip or not isinstance(sentence, str) or not sentence:
                raise ValueError('Empty query clip/sentence')
            if clip in all_clips:
                raise ValueError('Repeated query clip across cache')
            if speaker in mapping and mapping[speaker] != sid:
                raise ValueError('Query speaker ID mapping differs')
            if sid in mapping.values() and speaker not in mapping:
                raise ValueError('Speaker ID belongs to multiple names')
            mapping[speaker] = sid; all_clips.add(clip); all_sentences.add(sentence)
    return mapping, all_clips, all_sentences


def _reference_mean(row, native_root):
    root = Path(native_root).resolve(); path = (root / row['artifact']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Enrollment artifact leaves the explicitly allowed native root')
    digest = sha(path)
    if not isinstance(row.get('artifact_sha256'), str) or row['artifact_sha256'] != digest:
        raise ValueError('Enrollment native artifact hash differs')
    with np.load(path, allow_pickle=False) as source:
        if not {'motion', 'mask', 'channel_mask', 'provenance'}.issubset(source.files):
            raise ValueError('Incomplete native reference artifact')
        x = np.asarray(source['motion'], dtype=np.float64)
        valid = np.asarray(source['mask']); channels = np.asarray(source['channel_mask'])
        provenance = json.loads(str(source['provenance'].item()))
    if (not str(provenance.get('schema', '')).startswith('native_affect_style_v4')
            or provenance.get('clock_evidence') != 'embedded_video'):
        raise ValueError('Native reference schema/clock differs')
    if (x.ndim != 2 or x.shape[1] != 52 or len(x) < 1
            or valid.shape != (len(x),) or channels.shape != (52,)
            or not np.isin(valid, [0, 1]).all() or not np.isin(channels, [0, 1]).all()):
        raise ValueError('Invalid native reference motion/mask shapes')
    observed = valid.astype(bool)[:, None] & channels.astype(bool)[None]
    if not np.isfinite(x[observed]).all():
        raise ValueError('Nonfinite observed reference motion')
    count = observed.sum(0)
    mean = np.where(observed, x, 0).sum(0) / np.maximum(count, 1)
    mean[count == 0] = np.nan
    return mean, {'clip_id': row['clip_id'], 'speaker': row['speaker'],
        'speaker_id': int(row['speaker_id']), 'sentence': row['sentence'], 'split': row['split'],
        'artifact': str(path), 'artifact_sha256': digest, 'observed_frames_per_channel': count.tolist()}


def prepare_targets(cache_path, enrollment_path, native_root, output, *, min_references=2, scale_floor=.02):
    cache_path, enrollment_path, native_root, output = map(Path, (cache_path, enrollment_path, native_root, output))
    if output.exists():
        raise FileExistsError('Fresh output directory required')
    if min_references < 2:
        raise ValueError('At least two independent neutral references are required')
    sources = {'cache': sha(cache_path), 'enrollment': sha(enrollment_path), 'preparation_script': sha(__file__),
        'intensity_module': sha(Path(__file__).resolve().parents[1]/'kinetalk_b0/label_guided_intensity.py')}
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    mapping, query_clips, query_sentences = _query_metadata(cache)
    rows = [json.loads(line) for line in enrollment_path.read_text(encoding='utf8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty enrollment manifest')
    refs = {speaker: [] for speaker in mapping}; reference_clips = set(); reference_sentences = {s: set() for s in mapping}
    # Validate every manifest row before opening any referenced artifact.
    for row in rows:
        if row.get('split') != 'train' or row.get('emotion_id', row.get('emotion')) != 0:
            raise ValueError('Only explicitly supplied native-TRAIN neutral enrollment is allowed')
        if 'emotion' in row and row['emotion'] not in (0, 'neutral'):
            raise ValueError('Enrollment emotion fields disagree')
        speaker = row.get('speaker')
        if speaker not in mapping or int(row.get('speaker_id', -1)) != mapping[speaker]:
            raise ValueError('Enrollment speaker ID mapping differs or speaker is not a query identity')
        clip, sentence = row.get('clip_id'), row.get('sentence')
        if not isinstance(clip, str) or not clip or not isinstance(sentence, str) or not sentence:
            raise ValueError('Missing enrollment clip/sentence metadata')
        if clip in query_clips or sentence in query_sentences:
            raise ValueError('Reference and query clip/sentence are not disjoint')
        if clip in reference_clips or sentence in reference_sentences[speaker]:
            raise ValueError('Repeated enrollment clip or per-person sentence')
        reference_clips.add(clip); reference_sentences[speaker].add(sentence); refs[speaker].append(row)
    by_speaker, reference_bindings = {}, []
    for speaker, reference_rows in refs.items():
        if len(reference_rows) < min_references:
            raise ValueError('Too few independent neutral references for ' + speaker)
        means = []
        for row in reference_rows:
            mean, binding = _reference_mean(row, native_root)
            means.append(mean); reference_bindings.append(binding)
        means = np.stack(means)
        count = np.isfinite(means).sum(0); anchor = np.full(52, np.nan)
        for channel in range(52):
            if count[channel] >= min_references:
                anchor[channel] = np.median(means[np.isfinite(means[:, channel]), channel])
        if any(not np.isfinite(anchor[list(channels)]).any() for channels in (BROWS, EYES)):
            raise ValueError('References do not cover both expression groups for ' + speaker)
        by_speaker[speaker] = {'speaker_id': mapping[speaker], 'anchor': torch.from_numpy(anchor).float(),
            'anchor_valid': torch.from_numpy(np.isfinite(anchor)), 'reference_count_per_channel': torch.from_numpy(count),
            'reference_clip_ids': [row['clip_id'] for row in reference_rows]}
    splits = {}
    for name in ('train', 'validation'):
        q = cache['splits'][name]['q']
        anchors = torch.stack([by_speaker[s]['anchor'] for s in q['speaker']])
        anchor_valid = torch.stack([by_speaker[s]['anchor_valid'] for s in q['speaker']])
        observed = q['valid'][..., None] & q['channel_mask'][:, None] & anchor_valid[:, None]
        splits[name] = {'anchors': anchors, 'anchor_valid': anchor_valid, 'observed': observed,
            'clip_id': list(q['clip_id']), 'speaker': list(q['speaker']), 'speaker_id': q['speaker_id'].clone()}
    train = splits['train']
    scales = fit_intensity_scales(cache['splits']['train']['q']['motion'], train['observed'], train['anchors'], floor=scale_floor)
    for name, split in splits.items():
        motion = cache['splits'][name]['q']['motion']
        split['intensity'], split['valid'] = regional_intensity(motion, split['observed'], split['anchors'], scales)
        split['activity'], split['activity_valid'] = regional_window_activity(motion, split['observed'], split['anchors'], scales)
        del split['observed']
    provenance = {'schema': SCHEMA, 'renderer_cache_sha256': sources['cache'], 'sources': {'cache': str(cache_path.resolve()),
        'enrollment': str(enrollment_path.resolve()), 'native_root': str(native_root.resolve())},
        'source_sha256': sources, 'references': reference_bindings,
        'anchor': 'Full native observed mean per neutral reference, then per-channel median across references; even count uses midpoint.',
        'minimum_references_per_channel': min_references, 'scale': 'Training-only RMS(raw motion minus independent neutral anchor), no clip centering.',
        'scale_floor': scale_floor, 'groups': {'brows': list(BROWS), 'eyes_expression': list(EYES)},
        'group_weight': .5, 'frame_valid': 'At least one observed channel with valid anchor in each group.',
        'reference_query_clip_disjoint': True, 'reference_query_sentence_disjoint': True,
        'native_reference_splits': ['train'], 'fit_target_splits': ['train'], 'query_mean_used_as_anchor': False,
        'emotion_label_used_for_query_intensity': False, 'inference_target_use': 'Forbidden except explicitly labeled oracle diagnostic.',
        'test_loaded': False, 'cache_provenance': cache.get('provenance', {})}
    # Detect source mutation before committing any output.
    if sources['cache'] != sha(cache_path) or sources['enrollment'] != sha(enrollment_path):
        raise ValueError('Preparation source changed while reading')
    for record in reference_bindings:
        if sha(record['artifact']) != record['artifact_sha256']:
            raise ValueError('Reference artifact changed while preparing')
    payload = {'schema': SCHEMA, 'splits': splits, 'scales': scales, 'by_speaker': by_speaker, 'provenance': provenance}
    output.mkdir(parents=True)
    torch.save(payload, output/'targets.pt')
    provenance = {**provenance, 'output_sha256': {'targets.pt': sha(output/'targets.pt')}}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--enrollment', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-references', type=int, default=2)
    parser.add_argument('--scale-floor', type=float, default=.02)
    args = parser.parse_args()
    result = prepare_targets(args.cache, args.enrollment, args.native_root, args.output,
        min_references=args.min_references, scale_floor=args.scale_floor)
    print(json.dumps({'output': str(args.output.resolve()), 'splits': {name: {
        'clips': len(split['clip_id']), 'valid_frames': int(split['valid'].sum())} for name, split in result['splits'].items()}}))


if __name__ == '__main__':
    main()
