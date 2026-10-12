"""Bind quality-filtered TRAIN/val windows to available remote artifacts.

Only metadata is read for excluded test/quarantine rows. Source raw targets and
features are opened later by the verified loader, never by this manifest step.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    root = Path(a.source_root)
    receipts = {r['source_id']: r for r in json.loads(Path(a.receipts).read_text(encoding='utf8'))}
    output = Path(a.output)
    if output.exists(): raise FileExistsError('Fresh HDTF manifest required')
    output.mkdir(parents=True)
    rows, rejected = [], Counter()
    source_records = {}
    source_roles, person_roles = {}, {}
    for line in (root / 'training_windows.jsonl').read_text(encoding='utf8').splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r['split'] not in ('train', 'val'):
            rejected['excluded_sealed_or_quarantine_metadata'] += 1
            continue
        sid = r['source_id']
        if sid not in receipts:
            rejected['no_available_feature_receipt'] += 1
            continue
        receipt = receipts[sid]
        if r['split'] != receipt['split'] or r['speaker_group'] != receipt['speaker_group']:
            raise ValueError('Window/receipt split or identity mismatch: ' + sid)
        if sid not in source_records:
            record = json.loads((root / 'sources' / (sid + '.json')).read_text(encoding='utf8'))
            if record['split'] not in ('train', 'val'):
                raise ValueError('Unexpected sealed source record')
            for key in ('motion_sha256', 'audio_sha256'):
                if record[key] != receipt[key]: raise ValueError('Remote/local raw lineage mismatch: ' + sid)
            if record['clock_status'] != 'duration_consistent_sync_unverified':
                raise ValueError('Source duration mismatch or unknown clock policy: ' + sid)
            if abs(float(record['audio_motion_duration_error'])) > .04:
                raise ValueError('Audio/motion duration mismatch: ' + sid)
            source_records[sid] = record
        if float(r['quality_ratio']) < a.min_quality:
            rejected['low_window_quality'] += 1
            continue
        if int(r['end_frame']) > int(receipt['n_frames']):
            rejected['outside_available_source'] += 1
            continue
        for mapping, group in ((source_roles, r['source_group']), (person_roles, r['speaker_group'])):
            mapping.setdefault(group, set()).add(r['split'])
            if len(mapping[group]) != 1: raise ValueError('Source/person split leak')
        r.update(motion_path=receipt['motion_path'], audio_path=receipt['audio_path'],
                 motion_sha256=receipt['motion_sha256'], audio_sha256=receipt['audio_sha256'],
                 features_sha256=receipt['features_sha256'],
                 emotion_id=-1, emotion_valid=False, intensity_id=-1, intensity_valid=False,
                 neutral_target_valid=False)
        rows.append(r)
    if not rows: raise ValueError('No bound windows')
    manifest = output / 'training_windows.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf8')
    summary = dict(schema='hdtf_quality_binding_v5', source_root=str(root.resolve()),
        available_receipts_sha256=sha(a.receipts), local_manifest_sha256=sha(root / 'training_windows.jsonl'),
        bound_manifest_sha256=sha(manifest), min_window_quality=a.min_quality,
        motion_policy='raw', split_windows=dict(Counter(r['split'] for r in rows)),
        split_sources={s: len({r['source_id'] for r in rows if r['split'] == s}) for s in ('train', 'val')},
        split_people={s: len({r['speaker_group'] for r in rows if r['split'] == s}) for s in ('train', 'val')},
        rejected_metadata_counts=dict(rejected), source_disjoint=True, person_disjoint=True,
        test_target_loaded=False, emotion_targets=0, intensity_targets=0, neutral_targets=0,
        reference_policy='prefer other same-person recording, otherwise source-time-disjoint with >=0.5s gap',
        event_policy='parse surrounding source context before cropping; boundary events censored',
        clock_policy='source timestamps and durations consistent; perceptual audio/lip sync remains unverified')
    (output / 'binding.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', required=True); p.add_argument('--receipts', required=True)
    p.add_argument('--output', required=True); p.add_argument('--min-quality', type=float, default=.95)
    main(p.parse_args())
