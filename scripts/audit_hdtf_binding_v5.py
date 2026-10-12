"""Audit all bound TRAIN source arrays and independent reference metadata."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v2.hdtf_dataset import HDTFSplit


def main(a):
    out = Path(a.output)
    if out.exists(): raise FileExistsError('Fresh audit output required')
    out.mkdir(parents=True)
    dataset = HDTFSplit(a.manifest, a.features, split='train', source_root=a.source_root,
                       verify_hashes=True, cache_sources=2, motion_policy='raw')
    representatives = {}
    for i, row in enumerate(dataset.rows): representatives.setdefault(row['source_id'], i)
    rows = []; accepted_windows = []; dropped_windows = []; begin = time.monotonic()
    for index, (sid, i) in enumerate(sorted(representatives.items())):
        source = dataset._source(dataset.rows[i])
        valid = source['motion_valid'] & source['feature_valid']
        support = source['channel_mask']
        if not np.isfinite(source['audio_features'][valid]).all(): raise ValueError('Nonfinite audio: ' + sid)
        if not np.isfinite(source['motion'][valid][:, support]).all(): raise ValueError('Nonfinite observed motion: ' + sid)
        if not np.allclose(np.diff(source['times']), .04, atol=1e-7, rtol=1e-4):
            raise ValueError('Non-native source clock: ' + sid)
        if not valid.any(): raise ValueError('No usable observations: ' + sid)
        # A partly failed long recording may still contain high-quality windows.
        # Apply the threshold to actual query windows, not its entire duration.
        for window in dataset.rows:
            if window['source_id'] != sid: continue
            fraction = float(valid[int(window['start_frame']):int(window['end_frame'])].mean())
            if fraction >= .94:
                copied = dict(window, combined_motion_feature_valid_ratio=fraction)
                accepted_windows.append(copied)
            else:
                dropped_windows.append(dict(clip_id=window['clip_id'], combined_valid_ratio=fraction))
        rows.append(dict(source_id=sid, frames=len(valid), valid_ratio=float(valid.mean()),
                         observed_channels=int(support.sum())))
        if index % 16 == 0:
            state = dict(status='verifying', sources=index + 1, total=len(representatives),
                         elapsed_seconds=time.monotonic() - begin)
            (out / 'state.json').write_text(json.dumps(state))
            print(json.dumps(state), flush=True)
    verified = out / 'training_windows_verified.jsonl'
    all_metadata = [json.loads(line) for line in Path(a.manifest).read_text().splitlines() if line.strip()]
    # Validation metadata is preserved without opening its target arrays.
    verified_rows = accepted_windows + [r for r in all_metadata if r['split'] == 'val']
    verified.write_text(''.join(json.dumps(r) + '\n' for r in verified_rows))
    dataset = HDTFSplit(verified, a.features, split='train', source_root=a.source_root,
                       verify_hashes=True, cache_sources=2, motion_policy='raw')
    eligible = dataset.eligible_reference_indices()
    by_source = Counter(dataset['source_id'][i] for i in eligible)
    missing = set(representatives) - set(by_source)
    # Quality rejection can leave a source with too little independent support.
    # Exclude these windows visibly instead of turning a data filter into a crash.
    verified_rows = [dataset.rows[i] for i in eligible] + [r for r in all_metadata if r['split'] == 'val']
    verified.write_text(''.join(json.dumps(r) + '\n' for r in verified_rows))
    dataset = HDTFSplit(verified, a.features, split='train', source_root=a.source_root,
                       verify_hashes=True, cache_sources=2, motion_policy='raw')
    eligible = dataset.eligible_reference_indices()
    if len(eligible) != len(dataset): raise ValueError('Filtered references are not closed')
    # Validate one independently selected reference for each available source.
    for i in eligible:
        candidates = dataset.reference_candidates(i)
        if not candidates: raise ValueError('No independent reference')
        query, ref = dataset.rows[i], dataset.rows[candidates[0]]
        if query['clip_id'] == ref['clip_id']: raise ValueError('Reference identity leak')
        if query['source_group'] == ref['source_group']:
            if not (ref['end_seconds'] + .5 <= query['start_seconds'] or
                    query['end_seconds'] + .5 <= ref['start_seconds']):
                raise ValueError('Reference source-time leak')
    report = dict(status='complete', schema='hdtf_train_binding_audit_v5', provenance=dataset.provenance,
        sources=len(set(dataset['source_id'])), verified_available_sources=len(rows),
        excluded_sources_without_references=sorted(missing),
        people=len(set(dataset['speaker_group'])), windows=len(dataset),
        eligible_windows=len(eligible), frames=sum(r['frames'] for r in rows),
        eligible_indices=eligible, source_rows=rows, dropped_windows=dropped_windows,
        combined_valid_threshold=.94, verified_manifest=str(verified),
        elapsed_seconds=time.monotonic() - begin,
        test_loaded=False, emotion_targets=0, intensity_targets=0,
        hashes_verified=True, exact_feature_motion_clock=True,
        perceptual_sync_verified=False, candidate_limit=None)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    (out / 'complete.json').write_text(json.dumps({k: report[k] for k in
        ('status', 'sources', 'people', 'windows', 'eligible_windows', 'frames', 'test_loaded', 'elapsed_seconds')}))
    print((out / 'complete.json').read_text(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True); p.add_argument('--features', required=True)
    p.add_argument('--source-root', required=True); p.add_argument('--output', required=True)
    main(p.parse_args())
