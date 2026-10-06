"""Rebuild the incremental comparison from hash-verified sealed reports.

This is an incomplete progress summary, not the final paper-table gate.
Missing reports have no metrics and imply no claim about live remote jobs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


METHODS = (
    ('faceformer', 'FaceFormer (ARKit, shared audio)'),
    ('facediffuser', 'FaceDiffuser (ARKit, frozen KineTalk conditions)'),
    ('voca_core', 'VOCA-core (ARKit, shared audio)'),
    ('emotalk_core', 'EmoTalk-core (ARKit, shared audio)'),
    ('ours', 'KineTalk'),
)
METRICS = ('MBE', 'LBE', 'FDD_abs', 'LVE_mm', 'EVE_mm', 'FDD_mm2', 'emotion_macro_f1')
COLUMNS = ('method', 'status', 'clips', *METRICS, 'test_loaded')


def build(root: Path):
    rows = []
    provenance = []
    for directory, label in METHODS:
        report_path = root / directory / 'report.json'
        complete_path = root / directory / 'complete.json'
        row = dict(method=label, status='awaiting_verified_report', test_loaded=False)
        if report_path.is_file() and complete_path.is_file():
            raw = report_path.read_bytes()
            report = json.loads(raw)
            complete = json.loads(complete_path.read_text(encoding='utf8'))
            digest = hashlib.sha256(raw).hexdigest()
            if complete.get('report_sha256') != digest:
                raise ValueError(f'Report hash mismatch: {report_path}')
            if complete.get('status') != 'complete' or complete.get('test_loaded') is not True or report.get('test_loaded') is not True:
                raise ValueError(f'Incomplete/non-test report: {report_path}')
            if complete.get('clips') != 1622 or report.get('clips') != 1622:
                raise ValueError(f'Unexpected clip count: {report_path}')
            if directory in ('voca_core', 'emotalk_core'):
                if report.get('baseline_adaptation', {}).get('version') != 'core_arkit_v1':
                    raise ValueError(f'Legacy simplified baseline forbidden: {report_path}')
            summary, vertex = report['summary'], report['vertex']
            row.update(status='test_complete', clips=report['clips'], test_loaded=True,
                       MBE=summary['arkit_mbe']['value'], LBE=summary['arkit_lbe']['value'],
                       FDD_abs=summary['arkit_fdd_absolute']['value'],
                       LVE_mm=vertex['vertex_lve_sqrt_mean'],
                       EVE_mm=vertex['eye_forehead_eve_sqrt_mean'],
                       FDD_mm2=vertex['vertex_fdd_absolute_mm2_mean'],
                       emotion_macro_f1=report['emotion_macro_f1'])
            if not all(math.isfinite(float(row[key])) for key in METRICS):
                raise ValueError(f'Nonfinite metric: {report_path}')
            provenance.append(dict(method=label, report=str(report_path), report_sha256=digest))
        rows.append(row)

    # Always serialize with csv, since disclosed method names contain commas.
    path = root / 'adapted_baselines.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        decoded = list(csv.DictReader(handle))
    if len(decoded) != len(rows) or any(a['method'] != b['method'] or None in a for a, b in zip(decoded, rows)):
        raise ValueError('CSV round-trip failed')

    lines = ['# Adapted baseline metrics', '',
             'Incremental summary from hash-verified sealed reports. Baselines are declared ARKit adaptations; none claims official end-to-end reproduction. Legacy simplified VOCA/EmoTalk results are excluded.', '',
             'Missing metrics mean no verified local report yet. They do not establish the current state of a remote training process. Final Table 0/1/2/3 still require all methods and ablations.', '',
             '| Method | Status | MBE | LBE | Coeff FDD | LVE mm | EVE mm | Vertex FDD mm² | Emotion F1 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        numbers = [f'{row[key]:.6f}' if key in row else '—' for key in METRICS]
        lines.append('| ' + ' | '.join([row['method'], row['status'], *numbers]) + ' |')
    lines += ['', 'Completed rows each cover 1,622 test clips. Metric values come directly from unchanged report.json files, validated against complete.json hashes. Emotion F1 uses the independent real-motion probe. Vertex values use the common fixed ARKit rig.', '']
    (root / 'adapted_baselines.md').write_text('\n'.join(lines), encoding='utf8')
    (root / 'adapted_baselines_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf8')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('final_experiment/evaluation/sealed_20260923'))
    args = parser.parse_args()
    result = build(args.root)
    print(json.dumps({'verified_reports': sum(r['status'] == 'test_complete' for r in result), 'rows': len(result)}))
