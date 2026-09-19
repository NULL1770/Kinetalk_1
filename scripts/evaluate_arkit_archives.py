"""Reevaluate completed generation archives without changing historical gates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.arkit_benchmark_report import build_report, score_fullface, write_report
from scripts.evaluate_paper_coefficients import ARMS, cluster_interval, sha, write_csv
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES


def run(root, reference, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    refs = torch.load(reference, map_location='cpu', weights_only=False)['clips']
    ids = sorted(refs)
    anchor = root / 'outer/audio_generation/holdout'
    reports, summary_rows, per_clip_rows = {}, [], []
    for arm, stage in ARMS.items():
        directory = anchor if stage is None else root / 'outer' / stage / 'holdout'
        old = json.loads((directory / 'result.json').read_text(encoding='utf8'))
        if {r['clip_id'] for r in old['per_clip_scores']} != set(ids):
            raise ValueError('Different evaluation membership: ' + arm)
        rows, provenance = [], {}
        for cid in ids:
            path = directory / 'npz' / (cid + '.npz')
            with np.load(path, allow_pickle=False) as z:
                clip = {**refs[cid], 'clip_id': cid}
                np.testing.assert_array_equal(z['channels'], ARKIT_NAMES)
                np.testing.assert_array_equal(z['times'], clip['times'])
                np.testing.assert_array_equal(z['native_valid'], clip['valid'])
                np.testing.assert_array_equal(z['motions'][1], clip['baseline52'])
                samples = z['motions'][1:2] if stage is None else z['motions'][2:]
                if stage is not None and len(samples) != 4:
                    raise ValueError('Expected all four archived draws')
                row = score_fullface(samples, clip)
            rows.append(row)
            provenance[cid] = sha(path)
            per_clip_rows.append({'arm': arm, 'clip_id': cid, 'sentence': row['sentence'],
                                  **{k: v['value'] for k, v in row['metrics'].items()}})
        report = build_report(rows, scope='64 historically exposed development clips; not sealed test',
                              sources={'reference_sha256': sha(reference), 'npz_sha256': provenance,
                                       'historical_result_sha256': sha(directory / 'result.json')})
        for name, value in report['summary'].items():
            if value['status'] in ('computed', 'partial'):
                value['sentence_cluster_interval'] = cluster_interval(
                    [r['metrics'][name]['value'] for r in rows], [r['sentence'] for r in rows])
        reports[arm] = report
        write_report(output / (arm + '.json'), report)
        summary_rows.append({'arm': arm, **{k: v['value'] for k, v in report['summary'].items()}})
    write_csv(output / 'main_metrics.csv', summary_rows)
    write_csv(output / 'per_clip.csv', per_clip_rows)
    write_report(output / 'all_arms.json', reports)
    headers = ['arm', 'arkit_mbe', 'arkit_lbe', 'arkit_fdd_signed', 'arkit_fdd_absolute', 'supp_upper9_fdd_absolute']
    lines = ['# ARKit coefficient literature metrics — development evaluation', '',
             'Same 64 exposed development clips; four fixed draws (base deterministic). No external baseline scores.', '',
             '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in summary_rows:
        lines.append('| ' + ' | '.join(str(row[k]) if k == 'arm' else f'{row[k]:.6f}' for k in headers) + ' |')
    lines += ['', 'FDD is signed GT minus prediction energy-std; zero is the reference. Absolute FDD is lower-is-better.',
              'Official main FDD region contains eyes/nose, not brows. Upper9 is a declared supplemental adaptation.',
              'MBE omits unobserved coefficients; missing tongue is not a zero-valued target.',
              'AV offset/confidence, Multimodality, FD and WInD remain pending their common render/SyncNet or trained/frozen evaluation encoder.',
              'These statistics do not establish audio timing or permit ranking against original-paper tables.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'output': str(output.resolve()), 'arms': len(reports), 'clips': len(ids)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.reference, a.output)
