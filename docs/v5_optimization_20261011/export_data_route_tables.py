"""Export equal-scope data-route tables without overwriting earlier evidence."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent
KEYS = dict(MBE='arkit_mbe', LBE='arkit_lbe', lip_mean_distance_mm='lve_mean_mm_mean',
            EVE_mm='eye_forehead_eve_sqrt_mean', closure_proxy_F1='jaw_closure_frame_f1',
            jaw_correlation='jawOpen/centered_correlation', brow_bias_MSE='brows/mean_bias_mse',
            brow_centered_MSE='brows/centered_mse')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(scope, inputs):
    missing = [name for name in inputs.values() if not (ROOT / name).exists()]
    if missing:
        print(json.dumps(dict(scope=scope, status='pending', missing=missing)))
        return
    reports = {arm:json.loads((ROOT/name).read_text(encoding='utf8')) for arm,name in inputs.items()}
    initial = next(iter(reports.values()))
    if any(d['indices'] != initial['indices'] or d['baseline_sha256'] != initial['baseline_sha256']
           or d['test_loaded'] for d in reports.values()):
        raise ValueError('Different query selection/baseline or unexpected test access')
    rows = []
    for arm, report in [('Phase53_default', initial), ('Phase53_reference_A', initial)] + list(reports.items()):
        prefix = 'phase53_default' if arm == 'Phase53_default' else 'phase53_reference_A' if arm == 'Phase53_reference_A' else None
        for mode in ([prefix] if prefix else ['v4_sample', 'v4_modal']):
            for policy in ('raw', 'clip_all'):
                r = report['results'][mode + '/' + policy]
                probe = r['probes'][0]
                row = dict(arm=arm, mode=mode, policy=policy, clips=report['clips'],
                           **{k:r['metrics'][v] for k,v in KEYS.items()},
                           emotion_probe_F1=probe['macro_f1'], angry_probe_F1=probe['per_class']['angry']['f1'])
                rows.append(row)
    target = ROOT / ('data_route_comparison_' + scope + '.csv')
    with target.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    source = {arm:dict(file=name, report_sha256=sha(ROOT/name), checkpoint_sha256=reports[arm]['checkpoint_sha256'],
                      binding_sha256=reports[arm]['training_protocol']['binding_sha256'],
                      source_checkpoint_sha256=reports[arm]['training_protocol']['source_checkpoint_sha256'])
              for arm,name in inputs.items()}
    provenance = dict(schema='v5_data_route_comparison_v1', scope=scope, clips=initial['clips'],
        sources=source, rows=rows, test_loaded=False, promotion=False,
        limits=['Primary reported view must declare sample/modal and raw/clipped policy; every view is retained.',
                'MEAD vs mixed replaces four MEAD examples with HDTF, rather than matching MEAD exposure.',
                'All-factor vs content-style HDTF arms share composition/update budget; shared decoder can still change affect.',
                'Emotion probe is whole-clip motion statistics; lip mean distance is not max-vertex paper LVE.',
                'Development comparisons and sensitivity do not establish novelty, style accuracy or test performance.'])
    (ROOT / ('data_route_comparison_' + scope + '_provenance.json')).write_text(json.dumps(provenance,indent=2),encoding='utf8')
    print(json.dumps(dict(scope=scope,status='complete',clips=initial['clips'],rows=len(rows),file=str(target))))


if __name__ == '__main__':
    export('subset', dict(MEAD='mead_matched_report.json', MEAD_HDTF_all='mixed_matched_report.json',
                         MEAD_HDTF_content_style='route_matched_report.json'))
    export('full', dict(MEAD='mead_full_report.json', MEAD_HDTF_all='mixed_full_report.json',
                       MEAD_HDTF_content_style='route_full_report.json'))
