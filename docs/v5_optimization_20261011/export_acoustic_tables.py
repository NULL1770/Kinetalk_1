"""Build matched acoustic-receiver tables from immutable development reports."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent
FILES = {
    'phase53': 'posture_control_full_report.json',  # Phase53 values are embedded in this same-scope report.
    'control': 'posture_control_full_report.json',
    'coarse': 'geometry_coarse_full_report.json',
    'hubert': 'geometry_acoustic_full_report.json',
}
GEOMETRY = {
    'control': 'geometry_posture_control.json',
    'coarse': 'geometry_coarse_geometry_report.json',
    'hubert': 'geometry_acoustic_geometry_report.json',
}
DYNAMICS = {
    'control': 'posture_control_dynamics.json',
    'hubert': 'geometry_acoustic_dynamics.json',
}
CONTROLS = {
    'control': 'posture_control_factor_controls.json',
    'hubert': 'geometry_acoustic_controls_report.json',
}
METRICS = {
    'lip_mean_distance_mm': 'lve_mean_mm_mean',
    'max_vertex_LVE_mm': 'vertex_lve_sqrt_mean',
    'upper_face_EVE_mm': 'eye_forehead_eve_sqrt_mean',
    'jaw_closure_proxy_F1': 'jaw_closure_frame_f1',
    'MBE': 'arkit_mbe',
    'LBE': 'arkit_lbe',
}


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (ROOT/name).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    reports = {arm: read(name) for arm, name in FILES.items()}
    hubert = reports['hubert']
    if hubert['clips'] != 1367 or hubert['test_loaded']:
        raise ValueError('Native receiver report is incomplete or includes test')
    common_indices = hubert['indices']
    common_baseline = hubert['baseline_sha256']
    rows = []
    for arm, report in reports.items():
        if report['indices'] != common_indices or report['baseline_sha256'] != common_baseline or report['test_loaded']:
            raise ValueError('Full-development clip or Phase53 binding differs')
        if arm == 'phase53':
            value = report['results']['phase53_default/clip_all']
            checkpoint = '80a7ca69b0f868bad6361381e1ca8e4e168ca227a0548221f8efa71b42169c41'
        else:
            value = report['results']['v4_sample/clip_all']
            checkpoint = report['checkpoint_sha256']
        probe = value['probes'][0]
        rows.append(dict(arm=arm, clips=report['clips'], checkpoint_sha256=checkpoint,
            **{name: value['metrics'].get(key) for name, key in METRICS.items()},
            emotion_probe_F1=probe['macro_f1'], angry_probe_F1=probe['per_class']['angry']['f1']))
    write_csv('acoustic_full_development_comparison.csv', rows)

    geom_rows = []
    geometry_reports = {arm: read(name) for arm, name in GEOMETRY.items()}
    first = geometry_reports['hubert']
    for arm, report in geometry_reports.items():
        if report['indices'] != first['indices'] or report['binding_sha256'] != first['binding_sha256'] or report['test_loaded']:
            raise ValueError('Geometry subset/binding differs')
        for emotion in ('ALL', 'angry', 'happy'):
            for stage, values in report['summary'][emotion].items():
                for axis, metrics in values.items():
                    geom_rows.append(dict(arm=arm, clips=report['clips'], emotion=emotion, stage=stage,
                        coordinate=axis, MAE_mm=metrics['MAE'], bias_mm=metrics['bias'],
                        centered_MSE=metrics['centered_MSE'], supported_clips=metrics['supported_clips']))
    write_csv('acoustic_geometry_comparison.csv', geom_rows)

    dynamic_rows = []
    dynamic_reports = {arm: read(name) for arm, name in DYNAMICS.items()}
    for arm, report in dynamic_reports.items():
        for mode, result in report['results'].items():
            direction = result.get('expression_direction')
            events = result.get('realized_event_distributions', {})
            for event, value in events.items():
                dynamic_rows.append(dict(arm=arm, mode=mode, event=event, clips=report['clips'],
                    sample_seeds='47,91,137' if mode.startswith('v4_sample') else '',
                    direction_mean_axis_MAE=direction['mean_axis_mae'] if direction else None,
                    direction_cosine_error=direction['direction_cosine_error'] if direction else None,
                    event_rate_per_minute=value['complete_events_per_minute'],
                    event_rate_abs_error=value.get('complete_events_per_minute_abs_error'),
                    duration_mean_seconds=value.get('duration_seconds_mean'),
                    amplitude_mean=value.get('amplitude_mean'),
                    duration_W1=value.get('duration_seconds_wasserstein1'),
                    amplitude_W1=value.get('amplitude_wasserstein1')))
    write_csv('acoustic_dynamic_comparison.csv', dynamic_rows)

    control_rows = []
    for arm, name in CONTROLS.items():
        report = read(name)
        for condition in ('same_person_B', 'donor', 'same_person_B_fixed_schedule', 'donor_fixed_schedule'):
            summary = report['summary'][condition]
            for metric, value in summary['content_proxies'].items():
                if isinstance(value, dict) and 'mean' in value:
                    control_rows.append(dict(arm=arm, condition=condition, metric=metric,
                        mean=value['mean'], median=value['median'], clips=value['n']))
    write_csv('acoustic_reference_control_comparison.csv', control_rows)

    sources = sorted(set(FILES.values()) | set(GEOMETRY.values()) | set(DYNAMICS.values()) | set(CONTROLS.values()))
    provenance = dict(schema='acoustic_receiver_evidence_v1', sources={name: sha(ROOT/name) for name in sources},
        full_indices_sha256=hashlib.sha256(json.dumps(common_indices).encode()).hexdigest(),
        geometry_indices_sha256=hashlib.sha256(json.dumps(first['indices']).encode()).hexdigest(),
        table_rows={name: sum(1 for _ in (ROOT/name).open(encoding='utf-8-sig'))-1 for name in
            ('acoustic_full_development_comparison.csv','acoustic_geometry_comparison.csv',
             'acoustic_dynamic_comparison.csv','acoustic_reference_control_comparison.csv')},
        failed_optional_arm='geometry_acoustic_mean', test_loaded=False, promotion=False,
        limitations=['One training continuation seed; sampling seeds are not training seeds.',
            'Motion probes are whole-clip statistics, not framewise emotion truth.',
            'Jaw closure threshold and shared-rig geometry are proxies, not phoneme/contact ground truth.',
            'Style controls measure sensitivity/directional proxies, not perceived style accuracy.'])
    (ROOT/'acoustic_tables_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf8')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
