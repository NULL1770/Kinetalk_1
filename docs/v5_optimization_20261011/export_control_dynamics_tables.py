"""Source-scoped factor sensitivity and weak dynamics tables; no novelty gate."""
from pathlib import Path
import csv
import hashlib
import json
ROOT = Path(__file__).resolve().parent


def export(filename, rows, sources, limits):
    if not rows:
        return
    with (ROOT / filename).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    provenance = dict(schema='v5_descriptive_control_dynamics_table_v1', sources={
        n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},
        test_loaded=False, promotion=False, rows=rows, limits=limits)
    (ROOT / (Path(filename).stem + '_provenance.json')).write_text(json.dumps(provenance,indent=2),encoding='utf8')
    print(json.dumps(dict(file=filename,rows=len(rows))))


def main():
    control_files = dict(MEAD='mead_factor_controls.json', HDTF_all='mixed_factor_controls.json',
                         HDTF_route='route_factor_controls.json')
    dynamics_files = dict(MEAD='mead_matched_dynamics.json', HDTF_all='mixed_matched_dynamics.json',
                          HDTF_route='route_matched_dynamics.json')
    rows = []
    clips = None
    for arm,name in control_files.items():
        data = json.loads((ROOT/name).read_text(encoding='utf8'))
        assert not data['test_loaded']
        selection = [r['clip_id'] for r in data['records']]
        if clips is None: clips = selection
        assert clips == selection
        for condition in ('same_person_B','donor','same_person_B_fixed_schedule','donor_fixed_schedule'):
            values = data['summary'][condition]['content_proxies']
            rows.append(dict(arm=arm,condition=condition,clips=data['clips'],
                closure_threshold_flip_fraction=values['jaw_closure_proxy_flip_fraction_at_05']['mean'],
                jaw_derivative_correlation=values['jaw_derivative_correlation']['mean'],
                jaw_zero_lag_correlation=values['jaw_zero_lag_correlation']['mean'],
                jaw_range_ratio=values['jaw_range_ratio']['mean'],
                mouth_coefficient_pairwise_mae=values['mouth_coefficient_pairwise_mae']['mean']))
    export('factor_reference_sensitivity.csv', rows, control_files.values(), [
        'jawOpen < .05 is a weak coefficient closure proxy, not phoneme/contact correctness.',
        'Legitimate personal amplitude changes can cross the threshold; changes are diagnostic, not accuracy.',
        'Same-person references are separate inputs, never simultaneous requirements.',
        'Target style transfer accuracy and trained architectural ablations are not established.'])
    direction_rows, event_rows, region_rows = [], [], []
    indices = None
    for arm,name in dynamics_files.items():
        data = json.loads((ROOT/name).read_text(encoding='utf8'))
        assert not data['test_loaded']
        if indices is None: indices = data['selected_indices']
        assert data['selected_indices'] == indices
        for mode,result in data['results'].items():
            if 'expression_direction' in result:
                direction = result['expression_direction']
                direction_rows.append(dict(arm=arm,mode=mode,clips=data['clips'],
                    state_MAE=direction['mean_axis_mae'], direction_cosine_error=direction['direction_cosine_error']))
            for region,values in result['regional_dynamics_macro_clip'].items():
                region_rows.append(dict(arm=arm,mode=mode,region=region,clips=data['clips'],
                    std=values['within_run_population_std'],velocity=values['mean_abs_velocity_per_second'],
                    acceleration=values['mean_abs_acceleration_per_second2']))
            for event,values in result['realized_event_distributions'].items():
                event_rows.append(dict(arm=arm,mode=mode,event=event,clips=data['clips'],
                    rate_per_minute=values['complete_events_per_minute'],
                    rate_absolute_error=values.get('complete_events_per_minute_abs_error'),
                    duration_mean_seconds=values['duration_seconds_mean'], amplitude_mean=values['amplitude_mean'],
                    duration_wasserstein1=values.get('duration_seconds_wasserstein1'),
                    amplitude_wasserstein1=values.get('amplitude_wasserstein1')))
    limits = ['Weak ARKit parser and motion coordinates; no psychological affect-dynamics ground truth.',
              'All predeclared sampling seeds retained; no best-of-N and no independent training-seed claim.',
              'Event execution consistency is distinct from GT rate/duration/amplitude agreement.',
              'Regional motion std/velocity/acceleration are custom diagnostics, not canonical paper FDD.']
    export('data_route_expression_direction.csv', direction_rows, dynamics_files.values(), limits)
    export('data_route_event_distributions.csv', event_rows, dynamics_files.values(), limits)
    export('data_route_regional_dynamics.csv', region_rows, dynamics_files.values(), limits)


if __name__ == '__main__':
    main()
