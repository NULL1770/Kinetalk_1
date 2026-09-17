"""Paired feature comparison from saved development and locked-audit results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_audio_dynamic_context_features import bootstrap, per_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/neutral_affect_pilot_20260916'))
    args = parser.parse_args()
    result = {'definition': 'Positive paired improvement means lower MSE for emotion2vec. Seeds averaged inside clip.',
              'limitations': 'Descriptive intervals; only four enrolled people, repeated sentences, no multiplicity correction.',
              'splits': {}}
    for split, suffix in [('development', '_eval'), ('audit26', '_audit26')]:
        entries = {name: json.loads((args.root / (name + suffix) / 'heldout_summary.json').read_text())
                   for name in ('run07_acoustic_probe', 'run09_emotion2vec_probe')}
        acoustic, affect = entries.values()
        for key in ('clips', 'speaker_ids', 'sentences'):
            assert acoustic[key] == affect[key]
        metrics = {}
        for name, entry in entries.items():
            full = entry['aggregate']['audio_full']['mean']
            metrics[name] = {'motion': {k: full[k] for k in ('masked_mse', 'upper_pred_corr', 'brows_pred_corr',
                                                           'jaw17_pred_corr', 'mouth_pred_speed_corr', 'upper_pred_std_ratio')},
                             'mean_mse': entry['aggregate']['audio_mean']['mean']['masked_mse'],
                             'reverse_mse': entry['aggregate']['audio_reverse']['mean']['masked_mse'],
                             'affect': entry['affect_diagnostics'],
                             'full_vs_mean': entry['paired_improvement']['comparisons']['audio_full_vs_audio_mean']['mse'],
                             'full_vs_reverse': entry['paired_improvement']['comparisons']['audio_full_vs_audio_reverse']['mse']}
        comparisons = {region: bootstrap(per_clip(acoustic, 'audio_full', region) - per_clip(affect, 'audio_full', region),
                                         acoustic['speaker_ids'], acoustic['sentences'], seed=9173, repetitions=10000)
                       for region in ('all', 'upper', 'brows', 'mouth', 'jaw17')}
        result['splits'][split] = {'clips': len(acoustic['clips']), 'metrics': metrics, 'paired_emotion2vec_vs_acoustic': comparisons,
                                  'b0': {k: acoustic['b0'][k] for k in ('masked_mse', 'jaw17_pred_corr', 'mouth_pred_speed_corr')}}
    provenance = {name: json.loads((args.root / name / 'provenance.json').read_text()) for name in
                  ('run07_acoustic_probe', 'run08_content_probe', 'run09_emotion2vec_probe')}
    for key in ('teacher_sha256', 'initial_frozen_state_sha256', 'scales_sha256'):
        assert len({p[key] for p in provenance.values()}) == 1, key
    assert provenance['run08_content_probe']['initial_audio_state_sha256'] == provenance['run09_emotion2vec_probe']['initial_audio_state_sha256']
    summaries = [json.loads((args.root / name / 'summary.json').read_text()) for name in provenance]
    assert len({s['minibatch_sha256'] for s in summaries}) == 1
    assert all(s['frozen_state_unchanged'] for s in summaries)
    result['integrity'] = {'teacher_frozen_scales_minibatches_matched': True, 'hubert_emotion2vec_initial_audio_bitwise_same': True,
                           'non_audio_parameters_unchanged': True, 'acoustic_to_768_input_projection_extra_parameters': 65760}
    output = args.root / 'emotion2vec_feature_comparison.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
