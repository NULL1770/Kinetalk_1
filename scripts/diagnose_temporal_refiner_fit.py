"""Read-only final-checkpoint fit/OOF and learned-neighbour intervention."""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.predictable_motion import predict_motion, weighted_clip_center
from kinetalk_b0.temporal_motion_refiner import FrozenRidgeRefiner, fit_control_scale
from scripts.train_temporal_audio_refiner_probe import load_sources, predict_batches
from scripts.train_predictable_renderer import audio_features, sha
from scripts.audit_predictable_motion_predictions import clip_statistics, scalar_summary, paired_summary
from scripts.probe_scaled_motion_basis import motion_groups, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    out = args.study / 'fit_and_side_taps.json'
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    report = json.loads((args.study / 'paired_audit.json').read_text())
    for name, digest in report['input_output_sha256'].items():
        if sha(args.study / name) != digest:
            raise ValueError('Changed input')
    recipe = report['provenance']; sa = recipe['args']
    bundle, channels, speakers, folds, saved, old, _, _ = load_sources(
        Path(sa['bundle']), Path(sa['weights']), Path(sa['basis_study']))
    x, y, w = audio_features(bundle), bundle['motion_bins'][..., channels], bundle['weight']
    curves = torch.load(args.study / 'oof_predictions.pt', map_location='cpu', weights_only=False)
    _, groups = motion_groups(channels)
    fit_report, tap_report = [], []
    removed = torch.zeros_like(curves['target'])
    for fold, (fit, val) in enumerate(folds):
        state = saved['states'][f'fold{fold}_native']['transformed_model']
        scale = fit_control_scale(y, w, fit, state)
        target = weighted_clip_center(y[fit], w[fit])
        predictions = {'ridge': predict_motion(x[fit], w[fit], state)}
        for arm in ('pointwise', 'temporal'):
            ck = torch.load(args.study / f'fold{fold}_{arm}/last.pt', map_location='cpu', weights_only=False)
            model = FrozenRidgeRefiner(state, scale, arm)
            model.load_state_dict(ck['model'], strict=True); model.cuda()
            predictions[arm] = predict_batches(model, x[fit], w[fit], 'cuda')
            if arm == 'temporal':
                taps = []
                with torch.no_grad():
                    for block in model.refiner.blocks:
                        kernel = block.depthwise.weight
                        taps.append({'center_rms': float(kernel[..., 1].square().mean().sqrt()),
                            'learned_side_rms': float(kernel[..., [0, 2]].square().mean().sqrt())})
                        kernel[..., [0, 2]] = 0
                removed[val] = predict_batches(model, x[val], w[val], 'cuda')
                tap_report.append({'fold': fold, 'kernels': taps})
            model.cpu()
        scores = {}
        for arm, pred in predictions.items():
            scores[arm] = {}
            for pop, mask in [('all', torch.ones(len(fit), dtype=torch.bool)),
                              ('nonneutral', bundle['emotion_id'][fit] != 0),
                              ('neutral', bundle['emotion_id'][fit] == 0)]:
                scores[arm][pop] = {g: scalar_summary(clip_statistics(pred[mask], target[mask], w[fit][mask], cc))
                                     for g, cc in groups.items()}
        fit_report.append({'fold': fold, 'fit_clips': len(fit), 'scores': scores})
    target = curves['target']
    left = {g: clip_statistics(curves['predictions']['temporal']['full'], target, w, cc) for g, cc in groups.items()}
    right = {g: clip_statistics(removed, target, w, cc) for g, cc in groups.items()}
    comparisons = {}
    for pop, mask in [('all', torch.ones(len(w), dtype=torch.bool)), ('nonneutral', bundle['emotion_id'] != 0),
                      ('neutral', bundle['emotion_id'] == 0)]:
        ids = mask.nonzero(as_tuple=True)[0].tolist()
        comparisons[pop] = {g: paired_summary(left[g], right[g], bundle['sentence_id'], ids, samples=5000, seed=45) for g in groups}
    result = {'schema': 'temporal_refiner_fit_and_taps_v1', 'source_audit_sha256': sha(args.study / 'paired_audit.json'),
        'script_sha256': sha(__file__), 'fit_scores': fit_report, 'learned_taps': tap_report,
        'temporal_full_vs_side_taps_removed': comparisons,
        'scope': 'Posthoc read-only diagnosis, no training or reselection. Fit clips repeat across folds and are not independent. Removing learned side taps is an intervention in the final model, not a separately trained baseline. No final-test or development targets accessed.'}
    write_json(out, result)
    print(json.dumps({'fit_mse': [{a: f['scores'][a]['all']['all_expression']['native_mse'] for a in f['scores']} for f in fit_report],
        'nonneutral_tap_delta_r2': {g: comparisons['nonneutral'][g]['r2_improvement'] for g in groups}}), flush=True)


if __name__ == '__main__':
    main()
