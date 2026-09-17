"""Fixed8epoch audit of a single train-only loss metric change."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_audio_activity_gate import SEEDS, write_json
from scripts.audit_formal_projection import assert_metric_agreement
from scripts.audit_projection_rollout_probe import audit_two_arm_epoch, load_rollout
from scripts.audit_teacher_schedule_probe import load_arm, validate_training_inputs, validate_curves
from scripts.evaluate_interrupted_centered_probe import audit_single
from scripts.train_projection_scaled_centered_probe import SCHEMA, LOSS, fit_training_channel_metric, channel_metric_report
from scripts.train_predictable_renderer import basic_metrics, sha, state_hash
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_projection_centered_rollout_probe import SCHEMA as OLD_SCHEMA, ARM as OLD_ARM, LOSS as OLD_LOSS

ARMS = ('train_rms', 'uniform')
EPOCHS = (2, 8)


def validate_metric_pair(scaled, uniform):
    for recipe, arm in ((scaled, 'train_rms'), (uniform, 'uniform')):
        if recipe.get('schema') != SCHEMA or recipe['args'].get('arm') != arm or recipe.get('loss') != LOSS:
            raise ValueError('Unexpected metric arm/loss')
        if recipe['args'].get('epochs') != 8 or recipe.get('teacher_probability') != .5 or recipe.get('decode_steps') != 12:
            raise ValueError('Unexpected budget/teacher/decoding')
        if recipe.get('trainable') != ['local_projection.weight'] or recipe.get('mean_anchoring_loss') is not False:
            raise ValueError('Extra trainable module/loss')
    left, right = copy.deepcopy(scaled), copy.deepcopy(uniform)
    for recipe in (left, right):
        for key in ('arm', 'output'): recipe['args'].pop(key)
        recipe.pop('selected_weight_sha256')
    if left != right:
        raise ValueError('Matched metric arms differ beyond selected fixed channel weights')


def validate_historical_reproduction(uniform, old_dir):
    old_provenance = json.loads((old_dir / 'provenance.json').read_text())
    old_recipe = old_provenance['recipe']
    if old_provenance.get('recipe_sha256') != canonical_hash(old_recipe) or (old_recipe['schema'],old_recipe['args']['arm'],old_recipe['loss']) != (OLD_SCHEMA,OLD_ARM,OLD_LOSS):
        raise ValueError('Wrong historical centered provenance')
    new_recipe = uniform['recipe']
    for key in ('input_sha256', 'data_scope', 'loss_centering', 'mean_anchoring_loss', 'teacher_probability', 'decode_steps', 'head_basis_scale_frozen'):
        if new_recipe[key] != old_recipe[key]: raise ValueError('Historical centered protocol changed: '+key)
    for name, digest in old_recipe['source_sha256'].items():
        if new_recipe['source_sha256'].get(name) != digest: raise ValueError('Historical shared training source changed')
    for key, value in old_recipe['args'].items():
        if key not in ('arm', 'output', 'epochs') and new_recipe['args'][key] != value:
            raise ValueError('Historical training argument changed')
    for epoch in range(1, 9):
        old = json.loads((old_dir / f'epoch{epoch:03d}.json').read_text())
        new = uniform['records'][epoch]
        for key in ('minibatch_sha256', 'noise_time_sha256', 'teacher_choice_draw_sha256', 'step', 'samples_seen', 'teacher_fraction'):
            if new[key] != old[key]: raise ValueError(f'Historical RNG/coverage differs at{epoch}/{key}')
        if epoch % 2 == 0:
            recorded = json.loads((old_dir / f'development_epoch{epoch:03d}.json').read_text())['noise_reports']
            current = json.loads((Path(new_recipe['args']['output']) / f'development_epoch{epoch:03d}.json').read_text())['noise_reports']
            for seed in SEEDS:
                for mode in ('full', 'zero'):
                    if epoch in EPOCHS:
                        assert_metric_agreement(current[str(seed)][mode], uniform['reports'][epoch][str(seed)][mode])
                    assert_metric_agreement(current[str(seed)][mode], recorded[str(seed)][mode])
                    if current[str(seed)][mode]['frozen_teacher_emotion_accuracy'] != recorded[str(seed)][mode]['frozen_teacher_emotion_accuracy']:
                        raise ValueError('Historical frozen teacher readout differs')
    return {str(p):sha(p) for p in [old_dir/'provenance.json'] +
        [folder/f'epoch{e:03d}.json' for folder in (old_dir,Path(new_recipe['args']['output'])) for e in range(1,9)] +
        [folder/f'development_epoch{e:03d}.json' for folder in (old_dir,Path(new_recipe['args']['output'])) for e in (2,4,6,8)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scaled', 'uniform', 'flow-control', 'historical-centered', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    detail_paths = {arm:args.output.with_name(args.output.stem+'_'+arm+'_raw_mean.json') for arm in ARMS}
    if args.output.exists() or any(path.exists() for path in detail_paths.values()):
        raise ValueError('Fresh audit outputs required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    arms = {'train_rms': load_rollout(args.scaled, schema=SCHEMA, arm='train_rms', final_epoch=8),
        'uniform': load_rollout(args.uniform, schema=SCHEMA, arm='uniform', final_epoch=8)}
    validate_metric_pair(arms['train_rms']['recipe'], arms['uniform']['recipe'])
    reproduction_hashes = validate_historical_reproduction(arms['uniform'], args.historical_centered)
    for epoch in range(1, 9):
        for key in ('minibatch_sha256','noise_time_sha256','teacher_choice_draw_sha256','step','samples_seen','teacher_fraction'):
            if arms['train_rms']['records'][epoch][key] != arms['uniform']['records'][epoch][key]:
                raise ValueError('Metric arms RNG/schedule mismatch')
    flow = load_arm(args.flow_control, 'constant_teacher')
    for data in arms.values():
        recipe = data['recipe']
        if recipe['input_sha256'] != flow['recipe']['input_sha256'] or recipe['data_scope'] != flow['recipe']['data_scope']:
            raise ValueError('Metric arms do not use audited original flow inputs/scope')
        for key,digest in recipe['input_sha256'].items():
            if recipe['args'][key] != flow['recipe']['args'][key] or sha(recipe['args'][key]) != digest:
                raise ValueError('Metric arm actual input changed')
        for path,digest in recipe['source_sha256'].items():
            if sha(path) != digest: raise ValueError('Training source changed')
    reference, lock, scope, fixed = validate_training_inputs({'constant_teacher': flow})
    recipe = arms['uniform']['recipe']
    cache = torch.load(recipe['args']['cache'], map_location='cpu', weights_only=False, mmap=True)
    metric = fit_training_channel_metric(cache)
    for arm, data in arms.items():
        if data['recipe']['channel_metric'] != channel_metric_report(metric):
            raise ValueError('Loss metric was not derived from fixed train-only tensors')
        chosen = state_hash({'weights': metric[arm+'_weights']})
        if data['recipe']['selected_weight_sha256'] != chosen or data['summary'].get('selected_weight_sha256') != chosen:
            raise ValueError('Wrong selected channel weight')
        if data['summary'].get('generated_output_centered') is not False:
            raise ValueError('Raw generated output required')
        metric_path = Path(data['recipe']['args']['output']) / 'channel_metric.pt'
        if data['summary'].get('channel_metric_sha256') != sha(metric_path) or state_hash(torch.load(metric_path, map_location='cpu', weights_only=False)) != state_hash(metric):
            raise ValueError('Saved channel metric binding differs')
        for epoch in EPOCHS:
            ck = data['checkpoints'][epoch]
            if ck['head_sha256'] != fixed['fixed_head'] or ck['frozen_state_sha256'] != fixed['frozen_backbone']:
                raise ValueError('Frozen head/backbone changed')
            if state_hash(ck['channel_metric']) != state_hash(metric) or ck['selected_weight_sha256'] != chosen:
                raise ValueError('Checkpoint metric changed')
            validate_curves(data['curves'][epoch], reference)
            for seed in SEEDS:
                for mode in ('full','zero','reverse','oracle'):
                    assert_metric_agreement(basic_metrics(data['curves'][epoch]['motion'][str(seed)][mode], reference),
                        data['reports'][epoch][str(seed)][mode])
    epochs = {}
    for epoch in EPOCHS:
        epochs[str(epoch)] = audit_two_arm_epoch({a:v['curves'][epoch] for a,v in arms.items()}, reference, samples=5000, arms=ARMS)
        epochs[str(epoch)]['teacher_readout'] = {arm: {mode: float(np.mean([data['reports'][epoch][str(seed)][mode]['frozen_teacher_emotion_accuracy']
            for seed in SEEDS])) for mode in ('full','zero','reverse','oracle')} for arm,data in arms.items()}
    write_json(args.output, {'schema':'scaled_centered_fixed8_audit_v1','source_sha256':sha(__file__),
        'arm_hashes':{a:d['hashes'] for a,d in arms.items()},'data_scope':scope,'frozen_hashes':fixed,
        'historical_reproduction_source_hashes':reproduction_hashes,
        'same_rng_all8epochs':True,'uniform_reproduces_historical_epochs2_4_6_8':True,
        'train_only_weights_recomputed':True,'channel_metric':channel_metric_report(metric),
        'primary_epoch':8,'auxiliary_epoch':2,'epochs':epochs,
        'same_arm_checks_scope':'Dynamic/neutral/mouth engineering checks; raw/mean nonneutral decomposition separately required. Not deployment certification.',
        'test_loaded':False,'outer280_loaded':False,'new_identity439_loaded':False,'default_replaced':False})
    for arm, data in arms.items():
        detail = audit_single(data['curves'][8], reference, samples=5000)
        write_json(detail_paths[arm],
            {'arm':arm,'epoch':8,'parent_audit_sha256':sha(args.output),'detail':detail})
    print(json.dumps({'complete':True,'epoch':8,'checks':epochs['8']['same_arm_checks'],
        'full_vs_zero':{a:{g:r['r2_improvement'] for g,r in epochs['8']['full_vs_same_arm'][a]['zero']['nonneutral'].items()} for a in ARMS}}), flush=True)


if __name__=='__main__': main()
