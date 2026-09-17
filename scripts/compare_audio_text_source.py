"""Additional eight-seed comparison against the inherited old full-local model.

Reads immutable saved predictions only. Authenticates the uniform source model
and the historical audio-flow epoch0 eight-seed expansion; additionally compares
the original three-seed uniform curves exactly. No generation or training.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_audio_text_dynamics as audit
from scripts.audit_projection_readiness import relative_error_audit
from scripts.audit_teacher_schedule_probe import GROUPS, audit_populations
from scripts.train_formal_predictable_projection import save_json
from scripts.train_predictable_renderer import sha

SCHEMA = 'audio_text_source_comparison_v1'
DEFAULT_SOURCE = Path('/root/kinetalk_runs/teacher_schedule_v1/audio_flow_v1/audio_local/epoch000_curves.pt')


def validate_source_reproduction(historical, uniform, arms):
    """Bind the old eight-seed expansion to original three-seed predictions."""
    if uniform.get('noise_seeds') != [42,123,2026] or uniform.get('decode_steps') != 12:
        raise ValueError('Original uniform source sampling differs')
    if set(uniform.get('motion',{})) != {'42','123','2026'}:
        raise ValueError('Original uniform seed keys differ')
    for seed in (42,123,2026):
        if set(uniform['motion'][str(seed)]) != {'full','zero','reverse','oracle'}:
            raise ValueError('Original uniform modes differ')
        for mode in ('full','zero','reverse','oracle'):
            torch.testing.assert_close(historical['motion'][str(seed)][mode],uniform['motion'][str(seed)][mode],rtol=0,atol=0)
    audit.validate_common_initial(arms)
    for seed in audit.SEEDS:
        torch.testing.assert_close(historical['motion'][str(seed)]['zero'],
            arms['text']['epoch0']['motion'][str(seed)]['zero'],rtol=0,atol=0)
    return {'historical_first_three_seeds_all_modes_equal_original_uniform':True,
        'historical_all_eight_zero_equal_new_matched_epoch0':True}


def analyze(historical, arms, reference, samples=2000):
    motion={str(seed):{'old_full_local':historical['motion'][str(seed)]['full'],
        **{arm+'/full':data['final']['motion'][str(seed)]['full'] for arm,data in arms.items()}}
        for seed in audit.SEEDS}
    stats,velocity,components=audit._condition_statistics(motion,reference,audit.SEEDS)
    q=reference['q'];populations=audit_populations(q);comparisons={}
    for arm in audit.ARMS:
        name=arm+'/full';baseline='old_full_local'
        row=audit._paired(stats,velocity,components,name,baseline,q,populations,samples)
        row['centered_relative']={pop:{group:relative_error_audit(
            stats[name,'centered_residual',group][:,[0,3]],stats[baseline,'centered_residual',group][:,[0,3]],
            q['sentence_id'],ids,samples=samples) for group in GROUPS} for pop,ids in populations.items()}
        comparisons[name+'_vs_'+baseline]=row
    return {'scores':audit._score_table(stats,populations),'comparisons':comparisons,
        'domain':audit.domain_report(motion,q,populations),
        'population_counts':{name:{'clips':len(ids),'sentences':len({q['sentence_id'][i] for i in ids})}
            for name,ids in populations.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-curves',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--device',default='cpu');p.add_argument('--samples',type=int,default=2000)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Fresh additional result required')
    if args.samples<0:raise ValueError('Bootstrap samples must be nonnegative')
    torch.set_num_threads(4)
    arms={arm:audit.load_arm(args.root/arm,arm) for arm in audit.ARMS}
    recipes=[copy.deepcopy(v['recipe']) for v in arms.values()]
    for value in recipes:value.pop('arm')
    if recipes[0]!=recipes[1]:raise ValueError('Matched recipes differ beyond arm')
    recipe=arms['text']['recipe'];loaded=audit.runner.load_source(recipe['source_run'],args.device)
    for key in ('input_sha256','source_adapter_sha256','data_scope'):
        if loaded[key]!=recipe[key]:raise ValueError('Rebuilt inherited source differs: '+key)
    reference=loaded['cache']['splits']['validation']
    for arm,data in arms.items():
        audit.validate_saved_curves(data['epoch0'],reference,arm,initial=True)
        audit.validate_saved_curves(data['final'],reference,arm)
    historical,binding=audit.common.load_historical_source_curves(args.source_curves,loaded,recipe,reference)
    historical_recipe=audit.read(args.source_curves.parent/'provenance.json')['recipe']
    if historical_recipe.get('audio_activity_gate') is not True:
        raise ValueError('Historical full-local gate protocol differs')
    uniform_path=Path(recipe['source_run'])/'final_epoch008_curves.pt'
    if sha(uniform_path)!=loaded['source_curve_sha256']:raise ValueError('Uniform source curves changed during loading')
    uniform=torch.load(uniform_path,map_location='cpu',weights_only=False,mmap=True)
    reproduction=validate_source_reproduction(historical,uniform,arms)
    rng=audit.common.validate_rng(arms,loaded)
    del uniform,loaded['system'],loaded['head']
    report=analyze(historical,arms,reference,args.samples)
    report.update(schema=SCHEMA,source_binding=binding,source_reproduction=reproduction,
        original_uniform_curves={'path':str(uniform_path),'sha256':loaded['source_curve_sha256']},
        input_sha256=recipe['input_sha256'],source_adapter_sha256=recipe['source_adapter_sha256'],
        arm_inventories={arm:data['hashes']['inventory'] for arm,data in arms.items()},training_rng=rng,
        script_sha256=sha(__file__),audit_helper_sha256=sha(audit.__file__),
        scope={'noise_seeds':audit.SEEDS,'decode_steps':12,'bootstrap_samples':args.samples,
            'baseline':'Original uniform epoch8 full-local audio head with its original activity gate. Historical audio-flow epoch0 expands the same source to eight seeds; original three-seed predictions agree exactly.',
            'paired_unit':'Sentence-cluster bootstrap; sufficient statistics average eight seed errors, never trajectories.',
            'sign':'Relative MSE increase below zero is better. Centered R2 improvement above zero is better.',
            'generation_performed':False,'training_performed':False,'test_loaded':False,'checkpoint_selection_performed':False,
            'limits':'Internal development405; no perceptual lip-sync/identity certification or publication claim. This supplements, and does not replace, the existing zero-local baseline audit.'})
    save_json(args.output,report)
    print(json.dumps({'output':str(args.output.resolve()),'schema':SCHEMA,'source_reproduction':reproduction}),flush=True)


if __name__=='__main__':main()
