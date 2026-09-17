"""Read-only weight/curve integrity audit and fixed metadata visual subset."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_temporal_repair import read, sha, load_pt, metadata_equal, summarize
from scripts import train_prefix_upper as p
from scripts.train_audio_prefix_adaptation import protected_local_state, compose_dc
from scripts.evaluate_audio_prefix_adaptation import _composition_invariants
from scripts.evaluate_prefix_formal import _same_bits

MODES = ('Old audio mean + flow', 'Previous prefix + audio mean',
         'New frozen local + audio mean', 'New adapted local + audio mean', 'New adapted local raw')


def compare_tree(a, b, where='root'):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b): raise ValueError('Keys differ: '+where)
        for key in a: compare_tree(a[key], b[key], where+'/'+key)
    elif isinstance(a, list):
        if len(a) != len(b): raise ValueError('Length differs: '+where)
        for i,(x,y) in enumerate(zip(a,b)):compare_tree(x,y,where+'/'+str(i))
    elif isinstance(a, (float,int)) and not isinstance(a,bool):
        if not np.isclose(a,b,rtol=1e-8,atol=1e-10,equal_nan=True): raise ValueError('Value differs: '+where)
    elif a != b: raise ValueError('Value differs: '+where)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root;run=root/'audio_prefix12';torch.set_num_threads(4)
    status=read(run/'status.json');process=read(root/'audio_prefix12_process.json')
    if status['status']!='complete' or process['exit_code']!=0:raise ValueError('Run incomplete')
    initial=load_pt(run/'initial.pt');shared={k:p.state_hash(v) for k,v in initial.items()}
    sourcepath=root/'context12/chunk_teacher/final.pt';source=load_pt(sourcepath)
    manifest=read(sourcepath.with_name('complete.json'))
    if sha(sourcepath)!=manifest['final_sha256'] or shared['upper']!=p.state_hash(source['upper']):raise ValueError('Warmstart differs')
    bases=load_pt(root/'repair12/centered_prior/white/curves.pt')
    selection_path=root/'context12/fixed_visual_selection.json';selection=read(selection_path)
    picks=selection['nine_plot_clips'];bindings={};curves={};audit={}
    for arm in ('frozen_local','adapt_local'):
        folder=run/arm;complete=read(folder/'complete.json');record=read(folder/'provenance.json');recipe=record['recipe']
        if record['recipe_sha256']!=p.canonical_hash(recipe) or complete['recipe_sha256']!=record['recipe_sha256']:
            raise ValueError('Recipe mismatch')
        if recipe['initial']!=shared or recipe['initial_file_sha256']!=sha(run/'initial.pt'):
            raise ValueError('Initial snapshot mismatch')
        if recipe['source_sha256']!=sha(sourcepath) or recipe['source_recipe_sha256']!=manifest['recipe_sha256']:
            raise ValueError('Source binding mismatch')
        if sha(root/'repair12/centered_prior/white/curves.pt')!=recipe['baseline_curves_sha256']:
            raise ValueError('Base hash mismatch')
        code_root=Path(__file__).resolve().parents[1]
        for name,digest in recipe['code_sha256'].items():
            if sha(code_root/name)!=digest:raise ValueError('Code changed since training: '+name)
        if complete['completed_epochs']!=12 or complete['total_steps']!=1740 or recipe['test_loaded'] is not False:
            raise ValueError('Budget or test contract differs')
        for file,key in (('final','final_sha256'),('curves','curves_sha256'),('dc_curves','dc_curves_sha256'),('fit_curves','fit_curves_sha256')):
            if sha(folder/(file+'.pt'))!=complete[key]:raise ValueError('File hash mismatch: '+file)
        final=load_pt(folder/'final.pt')
        if final['recipe_sha256']!=complete['recipe_sha256'] or final['frozen']!=recipe['frozen'] or final['total_steps']!=1740:
            raise ValueError('Final state contract differs')
        protected=lambda state:{k:v for k,v in state.items() if not k.startswith(('input.','blocks.','local_head.'))}
        if p.state_hash(protected(final['local']))!=p.state_hash(protected(initial['local'])):
            raise ValueError('Protected local heads changed')
        changed=p.state_hash(final['local'])!=shared['local']
        if changed!=(arm=='adapt_local') or changed!=complete['local_changed']:raise ValueError('Local update policy differs')
        raw,dc=load_pt(folder/'curves.pt'),load_pt(folder/'dc_curves.pt')
        metadata_equal(bases,raw);metadata_equal(raw,dc)
        if not torch.equal(raw['static_upper'],dc['static_upper']):raise ValueError('Static audio means differ')
        if any('oracle' in key for key in dc['predictions']):raise ValueError('Oracle entered deployment composition')
        for label,c in (('raw',raw),('dc',dc)):
            for key,pred in c['predictions'].items():
                base=bases['predictions'][key.split('/')[0]+'/base'];valid=c['valid']
                if not _same_bits(pred[...,list(p.r.NOT_UPPER)],base[...,list(p.r.NOT_UPPER)]) or not _same_bits(pred[~valid],base[~valid]):
                    raise ValueError('Protected channels differ')
                if label=='dc':
                    expected=compose_dc(base,raw['predictions'][key][...,p.CC],raw['static_upper'],valid)
                    if not torch.allclose(expected,pred,atol=2e-6,rtol=1e-6):raise ValueError('DC recomposition mismatch')
                    _composition_invariants(raw['predictions'][key][...,p.CC],pred[...,p.CC],raw['static_upper'],valid)
            report=read(folder/('evaluation.json' if label=='raw' else 'dc_evaluation.json'))
            deploy={**c,'predictions':{key:value for key,value in c['predictions'].items() if '/oracle_' not in key}}
            recalculated=summarize(deploy,c['emotion_id']);interventions=recalculated.pop('single_seed_interventions')
            compare_tree(recalculated,report['distribution']);compare_tree(interventions,report['deployable_interventions_seed42'])
            if label=='raw':
                full=c['predictions']['42/full']
                for mode in ('empty','reverse_history','oracle_history','oracle_reverse_history'):
                    if not _same_bits(full[:,:16],c['predictions']['42/'+mode][:,:16]):raise ValueError('First chunk GT history leakage')
        fit=load_pt(folder/'fit_curves.pt');fit_selection=read(run/'fit_selection.json')
        if fit['clip_id']!=[x['clip_id'] for x in fit_selection['clips']] or len(fit['clip_id'])!=128 or fit['nonupper_placeholders'] is not True:
            raise ValueError('Fixed fit diagnostic differs')
        if any(torch.count_nonzero(value[...,list(p.r.NOT_UPPER)]) for value in fit['predictions'].values()):raise ValueError('Fit placeholders nonzero')
        rows=[read(folder/f'epoch{i:03d}.json') for i in range(1,13)]
        audit[arm]={'hashes_valid':True,'local_changed':changed,'nonupper_invalid_exact':True,'DC_recomposed':True,
            'distribution_and_interventions_recomputed':True,'source_first_chunk_invariants':True,
            'epoch_draws':[row['draw_sha256'] for row in rows], 'steps':[row['total_steps'] for row in rows]}
        bindings[arm+'/raw']=complete['curves_sha256'];bindings[arm+'/dc']=complete['dc_curves_sha256']
        curves[arm+'/raw']=raw;curves[arm+'/dc']=dc
    if audit['frozen_local']['epoch_draws']!=audit['adapt_local']['epoch_draws'] or audit['frozen_local']['steps']!=list(range(145,1741,145)) or audit['adapt_local']['steps']!=audit['frozen_local']['steps']:
        raise ValueError('Paired updates/RNG differ')
    recorded=read(run/'matched_audit.json')
    if not recorded['equal'] or recorded['arms']['frozen_local']!=recorded['arms']['adapt_local']:raise ValueError('Matching manifest differs')
    oldpath=root/'repair12/centered_prior/state_mean/state_white_curves.pt'
    previouspath=root/'origin_diagnostic/audio_mean_curves.pt'
    if sha(oldpath)!=read(oldpath.parent/'complete.json')['state_white'] or sha(previouspath)!=read(previouspath.parent/'complete.json')['curves_sha256']:
        raise ValueError('Historical deployment binding differs')
    old,previous=load_pt(oldpath),load_pt(previouspath);reference=curves['adapt_local/raw']
    metadata_equal(reference,old);metadata_equal(reference,previous)
    bindings.update(old_white=sha(oldpath),previous_context_dc=sha(previouspath))
    indices=torch.tensor([reference['clip_id'].index(pick['clip_id']) for pick in picks])
    sources=[old,previous,curves['frozen_local/dc'],curves['adapt_local/dc'],reference]
    np.savez_compressed(run/'new_nine_clips.npz',motions=np.stack([c['predictions']['42/full'][indices].numpy() for c in sources]),
        target=reference['target'][indices].numpy(),mode_names=np.asarray(MODES),clip_id=np.asarray([pick['clip_id'] for pick in picks]),
        **{key:reference[key][indices].numpy() for key in ('times','valid','channel_mask')})
    p.save_json(run/'new_nine_clips_manifest.json',{'output_sha256':sha(run/'new_nine_clips.npz'),
        'selection_sha256':sha(selection_path),'source_curves':bindings,'noise_seed':42,'oracle_included':False,
        'raw_clamped':False,'test_loaded':False,'mode_names':list(MODES)})
    p.save_json(run/'integrity_audit.json',{'status':'passed','audit_source_sha256':sha(__file__),'source_sha256':sha(sourcepath),
        'arms':audit,'matched_rng_and_updates':True,'initial_snapshot_and_source_verified':True,
        'all_405_metadata_equal':True,'frozen_system_audio_record_hashes_checked':True,
        'frozen_system_audio_source_tensors_reloaded':False,'test_loaded':False,
        'limitations':'Integrity is not quality; system/audio source tensors rely on training checks; three seeds and repeated development only.'})
    print('AUDIO_PREFIX_INTEGRITY_PASSED',flush=True)


if __name__=='__main__':main()
