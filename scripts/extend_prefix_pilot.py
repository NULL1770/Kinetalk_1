"""One bounded paired continuation of prefix pilot, preserving original outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_prefix_upper as p
from scripts.train_formal_predictable_projection import restore_rng
from scripts.audit_temporal_repair import read, sha, metadata_equal


def continuation_report(pred, target, valid, channel_mask, supplied_past):
    pairs = valid[:, 1:] & valid[:, :-1]
    pairs &= (torch.arange(1, valid.shape[1], device=valid.device) % p.CHUNK == 0)[None]
    report = {}
    for name in ('brows', 'eyes_expression'):
        cc = list(p.GROUPS[name]); observed = pairs[..., None] & channel_mask[:, None, cc]
        step = (pred[:, 1:, cc]-supplied_past[:, :-1, cc])[observed].double()
        gt_step = (target[:, 1:, cc]-target[:, :-1, cc])[observed].double()
        report[name] = {'observed_channel_pairs': int(observed.sum()),
            'rms': float(step.square().mean().sqrt()), 'reference_rms': float(gt_step.square().mean().sqrt()),
            'displacement_mse': float((step-gt_step).square().mean())}
    return report


def corrected_gate(curves, reports):
    ref = curves['no_prefix']; teacher = curves['teacher_prefix']; metadata_equal(ref, teacher)
    target, valid, mask = ref['target'], ref['valid'], ref['channel_mask']
    base = continuation_report(ref['predictions']['42/full'], target, valid, mask, target)
    oracle = continuation_report(teacher['predictions']['42/oracle_history'], target, valid, mask, target)
    full = continuation_report(teacher['predictions']['42/full'], target, valid, mask, teacher['predictions']['42/full'])
    checks = {}
    for name in ('brows', 'eyes_expression'):
        checks[name] = {'oracle_boundary_rms_ratio': oracle[name]['rms']/max(oracle[name]['reference_rms'], 1e-12),
            'boundary_error_ratio_to_no_prefix': oracle[name]['displacement_mse']/max(base[name]['displacement_mse'], 1e-12),
            'centered_mse_not_worse': reports['teacher_prefix']['modes']['42/oracle_history']['metrics'][name]['centered_mse'] <=
                reports['no_prefix']['modes']['42/full']['metrics'][name]['centered_mse']}
    return {'passed': all(row['oracle_boundary_rms_ratio'] <= 2 and row['boundary_error_ratio_to_no_prefix'] <= .5
        and row['centered_mse_not_worse'] for row in checks.values()), 'checks': checks,
        'teacher_to_actual_supplied_gt_prefix': oracle, 'no_prefix_to_same_gt_DIAGNOSTIC_NOT_INPUT': base,
        'generated_to_actual_supplied_generated_prefix': full,
        'post_result_metric_semantics_correction': True, 'threshold_values_unchanged': True,
        'scope': 'Only eight fit reconstruction clips; no audio/generalization success claim.'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('pilot-run', 'output', 'source-run', 'audio', 'targets', 'enrollment', 'native-root', 'trained-run', 'history-run', 'centered-run', 'pilot-selection'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); args.pilot=True; args.batch_size=8; torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True; started=time.monotonic()
    if args.output.exists(): raise FileExistsError('Fresh continuation output required')
    if read(args.pilot_run/'status.json')['status'] != 'complete': raise ValueError('Pilot is incomplete')
    data, system, audio, local, identities, source, source_recipe, source_sha, steps, selection, frozen = p.load_context(args)
    train = data['splits']['train']; scales=source['scales'].to(args.device)
    b=p.r.subset(train,torch.arange(len(train['valid'])),args.device); ident=p.r.batch_identity(identities,b)
    native=b['prefix_local']; target=p.h.normalized_target(b,scales)
    pairs=[(i,start) for i in range(len(b['valid'])) for start in range(0,96,p.CHUNK) if b['valid'][i,start:start+p.CHUNK].any()]
    original_curves=torch.load(args.pilot_run/'no_prefix/curves.pt',map_location='cpu',weights_only=False)
    old_pairs=[(i,start) for i in range(len(original_curves['valid'])) for start in range(0,96,p.CHUNK) if original_curves['valid'][i,start:start+p.CHUNK].any()]
    if pairs!=old_pairs or (len(pairs)+7)//8!=6:raise ValueError('Extension window order or six updates per epoch differs')
    if train['clip_id']!=original_curves['clip_id'] or not torch.equal(train['valid'],original_curves['valid']):
        raise ValueError('Extension masks or clip order differ from original pilot')
    args.output.mkdir(parents=True); matches={}; reports={}; curves_all={}; old_reports={}; old_curves={}
    protocol=Path(__file__).resolve().parents[1]/'docs/PREFIX_PILOT_EXTENSION_PROTOCOL_20260917.md'
    for arm in ('no_prefix','teacher_prefix'):
        previous=args.pilot_run/arm; record=read(previous/'provenance.json'); recipe=record['recipe']
        if (p.canonical_hash(recipe)!=record['recipe_sha256'] or recipe['fit_ids']!=train['clip_id'] or
            recipe['pilot_selection_sha256']!=sha(args.pilot_selection) or recipe['pilot_selection']!=selection or
            recipe['source_sha256']!=source_sha or recipe['batch_size']!=8 or recipe['epochs']!=15 or
            not recipe['pilot'] or recipe['arm']!=arm):raise ValueError('Original pilot membership or recipe differs')
        if recipe['code_sha256']['scripts/train_prefix_upper.py'] != sha(Path(p.__file__)):
            raise ValueError('Original training source changed')
        complete=read(previous/'complete.json')
        if sha(previous/'final.pt')!=complete['final_sha256'] or sha(previous/'curves.pt')!=complete['curves_sha256']:
            raise ValueError('Completed pilot artifact binding differs')
        last=torch.load(previous/'last.pt',map_location='cpu',weights_only=False)
        final=torch.load(previous/'final.pt',map_location='cpu',weights_only=False)
        if (last['completed_epochs']!=15 or last['total_steps']!=90 or last['source_sha256']!=source_sha or
            last['recipe_sha256']!=record['recipe_sha256'] or complete['recipe_sha256']!=record['recipe_sha256'] or
            final['recipe_sha256']!=record['recipe_sha256'] or final['completed_epochs']!=15 or final['total_steps']!=90 or
            not torch.equal(last['scales'],source['scales']) or not torch.equal(last['scales'],torch.tensor(recipe['scales'])) or
            p.state_hash(last['upper'])!=p.state_hash(final['upper']) or recipe['frozen']!=frozen):
            raise ValueError('Cannot bind last to completed endpoint')
        upper=p.PrefixUpperFlow(data['config']).to(args.device).eval(); upper.load_state_dict(last['upper'])
        params=list(upper.parameters()); optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-5); optimizer.load_state_dict(last['optimizer'])
        generator=torch.Generator(); restore_rng(last['rng'],generator)
        output=args.output/arm; output.mkdir(); draws=[]
        provenance={'schema':'prefix_pilot_extension_v1','arm':arm,'completed_epochs_before':15,'additional_epochs':15,
            'fixed_total_epochs':30,'windows_per_epoch':len(pairs),'source_last_sha256':sha(previous/'last.pt'),'source_final_sha256':complete['final_sha256'],
            'source_recipe_sha256':record['recipe_sha256'],'frozen':frozen,'selection':selection,'extension_protocol_sha256':sha(protocol),
            'extension_script_sha256':sha(Path(__file__)),'test_loaded':False,'default_replaced':False}
        digest=p.canonical_hash(provenance); p.save_json(output/'provenance.json',{'recipe':provenance,'recipe_sha256':digest})
        for epoch in range(15,30):
            t0=time.monotonic(); losses=[]; draw=hashlib.sha256()
            order=torch.randperm(len(pairs),generator=generator)
            for selected in order.split(8):
                pair_tensor=torch.tensor([pairs[i] for i in selected],device=args.device); ids,starts=pair_tensor[:,0],pair_tensor[:,1]
                noise=torch.randn(len(ids),24,9,generator=generator); times=torch.rand(len(ids),generator=generator)
                for tensor in (selected,noise,times):draw.update(tensor.numpy().tobytes())
                loss=p.patch_loss(upper,b,ident,native,target,target,ids,starts,noise.to(args.device),times.to(args.device),empty=arm=='no_prefix')
                p.r.optimize(loss,optimizer,params); losses.append(float(loss.detach()))
            draws.append(draw.hexdigest()); row={'epoch':epoch+1,'arm':arm,'loss':sum(losses)/len(losses),'total_steps':(epoch+1)*6,
                'seconds':time.monotonic()-t0,'draw_sha256':draw.hexdigest()}
            p.save_json(output/f'epoch{epoch+1:03d}.json',row); print(json.dumps(row),flush=True)
        report,curves=p.evaluate(upper,b,ident,native,scales,steps=steps,use_prefix=arm!='no_prefix')
        report.update(evaluation_role='eight_fit_reconstruction_extended',total_epochs=30,total_steps=180)
        reports[arm]=report; curves_all[arm]=curves
        old_reports[arm]=read(previous/'evaluation.json'); old_curves[arm]=torch.load(previous/'curves.pt',map_location='cpu',weights_only=False)
        p.save_json(output/'evaluation.json',report); p.save_checkpoint(output/'curves.pt',curves)
        p.save_checkpoint(output/'final.pt',{'upper':upper.state_dict(),'completed_epochs':30,'total_steps':180,'recipe_sha256':digest})
        p.save_json(output/'complete.json',{'curves_sha256':sha(output/'curves.pt'),'final_sha256':sha(output/'final.pt'),'completed_epochs':30,'recipe_sha256':digest})
        matches[arm]=draws
    if matches['no_prefix']!=matches['teacher_prefix']:raise RuntimeError('Continued random streams differ')
    for name,module in [('system',system),('audio',audio),('local',local)]:
        if p.state_hash(module.state_dict())!=frozen[name] or any(param.grad is not None for param in module.parameters()):raise RuntimeError('Frozen module changed')
    p.save_json(args.output/'corrected_epoch15_gate.json',corrected_gate(old_curves,old_reports))
    p.save_json(args.output/'receiver_gate.json',corrected_gate(curves_all,reports))
    p.save_json(args.output/'matched_audit.json',{'random_streams_equal':True,'draws':matches})
    p.save_json(args.output/'status.json',{'status':'complete','total_epochs':30,'additional_epochs':15,'seconds':time.monotonic()-started})
    print('PREFIX_EXTENSION_COMPLETE',flush=True)


if __name__=='__main__':main()
