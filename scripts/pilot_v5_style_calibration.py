"""TRAIN-only native reference calibration and read-only held factor audits."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.factor_rebuild import V5FactorRebuild
from kinetalk_v5.style_calibration import V5CalibratedStyle
from scripts.pilot_v5_factor_rebuild import raw_audio,reference,evaluate
from scripts.train_v7_pilot import load,batch,sha
from scripts.train_expression_response import configure,load_runtime,write,save,state_digest


@torch.no_grad()
def run(a):
    configure(a.seed);start=time.time();out=Path(a.output);device=torch.device(a.device)
    if out.exists():raise FileExistsError('fresh output required')
    out.mkdir(parents=True)
    if sha(a.checkpoint)!=a.sha256:raise ValueError('checkpoint mismatch')
    prior=V5FactorRebuild.from_checkpoint(torch.load(a.checkpoint,map_location='cpu',weights_only=False))
    prior.flow.sample_gain.zero_();prior.style_response.basis[-1].weight.zero_();prior.style_response.basis[-1].bias.zero_()
    model=V5CalibratedStyle(prior).to(device).eval().requires_grad_(False)
    core_before=state_digest(model.core);teacher_before=state_digest(model.teacher)
    binding=json.loads(Path(a.binding).read_text());runtime,base=load_runtime(binding,torch.device('cpu'))
    mean,std=runtime['feature_stats']['mean'].cpu(),runtime['feature_stats']['std'].cpu();del runtime,base
    fit=raw_audio(load(Path(a.data)/'fit.npz'),mean,std)
    records=[r for r in json.loads((Path(a.data)/'provenance.json').read_text())['records'] if r['role']=='fit']
    seen=set(); features=[];codes=[];people=[]
    # Deduplicate identical windows; repeated enrollment must not weight the
    # nuisance effect artificially. References stay inside prepared TRAIN.
    for offset in range(0,len(fit['motion']),8):
        ids=list(range(offset,min(offset+8,len(fit['motion'])))); b,refs=batch(fit,ids,device)
        for slot in (0,1):
            r=reference(refs,slot);x=model.reference_features(r)
            s=super(V5CalibratedStyle,model).encode_style(r)
            for j,i in enumerate(ids):
                rec=next(q for q in records if q['row']==i)
                if slot>=len(rec['references']) or not refs['valid'][j,slot].any():continue
                ref=rec['references'][slot]
                key=(ref['speaker'],ref['clip_id'],ref['start_frame'],ref['end_frame'])
                if key in seen:continue
                seen.add(key);features.append(x[j]);codes.append(s[j]);people.append(int(b['speaker_id'][j])+int(b['dataset_id'][j])*100000)
    model.reference_calibration.fit(torch.stack(features),torch.stack(codes),
                                     torch.tensor(people,device=device),ridge=a.ridge)
    results={}
    for strength in (0.,.5,1.):
        name='calibration_'+str(strength);model.reference_calibration.strength.fill_(strength)
        dest=out/name;dest.mkdir();results[name]={}
        for role in ('held_sentence','unseen_speaker'):
            data=raw_audio(load(Path(a.data)/(role+'.npz')),mean,std)
            results[name][role]=evaluate(model,data,device,binding,a.seed,dest,role)
    if state_digest(model.core)!=core_before or state_digest(model.teacher)!=teacher_before:
        raise RuntimeError('frozen core/teacher changed')
    save(out/'calibration.pt',{'schema':'v5_reference_calibration_diagnostic_v1',
                             'model':model.state_dict(),'base_checkpoint':a.checkpoint})
    write(out/'report.json',dict(schema='v5_reference_calibration_diagnostic_v1',
        elapsed_seconds=time.time()-start,base_checkpoint=a.checkpoint,base_sha256=a.sha256,
        unique_fit_references=len(seen),fit_person_count=len(set(people)),ridge=a.ridge,
        results=results,sealed_test_loaded=False,independent_probes_used_in_training=False,
        frozen_core_verified=True,paper_ready=False,
        limits=['known linear fixed-effect calibration is a diagnostic not novelty',
                'native reference code still may contain nonlinear nuisances',
                'HDTF unknown labels untouched; same-source identity limitation retained',
                'held output stability does not itself prove natural donor-style transfer']))
    print(json.dumps({'completed':str(out),'elapsed_seconds':time.time()-start}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('checkpoint','sha256','binding','data','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--ridge',type=float,default=10.);p.add_argument('--seed',type=int,default=47)
    p.add_argument('--device',default='cuda');run(p.parse_args())
