"""No training: strict composition of frozen native mean/style improvements."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.joint_factor_repair import V5JointFactorRepair
from scripts.pilot_v5_factor_rebuild import raw_audio,evaluate
from scripts.train_v7_pilot import load,sha
from scripts.train_expression_response import configure,load_runtime,write,save,state_digest


@torch.no_grad()
def run(a):
    configure(a.seed);start=time.time();out=Path(a.output);device=torch.device(a.device)
    if out.exists():raise FileExistsError('fresh output required')
    out.mkdir(parents=True)
    mean_ck=torch.load(a.trend_checkpoint,map_location='cpu',weights_only=False)
    style_ck=torch.load(a.style_checkpoint,map_location='cpu',weights_only=False)
    model=V5JointFactorRepair.from_parts(mean_ck,style_ck).to(device).eval().requires_grad_(False)
    binding=json.loads(Path(a.binding).read_text());runtime,base=load_runtime(binding,torch.device('cpu'))
    mean,std=runtime['feature_stats']['mean'].cpu(),runtime['feature_stats']['std'].cpu();del runtime,base
    results={}
    for weight in (0.,.25,.5,1.):
        arm='style_weight_'+str(weight);dest=out/arm;dest.mkdir();results[arm]={}
        model.reference_blend_weight.fill_(weight);before=state_digest(model)
        for role in ('held_sentence','unseen_speaker'):
            data=raw_audio(load(Path(a.data)/(role+'.npz')),mean,std)
            results[arm][role]=evaluate(model,data,device,binding,a.seed,dest,role)
        if state_digest(model)!=before:raise RuntimeError('read-only evaluation drift')
        save(out/(arm+'.pt'),model.checkpoint())
    write(out/'report.json',dict(schema='v5_joint_factor_repair_eval_v1',elapsed_seconds=time.time()-start,
        trend_checkpoint_sha256=sha(a.trend_checkpoint),style_checkpoint_sha256=sha(a.style_checkpoint),
        independent_probes_used_in_training=False,read_only=True,sealed_test_loaded=False,results=results,
        flow_enabled=False,personal_response_enabled=False,paper_ready=False,
        limits=['held blend validation is model selection, not independent final benchmark',
                'weights control two frozen reference predictions, not hidden channel partitions',
                'donor-style direction and phonetic preservation still need external evidence',
                'teacher raw-state distillation failed; no teacher-success claim from output-only repairs']))
    print(json.dumps({'completed':str(out),'elapsed_seconds':time.time()-start}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('trend-checkpoint','style-checkpoint','binding','data','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=47)
    run(p.parse_args())
