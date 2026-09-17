"""Development-selected candidate: aligned upper, original-local non-upper.

No sample-specific gating or new training. The rule was chosen after inspecting
development; these scores must not be presented as untouched test evidence.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import train_temporal_repair as r
from scripts.train_formal_predictable_projection import save_json,save_checkpoint
from scripts.audit_temporal_repair import summarize


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ('source-run','audio','targets','enrollment','native-root','trained-run','align-run','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh candidate output required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    data=r.load_training_inputs(a.source_run,a.audio,a.targets,a.enrollment,a.native_root)
    ck,bindings,steps,stride=r.matching_checkpoints(a.trained_run,data)
    system=data['system'].to(a.device).eval().requires_grad_(False);system.load_state_dict(ck['teacher']['system'])
    audio=r._make_audio(ck['audio']['audio'],stride,a.device)
    path=a.align_run/'final.pt';complete=json.loads((a.align_run/'complete.json').read_text())
    if r.sha(path)!=complete['final_sha256']:raise ValueError('Alignment hash differs')
    aligned=torch.load(path,map_location='cpu',weights_only=False)
    if aligned['source_bindings']!=bindings:raise ValueError('Alignment source differs')
    adapter=r.AlignedAudioLocal(audio.feature_mean,audio.feature_std,hidden=audio.input.out_features,rank=system.motion_teacher.rank,stride=system.motion_teacher.stride).to(a.device).eval().requires_grad_(False)
    adapter.load_state_dict(aligned['adapter'])
    r.cache_current_base(system,data,a.device,a.batch_size);identities=r.identity_cache(system,data,a.device)
    q=data['splits']['validation'];predictions={};metrics={};nonupper=list(r.NOT_UPPER);upper=list(r.UPPER_INDICES)
    for seed in r.SEEDS:
        noise=torch.randn(len(q['valid']),q['valid'].shape[1],52,generator=torch.Generator().manual_seed(seed))
        outs=[];baselines=[];emotions=[]
        for ix in torch.arange(len(q['valid'])).split(a.batch_size):
            b=r.subset(q,ix,a.device);ident=r.batch_identity(identities,b);affect=audio(b['audio_features'],b['valid'])
            base={'b0':b['b0'],'h0':b['h0']};n=noise[ix].to(a.device)
            original=system.generate(b['content'],b['valid'],ident,affect,initial_noise=n,steps=steps,base=base)['motion']
            transferred=system.project_affect({**affect,**adapter(b['audio_features'],b['valid'])},b['valid'])
            aligned_motion=system.generate(b['content'],b['valid'],ident,transferred,initial_noise=n,steps=steps,base=base)['motion']
            candidate=r.compose_upper_face(original,aligned_motion[...,upper],b['valid'])
            assert torch.equal(candidate[...,nonupper],original[...,nonupper])
            t=system.encode_motion(torch.where(r.obs(b),candidate-b['b0']-ident['baseline'][:,None],0.),b['valid'])
            emotions.extend((t['emotion_logits'].argmax(-1)==b['emotion_id']).cpu().tolist())
            outs.append(candidate.cpu());baselines.append(original.cpu())
        predictions[f'{seed}/full']=torch.cat(outs);predictions[f'{seed}/original_local']=torch.cat(baselines)
        metrics[str(seed)]={'candidate':r.populations(predictions[f'{seed}/full'],q),
            'original':r.populations(predictions[f'{seed}/original_local'],q),
            'generated_emotion_accuracy_nonindependent':sum(emotions)/len(emotions)}
    curves={'schema':'regional_aligned_candidate_v1','clip_id':q['clip_id'],'target':q['motion'],'valid':q['valid'],
        'times':q['times'],'channel_mask':q['channel_mask'],'b0':q['b0'],'emotion_id':q['emotion_id'],'speaker_id':q['speaker_id'],
        'predictions':predictions,'noise_seeds':list(r.SEEDS),'decode_steps':steps}
    a.output.mkdir(parents=True)
    save_checkpoint(a.output/'curves.pt',curves)
    report={'schema':'regional_aligned_candidate_v1','metrics':metrics,'distribution':summarize(curves,q['emotion_id']),
        'source_bindings':bindings,'align_checkpoint_sha256':r.sha(path),'identity':r.identity_report(system,data,a.device),
        'nonupper_original_local_exact':True,'development_selected_rule':True,'selection_rule':'Same fixed region rule for every clip/seed; aligned upper from Stage3 renderer, original Stage4 local for other channels in Stage3 renderer',
        'test_loaded':False,'default_replaced':False,'training_performed':False,'script_sha256':r.sha(__file__),
        'deploy':'Need only frozen Stage3 system, Stage4 audio, aligned adapter and independent references; no GT query'}
    save_json(a.output/'evaluation.json',report)
    save_json(a.output/'complete.json',{'curves_sha256':r.sha(a.output/'curves.pt'),'evaluation_sha256':r.sha(a.output/'evaluation.json')})
    print('REGIONAL_CANDIDATE_COMPLETE',flush=True)


if __name__=='__main__':main()
