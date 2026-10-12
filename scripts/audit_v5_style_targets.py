"""Independent single-reference target-style direction on matched development.

Each donor utterance matches sentence/emotion/intensity, but is a different
performance: only clip statistics are compared, never aligned frame targets.
Native prediction receives audio/B0 and one independent neutral reference;
query/donor motion enters scoring after generation. All matches are declared
before model outputs and balanced by query speaker and emotion.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v4.unified_model_v4 import KineTalkV4Config,KineTalkV4Model
from scripts.load_candidate_model import load_candidate_model
from scripts.train_expression_response import configure,load_runtime,sha,state_digest,write
from scripts.train_unified_v4 import select_balanced,build_base_cache,attach_base
from scripts.audit_reference_style_swap import summary,change,REGIONS,STATS,correlation
from scripts.evaluate_expression_response import aggregate,finite
from kinetalk_b0.emotion_probe import MotionEmotionProbe,motion_features


def independent_pairs(q,refs,limit):
    groups={};ambiguous=[]
    for i in range(len(q['clip_id'])):
        key=(str(q['sentence_id'][i]),int(q['emotion_id'][i]),int(q['intensity_id'][i]))
        groups.setdefault(key,[]).append(i)
    pairs=[]
    for key,ids in groups.items():
        people=[int(q['speaker_id'][i]) for i in ids]
        if len(people)!=len(set(people)):
            ambiguous.append(key);continue
        for i in ids:
            sid=int(q['speaker_id'][i])
            for j in ids:
                donor=int(q['speaker_id'][j])
                if donor==sid:continue
                if any(q['clip_id'][k] in refs[p]['clip_id']
                       or q['sentence_id'][k] in refs[p].get('sentence_id',[])
                       for k in (i,j) for p in (sid,donor)):
                    continue
                pairs.append((i,j))
    balanced={}
    import hashlib
    for i,j in pairs:
        balanced.setdefault((int(q['speaker_id'][i]),int(q['emotion_id'][i])),[]).append((i,j))
    for values in balanced.values():
        values.sort(key=lambda ij:hashlib.sha256((q['clip_id'][ij[0]]+'|'+q['clip_id'][ij[1]]).encode()).hexdigest())
    chosen=[]
    while len(chosen)<limit and any(balanced.values()):
        for group in sorted(balanced):
            if balanced[group]:chosen.append(balanced[group].pop(0))
            if len(chosen)==limit:break
    if len(chosen)<limit:raise ValueError('Insufficient exact independent style targets')
    return chosen,ambiguous


def direction(own,swapped,donor,mask,scales):
    result={}
    for region,channels in REGIONS.items():
        observed=[k for k in channels if mask[k]]
        if not observed:continue
        scale=scales[observed]
        for axis,name in enumerate(STATS):
            before=np.mean(((own[axis,observed]-donor[axis,observed])/scale)**2)
            after=np.mean(((swapped[axis,observed]-donor[axis,observed])/scale)**2)
            result[region+'/'+name+'_before']=float(before)
            result[region+'/'+name+'_after']=float(after)
            result[region+'/'+name+'_improvement']=float(before-after)
            result[region+'/'+name+'_improved']=float(after<before)
    return result


@torch.no_grad()
def main(args):
    configure(47);device=torch.device(args.device)
    root=Path(args.output)
    if root.exists():raise FileExistsError('Fresh target-style audit required')
    root.mkdir(parents=True)
    binding=json.loads(Path(args.binding).read_text());data,base=load_runtime(binding,device)
    q=data['splits']['validation'];pairs,ambiguous=independent_pairs(q,data['refs'],args.limit)
    cache=build_base_cache(q,base,sorted({i for i,j in pairs}),device)
    networks=[]
    for spec in binding['probes']:
        if sha(spec['path'])!=spec['sha256']:raise ValueError('Probe hash differs')
        saved=torch.load(spec['path'],map_location='cpu',weights_only=False)
        net=MotionEmotionProbe(saved['feature_dim'],saved['hidden'],len(data['config']['data']['emotion_classes'])).eval()
        net.load_state_dict(saved['model'])
        networks.append((net,saved.get('feature_mask',torch.ones(saved['feature_dim'],dtype=torch.bool))))
    def classify(x,valid):
        values=motion_features(torch.tensor(x[:,:51],dtype=torch.float32),torch.tensor(valid))
        return [int(net(values[None,mask]).argmax(-1)) for net,mask in networks]
    for checkpoint in args.checkpoint:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);s=ck['model']
        model=load_candidate_model(ck,device);model.eval().requires_grad_(False)
        before=state_digest(model),state_digest(base.stage1);rows=[]
        for ordinal,(i,j) in enumerate(pairs):
            batch=attach_base(q.batch(torch.tensor([i]),device,
                keys=('audio_features','content','valid','channel_mask','times')),[i],cache)
            assert 'motion' not in batch
            refs={}
            for name,speaker in [('own',int(q['speaker_id'][i])),('donor',int(q['speaker_id'][j]))]:
                r=data['refs'][speaker]
                refs[name]={k:r[k][0:1].to(device) for k in ('motion','valid','channel_mask')}
            outputs={name:model.predict(batch['audio_features'],batch['h0'],batch['b0'],batch['valid'],ref,
                times=batch['times'],seed=47+ordinal) for name,ref in refs.items()}
            # Complete generation before loading either observed target sequence.
            target=q.batch(torch.tensor([j]),keys=('motion','valid','channel_mask','times'))
            valid=batch['valid'][0].cpu().numpy();times=batch['times'][0].cpu().numpy()
            query_mask=batch['channel_mask'][0].cpu().numpy();mask=query_mask&target['channel_mask'][0].numpy();mask[51]=False
            for policy in ('raw','clip_all'):
                a=outputs['own']['motion'][0].cpu().numpy();b=outputs['donor']['motion'][0].cpu().numpy()
                if policy=='clip_all':a=np.clip(a,0,1);b=np.clip(b,0,1)
                own=summary(a,valid,query_mask,times);swapped=summary(b,valid,query_mask,times)
                gt=summary(target['motion'][0].numpy(),target['valid'][0].numpy(),
                    target['channel_mask'][0].numpy(),target['times'][0].numpy())
                pa,pb=classify(a,valid),classify(b,valid)
                semantic={}
                for k,(left,right) in enumerate(zip(pa,pb)):
                    semantic['probe'+str(k+1)+'_label_preserved']=float(left==right)
                    semantic['probe'+str(k+1)+'_emotion_correct_after']=float(right==int(q['emotion_id'][i]))
                row=dict(source_clip=q['clip_id'][i],target_clip=q['clip_id'][j],policy=policy,
                    source_speaker=int(q['speaker_id'][i]),target_speaker=int(q['speaker_id'][j]),
                    emotion=int(q['emotion_id'][i]),intensity=int(q['intensity_id'][i]),
                    sentence=str(q['sentence_id'][i]),deployment_query_motion=False,score='separate-performance clip statistics',
                    measurements={**direction(own,swapped,gt,mask,model.scales.cpu().numpy()),
                                  **change(a,b,valid,query_mask,times),**semantic})
                rows.append(row)
            write(root/'state.json',dict(status='scoring',pairs=ordinal+1,total=len(pairs),checkpoint=sha(checkpoint)))
        if before!=(state_digest(model),state_digest(base.stage1)):raise RuntimeError('Style audit mutated weights')
        report=dict(schema='v5_target_style_direction_v1',matched_pairs=len(pairs),test_loaded=False,promotion=False,
            checkpoint_sha256=sha(checkpoint),binding_sha256=sha(args.binding),source_sha256=sha(__file__),
            selection=pairs,ambiguous_keys_excluded=ambiguous,reference='one neutral slot0 per generation',
            query_motion_role='no query target loaded for deployment; donor motion scoring only',
            summary={policy:aggregate([r['measurements'] for r in rows if r['policy']==policy]) for policy in ('raw','clip_all')},
            records=rows,limits=['Exact sentence/emotion/intensity matches, different utterance performances; no framewise counterfactual truth.',
                'Three development speakers are limited style generalization, not a25-person or HDTF external benchmark.',
                'Whole-clip frozen probes and jaw closure agreement do not measure phonetic intelligibility or emotional timing.',
                'Target-style direction is a diagnostic and does not establish complete factor disentanglement.'])
        write(root/(sha(checkpoint)[:12]+'.json'),finite(report))
    write(root/'complete.json',dict(status='complete',matched_pairs=len(pairs),checkpoints=len(args.checkpoint),test_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',required=True);p.add_argument('--checkpoint',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=64);p.add_argument('--device',default='cpu')
    main(p.parse_args())
