"""Fit an independent motion-only probe on train; select only on validation."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.packed_trainval_cache import load_packed
from kinetalk_b0.emotion_probe import MotionEmotionProbe, motion_features, classification_metrics
from scripts.train_formal_predictable_projection import save_json, save_checkpoint
from scripts.extract_emotion2vec_pilot import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--hidden',type=int,default=128);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed);data=load_packed(args.data)
    args.output.mkdir(parents=True,exist_ok=False)
    support=data['splits']['train']['channel_mask'].all(0)
    features={};labels={}
    for role,split in data['splits'].items():
        if not split['channel_mask'][:,support].all():raise ValueError('Probe fixed train channel support absent')
        rows=[]
        for i in range(len(split['valid'])):
            b=split.batch(torch.tensor([i]),keys=('motion','valid'))
            rows.append(motion_features(b['motion'][0][:,support],b['valid'][0]))
        features[role]=torch.stack(rows);labels[role]=split['emotion_id'].clone()
        print(json.dumps({'event':'features','role':role,'clips':len(rows)}),flush=True)
    train=features['train'];target=labels['train'];names=data['config']['data']['emotion_classes']
    model=MotionEmotionProbe(train.shape[-1],args.hidden,len(names));model.fit_normalization(train);model.to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    counts=torch.bincount(target,minlength=len(names)).float();weights=(len(target)/len(names)/counts).to(args.device)
    best=-1.;history=[];generator=torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        for ids in torch.randperm(len(train),generator=generator).split(128):
            logits=model(train[ids].to(args.device));loss=torch.nn.functional.cross_entropy(logits,target[ids].to(args.device),weight=weights)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():pred=model(features['validation'].to(args.device)).argmax(-1).cpu()
        score=classification_metrics(labels['validation'],pred,names)
        history.append({'epoch':epoch+1,**score})
        if score['macro_f1']>best:
            best=score['macro_f1'];best_epoch=epoch+1
            save_checkpoint(args.output/'probe.pt',dict(kind='independent_native_motion_statistics_probe_v1',
                model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},hidden=args.hidden,feature_dim=train.shape[-1],classes=names,
                channel_support=support,test_used_for_selection=False,epoch=best_epoch,
                train_manifest_sha256=data['provenance']['manifest_sha256'],selection='validation macro-F1; earliest tie',
                normalization='train only',generator_outputs_used_for_fitting=False,source_sha256=sha(__file__)))
        print(json.dumps({'event':'probe_epoch','epoch':epoch+1,'macro_f1':score['macro_f1'],'best':best}),flush=True)
    save_json(args.output/'report.json',dict(status='complete',best_epoch=best_epoch,validation_macro_f1=best,
        history=history,test_loaded=False,checkpoint_sha256=sha(args.output/'probe.pt')))


if __name__=='__main__':main()
