"""Read-only motion prediction before/after fixed-budget renderer training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_predictable_renderer import PredictableAudioHead, audio_features, sha
from scripts.probe_predictable_motion import score_fields


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('bundle','weights','rrr','pca','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    if args.output.exists():raise FileExistsError('Fresh output required')
    source=torch.load(args.bundle,map_location='cpu',weights_only=False)
    weights=torch.load(args.weights,map_location='cpu',weights_only=False)
    train=source['bundles']['internal'];dev=source['bundles']['external_dev']
    features=audio_features(dev);w=dev['weight'].float();y=dev['motion_bins'].float()
    report={'schema':'predictable_head_drift_v1','scope':'Frozen-basis deterministic dev diagnosis, no fitting or model selection',
        'bundle_sha256':sha(args.bundle),'arms':{}}
    for method in ('rrr','pca'):
        state=weights['states'][method+'_rank8']
        head=PredictableAudioHead(state,train['motion_bins'],train['weight'])
        delta=torch.load(getattr(args,method)/'renderer_delta.pt',map_location='cpu',weights_only=False)
        parts={}
        for stage in ('before','after'):
            if stage=='after':head.load_state_dict(delta['head'],strict=True)
            with torch.no_grad():
                controls=head(features,w)*head.target_scale
                pred=torch.zeros_like(y)
                pred[...,head.channels]=controls@head.basis.T
            parts[stage]=score_fields(pred,y,w,dev['sentence_id'],dev['emotion_id'],dev['groups'],bootstrap=1000)
        report['arms'][method]=parts
        print(json.dumps({'method':method,**{stage:{g:parts[stage]['nonneutral'][g]['r2_against_zero'] for g in ('upper_expression','mouth','jaw17')} for stage in parts}}),flush=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')


if __name__=='__main__':main()
