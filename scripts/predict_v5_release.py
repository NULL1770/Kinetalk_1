"""Frozen release inference: raw audio + aligned carrier + one own-context ref.

Automatic mode consumes no query label/motion. Emotion control is a separate,
explicit user parameter, never silently selected from an evaluation target.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.joint_factor_repair import V5JointFactorRepair
from kinetalk_b0.protocol import EMOTIONS


@torch.no_grad()
def main(a):
    sha=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    if a.sha256 and sha!=a.sha256:raise ValueError('release checkpoint hash differs')
    model=V5JointFactorRepair.from_checkpoint(torch.load(a.checkpoint,map_location='cpu',weights_only=False)).to(a.device).eval().requires_grad_(False)
    if float(model.flow.sample_gain)!=0:raise ValueError('this release excludes the rejected StateFlow')
    with np.load(a.input,allow_pickle=False) as z:
        if any(key in z for key in ('motion','emotion_id','intensity')):
            raise ValueError('query ground truth is not accepted by release inference')
        keys=('audio_features','h0','b0','valid','times','ref_motion','ref_valid','ref_channel_mask','ref_h0','ref_audio_features')
        x={key:torch.from_numpy(np.array(z[key])).to(a.device) for key in keys}
    if x['audio_features'].shape[-1]!=772 or x['ref_audio_features'].shape[-1]!=772:
        raise ValueError('raw 772D own-audio inputs required; not normalized pilot arrays')
    refs={key:x['ref_'+key] for key in ('motion','valid','channel_mask','h0','audio_features')}
    emo=None
    if a.emotion:
        emo=torch.zeros(len(x['valid']),len(EMOTIONS),device=a.device)
        emo[:,EMOTIONS.index(a.emotion)]=1.
    intensity=None if a.intensity is None else torch.full((len(x['valid']),),a.intensity,device=a.device)
    seeds=[int(seed) for seed in a.seeds.split(',')]
    outputs=[model.predict(x['audio_features'],x['h0'],x['b0'],x['valid'],refs,times=x['times'],seed=seed,
        mode=a.mode,sample_flow=False,emotion=emo,intensity=intensity)['motion'].cpu().numpy() for seed in seeds]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,motions=np.stack(outputs),valid=x['valid'].cpu().numpy(),times=x['times'].cpu().numpy(),seeds=np.array(seeds))
    out.with_suffix('.json').write_text(json.dumps(dict(checkpoint_sha256=sha,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mode=a.mode,
        semantic_mode='automatic' if emo is None else 'explicit_user_control',emotion=a.emotion,
        intensity=a.intensity,seeds=seeds,new_state_flow_enabled=False,query_motion_loaded=False,
        hidden_postprocessing=False),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('checkpoint','input','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--sha256');p.add_argument('--device',default='cuda')
    p.add_argument('--emotion',choices=EMOTIONS);p.add_argument('--intensity',type=float)
    p.add_argument('--seeds',default='47,138,229,320');p.add_argument('--mode',choices=('modal','sample'),default='sample')
    main(p.parse_args())
