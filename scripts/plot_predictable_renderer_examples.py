"""Prespecified examples from stored generation, never ranked by prediction quality."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_predictable_renderer import center


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    ref=torch.load(args.run/'validation_reference.pt',map_location='cpu',weights_only=False)
    q=ref['q'];full=torch.load(args.run/'final_seed42_curves.pt',map_location='cpu',weights_only=False)['motion']
    old=torch.load(args.run/'original_seed42_curves.pt',map_location='cpu',weights_only=False)['motion']['full']
    picks=[]
    for eid,name in ((0,'neutral'),(1,'angry'),(5,'happy'),(6,'sad')):
        eligible=(q['emotion_id']==eid).nonzero(as_tuple=True)[0].tolist()
        idx=min(eligible,key=lambda i:q['clip_id'][i]);picks.append({'index':idx,'clip_id':q['clip_id'][idx],'emotion':name})
    channels=[(41,'browDownLeft'),(43,'browInnerUp'),(5,'eyeSquintLeft'),(6,'eyeWideLeft'),(17,'jawOpen')]
    for pick in picks:
        i=pick['index'];v=q['valid'][i];t=q['times'][i][v];t=t-t[0]
        fig,axes=plt.subplots(5,1,figsize=(11,10),sharex=True)
        for ax,(ch,label) in zip(axes,channels):
            for data,color,caption,width in [(q['motion'],'black','observed motion',2),(old,'#94a3b8','original audio',1),
                    (full['full'],'#2563eb','new audio',1.5),(full['zero'],'#d97706','zero local',1.2),(full['oracle'],'#16a34a','motion oracle',1)]:
                y=data[i,v,ch].double();y=y-y.mean()
                ax.plot(t,y,color=color,label=caption,lw=width)
            ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[0].legend(ncol=5,fontsize=8)
        axes[-1].set_xlabel('Time (s); each curve has its own clip mean removed')
        fig.suptitle(pick['clip_id']+' | fixed noise 42; first clip by ID in this emotion',fontsize=10)
        fig.tight_layout();fig.savefig(args.output/(pick['emotion']+'.png'),dpi=130);plt.close(fig)
    (args.output/'selection.json').write_text(json.dumps({'selection':'First lexicographic clip ID per emotion, before viewing curves; fixed channels and seed42',
        'scope':'Coefficient traces only, not face rendering or tracking-quality validation','clips':picks},indent=2),encoding='utf8')


if __name__=='__main__':main()
