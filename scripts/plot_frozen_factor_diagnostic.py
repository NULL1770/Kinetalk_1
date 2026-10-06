"""Display predeclared first-speaker neutral/happy-L3 diagnosis curves."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root, output):
    fig, axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
    clips=['mead_M025_neutral_L1_001','mead_M025_happy_L3_001']
    z=np.load(root/'factors_neutral_v2/fixed_exports.npz',allow_pickle=False)
    colors={'target':'#111111','b0':'#777777','original':'#0072B2','ua_static':'#D55E00',
            'id_own_A':'#009E73','id_own_B':'#56B4E9','id_other_1':'#CC79A7','id_other_2':'#E69F00'}
    for row,cid in enumerate(clips):
        valid=z[cid+'/valid'];t=z[cid+'/times'].copy();t-=t[0]
        for col,(channel,modes,title) in enumerate([
            (17,['target','b0','original','ua_static'],'jawOpen: temporal condition'),
            (17,['original','id_own_A','id_own_B','id_other_1','id_other_2'],'jawOpen: identity reference'),
            (23,['target','original','id_own_A','id_own_B','id_other_1','id_other_2'],'mouthSmileLeft: identity reference')]):
            ax=axes[row,col]
            for mode in modes:
                x=z[cid+'/'+mode][:,channel].clip(0,1).copy();x[~valid]=np.nan
                ax.plot(t,x,label=mode,color=colors[mode],linewidth=1.4,alpha=.9)
            ax.set_title(cid.replace('mead_M025_','')+' | '+title,fontsize=10)
            ax.set_xlabel('native time (s)');ax.set_ylabel('coefficient')
            ax.set_ylim(-.02,1.02);ax.grid(alpha=.15);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Frozen neutral-B0 model | original noise draw42 | same content/global/intensity/noise\n'
                 'Fixed M025 clips, clip_all display; response does not prove correct target identity.',fontsize=12)
    fig.savefig(output,dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.root,a.output)
