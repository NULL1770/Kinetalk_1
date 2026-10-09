"""Reproducible Phase53 reference figures: real embeddings and native renders."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image

COLORS=['#718096','#d45b47','#9567a3','#42a897','#8379ba','#e6ac32','#528bb8','#d97ca7']
INK='#223247';SUB='#627184';BLUE='#347daf';TEAL='#209488';GOLD='#cb993e';PURPLE='#8670ad'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(fig,out,name):
    for ext in ('png','pdf','svg'):fig.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def tsne(root):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    out=root/'figures';out.mkdir(exist_ok=True)
    z=np.load(root/'data/embeddings.npz',allow_pickle=False)
    meta=json.loads((root/'data/metadata.json').read_text());labels=z['label'];names=meta['classes']
    joined=np.concatenate([z[k] for k in ('gt_embedding','style_only_embedding','joint_embedding')])
    pca=PCA(n_components=50,svd_solver='full').fit(joined)
    coords=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=47,max_iter=1000,n_jobs=2).fit_transform(pca.transform(joined))
    audio=z['audio_g']
    # The shared normalization for motion is the frozen TRAIN probe. No new
    # classifier or model parameter is fitted. PCA/t-SNE are visualization only.
    ga=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=47,max_iter=1000,n_jobs=2).fit_transform(audio)
    np.savez_compressed(root/'data/projection_coordinates.npz',motion=coords,audio_global=ga,labels=labels,speaker=z['speaker'],clip_id=z['clip_id'],pca_components=pca.components_,pca_mean=pca.mean_)
    fig,axes=plt.subplots(1,3,figsize=(13.3,4.25));n=len(labels)
    titles=['(a) Real motion','(b) Phase53 style-only','(c) Phase53 joint']
    for k,ax in enumerate(axes):
        xy=coords[k*n:(k+1)*n]
        for e,name in enumerate(names):ax.scatter(*xy[labels==e].T,s=7,alpha=.67,c=COLORS[e],linewidths=0,label=name)
        ax.set(xlim=(coords[:,0].min()-3,coords[:,0].max()+3),ylim=(coords[:,1].min()-3,coords[:,1].max()+3),xticks=[],yticks=[])
        ax.set_title(titles[k],loc='left',fontsize=12,fontweight='bold',color=INK,pad=10)
        ax.spines[['left','bottom']].set_visible(False)
    handles=[Line2D([],[],marker='o',ls='',color=COLORS[e],label=name,markersize=5) for e,name in enumerate(names)]
    fig.legend(handles=handles,loc='lower center',ncol=8,frameon=False,bbox_to_anchor=(.5,.025))
    fig.suptitle('Emotion structure in a shared frozen motion-feature space',x=.065,y=1.04,ha='left',fontsize=15,fontweight='bold',color=INK)
    fig.text(.065,.945,'1,367 development clips per panel  •  one joint, unlabelled t-SNE projection  •  seed 47',fontsize=9,color=SUB)
    fig.subplots_adjust(left=.045,right=.985,top=.85,bottom=.17,wspace=.1)
    save(fig,out,'01_motion_emotion_tsne')
    fig,axes=plt.subplots(1,2,figsize=(9,4.3))
    for e,name in enumerate(names):axes[0].scatter(*ga[labels==e].T,s=8,alpha=.65,c=COLORS[e],linewidths=0,label=name)
    for i,speaker in enumerate(np.unique(z['speaker'])):
        axes[1].scatter(*ga[z['speaker']==speaker].T,s=8,alpha=.6,c=['#347daf','#cf8c38','#8f75ac'][i],label='Speaker '+str(speaker),linewidths=0)
    for ax,title in zip(axes,['(a) Colour by emotion','(b) Colour by speaker']):
        ax.set_title(title,loc='left',fontweight='bold',color=INK);ax.set(xticks=[],yticks=[]);ax.spines[['left','bottom']].set_visible(False)
    axes[0].legend(frameon=False,ncol=4,fontsize=8,loc='lower center',bbox_to_anchor=(.5,-.17))
    axes[1].legend(frameon=False,ncol=3,fontsize=8,loc='lower center',bbox_to_anchor=(.5,-.1))
    fig.suptitle('Audio global affect representation, g (32D)',x=.065,ha='left',fontsize=14,fontweight='bold',color=INK)
    fig.subplots_adjust(top=.84,bottom=.2,wspace=.17);save(fig,out,'01b_audio_affect_tsne')
    report=dict(seed=47,perplexity=30,max_iter=1000,motion_joint_points=len(coords),clips_per_panel=n,
        pca_dimensions=50,motion_pca_explained_variance=float(pca.explained_variance_ratio_.sum()),
        source_sha256=sha(root/'data/embeddings.npz'),coordinates_sha256=sha(root/'data/projection_coordinates.npz'),
        protocol='All development clips, no class filtering or seed search. Shared TRAIN-fitted frozen probe. Joint unlabelled visualization PCA/t-SNE only; no generator fit.',
        limitations='2D overlap or separation neither certifies generation quality nor proves emotion/content/style disentanglement.')
    (out/'tsne_protocol.json').write_text(json.dumps(report,indent=2))


def box(ax,x,y,w,h,title,body='',color=BLUE,fill='#edf5fb',fs=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.14',fc=fill,ec=color,lw=1.15))
    ax.text(x+w/2,y+h*.68 if body else y+h/2,title,ha='center',va='center',fontsize=fs,fontweight='bold',color=INK)
    if body:ax.text(x+w/2,y+h*.3,body,ha='center',va='center',fontsize=fs-1.6,color=SUB,linespacing=1.35)


def arrow(ax,points,color=INK,dashed=False):
    for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.35,ls='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=11,lw=1.35,color=color,ls='--' if dashed else '-'))


def architecture(root):
    out=root/'figures';out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(16.5,9));ax.set(xlim=(0,17),ylim=(0,9.3));ax.axis('off')
    ax.text(.2,9.05,'KineTalk',fontsize=23,fontweight='bold',color=INK)
    ax.text(2.75,9.05,'Neutral articulation + audio expression + reference motion style',fontsize=13,color=SUB)
    ax.text(.2,8.6,'INFERENCE',fontsize=10,fontweight='bold',color=SUB)
    box(ax,.2,5.8,1.65,1.1,'Query audio','one utterance',color='#8796a6',fill='#f3f6f8')
    box(ax,2.4,7.35,2.45,1,'Content features','HuBERT layer 6 · 768D')
    box(ax,5.4,7.35,3.1,1,'Frozen neutral B0','4 TCN + 2 Transformer\n3-layer motion decoder')
    box(ax,9.1,7.35,2.05,1,'Neutral motion','52 coefficients')
    arrow(ax,[(1.85,6.65),(2.1,6.65),(2.1,7.85),(2.4,7.85)],BLUE)
    arrow(ax,[(4.85,7.85),(5.4,7.85)],BLUE);arrow(ax,[(8.5,7.85),(9.1,7.85)],BLUE)
    box(ax,2.4,5.45,2.45,1.15,'Affect + prosody','emotion2vec 768D\nF0 / RMS / periodicity / voicing',TEAL,'#eaf7f3',9.5)
    box(ax,5.4,5.45,3.1,1.15,'Audio prior p','4 TCN + 2 Transformer\nhidden 128 · Gaussian g/u',TEAL,'#eaf7f3')
    box(ax,9.1,5.45,2.05,1.15,'g32 + u16','global + local\nu: native 25 fps, centred',TEAL,'#eaf7f3',9.5)
    arrow(ax,[(1.85,6.05),(2.4,6.05)],TEAL);arrow(ax,[(4.85,6.05),(5.4,6.05)],TEAL);arrow(ax,[(8.5,6.05),(9.1,6.05)],TEAL)
    box(ax,.2,3.5,2.25,1.1,'Neutral references','2 independent utterances\nreference audio + motion',GOLD,'#fff7e8',9.5)
    box(ax,3,3.5,2.35,1.1,'Reference statistics','motion / B0 / masks',GOLD,'#fff7e8',9.5)
    box(ax,5.9,3.5,2.6,1.1,'Reference encoder','MLPs → style64\nposture32 / response32',GOLD,'#fff7e8',9.5)
    box(ax,9.1,3.5,2.05,1.1,'Style code s','shared-rig motion habits',GOLD,'#fff7e8',9.5)
    for a,b in [(2.45,3),(5.35,5.9),(8.5,9.1)]:arrow(ax,[(a,4.05),(b,4.05)],GOLD)
    box(ax,12.0,4.55,2.45,2.2,'Response decoder','hidden 192 · 4 TCN\naffect / style modulation\n52D expressive residual',PURPLE,'#f3eff9',10)
    box(ax,15.05,5.15,1.7,1.2,'Facial motion','B0 + residual\n+ style offset',PURPLE,'#f3eff9',9.5)
    arrow(ax,[(11.15,7.85),(13.22,7.85),(13.22,6.75)],BLUE)
    arrow(ax,[(11.15,6.05),(12,6.05)],TEAL)
    arrow(ax,[(11.15,4.05),(13.22,4.05),(13.22,4.55)],GOLD)
    arrow(ax,[(14.45,5.75),(15.05,5.75)],PURPLE)
    ax.text(14.4,7.92,'Mean conditions at deployment',fontsize=9,ha='center',color=SUB)
    ax.plot([.2,16.8],[3.04,3.04],color='#d9dfe6',lw=1)
    ax.text(.2,2.72,'TRAINING ONLY',fontsize=10,fontweight='bold',color=SUB)
    box(ax,.2,.97,3.3,1.25,'Motion posterior q','GT − B0, B0, residual displacement\nobservations + detached reference style\n3 TCN + 2 Transformer → g32 / u16',PURPLE,'#f6f3fa',9.5)
    box(ax,4.1,.97,2.55,1.25,'Distribution matching','KL(q || p)\n+ emotion / intensity labels',TEAL,'#edf8f5',9.3)
    box(ax,7.25,.97,4.1,1.25,'Motion reconstruction','posterior sample → response decoder\nposition + 0.5 × adjacent displacement',PURPLE,'#f6f3fa',9.5)
    box(ax,12,.97,4.75,1.25,'Prior / reference adaptation','detached audio conditions → reconstruction\nPhase53: prior, posterior and B0 frozen',GOLD,'#fff7e8',9.5)
    arrow(ax,[(3.5,1.6),(4.1,1.6)],PURPLE,True)
    arrow(ax,[(1.85,.97),(1.85,.6),(9.3,.6),(9.3,.97)],PURPLE,True)
    ax.text(.2,.14,'Solid: inference dataflow     Dashed: training relation     Student receives no direct motion-reconstruction gradient',fontsize=9,color=SUB)
    ax.text(16.8,.14,'Phase53 reference · implementation diagram',ha='right',fontsize=9,color=SUB)
    save(fig,out,'03_model_architecture')


def white_image(path):
    im=Image.open(path).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);return bg.convert('RGB')


def montage(root):
    out=root/'figures';renders=root/'stills_transparent';white=root/'stills_white';white.mkdir(exist_ok=True)
    manifest=json.loads((root/'stills_manifest.json').read_text())
    receipt=json.loads((renders/'render_receipt.json').read_text());assert len(receipt['stills'])==len(manifest['stills'])
    for row in receipt['stills']:
        p=renders/(row['name']+'.png');assert sha(p)==row['image_sha256'];assert Image.open(p).size==(1024,1024)
        white_image(p).save(white/p.name)
    emotions=['happy','angry','fear','disgust','sad','surprise','contempt','neutral']
    methods=['GT','Neutral B0','VOCA-core\n(adapted)','EmoTalk-core\n(adapted)','FaceFormer\n(adapted)','FaceDiffuser\n(adapted, seed42)','KineTalk\nPhase53 joint']
    order=[0,1,2,3,4,6,5]
    fig,axes=plt.subplots(7,8,figsize=(16,13.5))
    for row,mode in enumerate(order):
        for col,e in enumerate(emotions):
            ax=axes[row,col];ax.imshow(Image.open(white/f'{e}_peak_gt_jaw_m{mode}.png'));ax.axis('off')
            if row==0:ax.set_title(e.capitalize(),fontsize=12,color=INK)
            if col==0:ax.text(-.07,.5,methods[row],ha='right',va='center',transform=ax.transAxes,fontsize=10,color=INK)
    fig.suptitle('Emotion and articulation • fixed native timestamps',fontsize=18,fontweight='bold',color=INK,y=.975)
    fig.text(.12,.943,'Eight existing development clips · earliest GT jaw maximum · identical rig / lighting / camera',fontsize=10,color=SUB)
    fig.subplots_adjust(left=.12,right=.99,top=.91,bottom=.03,hspace=.05,wspace=.025);save(fig,out,'02a_emotion_comparison')
    fig,axes=plt.subplots(7,4,figsize=(9.5,13.5))
    for row,mode in enumerate(order):
        for col,pct in enumerate((20,40,60,80)):
            ax=axes[row,col];name=f'happy_sequence_{pct}_m{mode}'
            ax.imshow(Image.open(white/(name+'.png')));ax.axis('off')
            r=next(x for x in manifest['stills'] if x['name']==name)
            if row==0:ax.set_title(f"t = {r['time_seconds']:.2f} s",fontsize=12)
            if col==0:ax.text(-.07,.5,methods[row],ha='right',va='center',transform=ax.transAxes,fontsize=10)
    fig.suptitle('Articulation along one utterance',fontsize=17,fontweight='bold',color=INK,y=.975)
    fig.text(.18,.943,'Fixed 20 / 40 / 60 / 80% native positions; words not annotated',fontsize=9,color=SUB)
    fig.subplots_adjust(left=.19,right=.99,top=.91,bottom=.03,hspace=.05,wspace=.02);save(fig,out,'02b_native_time_sequence')
    style=[x for x in manifest['stills'] if x['name'].startswith('style_')]
    fig,axes=plt.subplots(1,6,figsize=(15,3))
    for ax,row in zip(axes,style):
        ax.imshow(Image.open(white/(row['name']+'.png')));ax.axis('off');ax.set_title(row['method'].replace('candidate_',''),fontsize=9)
    fig.suptitle('Reference motion style • fixed source audio and expression conditions',fontsize=14,fontweight='bold',color=INK)
    fig.subplots_adjust(top=.78,bottom=.02,wspace=.03);save(fig,out,'02c_reference_style')


def teaser(root):
    from scipy.io import wavfile
    out=root/'figures';repo=Path(__file__).resolve().parents[1]
    fig,ax=plt.subplots(figsize=(12.8,8));ax.set(xlim=(0,13),ylim=(0,8));ax.axis('off')
    ax.text(.2,7.7,'From expressive speech to facial motion',fontsize=21,fontweight='bold',color=INK)
    ax.text(.2,7.26,'Audio expression meets neutral articulation and reference motion style',fontsize=11.5,color=SUB)
    ax.text(.2,6.77,'AUDIO INPUT',fontsize=9,fontweight='bold',color=SUB)
    ax.text(8.1,6.77,'GENERATED MOTION',fontsize=9,fontweight='bold',color=SUB)
    box(ax,4.55,2.1,2.3,4.15,'KineTalk','neutral B0\n\naudio g + u\n\nreference style',TEAL,'#edf7f4',16)
    data=json.loads((root/'data/metadata.json').read_text())
    emotions=['happy','angry','sad','surprise']
    for i,e in enumerate(emotions):
        y=5.8-i*1.18;color=COLORS[data['classes'].index(e)]
        cid=data['display'][e]['clip_id']
        audio=repo/f'final_experiment/data/mead_media_v1/wav/{cid}.wav'
        if e=='happy':audio=repo/f'final_experiment/evaluation/render_inputs/{cid}.wav'
        sr,w=wavfile.read(audio);w=w.astype(float)
        if w.ndim>1:w=w.mean(1)
        blocks=np.array_split(w,180);amplitude=np.array([np.sqrt((b*b).mean()) for b in blocks]);amplitude/=max(amplitude.max(),1e-9)
        xx=np.linspace(.45,3.2,len(amplitude));ax.vlines(xx,y-amplitude*.25,y+amplitude*.25,color=color,lw=1)
        ax.text(.45,y-.48,e.capitalize(),fontsize=14,color=color,fontweight='bold')
        arrow(ax,[(3.4,y),(4.55,y)],color)
        arrow(ax,[(6.85,y),(7.65,y)],color)
        im=Image.open(root/'stills_transparent'/f'{e}_peak_gt_jaw_m5.png')
        # The source stills are square.  Keep equal data units here so the
        # teaser never stretches the rendered face horizontally.
        ax.imshow(im,extent=(8.1,9.18,y-.54,y+.54),aspect='equal',zorder=2)
        ax.text(10.25,y,e.capitalize(),fontsize=12,color=color,va='center')
    ax.plot([.2,12.8],[1.33,1.33],color='#d9dfe6')
    ax.text(.2,.88,'Reference-driven style',fontsize=13,fontweight='bold',color=GOLD)
    ax.text(.2,.4,'Same source audio and emotion conditions;\nchange only the neutral reference.',fontsize=9,color=SUB,linespacing=1.5)
    for j,mode in enumerate((1,2,3)):
        x=7.5+j*1.65
        ax.imshow(Image.open(root/'stills_transparent'/f'style_happy_m{mode}.png'),extent=(x,x+1.05,.2,1.25),aspect='equal',zorder=2)
        ax.text(x+.625,.04,['M025 AB','M037 AB','M039 AB'][j],fontsize=8,ha='center',color=SUB)
    save(fig,out,'04_overview_teaser')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['tsne','architecture','montage','teaser'],required=True)
    a=p.parse_args();globals()[a.stage](a.root.resolve())
