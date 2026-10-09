"""Source-bound, unsmoothed style plots; no model or selection by performance."""
import argparse,csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(fig,out,name):
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)


def build(report,out):
    r=json.loads(report.read_text(encoding='utf8'))
    assert not r['test_loaded'] and r['frozen_exact'] and r['people']==25
    out.mkdir(parents=True,exist_ok=True)
    people=sorted(r['references'],key=int)
    names=[r['references'][sid]['speaker'].replace('mead_','') for sid in people]
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    measures=[('own_A_B','mouth/mae','Same-person A/B mouth MAE'),
              ('cross_AB','jaw_closure_disagreement','Cross-person closure change fraction'),
              ('target_direction','mouth/mean_improvement','Mouth target-statistic improvement')]
    rows=[]
    for ax,(kind,metric,label) in zip(axes,measures):
        for method,color,shift in [('parent','#64748b',-.1),('candidate','#2563eb',.1)]:
            d=r['results'][f'{method}/clip_all/{kind}']['by_source_speaker']
            values=[]
            for sid in people:
                v=d[sid]['metrics'].get(metric) if d[sid]['n'] else None
                values.append(np.nan if v is None else v)
                rows.append(dict(source=names[people.index(sid)],role=r['references'][sid]['role'],method=method,
                    metric=metric,n=d[sid]['n'],value=v))
            ax.scatter(np.arange(len(people))+shift,values,label='Phase53' if method=='parent' else 'Phase64',s=24,color=color)
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
        for j,sid in enumerate(people):
            if r['references'][sid]['role']!='seen_held_sentence':ax.axvspan(j-.45,j+.45,color='#fbbf24',alpha=.15,zorder=-1)
    axes[-1].axhline(0,color='#555',lw=.8,ls='--');axes[0].legend(frameon=False,ncol=2)
    axes[-1].set_xticks(range(len(people)),names,rotation=60,ha='right')
    fig.suptitle('25 enrolled speakers: fixed driving audio, independent reference swaps\nShading: 5 unseen speakers; others: 20 seen speakers on held sentences',fontsize=11)
    save(fig,out,'style_25_people')
    with (out/'style_25_people.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    native={}
    emotions=('neutral','angry','contempt','disgust','fear','happy','sad','surprise')
    for key,rec in r['render_inputs'].items():
        p=report.parent/'render_inputs'/rec['path'];assert sha(p)==rec['sha256']
        with np.load(p,allow_pickle=False) as z:
            times=z['times'];motion=z['motions'].copy();labels=z['mode_names'].tolist();channels=z['channels'].tolist()
            assert z['valid'].all() and np.isclose(np.diff(times),.04,rtol=1e-5,atol=1e-7).all()
            cid=str(z['clip_id'].item());support=z['channel_mask']
        ix=[channels.index(c) for c in ('jawOpen','mouthSmileLeft','mouthSmileRight','browInnerUp')]
        assert support[ix].all();motion[1:]=motion[1:].clip(0,1)
        jaw,left,right,brow=[motion[:,:,i] for i in ix]
        traces=np.stack((jaw,(left+right)/2,brow,left-right),axis=-1)
        features=('Jaw opening','Smile mean','Inner brow raise','Smile L - R')
        fig,axes=plt.subplots(2,4,figsize=(14,5.5),sharex=True,sharey='col')
        for j,feature in enumerate(features):
            for k in range(1,6):axes[0,j].plot(times,traces[k,:,j],lw=1,label=labels[k])
            axes[0,j].plot(times,traces[0,:,j],lw=.8,ls='--',color='#555',label='Source GT')
            for k in (6,7):axes[1,j].plot(times,traces[k,:,j],lw=1.3,label=labels[k])
            axes[0,j].set_title(feature);axes[1,j].set_xlabel('Native time (s)')
            for ax in axes[:,j]:ax.grid(axis='y',alpha=.2)
        axes[0,0].set_ylabel('Reference swap');axes[1,0].set_ylabel('Same-person A/B')
        axes[0,0].legend(frameon=False,fontsize=6,ncol=2);axes[1,0].legend(frameon=False,fontsize=7)
        emotion=emotions[int(key)]
        fig.suptitle(f'{emotion}: one audio/expression, five reference identities; GT belongs only to the source',fontsize=11)
        save(fig,out,'style_multi_native_'+emotion)
        with (out/f'style_multi_native_{emotion}.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(['clip_id','mode','native_time_s',*features])
            for k,label in enumerate(labels):
                for i,t in enumerate(times):w.writerow([cid,label,float(t),*traces[k,i].tolist()])
        native[emotion]=dict(source_sha256=sha(p),clip_id=cid,frames=len(times),smoothed=False,retimed=False)
    receipt=dict(report_sha256=sha(report),native=native,test_loaded=False,
        interpretation='Shared-rig reference-conditioned motion tendencies; selected qualitative sources; no unseen-25 or counterfactual-GT claim.')
    (out/'sources.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.report,a.output)
