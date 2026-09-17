"""Fixed historical nine-clip export from saved repair trajectories only."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.export_full_staged_examples import bind_selection, ARKIT_NAMES, local_child
from scripts.audit_temporal_repair import sha, read, validate_curves, metadata_equal

MODES=('GT','previous run12','aligned local','direct upper','soft-state upper','original local')


def export(root,old,previous,output):
    if output.exists():raise FileExistsError('Fresh output required')
    curves={};bindings={}
    for mode,path in [('align',root/'align'),('direct',root/'direct'),('soft',root/'soft'),('old',old/'dynamics')]:
        file=path/'curves.pt';digest=sha(file)
        if read(path/'complete.json')['curves_sha256']!=digest:raise ValueError('Curve binding changed')
        curves[mode]=torch.load(file,map_location='cpu',weights_only=False,mmap=True)
        validate_curves(curves[mode],phase=mode if mode!='old' else None)
        bindings[mode]={'path':str(file.resolve()),'sha256':digest}
    reference=curves['align']
    for key,curve in curves.items():metadata_equal(reference,curve,compare_b0=key!='old')
    prior,picks=bind_selection(previous,reference)
    ids=[p['index'] for p in picks]
    data=[reference['target'],curves['old']['predictions']['42/full'],reference['predictions']['42/full'],
          curves['direct']['predictions']['42/full'],curves['soft']['predictions']['42/full'],reference['predictions']['42/original_local']]
    motions=torch.stack([v[ids] for v in data]).numpy()
    times,valid,channels=[reference[k][ids].numpy() for k in ('times','valid','channel_mask')]
    output.mkdir(parents=True);(output/'video_npz').mkdir();(output/'audio').mkdir()
    np.savez_compressed(output/'nine_clip_curves.npz',motions=motions,times=times,valid=valid,channel_mask=channels,
        channels=np.asarray(ARKIT_NAMES),mode_names=np.asarray(MODES),clip_id=np.asarray([p['clip_id'] for p in picks]))
    videos={v['clip']['clip_id']:v for v in prior['videos']};jobs=[];seen=set()
    for j,pick in enumerate(picks):
        if pick['speaker'] in seen:continue
        seen.add(pick['speaker']);name=pick['speaker'];video=videos[pick['clip_id']];audio=local_child(previous,video['audio_copy'])
        binding=video['audio_binding']
        if sha(audio)!=binding['audio_sha256']:raise ValueError('Audio hash differs')
        shutil.copy2(audio,output/'audio'/f'{name}.wav')
        np.savez_compressed(output/'video_npz'/f'{name}.npz',motions=motions[:,j],times=times[j],valid=valid[j],channel_mask=channels[j],
            channels=np.asarray(ARKIT_NAMES),mode_names=np.asarray(MODES),clip_id=np.asarray(pick['clip_id']))
        jobs.append({'speaker':name,'input':f'video_npz/{name}.npz','audio':f'audio/{name}.wav','audio_offset_seconds':binding['audio_offset_s']})
    report={'mode_names':MODES,'source_bindings':bindings,'nine_plot_clips':picks,'selection_uses_metadata_only':True,
            'outcome_based_selection':False,'jobs':jobs,'amplitude_rescaling':False,'raw_clamped':False,'test_loaded':False}
    (output/'provenance.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(output/'nine_clip_curves.npz',allow_pickle=False) as z:
        motions=z['motions'];times=z['times'];valid=z['valid'];channels=z['channel_mask'];modes=z['mode_names'].tolist()
    picks=read(output/'provenance.json')['nine_plot_clips']
    for centered in (False,True):
        fig,axes=plt.subplots(4,9,figsize=(25,11),squeeze=False)
        for c,p in enumerate(picks):
            for r,ch in enumerate((43,41,5,17)):
                mask=valid[c]&channels[c,ch];a=axes[r,c]
                for i,name in enumerate(modes):
                    values=motions[i,c,:,ch].astype(float)
                    if centered:values-=values[mask].mean()
                    values[~mask]=np.nan;a.plot(times[c]-times[c,0],values,label=name,lw=1)
                if r==0:a.set_title(p['speaker'].replace('mead_','')+'/'+p['emotion'],fontsize=8)
                if c==0:a.set_ylabel(ARKIT_NAMES[ch])
                if r==3:a.set_xlabel('seconds')
                a.grid(alpha=.15)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=6)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output/('centered.png' if centered else 'raw.png'),dpi=110);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plot-only',action='store_true')
    for name in ('repair-root','old-run','previous-visual','output'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if not a.plot_only:export(a.repair_root,a.old_run,a.previous_visual,a.output)
    plot(a.output);print('REPAIR_EXPORT_COMPLETE')
