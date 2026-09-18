"""Package 32 group-state diagnostics and hash-verified original event pixels."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_brow_states_v3 import STATE_NAMES, GROUP_NAMES, SCHEMA
from scripts.extract_brow_events_v2 import GROUPS, write_json, sha
from scripts.package_sparse_brow_teacher import selected_frames, make_strip, checked, href

COLORS = ['#ddd', '#8dd3c7', '#fdb462', '#fb8072', '#80b1d3', '#ccebc5', '#bc80bd', '#777']


def checked_source_metadata(tracking, nuisance):
    """Bind nuisance diagnostics and review metadata to the same tracking run."""
    tracking, nuisance = Path(tracking), Path(nuisance)
    read = lambda p: json.loads(p.read_text(encoding='utf8'))
    track_manifest = read(tracking/'manifest.json')
    nuisance_manifest = read(nuisance/'manifest.json')
    for root, manifest, names in (
        (tracking, track_manifest, ('selection.json', 'provenance.json')),
        (nuisance, nuisance_manifest, ('protocol.json',)),
    ):
        for name in names:
            if name not in manifest:
                raise ValueError('source metadata missing from manifest: '+name)
            checked(root/name, manifest[name])
    selection = read(tracking/'selection.json')
    provenance = read(tracking/'provenance.json')
    selection_hash = sha(tracking/'selection.json')
    if provenance.get('selection_sha256') != selection_hash:
        raise ValueError('tracking provenance selection mismatch')
    protocol = read(nuisance/'protocol.json')
    expected = {
        'source_manifest_sha256': sha(tracking/'manifest.json'),
        'selection_sha256': selection_hash,
        'source_provenance_sha256': sha(tracking/'provenance.json'),
    }
    for key, digest in expected.items():
        if protocol.get(key) != digest:
            raise ValueError('nuisance tracking source mismatch: '+key)
    return track_manifest, nuisance_manifest, selection, provenance


def package(states, state_audit, tracking, nuisance, ffmpeg, output):
    states, state_audit, tracking, nuisance, ffmpeg, output = map(Path, (states, state_audit, tracking, nuisance, ffmpeg, output))
    if output.exists() and any(output.iterdir()): raise FileExistsError('fresh review required')
    read = lambda p: json.loads(p.read_text(encoding='utf8'))
    rows = read(states/'clips.json'); score = read(state_audit)
    if score['schema'] != SCHEMA+'_audit' or sha(states/'manifest.json') != score['source_manifest_sha256']:
        raise ValueError('state audit mismatch')
    for name, rec in read(states/'manifest.json').items(): checked(states/name, rec)
    track_manifest, nuisance_manifest, selection, provenance = checked_source_metadata(tracking, nuisance)
    if sha(ffmpeg) != provenance['ffmpeg_sha256']: raise ValueError('RGB decoder differs')
    selected = {r['clip_id']: r for r in selection['clips']}
    if set(selected) != {r['source_id'] for r in rows}: raise ValueError('selection mismatch')
    output.mkdir(parents=True); (output/'curves').mkdir(); (output/'pixels').mkdir()
    sections, pixel_records = [], []
    for row in rows:
        cid = row['source_id']; trp = tracking/'arrays'/(cid+'_fresh_forward.npz')
        checked(trp, track_manifest['arrays/'+trp.name])
        with np.load(trp, allow_pickle=False) as z: tr = {k: z[k].copy() for k in z.files}
        with np.load(states/'arrays'/(cid+'.npz'), allow_pickle=False) as z: a = {k: z[k].copy() for k in z.files}
        npth = nuisance/'arrays'/(cid+'.npz'); checked(npth, nuisance_manifest['arrays/'+npth.name])
        with np.load(npth, allow_pickle=False) as z: nu = {k: z[k].copy() for k in z.files}
        times = tr['times']; fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
        for j,g in enumerate(GROUP_NAMES):
            raw = a['raw5'][:, GROUPS[g]].mean(1); sm = a['smooth5'][:, GROUPS[g]].mean(1)
            axes[j].plot(times, np.where(a['valid'],raw,np.nan), c='#a0a0a0', lw=.7, label='raw')
            axes[j].plot(times, np.where(a['valid'],sm,np.nan), c='#213c67', lw=1, label='SG5/2')
            axes[j].plot(times,a['reconstruction'][:,j], c='#dc6b26', lw=1.2, label='piecewise target-endpoint reconstruction')
            axes[j].set_ylim(-.01,1.01); axes[j].set_ylabel(g); axes[j].legend(fontsize=7,loc='upper right')
            labels = a['labels'][:,j]
            for f, code in enumerate(labels):
                axes[2].plot([times[f], times[f]+.04],[1-j,1-j],color=COLORS[int(code)],lw=7,solid_capstyle='butt')
        axes[2].set_ylim(-.5,1.5); axes[2].set_yticks([0,1],['down state','raise state'])
        axes[3].plot(times,nu['nuisance_values'][:,0],label='blink L')
        axes[3].plot(times,nu['nuisance_values'][:,1],label='blink R'); axes[3].set_ylim(0,1); axes[3].legend(fontsize=7)
        for j, name in enumerate(['yaw','pitch','roll']): axes[4].plot(times,nu['nuisance_values'][:,4+j],label=name)
        axes[4].legend(fontsize=7); axes[4].set_ylabel('pose deg'); axes[4].set_xlabel('native seconds; nuisance = same tracker only')
        fig.suptitle(cid+' | fixed coefficient axis; reconstruction is NOT audio prediction',fontsize=10)
        fig.tight_layout(); fig.savefig(output/'curves'/(cid+'.png'),dpi=110); plt.close(fig)
        episodes = [(g,i,e) for g in GROUP_NAMES for i,e in enumerate(row['groups'][g]['episodes'])]
        cards=[]
        if episodes:
            import cv2
            video=Path(selected[cid]['video'])
            if sha(video)!=selected[cid]['video_sha256']: raise ValueError('video changed')
            cap=cv2.VideoCapture(str(video)); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
            ids=sorted({f for _,_,e in episodes for f in [e['start'],(e['start']+e['peak'])//2,e['peak'],(e['release']+e['stop']-1)//2,e['stop']-1]})
            rgb=selected_frames(video,ids,w,h,ffmpeg)
            for f in ids:
                if hashlib.sha256(rgb[f].tobytes()).hexdigest()!=str(tr['rgb_sha256'][f]): raise ValueError('original RGB changed')
            lm=tr['landmarks'][tr['valid'],:,:2]; low=np.nanmin(lm,(0,1)); high=np.nanmax(lm,(0,1)); span=high-low
            low=np.maximum(0,low-.12*span); high=np.minimum(1,high+.12*span)
            box=(int(low[0]*w),int(low[1]*h),int(high[0]*w),int(high[1]*h))
            for g,i,e in episodes:
                frames=[e['start'],(e['start']+e['peak'])//2,e['peak'],(e['release']+e['stop']-1)//2,e['stop']-1]
                dest=output/'pixels'/f'{cid}_{g}_{i}.png'
                make_strip(rgb,frames,['observed start','rise mid','peak','fall mid','observed end'],box,dest)
                cards.append(f'<p>{g} {e["direction"]:+d}; left/right censored={e["left_censored"]}/{e["right_censored"]}; conflict={e["conflict"]}</p><img loading="lazy" src="{href(dest,output)}">')
                pixel_records.append({'source_id':cid,'group':g,'episode':i,'frames':frames,'rgb_sha256':[str(tr['rgb_sha256'][f]) for f in frames],'file':href(dest,output)})
        source=html.escape(Path(selected[cid]['video']).resolve().as_uri(),quote=True)
        sections.append(f'<section id="{cid}"><h2>{cid}</h2><details><summary>原视频（仅监督检查）</summary><video controls preload="none" src="{source}"></video></details><img loading="lazy" src="curves/{cid}.png">'+''.join(cards)+'</section>')
    legend=' '.join(f'<span style="color:{COLORS[i]};background:#222;padding:4px">{n}</span>' for i,n in enumerate(STATE_NAMES))
    intro='<h1>删失状态与眉眼共现审计</h1><p><b>这是动作监督检查，不是新训练模型。</b>早期85.1%覆盖率已作废。橙线使用真实动作边界和端点作分段五次重构，与静态中位数拼接不保证连续；只验证局部表示。同跟踪器眨眼/头姿不是独立真值。</p>'
    gr=score['groups']
    intro+=f'<p>抬眉/压眉阶段识别覆盖：{gr["raise"]["identified_phase_fraction"]:.2%} / {gr["down"]["identified_phase_fraction"]:.2%}（按各组有效内帧）；完整且无冲突事件{gr["raise"]["complete_nonconflicting_episodes"]} / {gr["down"]["complete_nonconflicting_episodes"]}个。静态眉形和方向已知但阶段未知的变化另列，未训练音频头。</p><p>'+legend+'</p>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>眉部状态监督修正</title><style>body{max-width:1350px;margin:24px auto;padding:20px;font:16px/1.6 system-ui;background:#eef2f7;color:#172235}section{background:white;padding:20px;margin:24px 0;border-radius:10px}img{width:100%}video{max-width:700px;width:100%}h2{font-size:18px}</style>'+intro+''.join(sections)+'</html>'
    (output/'index.html').write_text(page,encoding='utf8')
    write_json(output/'pixels_manifest.json',{'schema':SCHEMA+'_pixel_review','source_audit_sha256':sha(state_audit),'event_windows':pixel_records,'original_rgb_hashes_verified':True,'independent_blind_review_completed':False})
    return {'clips':len(rows),'episode_windows':len(pixel_records),'path':str(output/'index.html')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('states','state-audit','tracking','nuisance','ffmpeg','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); print(json.dumps(package(a.states,a.state_audit,a.tracking,a.nuisance,a.ffmpeg,a.output)))
