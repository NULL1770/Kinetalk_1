"""Render fixed clips using all matched arms and the same native clock/audio."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent/'dual_evidence'
PROJECT=ROOT.parents[2]
OUTPUT=PROJECT/'artifacts/v5_dual_timescale_review_20261012'
ARMS=('control','direct','guided')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_json_legacy(path):return json.loads(path.read_text(encoding='utf8'))


def main(emotions):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    receipt_path=OUTPUT/'native_review_receipt.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(
        schema='dual_timescale_native_review_v1',test_loaded=False,promotion=False,videos={})
    reports={a:json.loads((ROOT/'evaluations'/(a+'_full1367')/'report.json').read_text()) for a in ARMS}
    for a in ARMS:
        if sha(ROOT/'runs'/a/'final.pt')!=reports[a]['checkpoint_sha256']:raise ValueError('Checkpoint hash differs')
    for emotion in emotions:
        cid='mead_M025_'+emotion+('_L1_005' if emotion=='neutral' else '_L3_005')
        paths=[ROOT/'evaluations'/(a+'_full1367')/'render_inputs'/(cid+'.npz') for a in ARMS]
        data=[]
        for p in paths:
            with np.load(p,allow_pickle=False) as z:data.append({k:z[k].copy() for k in z.files})
        first=data[0]
        for d in data[1:]:
            if any(not np.array_equal(d[k],first[k]) for k in ('times','valid','channels','channel_mask','noise_seed')):
                raise ValueError('Clock/mask/seed differs')
            if not np.array_equal(d['motions'][:3],first['motions'][:3]):raise ValueError('Baselines differ')
        good=np.flatnonzero(first['valid']);lo,hi=int(good[0]),int(good[-1])+1
        if not first['valid'][lo:hi].all():raise ValueError('Internal clock gap')
        merged=OUTPUT/(emotion+'_inputs.npz')
        values=np.concatenate((first['motions'][:3],*[d['motions'][3:4] for d in data]))[:,lo:hi]
        mask=first['channel_mask'];mask=mask[lo:hi] if mask.ndim==2 else mask
        merged_data=dict(motions=values,mode_names=np.asarray(['GT','B0','Phase53','Matched control','V5 direct','V5 teacher']),
            times=first['times'][lo:hi],valid=first['valid'][lo:hi],channels=first['channels'],channel_mask=mask,
            clip_id=first['clip_id'],noise_seed=first['noise_seed'])
        if merged.exists():
            with np.load(merged,allow_pickle=False) as z:
                if any(not np.array_equal(z[k],v) for k,v in merged_data.items()):raise ValueError('Existing input differs')
        else:np.savez_compressed(merged,**merged_data)
        audio=PROJECT/'final_experiment/data/mead_media_v1/wav'/(cid+'.wav')
        output=OUTPUT/(emotion+'_render')
        if not output.exists():
            subprocess.run([sys.executable,str(PROJECT/'scripts/render_dynamic_rig_comparison.py'),
                '--input',str(merged),'--output',str(output),'--audio',str(audio),'--tile-size','320',
                '--columns','6','--samples','16'],check=True)
        display=read_json_legacy(output/'display_report.json');rig=read_json_legacy(output/'rig_audit.json')
        if display['input_sha256']!=sha(merged) or display['audio_sha256']!=sha(audio):raise ValueError('Existing display input binding differs')
        if display['status']!='complete' or not display['original_blend_unchanged'] or rig['frame_value_max_error']!=0:
            raise ValueError('Rig/display verification failed')
        video=output/'comparison.mp4'
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],capture_output=True,check=True)
        streams=json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries',
            'stream=codec_type,nb_read_frames,r_frame_rate','-of','json',str(video)],capture_output=True,text=True,check=True).stdout)['streams']
        stream=next(s for s in streams if s['codec_type']=='video');frames=hi-lo
        if int(stream['nb_read_frames'])!=frames or stream['r_frame_rate']!='25/1' or not any(s['codec_type']=='audio' for s in streams):
            raise ValueError('Frame/audio/native-clock verification failed')
        indices=sorted(set([round((frames-1)*v) for v in (0,.25,.5,.75,1)]));rows=[]
        for idx in indices:
            with Image.open(output/'frames'/f'{idx+1:06d}.png') as im:
                im=im.convert('RGB');im=im.resize((1440,round(im.height*1440/im.width)));rows.append((idx,im))
        sheet=Image.new('RGB',(1440,sum(im.height+28 for _,im in rows)),'white');draw=ImageDraw.Draw(sheet);y=0
        for idx,im in rows:
            draw.text((8,y+6),f'{emotion} native frame {lo+idx}, t={first["times"][lo+idx]:.2f}s',fill='black');y+=28;sheet.paste(im,(0,y));y+=im.height
        sheet.save(output/'uniform_time_inspection.png')
        receipt['videos'][emotion]=dict(video=str(video),sha256=sha(video),frames=frames,fps=25,full_decode=True,rig_error=0,
            source_hashes=[sha(p) for p in paths],checkpoint_hashes={a:reports[a]['checkpoint_sha256'] for a in ARMS},
            audio_sha256=sha(audio),merged_sha256=sha(merged),native_start_seconds=float(first['times'][lo]),
            removed_invalid_edge_indices=list(range(lo))+list(range(hi,len(first['times']))),inspection_frame_indices=indices)
        receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf8')
        print(json.dumps(dict(status='complete',emotion=emotion,frames=frames,video=str(video))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--emotions',nargs='+',default=['angry','happy'])
    main(p.parse_args().emotions)
