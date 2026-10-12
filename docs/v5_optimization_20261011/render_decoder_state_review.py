"""Render three matched state-coordinate arms with native baselines and audio."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OUTPUT=PROJECT/'artifacts/v5_decoder_state_review_20261012'
ARMS=('state_raw_direct','state_decoder_direct','state_decoder_privileged')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():raise FileExistsError('Fresh review directory required')
    reports={a:json.loads((ROOT/(a+'_full_eval_report.json')).read_text()) for a in ARMS}
    for arm in ARMS:
        if sha(ROOT/(arm+'_final.pt'))!=reports[arm]['checkpoint_sha256']:raise ValueError('Checkpoint/report differs')
    OUTPUT.mkdir();receipt=dict(schema='decoder_state_native_review_v1',test_loaded=False,promotion=False,videos={})
    for emotion,cid in [('angry','mead_M025_angry_L3_005'),('happy','mead_M025_happy_L3_005')]:
        paths=[ROOT/(arm+'_render_inputs')/(cid+'.npz') for arm in ARMS]
        data=[]
        for p in paths:
            with np.load(p,allow_pickle=False) as z:data.append({k:z[k].copy() for k in z.files})
        first=data[0]
        for d in data[1:]:
            if any(not np.array_equal(d[k],first[k]) for k in ('times','valid','channels','channel_mask','clip_id','noise_seed')):
                raise ValueError('Native clock/mask/seed differs')
            if not np.array_equal(d['motions'][:3],first['motions'][:3]):raise ValueError('Baselines differ')
        good=np.flatnonzero(first['valid']);lo,hi=int(good[0]),int(good[-1])+1
        if not first['valid'][lo:hi].all():raise ValueError('Internal gaps require explicit display accounting')
        motions=np.concatenate((first['motions'][:3],*[d['motions'][3:4] for d in data]))[:,lo:hi]
        cm=first['channel_mask'];cm=cm[lo:hi] if cm.ndim==2 else cm
        merged=OUTPUT/(emotion+'_inputs.npz')
        np.savez_compressed(merged,motions=motions,mode_names=np.asarray(['GT','B0','Phase53','Raw direct','Decoder direct','Decoder teacher']),
            times=first['times'][lo:hi],valid=first['valid'][lo:hi],channels=first['channels'],channel_mask=cm,
            clip_id=first['clip_id'],noise_seed=first['noise_seed'])
        audio=PROJECT/'final_experiment/data/mead_media_v1/wav'/(cid+'.wav');output=OUTPUT/(emotion+'_render')
        subprocess.run([sys.executable,str(PROJECT/'scripts/render_dynamic_rig_comparison.py'),
            '--input',str(merged),'--output',str(output),'--audio',str(audio),
            '--tile-size','320','--columns','6','--samples','16'],check=True)
        display=json.loads((output/'display_report.json').read_text(encoding='utf8'));rig=json.loads((output/'rig_audit.json').read_text())
        if display['status']!='complete' or not display['original_blend_unchanged'] or rig['frame_value_max_error']!=0:
            raise ValueError('Rig/display verification failed')
        video=output/'comparison.mp4';subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],capture_output=True,check=True)
        streams=json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_type,nb_read_frames,r_frame_rate',
            '-of','json',str(video)],capture_output=True,text=True,check=True).stdout)['streams']
        stream=next(s for s in streams if s['codec_type']=='video');frames=hi-lo
        if int(stream['nb_read_frames'])!=frames or stream['r_frame_rate']!='25/1' or not any(s['codec_type']=='audio' for s in streams):
            raise ValueError('Incomplete video/audio/native clock')
        indices=sorted(set([0,round((frames-1)*.25),round((frames-1)*.5),round((frames-1)*.75),frames-1]))
        rows=[]
        for idx in indices:
            with Image.open(output/'frames'/f'{idx+1:06d}.png') as im:
                im=im.convert('RGB');im=im.resize((1200,round(im.height*1200/im.width)));rows.append((idx,im))
        sheet=Image.new('RGB',(1200,sum(im.height+28 for _,im in rows)),'white');draw=ImageDraw.Draw(sheet);y=0
        for idx,im in rows:
            draw.text((8,y+6),f'{emotion} native frame {lo+idx}  t={float(first["times"][lo+idx]):.2f}s',fill='black');y+=28
            sheet.paste(im,(0,y));y+=im.height
        sheet.save(output/'uniform_time_inspection.png')
        receipt['videos'][emotion]=dict(video=str(video),sha256=sha(video),frames=frames,fps=25,full_decode=True,
            source_hashes=[sha(p) for p in paths],checkpoint_hashes={a:reports[a]['checkpoint_sha256'] for a in ARMS},
            merged_sha256=sha(merged),audio_sha256=sha(audio),removed_invalid_edge_indices=list(range(lo))+list(range(hi,len(first['times']))),
            native_start_seconds=float(first['times'][lo]),native_end_seconds=float(first['times'][hi-1]),rig_error=0,inspection_frame_indices=indices)
        (OUTPUT/'native_review_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(dict(status='complete',output=str(OUTPUT),videos=2)))


if __name__=='__main__':main()
