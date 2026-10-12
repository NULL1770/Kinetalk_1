"""Native fixed-clip GT/B0/Phase53/v5/v6 comparison; source coefficients unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image,ImageDraw

PROJECT=Path(__file__).resolve().parents[2]
EVIDENCE=Path(__file__).parent/'evidence'
OLD=PROJECT/'docs/v5_optimization_20261011/dual_evidence/evaluations/control_full1367'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main(a):
    root=PROJECT/'artifacts'/a.output_name
    root.mkdir(parents=True,exist_ok=True)
    receipt=root/'native_review_receipt.json'
    saved=json.loads(receipt.read_text(encoding='utf8')) if receipt.exists() else dict(
        schema='v6_native_fixed_clip_review_v1',test_loaded=False,promotion=False,videos={})
    reports={arm:json.loads((EVIDENCE/'evaluations'/(arm+'_full1367')/'report.json').read_text(encoding='utf8')) for arm in a.arms}
    for arm,report in reports.items():
        if sha(EVIDENCE/'runs'/arm/'final.pt')!=report['checkpoint_sha256']:raise ValueError('checkpoint differs')
    for emotion in a.emotions:
        cid=f'mead_M025_{emotion}_'+('L1_005' if emotion=='neutral' else 'L3_005')
        paths=[OLD/'render_inputs'/(cid+'.npz')]+[
            EVIDENCE/'evaluations'/(arm+'_full1367')/'render_inputs'/(cid+'.npz') for arm in a.arms]
        data=[npz(p) for p in paths];first=data[0]
        for item in data[1:]:
            for key in ('times','valid','channels','channel_mask','noise_seed'):
                if not np.array_equal(first[key],item[key]):raise ValueError('native contract differs '+key)
            if not np.array_equal(first['motions'][:3],item['motions'][:3]):raise ValueError('baseline curves differ')
        good=np.flatnonzero(first['valid']);lo=int(good[0]);hi=int(good[-1])+1
        if not first['valid'][lo:hi].all():raise ValueError('internal gap')
        values=np.concatenate((first['motions'],*[item['motions'][3:4] for item in data[1:]]))[:,lo:hi]
        names=['GT','B0','Phase53','V5 control']+[arm.replace('_',' ') for arm in a.arms]
        merged=root/(emotion+'_inputs.npz')
        mask=first['channel_mask'];mask=mask[lo:hi] if mask.ndim==2 else mask
        payload=dict(motions=values,mode_names=np.asarray(names),times=first['times'][lo:hi],
            valid=first['valid'][lo:hi],channels=first['channels'],channel_mask=mask,
            clip_id=first['clip_id'],noise_seed=first['noise_seed'])
        if merged.exists():
            previous=npz(merged)
            if any(not np.array_equal(previous[k],v) for k,v in payload.items()):raise ValueError('existing input differs')
        else:np.savez_compressed(merged,**payload)
        audio=PROJECT/'final_experiment/data/mead_media_v1/wav'/(cid+'.wav')
        output=root/(emotion+'_render')
        if not output.exists():
            subprocess.run([sys.executable,str(PROJECT/'scripts/render_dynamic_rig_comparison.py'),
                '--input',str(merged),'--output',str(output),'--audio',str(audio),
                '--tile-size','320','--columns',str(len(names)),'--samples','16'],check=True)
        display=json.loads((output/'display_report.json').read_text(encoding='utf8'))
        rig=json.loads((output/'rig_audit.json').read_text(encoding='utf8'))
        if display['input_sha256']!=sha(merged) or display['audio_sha256']!=sha(audio):raise ValueError('binding differs')
        if display['status']!='complete' or not display['original_blend_unchanged'] or rig['frame_value_max_error']!=0:
            raise ValueError('display/rig failed')
        video=output/'comparison.mp4'
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],capture_output=True,check=True)
        streams=json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries',
            'stream=codec_type,nb_read_frames,r_frame_rate','-of','json',str(video)],capture_output=True,text=True,check=True).stdout)['streams']
        stream=next(s for s in streams if s['codec_type']=='video')
        if int(stream['nb_read_frames'])!=hi-lo or stream['r_frame_rate']!='25/1' or not any(s['codec_type']=='audio' for s in streams):
            raise ValueError('clock/audio decode differs')
        indices=sorted(set(round((hi-lo-1)*v) for v in (0,.25,.5,.75,1)));rows=[]
        for i in indices:
            with Image.open(output/'frames'/f'{i+1:06d}.png') as im:
                im=im.convert('RGB');im=im.resize((1600,round(im.height*1600/im.width)));rows.append((i,im))
        sheet=Image.new('RGB',(1600,sum(im.height+26 for _,im in rows)),'white');draw=ImageDraw.Draw(sheet);y=0
        for i,im in rows:
            draw.text((8,y+5),f'{emotion}: native frame {lo+i}, t={float(first["times"][lo+i]):.2f}s',fill='black')
            y+=26;sheet.paste(im,(0,y));y+=im.height
        sheet.save(output/'uniform_time_inspection.png')
        saved['videos'][emotion]=dict(video=str(video),sha256=sha(video),frames=hi-lo,
            fps=25,full_decode=True,rig_error=0,native_start_seconds=float(first['times'][lo]),
            audio_sha256=sha(audio),source_hashes=[sha(p) for p in paths],input_sha256=sha(merged),
            mode_names=names,checkpoint_sha256={arm:r['checkpoint_sha256'] for arm,r in reports.items()},
            inspection_frames=indices)
        receipt.write_text(json.dumps(saved,indent=2),encoding='utf8',newline='\n')
        print(json.dumps(dict(status='rendered_verified',emotion=emotion,frames=hi-lo,video=str(video))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['direct_seed47'])
    p.add_argument('--emotions',nargs='+',default=['angry','happy'])
    p.add_argument('--output-name',default='v6_direct_review_20261012');main(p.parse_args())
