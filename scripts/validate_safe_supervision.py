"""Validate the safe supervision sidecars and fail closed on missing masks."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--min-event-rate',type=float,default=0.03); a=p.parse_args()
    rows=[json.loads(x) for x in a.manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    seen=set(); bad=[]; event_rates=[]; boundary_rates=[]
    for r in rows:
        for k in ('source_clip_id','reference_clip_id'):
            clip=str(r[k]);
            if clip in seen: continue
            seen.add(clip); path=a.root/f'{clip}.npz'
            if not path.is_file(): bad.append((clip,'missing')); continue
            with np.load(path,allow_pickle=False) as z:
                for key in ('mouth_event','mouth_event_mask','boundary','boundary_mask','viseme_id','viseme_mask'):
                    if key not in z.files: bad.append((clip,'missing_'+key))
                if 'mouth_event_mask' in z.files: event_rates.append(float(np.mean(z['mouth_event_mask']>0.5)))
                if 'boundary_mask' in z.files: boundary_rates.append(float(np.mean(z['boundary_mask']>0.5)))
    out={'safe_pairs':len(rows),'unique_clips':len(seen),'bad_sidecars':len(bad),'event_mask_rate_mean':float(np.mean(event_rates)) if event_rates else 0.0,'boundary_mask_rate_mean':float(np.mean(boundary_rates)) if boundary_rates else 0.0,'phoneme_boundary_ready':bool(boundary_rates and max(boundary_rates)>0)}
    (a.root/'validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
    if bad: print('first_bad',bad[:10]); raise SystemExit(2)
if __name__=='__main__': main()
