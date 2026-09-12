"""Audit per-pair DTW mouth amplitude preservation and event masks.

This is a release audit, not a training transform.  It refuses to report a
ratio when the denominator is nearly static and records those cases separately.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

MOUTH=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,51]
def qrange(x): return float(np.percentile(x,95)-np.percentile(x,5))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--pair-root',type=Path,required=True); p.add_argument('--bs-root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    rows=[json.loads(x) for x in a.manifest.read_text(encoding='utf-8').splitlines() if x.strip()]; out=[]; ratios=[]; jaws=[]; skipped=0
    for row in rows:
        art=Path(row.get('teacher_artifact','')); art=a.pair_root/art.name if not art.is_file() else art
        ref=a.bs_root/f"{row['reference_clip_id']}.npz"
        try:
            with np.load(art,allow_pickle=False) as z:
                teacher=z['neutral_teacher_on_source'].astype(np.float32); mask=z['native_teacher_mask'].astype(bool)
            with np.load(ref,allow_pickle=False) as z: motion=z['coeffs'].astype(np.float32)
        except (FileNotFoundError,KeyError):
            skipped+=1; continue
        if mask.sum()<10: skipped+=1; continue
        base=np.asarray([qrange(motion[:,i]) for i in MOUTH]); warped=np.asarray([qrange(teacher[mask,i]) for i in MOUTH]); active=base>0.01
        if not active.any(): skipped+=1; continue
        r=warped[active]/base[active]; ratios.extend(r.tolist())
        jaw_i=MOUTH.index(17)
        if base[jaw_i]>.01: jaws.append(float(warped[jaw_i]/base[jaw_i]))
        out.append(dict(row, mouth_amplitude_median=float(np.median(r)), mouth_amplitude_p05=float(np.percentile(r,5)), mouth_amplitude_p95=float(np.percentile(r,95)), jaw_open_ratio=float(warped[jaw_i]/base[jaw_i]) if base[jaw_i]>.01 else None, amplitude_gate=bool(np.percentile(r,5)>=0.60 and np.percentile(r,95)<=1.40)))
    summary={'pairs_read':len(out),'pairs_skipped':skipped,'amplitude_ratio':{str(k):float(np.percentile(ratios,k)) for k in (5,25,50,75,95)} if ratios else {},'jaw_open_ratio':{str(k):float(np.percentile(jaws,k)) for k in (5,25,50,75,95)} if jaws else {},'amplitude_gate_pass':sum(x['amplitude_gate'] for x in out)}
    a.out.write_text(json.dumps({'summary':summary,'pairs':out},ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
