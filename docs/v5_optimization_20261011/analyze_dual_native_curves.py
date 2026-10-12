"""Quantify fixed-video trajectories and inherited/new parameter change; no fitting."""
import csv
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OUTPUT=PROJECT/'artifacts/v5_dual_timescale_review_20261012'
rows=[]
for emotion in ('angry','happy'):
    with np.load(OUTPUT/(emotion+'_inputs.npz'),allow_pickle=False) as z:
        names=z['mode_names'].tolist();channels=z['channels'].tolist();y=np.clip(z['motions'],0,1)
        for i,name in enumerate(names):
            for channel in ('jawOpen','mouthSmileLeft','mouthSmileRight','mouthStretchLeft','mouthStretchRight','browDownLeft','browInnerUp','eyeBlinkLeft'):
                x=y[i,:,channels.index(channel)]
                rows.append(dict(emotion=emotion,mode=name,channel=channel,mean=float(x.mean()),
                                 std=float(x.std()),range_q90_q10=float(np.quantile(x,.9)-np.quantile(x,.1))))
with (ROOT/'dual_native_curves.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
stats={}
for arm in ('direct','guided'):
    ck=torch.load(ROOT/'dual_evidence/runs'/arm/'final.pt',map_location='cpu',weights_only=False)
    stats[arm]={}
    for name in ('envelope_head.weight','envelope_head.bias','backbone.decoder.affect_adapter.weight',
                 'backbone.decoder.phonetic_adapter.weight'):
        x=ck['model'][name]
        stats[arm][name]=dict(rms=float(x.square().mean().sqrt()),absolute_max=float(x.abs().max()))
    for name in ('backbone.decoder.affect_gate','backbone.decoder.phonetic_gate'):
        stats[arm][name]=float(ck['model'][name].sigmoid())
(ROOT/'dual_new_parameter_diagnostics.json').write_text(json.dumps(dict(
    schema='dual_candidate_parameter_magnitude_v1',statistics=stats,test_loaded=False,
    limits='Parameter magnitudes verify trained nonzero heads, not their causal effect or semantic separation.'),indent=2))
print(json.dumps(dict(status='complete',native_rows=len(rows),arms=list(stats))))
