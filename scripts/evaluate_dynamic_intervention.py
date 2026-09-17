"""Measure whether a saved fixed-noise generation responds to dynamic-field interventions.

This is a post-hoc diagnostic: it does not fit a model and does not claim that a
non-zero response is correct emotion.  A useful dynamic path should respond to
full/zero/reversed local fields while preserving the same noise, B0 and identity.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

REGIONS = {"all": tuple(range(52)), "upper": tuple(range(14)),
           "brows": tuple(range(41,46)), "mouth": tuple(range(14,41)), "jaw17": (17,)}

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _mask(curves):
    valid = curves["valid"].bool()
    cm = curves["channel_mask"].bool()
    return valid[...,None] & cm[:,None,:]

def _mse(a,b,m):
    d=(a-b).square(); return float(d[m].mean()) if bool(m.any()) else None

def _region_mse(a,b,m,ids):
    ids=[i for i in ids if i<a.shape[-1]]
    return _mse(a[...,ids], b[...,ids], m[...,ids]) if ids else None

def _velocity(x, valid, times):
    pair=valid[:,1:] & valid[:,:-1]
    dt=times[:,1:]-times[:,:-1]
    pair &= torch.isfinite(dt) & (dt>0) & (dt<=.12)
    safe=torch.where(pair,dt,torch.ones_like(dt))
    return (x[:,1:]-x[:,:-1])/safe[...,None], pair[...,None]

def _response(curves, source):
    m=_mask(curves); full=curves[f"{source}_full"]; zero=curves[f"{source}_zero"]; mean=curves[f"{source}_mean"]; rev=curves[f"{source}_reverse"]
    target=curves["target"]; times=curves["times"].to(full.dtype); valid=curves["valid"].bool()
    out={"full_vs_zero_mse":_mse(full,zero,m), "full_vs_mean_mse":_mse(full,mean,m),
         "full_vs_reverse_mse":_mse(full,rev,m), "full_target_mse":_mse(full,target,m),
         "zero_target_mse":_mse(zero,target,m), "reverse_target_mse":_mse(rev,target,m)}
    target_energy=_mse(target,curves["b0"],m)
    out["response_over_target_energy"]=(out["full_vs_zero_mse"]/target_energy if target_energy and target_energy>0 else None)
    for name,pred in (("full",full),("zero",zero),("mean",mean),("reverse",rev)):
        vel,pair=_velocity(pred,valid,times)
        tv,tp=_velocity(target,valid,times)
        out[name+"_velocity_energy"]=float(vel[pair.expand_as(vel)].square().mean()) if bool(pair.any()) else None
        out[name+"_target_velocity_mse"]=_mse(vel,tv,pair.expand_as(vel))
        for region,ids in REGIONS.items():
            mm=m[...,list(ids)] if ids else m
            out[f"{name}_{region}_target_mse"]=_region_mse(pred,target,m,ids)
            out[f"{name}_{region}_response_mse"]=_region_mse(pred,zero,m,ids)
            out[f"{name}_{region}_reverse_gap_mse"]=_region_mse(pred,rev,m,ids)
    # Per-clip response is retained for uncertainty/paired analysis.
    per=[]
    for i in range(full.shape[0]):
        mi=m[i]
        per.append({"full_vs_zero_mse":_mse(full[i:i+1],zero[i:i+1],mi.unsqueeze(0)),
                    "full_target_mse":_mse(full[i:i+1],target[i:i+1],mi.unsqueeze(0)),
                    "zero_target_mse":_mse(zero[i:i+1],target[i:i+1],mi.unsqueeze(0))})
    out["per_clip"]=per
    out["clips"]=int(full.shape[0]); out["frames"]=int(full.shape[1]); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--curves",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args(); curves=torch.load(args.curves,map_location="cpu",weights_only=False)
    required=["target","b0","valid","channel_mask"]+[f"{s}_{k}" for s in ("audio","teacher") for k in ("full","zero","mean","reverse")]
    missing=[k for k in required if k not in curves]
    if missing: raise ValueError("curves missing: "+", ".join(missing))
    report={"schema":"dynamic_intervention_response_v1","curves":str(args.curves.resolve()),"curves_sha256":_sha(args.curves),"script_sha256":_sha(Path(__file__)),"sources":{s:_response(curves,s) for s in ("audio","teacher")},"interpretation":"Intervention response tests causal use of the local dynamic field under fixed noise; response magnitude alone is not evidence of correct emotion or generalization."}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,ensure_ascii=True),encoding="utf8"); print(args.output.resolve())
if __name__=="__main__": main()
