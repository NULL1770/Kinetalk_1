"""Read-only native audio-to-upper-face timing audit.

Validates prepared train/validation query shards and source audio provenance,
then computes simple lagged signal associations on the native 25 fps clock.
Lag and sign are fit on TRAIN only and applied unchanged on validation. This is
only a timing diagnostic; it is not an animation-quality or impossibility claim.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import defaultdict
import gc
from pathlib import Path
import numpy as np
import torch

FPS, DT = 25.0, 1.0 / 25.0
LAGS = tuple(range(-6, 7))
UPPER = (41,42,43,44,45,5,6,12,13)
MOUTH = tuple(range(14,41))
GROUPS = {"raise":(43,44,45), "down":(41,42), "squint":(5,12), "wide":(6,13)}

def sha256(path):
    """Stream a file SHA256 without loading the complete artifact into RAM."""
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def canonical_hash(value):
    """Use the same canonical JSON hash as the locked manifest producer."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def require(ok,msg):
    if not ok: raise ValueError(msg)

def arr(x,name,dtype=None):
    if torch.is_tensor(x): x=x.detach().cpu().numpy()
    x=np.asarray(x)
    if dtype is not None: x=x.astype(dtype,copy=False)
    require(np.isfinite(x).all(), f"{name} contains non-finite values")
    return x

def runs(mask):
    """Return all maximal contiguous true runs; invalid gaps are never crossed."""
    ids=np.flatnonzero(np.asarray(mask,bool))
    return [] if not len(ids) else list(np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1))

def diff_norm(x,valid):
    """Native first-difference norm, reset independently at every valid run."""
    x=np.asarray(x,float); out=np.zeros(len(x),float)
    for r in runs(valid):
        if len(r)>1: out[r[1:]]=np.linalg.norm(np.diff(x[r],axis=0),axis=1)
    return out

def lag_pairs(a,b,valid,lag,amask=None,bmask=None):
    """Align ``a[t]`` to ``b[t+lag]`` without gap crossing or time compression."""
    a=np.asarray(a,float); b=np.asarray(b,float); valid=np.asarray(valid,bool)
    amask=valid if amask is None else np.asarray(amask,bool)
    bmask=valid if bmask is None else np.asarray(bmask,bool)
    require(a.shape==b.shape==valid.shape==amask.shape==bmask.shape,"lag shapes differ")
    aa=[]; bb=[]
    for r in runs(valid):
        if len(r)<=abs(lag)+2: continue
        if lag>=0: src=r[:-lag] if lag else r; dst=r[lag:] if lag else r
        else: src,dst=r[-lag:],r[:lag]
        keep=amask[src]&bmask[dst]; aa.append(a[src[keep]]); bb.append(b[dst[keep]])
    return (np.concatenate(aa),np.concatenate(bb)) if aa else (np.empty(0),np.empty(0))

def stats(a,b):
    if len(a)<3: return {"corr":None,"dot":0.,"ae":0.,"be":0.,"n":int(len(a))}
    a=np.asarray(a,float); b=np.asarray(b,float); a=a-a.mean(); b=b-b.mean()
    dot=float(a.dot(b)); ae=float(a.dot(a)); be=float(b.dot(b)); den=math.sqrt(ae*be)
    return {"corr":dot/den if den>1e-12 else None,"dot":dot,"ae":ae,"be":be,"n":int(len(a))}

def lag_stat(a,b,v,lag,am=None,bm=None): return stats(*lag_pairs(a,b,v,lag,am,bm))

def pool_add(dst,s):
    for k in ("dot","ae","be"): dst[k]+=float(s[k])
    dst["n"]+=int(s["n"])

def pool_corr(s):
    d=math.sqrt(s["ae"]*s["be"]); return s["dot"]/d if d>1e-12 else None

def select(train,pair):
    """Fit one lag/sign on TRAIN only; never use validation to recalibrate.

    Each clip is centered before correlation.  We select the largest absolute
    mean signed TRAIN correlation, then freeze that mean's sign.  This avoids
    a sign-free objective that rewards inconsistent opposite associations.
    """
    choice={}
    for lag in LAGS:
        vals=[r["lags"][pair][str(lag)]["corr"] for r in train]
        vals=np.asarray([v for v in vals if v is not None and np.isfinite(v)])
        p={"dot":0.,"ae":0.,"be":0.,"n":0}
        for r in train: pool_add(p,r["lags"][pair][str(lag)])
        choice[int(lag)]={"mean_abs":float(np.abs(vals).mean()) if len(vals) else None,
                           "mean":float(vals.mean()) if len(vals) else None,
                           "pooled":pool_corr(p),"clips":int(len(vals)),"samples":int(p["n"])}
    cand=[(abs(v["mean"]),-abs(l),-l,l) for l,v in choice.items() if v["mean"] is not None]
    if not cand: return {"lag":0,"sign":1,"selection":choice,"pooled":None,"clips":0}
    _,_,_,lag=max(cand); pooled=choice[lag]["pooled"]
    return {"lag":int(lag),"sign":1 if choice[lag]["mean"]>=0 else -1,
            "selection":choice,"pooled":pooled,"clips":choice[lag]["clips"]}

def aggregate(records,fitted):
    out={}
    for pair,cfg in fitted.items():
        vals=[]; p={"dot":0.,"ae":0.,"be":0.,"n":0}
        for r in records:
            s=r["lags"][pair][str(cfg["lag"])]
            if s["corr"] is not None: vals.append(float(s["corr"])*cfg["sign"])
            pool_add(p,s)
        v=np.asarray(vals,float)
        out[pair]={"lag":cfg["lag"],"sign":cfg["sign"],"clips":int(len(v)),
                   "mean_signed":float(v.mean()) if len(v) else None,
                   "mean_abs":float(np.abs(v).mean()) if len(v) else None,
                   "median_signed":float(np.median(v)) if len(v) else None,
                   "pooled_signed":pool_corr(p)*cfg["sign"] if pool_corr(p) is not None else None,
                   "samples":int(p["n"])}
    return out

def validate(saved,row,shard,root):
    require(saved.get("row")==row,f"manifest row differs: {row['clip_id']}")
    valid=np.asarray(saved["valid"]); channel=np.asarray(saved["channel_mask"])
    require(valid.dtype==np.bool_ and channel.dtype==np.bool_,"valid/channel masks must be bool")
    motion=arr(saved["motion"],"motion",float); times=arr(saved["times"],"times",float)
    require(valid.ndim==1 and len(valid)==int(row["frames"]),f"frame shape differs: {row['clip_id']}")
    require(times.shape==valid.shape and (len(times)<2 or np.allclose(np.diff(times),DT,atol=1e-5,rtol=0)),f"clock differs: {row['clip_id']}")
    require(channel.shape==(52,) and motion.shape==(len(valid),52),f"motion shape differs: {row['clip_id']}")
    require(int(valid.sum())==int(row["valid_frames"]),f"valid count differs: {row['clip_id']}")
    require(channel[list(UPPER)].all() and channel[list(MOUTH)].all(),f"required channels masked: {row['clip_id']}")
    require(np.isfinite(motion[valid][:,channel]).all(),f"nonfinite motion: {row['clip_id']}")
    native=saved.get("native_provenance"); ext=saved.get("extraction_record")
    require(isinstance(native,dict) and isinstance(ext,dict),f"provenance missing: {row['clip_id']}")
    require("fps" in native and float(native["fps"])==FPS,"provenance fps differs")
    require(native.get("clock_evidence")=="embedded_video","clock evidence differs")
    require(all(k in native for k in ("audio_path","audio_sha256","audio_offset_s")),"audio provenance incomplete")
    audio=Path(str(native["audio_path"])); ah=str(native["audio_sha256"])
    require(audio.is_file() and sha256(audio)==ah,f"audio hash differs: {row['clip_id']}")
    require(ext.get("wave_sha256")==ah,f"extraction audio hash differs: {row['clip_id']}")
    require("audio_offset_s" in ext and math.isfinite(float(ext["audio_offset_s"])) and math.isfinite(float(native["audio_offset_s"])) and abs(float(ext["audio_offset_s"])-float(native["audio_offset_s"]))<=1e-8,f"audio offset differs: {row['clip_id']}")
    return valid,{"audio_path":str(audio),"audio_sha256":ah,"audio_offset_s":float(native["audio_offset_s"]),"shard":str(shard.relative_to(root))}

def record(saved,row,valid,motion,provenance=None):
    audio=arr(saved["audio"],"audio",float); content=arr(saved["content"],"content",float); middle=arr(saved["middle"],"middle",float); pro=arr(saved["prosody"],"prosody",float)
    require(audio.shape[0]==content.shape[0]==middle.shape[0]==len(valid),"feature clock differs")
    require(audio.ndim==2 and audio.shape[1]>=80 and content.shape[1]==middle.shape[1]==768 and pro.shape==(len(valid),4),"feature dimensions differ")
    require(np.isin(pro[valid,3],(0.,1.)).all(),"voicing is not binary")
    adj=np.zeros(len(valid),bool); adj[1:]=valid[1:]&valid[:-1]
    sig={"rms":pro[:,1],"f0":pro[:,0],"periodicity":pro[:,2],"voiced":pro[:,3],"audio_delta":diff_norm(audio[:,:80],valid),"content_delta":diff_norm(content,valid),"middle_delta":diff_norm(middle,valid)}
    smask={k:(adj if k.endswith("delta") else valid) for k in sig}; smask["f0"]=valid&(pro[:,3]>.5)
    target={};
    for name,ch in GROUPS.items(): target["upper_"+name]=motion[:,ch].mean(1); target["upper_"+name+"_speed"]=diff_norm(motion[:,ch],valid)
    target["jaw_open"]=motion[:,17]; target["mouth_speed"]=diff_norm(motion[:,MOUTH],valid)
    tmask={k:(adj if k.endswith("speed") or k=="mouth_speed" else valid) for k in target}
    pairs={}
    for an,av in sig.items():
        for tn,tv in target.items():
            key=an+"__"+tn; pairs[key]={str(l):lag_stat(av,tv,valid,l,smask[an],tmask[tn]) for l in LAGS}
    return {"clip_id":row["clip_id"],"role":row["source_split"],"speaker":row["speaker"],"sentence":row["sentence"],"emotion":row["emotion"],"intensity":row["intensity"],"frames":len(valid),"valid_frames":int(valid.sum()),"provenance":provenance or {},"lags":pairs}

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    root=args.data.resolve(); out=args.output.resolve(); require(root.is_dir() and not out.exists(),"data must exist and output must be fresh")
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf8")); index=json.loads((root/"index.json").read_text(encoding="utf8"))
    require(manifest.get("sealed_test_targets_loaded") is False and index.get("test_loaded") is False,"test targets must remain sealed")
    require(set(manifest.get("roles",{}))=={"train","val","test"},"explicit roles required")
    rows={r["clip_id"]:r for role in ("train","val") for r in manifest["roles"][role]["query"]}; recs={r["clip_id"]:r for r in index["records"] if r["kind"]=="query"}
    require(set(rows)==set(recs),"query membership differs")
    expected=sum(len(manifest["roles"][role]["query"]) for role in ("train","val"))
    require(len(rows)==expected and len(recs)==sum(r["kind"]=="query" for r in index["records"]),"duplicate query records")
    unsigned=dict(manifest); bound=unsigned.pop("manifest_sha256",None)
    require(canonical_hash(unsigned)==bound,"manifest canonical hash differs")
    require(index["recipe"]["manifest_sha256"]==bound,"index/manifest binding differs")
    out.mkdir(parents=True); items=[]
    for role in ("train","val"):
        for row in manifest["roles"][role]["query"]:
            rec=recs[row["clip_id"]]; require(rec["role"]==role and row["source_split"]==role,"index role differs")
            require(rec["frames"]==row["frames"] and rec["valid_frames"]==row["valid_frames"],"index frame support differs")
            shard=(root/rec["path"]).resolve(); require(shard.is_relative_to(root) and shard.is_file(),f"missing shard: {row['clip_id']}"); require(sha256(shard)==rec["sha256"],f"shard hash differs: {row['clip_id']}")
            saved=torch.load(shard,map_location="cpu",weights_only=False); valid,prov=validate(saved,row,shard,root); items.append(record(saved,row,valid,arr(saved["motion"],"motion",float),prov)); del saved
            if len(items)%250==0: print(json.dumps({"status":"auditing","clips":len(items),"total":expected}),flush=True)
    train=[x for x in items if x["role"]=="train"]; val=[x for x in items if x["role"]=="val"]; pairs=sorted(train[0]["lags"]) if train else []
    # Keep scalar lag statistics only until calibration; release no-longer-needed source tensors.
    gc.collect()
    fitted={p:select(train,p) for p in pairs}; fixed={"train":aggregate(train,fitted),"validation":aggregate(val,fitted)}
    all_lag={"train":{},"validation":{}}
    for role,rs in (("train",train),("validation",val)):
        for p in pairs:
            lag_view={}
            for lag in LAGS:
                pooled={"dot":0.,"ae":0.,"be":0.,"n":0}
                vals=[]
                for r in rs:
                    s=r["lags"][p][str(lag)]; pool_add(pooled,s)
                    if s["corr"] is not None: vals.append(float(s["corr"]))
                vv=np.asarray(vals,float)
                lag_view[str(lag)]={"clips":int(len(vv)),"mean_corr":float(vv.mean()) if len(vv) else None,"mean_abs":float(np.abs(vv).mean()) if len(vv) else None,"pooled_corr":pool_corr(pooled),"samples":int(pooled["n"])}
            all_lag[role][p]=lag_view
    speakers={}
    for role,rs in (("train",train),("validation",val)):
        by=defaultdict(list)
        for r in rs: by[r["speaker"]].append(r)
        speakers[role]={s:aggregate(v,fitted) for s,v in sorted(by.items())}
    public=[]
    for r in items:
        fm={}
        for p,cfg in fitted.items():
            s=r["lags"][p][str(cfg["lag"])]
            fm[p]={"lag":cfg["lag"],"sign":cfg["sign"],"corr":s["corr"],"signed_corr":float(s["corr"])*cfg["sign"] if s["corr"] is not None else None,"zero_corr":r["lags"][p]["0"]["corr"],"samples":s["n"]}
        public.append({**{k:v for k,v in r.items() if k!="lags"},"fixed_metrics":fm})
    (out/"perclip.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in public),encoding="utf8")
    summary={"schema":"prepared_audio_upper9_clock_audit_v2","data_root":str(root),"manifest_sha256":manifest.get("manifest_sha256"),"index_sha256":sha256(root/"index.json"),"train_clips":len(train),"validation_clips":len(val),"fps":FPS,"lag_frames":list(LAGS),"upper_indices":list(UPPER),"groups":{k:list(v) for k,v in GROUPS.items()},"pair_count":len(pairs),"fixed_train_lag_sign":fitted,"aggregate_by_fixed_lag_sign":fixed,"all_lag_aggregate":all_lag,"speaker_aggregate":speakers,"test_targets_loaded":False,"mask_policy":"all contiguous native valid runs; no gap crossing/compression/interpolation","selection_policy":"equal-clip absolute mean signed TRAIN correlation selects lag; its TRAIN sign is frozen; validation fixed","interpretation":"Diagnostic signal associations only; no dynamic success or impossibility claim.","script_sha256":sha256(Path(__file__).resolve())}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8"); print(json.dumps({"status":"complete","train":len(train),"validation":len(val),"pairs":len(pairs),"output":str(out)}),flush=True)
if __name__=="__main__": main()









