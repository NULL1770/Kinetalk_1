"""Independent coefficient-space probes for paper diagnostics.

Fits fixed alpha=1 class-balanced ridge probes on real motion statistics only.
It does not train or modify a generator.  It is a semantic diagnostic, not a
human emotion or perceptual identity metric.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
import torch

SCHEMA = "paper_motion_probe_v3"

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def arr(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def meta(row):
    m=row.get("metadata",{}) if isinstance(row.get("metadata",{}),dict) else {}
    return {k: row.get(k,m.get(k)) for k in ("clip_id","split","sentence","speaker","emotion")}

_EMOTION_NAMES = ("neutral", "angry", "contempt", "disgust", "fear", "happy", "sad", "surprise")

def label_value(value, kind):
    if value is None: return None
    s = str(value).strip().lower()
    if kind == "emotion":
        try:
            i = int(float(s))
            if 0 <= i < len(_EMOTION_NAMES): return _EMOTION_NAMES[i]
        except ValueError: pass
        return {"disgusted":"disgust", "fearful":"fear", "surprised":"surprise"}.get(s, s)
    return s

def rows(payload):
    x=payload.get("clips",payload.get("rows",payload.get("items"))) if isinstance(payload,dict) else payload
    if isinstance(x,dict): x=list(x.values())
    if not isinstance(x,list): raise ValueError("dataset clips list required")
    return [r for r in x if isinstance(r,dict)]

def target(row):
    for k in ("target52","target","motion","motion52"):
        if k in row:
            x=arr(row[k])
            if x.ndim==2 and x.shape[1]==52:return x.astype(float)
    raise ValueError("target52 absent")

def frame_mask(row,n):
    for k in ("valid","native_valid","mask"):
        if k in row:
            x=arr(row[k]).astype(bool).reshape(-1)
            if x.shape==(n,):return x
    raise ValueError("explicit native frame mask required")

def channel_mask(row,n):
    for k in ("channel_mask","motion_mask","target_mask"):
        if k not in row:continue
        x=arr(row[k]).astype(bool)
        if x.shape==(n,52):return x
        if x.shape==(52,):return np.broadcast_to(x,(n,52)).copy()
        if x.shape==(n,):return np.broadcast_to(x[:,None],(n,52)).copy()
    raise ValueError("explicit channel mask required")

def motion_features(x,support,valid):
    if x.ndim!=2 or x.shape[1]!=52 or support.shape!=x.shape:raise ValueError("[T,52] required")
    support=support & valid[:,None] & np.isfinite(x); out=np.full((52,6),np.nan)
    for c in range(52):
        ids=np.flatnonzero(support[:,c])
        if len(ids)==0:continue
        v=x[ids,c]; out[c,:4]=[v.mean(),v.std(),np.quantile(v,.1),np.quantile(v,.9)]
        if len(ids)>1:
            keep=ids[1:]==ids[:-1]+1; dv=(x[ids[1:],c]-x[ids[:-1],c])[keep]*25
            if len(dv):out[c,4:]=[np.abs(dv).mean(),dv.std()]
    return out.reshape(-1)

def usable(row):
    x=target(row);v=frame_mask(row,len(x));m=channel_mask(row,len(x));return motion_features(x,m,v)

class Probe:
    def __init__(self):self.labels=[];self.impute=self.mu=self.sd=self.w=None
    def fit(self,X,labels):
        self.labels=sorted(set(labels));
        if len(self.labels)<2:raise ValueError("at least two labels")
        if any(a is None for a in labels):raise ValueError("missing labels")
        y=np.array([self.labels.index(a) for a in labels]); finite=np.isfinite(X)
        self.impute=np.divide(np.where(finite,X,0.).sum(0),finite.sum(0),out=np.zeros(X.shape[1]),where=finite.sum(0)>0)
        Z=np.where(np.isfinite(X),X,self.impute);self.mu=Z.mean(0);self.sd=np.where(Z.std(0)>1e-8,Z.std(0),1.);Z=(Z-self.mu)/self.sd
        n=len(y);cnt=np.bincount(y,minlength=len(self.labels));w=n/(len(self.labels)*np.maximum(cnt[y],1));A=np.c_[np.ones(n),Z];Y=np.eye(len(self.labels))[y];s=np.sqrt(w)[:,None];R=np.eye(A.shape[1]);R[0,0]=0
        self.w=np.linalg.solve((A*s).T@(A*s)+R,(A*s).T@(Y*s));return self
    def predict(self,X):
        Z=np.where(np.isfinite(X),X,self.impute);return np.c_[np.ones(len(Z)),(Z-self.mu)/self.sd]@self.w
    def report(self,X,labels=None):
        p=self.predict(X).argmax(1);o={"n":len(X),"labels":self.labels,"prediction_ids":p.tolist()}
        if labels is None:return o
        known=np.array([a in self.labels for a in labels]);o.update(known_n=int(known.sum()),unknown_label_n=int((~known).sum()))
        if not known.any():return o
        y=np.array([self.labels.index(a) for a in labels if a in self.labels]);pp=p[known];C=np.zeros((len(self.labels),len(self.labels)),int)
        for a,b in zip(y,pp):C[a,b]+=1
        rec=np.divide(np.diag(C),C.sum(1),out=np.zeros(len(self.labels)),where=C.sum(1)>0);pre=np.divide(np.diag(C),C.sum(0),out=np.zeros(len(self.labels)),where=C.sum(0)>0);f1=np.divide(2*pre*rec,pre+rec,out=np.zeros(len(self.labels)),where=pre+rec>0)
        present=C.sum(1)>0
        o.update(accuracy=float((y==pp).mean()),balanced_accuracy=float(rec.mean()),macro_f1=float(f1.mean()),present_class_uar=float(rec[present].mean()),present_class_macro_f1=float(f1[present].mean()),supported_classes=int(present.sum()),macro_scope="all fit classes; absent true classes contribute zero",confusion_rows_true=C.tolist(),uniform_chance=1/len(self.labels),majority_baseline=float(np.bincount(y,minlength=len(self.labels)).max()/len(y)));return o

def split_rows(rs,name):return [r for r in rs if str(meta(r).get("split"))==name]

UPPER = [41,42,43,44,45,5,6,12,13]
STAGES = {"prior":"prior_generation", "audio":"audio_generation", "matched_static":"matched_static_generation", "static":"audio_static", "reverse":"audio_reverse", "mismatch":"audio_mismatch"}

def outer_rows(root, by_id, expected_ids):
    """Use raw dataset support/labels; display-filled NPZ is never a reference."""
    result={"reference":[],"base":[]}; sources={}
    for cid in expected_ids:
        r=by_id[cid]; y=target(r)
        if arr(r["baseline52"]).shape!=y.shape:raise ValueError("baseline shape")
        result["reference"].append(r)
        result["base"].append({**r,"target52":arr(r["baseline52"])})
    for arm,stage in STAGES.items():
        path=Path(root)/stage/"holdout"/"curves.pt"
        payload=torch.load(path,map_location="cpu",weights_only=False)
        curves=payload["clips"]; sources[arm]=sha256(path)
        if set(curves)!=set(expected_ids):raise ValueError("outer membership differs")
        for cid in expected_ids:
            r=by_id[cid];c=curves[cid];y=target(r);v=frame_mask(r,len(y));support=channel_mask(r,len(y))
            np.testing.assert_array_equal(arr(c["native_valid"]),v)
            np.testing.assert_array_equal(arr(c["score_mask"]),v & support[:,UPPER].all(1))
            np.testing.assert_allclose(arr(c["target"])[arr(c["score_mask"]).astype(bool)],y[:,UPPER][arr(c["score_mask"]).astype(bool)],atol=0,rtol=0)
            samples=arr(c["samples"])
            if samples.shape!=(4,len(y),9) or list(c["seeds"])!=[42,123,2026,77]:raise ValueError("four fixed draws required")
            for seed,x9 in zip(c["seeds"],samples):
                x=arr(r["baseline52"]).astype(float).copy();x[:,UPPER]=x9
                result.setdefault(f"{arm}/seed{seed}",[]).append({**r,"target52":x})
    return result,sources

def evaluate(dataset,output,protocol=None,outer_root=None):
    if output.exists():raise FileExistsError(output)
    payload=torch.load(dataset,map_location="cpu",weights_only=False);rs=rows(payload);fit=split_rows(rs,"train");dataset_hash=sha256(dataset)
    declared=json.loads(protocol.read_text(encoding="utf8")) if protocol else None
    if declared is None or declared.get("dataset_sha256")!=dataset_hash:raise ValueError("bound protocol and dataset SHA required")
    valid_ids=set((declared or {}).get("inner_validation_ids",(declared or {}).get("validation_ids",[])))
    fit_ids=set((declared or {}).get("inner_train_ids",(declared or {}).get("fit_ids",[])))
    if valid_ids and fit_ids:
        by_id={str(meta(r).get("clip_id")):r for r in rs}
        if len(by_id)!=len(rs):raise ValueError("duplicate dataset clip ids")
        if valid_ids & fit_ids or not valid_ids.issubset(by_id) or not fit_ids.issubset(by_id):raise ValueError("Declared nested split ids are invalid or overlap")
        fit=[by_id[c] for c in (declared or {}).get("inner_train_ids",(declared or {}).get("fit_ids",[]))]
        val=[by_id[c] for c in (declared or {}).get("inner_validation_ids",(declared or {}).get("validation_ids",[]))]
        split_source="declared nested ids"
    else:
        val=split_rows(rs,"inner_validation") or split_rows(rs,"val");split_source="row split"
    FX=[];fl=[];fs=[];fr=[]
    for r in fit:
        try:FX.append(usable(r));fl.append(label_value(meta(r).get("emotion"),"emotion"));fs.append(label_value(meta(r).get("speaker"),"speaker"))
        except ValueError as e:fr.append(str(meta(r).get("clip_id"))+":"+str(e))
    E=Probe().fit(np.stack(FX),fl);S=Probe().fit(np.stack(FX),fs) if len(set(fs))>1 else None
    def score(rr):
        X=[];el=[];sl=[];excluded=[]
        for r in rr:
            try:X.append(usable(r));el.append(label_value(meta(r).get("emotion"),"emotion"));sl.append(label_value(meta(r).get("speaker"),"speaker"))
            except ValueError as e:excluded.append(str(meta(r).get("clip_id"))+":"+str(e))
        if not X:return {"n":0,"excluded":excluded}
        return {"n":len(X),"excluded":excluded,"emotion":E.report(np.stack(X),el),"motion_speaker_signature":S.report(np.stack(X),sl) if S else None}
    out={"schema":SCHEMA,"alpha":1.0,"feature_schema":"52x(mean,std,q10,q90,abs_velocity_mean,velocity_std), dt=1/25; fit imputation; raw coefficients","dataset_sha256":dataset_hash,"source_sha256":sha256(Path(__file__)),"protocol_sha256":sha256(protocol),"protocol":declared,"split_source":split_source,"fit":{"n":len(FX),"excluded":fr,"emotion":E.report(np.stack(FX),fl),"motion_speaker_signature":S.report(np.stack(FX),fs) if S else None},"real_validation":score(val),"outer":{}}
    if outer_root:
        expected=declared["outer_ids"]
        if set(expected)&(fit_ids|valid_ids):raise ValueError("outer split overlap")
        grouped,sources=outer_rows(outer_root,by_id,expected)
        out["outer_sources_sha256"]=sources
        for name,items in grouped.items():
            out["outer"][name]={**score(items),"clips":[meta(i)["clip_id"] for i in items]}
        out["outer_arm_mean"]={}
        for arm in STAGES:
            group=[v for k,v in out["outer"].items() if k.startswith(arm+"/")]
            out["outer_arm_mean"][arm]={task:{metric:float(np.mean([g[task][metric] for g in group])) for metric in ("accuracy","balanced_accuracy","macro_f1","present_class_uar","present_class_macro_f1")} for task in ("emotion","motion_speaker_signature")}
    output.parent.mkdir(parents=True,exist_ok=True)
    state=output.with_suffix(".probe.npz")
    np.savez_compressed(state,**{f"{task}_{key}":value for task,p in (("emotion",E),("speaker",S)) if p for key,value in vars(p).items()})
    out["probe_state_sha256"]=sha256(state)
    out["limitations"]=["real-motion ridge diagnostic, not human emotion or appearance identity", "full52 features; scores do not localize defects to upper face", "current outer is historically exposed development", "all fit-class macro and supported-class macro reported separately"]
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8");return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--protocol",type=Path);p.add_argument("--outer-root",type=Path);a=p.parse_args();r=evaluate(a.dataset,a.output,a.protocol,a.outer_root);print(json.dumps({"schema":SCHEMA,"fit_n":r["fit"]["n"],"validation_n":r["real_validation"].get("n",0),"output":str(a.output)}))
if __name__=="__main__":main()
