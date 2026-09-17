"""Matched audio predictability probe for alternative low-rate affect ranks.

This isolates target capacity: the frozen rank-8 teacher/B0 checkpoint is only
used to build neutral-corrected residuals. A fresh audio student is trained
against an observed residual PCA of the requested rank. No renderer or global
path is changed.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import LowRateAffectEncoder, NeutralAffectSystem
from kinetalk_b0.neutral_data import NeutralAffectDataset
from kinetalk_b0.utils import freeze_module
from scripts.probe_dynamic_predictability import (sentence_split, training_only_audio,
    fit_pca_target, fit_target_scale, weighted_mse, target_metrics)
from scripts.train_neutral_affect_audio_ablation import state_hash
from scripts.train_neutral_affect_feature_probe import prepare_feature_batch
from scripts.train_neutral_affect_pilot import (affect_residual, cached_identity,
    device_batch, observed, optimize, prepare, select, sha, write_json)

def fresh(cfg, seed, inp, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); s=LowRateAffectEncoder(cfg, inp, motion=False)
    for n in ("global_head", "emotion_classifier", "intensity_classifier"):
        getattr(s,n).requires_grad_(False)
    return s.to(device)

def main():
    p=argparse.ArgumentParser()
    for n in ("config","data","checkpoint","output"): p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--rank",type=int,required=True); p.add_argument("--steps",type=int,default=800)
    p.add_argument("--seed",type=int,default=61)
    a=p.parse_args(); cfg=yaml.safe_load(a.config.read_text())
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if ck.get("stage")!="audio" or ck.get("audio_source")!="acoustic": raise ValueError("run07 acoustic checkpoint required")
    if any(cfg.get(k)!=ck.get("config",{}).get(k) for k in ("data","model")): raise ValueError("config/checkpoint mismatch")
    if a.rank<1: raise ValueError("rank must be positive")
    torch.manual_seed(a.seed); torch.set_num_threads(4); device=torch.device(cfg.get("device","cuda"))
    ds=NeutralAffectDataset(a.data/"train.pt"); tr_cpu,ho_cpu=sentence_split([q["sentence_id"] for q in ds.queries],a.seed)
    ids={"train":tr_cpu.to(device),"heldout":ho_cpu.to(device)}
    frozen=NeutralAffectSystem(cfg).to(device).eval(); frozen.load_state_dict(ck["model"],strict=True); freeze_module(frozen)
    with torch.no_grad():
        q,_=prepare(frozen,ds,device); q,stats=training_only_audio(q,ds.cache["audio_stats"],ids["train"])
        refs=ds.identity_references; ref=device_batch([r for s in sorted(refs) for r in refs[s]],device)
        rr=torch.where(observed(ref),ref["motion"]-frozen.base(ref["content"],ref["valid"])["b0"],0)
        ident=cached_identity(frozen,{"residual":rr.reshape(len(refs),len(refs[0]),*rr.shape[1:]),"valid":ref["valid"].reshape(len(refs),len(refs[0]),-1)})
        residual=affect_residual(q,select(ident,q["speaker_id"]))
        target,w,basis=fit_pca_target(residual,q["valid"],q["channel_mask"],ids["train"],a.rank,frozen.motion_teacher.stride)
        scale=fit_target_scale(target,w,ids["train"]); target=target/scale
    scfg=json.loads(json.dumps(cfg)); scfg["model"]["affect_rank"]=a.rank
    student=fresh(scfg,a.seed,q["audio"].shape[-1],device); init=state_hash(student.state_dict())
    a.output.mkdir(parents=True,exist_ok=False); (a.output/"effective_config.yaml").write_text(yaml.safe_dump(scfg,sort_keys=False))
    prov={"schema":"dynamic_rank_ablation_v1","rank":a.rank,"steps":a.steps,"seed":a.seed,"checkpoint_sha256":sha(a.checkpoint),"train_sha256":sha(a.data/"train.pt"),"split_seed":a.seed,"split":{"train":tr_cpu.tolist(),"heldout":ho_cpu.tolist()},"target":"training-only residual PCA","frozen_state_sha256":state_hash(frozen.state_dict()),"initial_student_sha256":init,"feature_stats":{k:(v.tolist() if torch.is_tensor(v) else v) for k,v in stats.items()}}
    write_json(a.output/"provenance.json",prov)
    params=[x for x in student.parameters() if x.requires_grad]; opt=torch.optim.AdamW(params,lr=float(cfg["training"].get("audio_lr",.0005)),weight_decay=1e-5)
    rng=torch.Generator(device="cpu").manual_seed(a.seed); digest=hashlib.sha256(); student.train()
    for step in range(a.steps):
        ix=torch.randint(len(tr_cpu),(int(cfg["training"].get("batch_size",8)),),generator=rng); digest.update(ix.numpy().tobytes()); b=ids["train"][ix.to(device)]
        pred=3.0*student(q["audio"][b],q["valid"][b])["controls"]; loss=weighted_mse(pred,target[b],w[b]); optimize(loss,opt,params)
        if step%200==0 or step+1==a.steps: print({"step":step+1,"loss":float(loss.detach())},flush=True)
    student.eval();
    with torch.no_grad(): pred=3.0*student(q["audio"],q["valid"])["controls"]
    report={}
    for s in ids:
        qq=select(q, ids[s])
        report[s]=target_metrics(pred[ids[s]], target[ids[s]], w[ids[s]], qq["clip_id"], qq["sentence_id"], scale)
    report["minibatch_sha256"]=digest.hexdigest(); report["rank"]=a.rank; report["target_scale"]=scale.cpu().tolist(); write_json(a.output/"summary.json",report)
    torch.save({"model":student.cpu().state_dict(),"config":scfg,"rank":a.rank,"summary":report,"basis":{k:(v.cpu() if torch.is_tensor(v) else v) for k,v in basis.items()}},a.output/"student.pt")

if __name__=="__main__": main()
