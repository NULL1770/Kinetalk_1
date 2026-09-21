"""Train and evaluate a signed, four-state audio timing predictor.

This runner is deliberately separate from the historical timing experiments.
It predicts only the centered, slow upper-face state (raise/down/squint/wide)
from relative acoustic features.  The Stage4 renderer, identity, emotion and
per-clip upper mean are frozen.  Speaker/sentence OOF selection is performed
on TRAIN, followed by a fixed-epoch refit on all TRAIN clips.  Development is
used only once for the final comparison against static, reverse, shuffled and
mismatched audio conditions.

The script is a gate, not a claim that audio dynamics work.  It writes the
state metrics, regional envelope diagnostics, and full-face protection checks;
it does not add stochastic motion or alter the default renderer.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinetalk_b0.models.relative_audio_timing import RelativeAudioTiming, relative_features
from kinetalk_b0.models.regional_intensity_gain import regional_envelope
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, readout_slow_state, masked_slow_state
from kinetalk_b0.label_guided_intensity import fit_intensity_scales
from scripts.extract_emotion2vec_pilot import sha
from scripts.mouth_protection import protection_report
from scripts.train_formal_predictable_projection import canonical_hash, save_checkpoint, save_json
from scripts.train_full_staged import NOT_UPPER, batch_identity, obs, subset
from scripts.train_isolated_audio_state import load_context, old_prediction, state_metrics, trim


STATE_NAMES = ("raise", "down", "squint", "wide")
SEEDS = (42, 123, 2026)


def _device(value: str) -> torch.device:
    d = torch.device(value)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return d


def _masked_center(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid[..., None]
    clean = torch.where(mask, value, 0.)
    mean = clean.sum(1, keepdim=True) / valid.sum(1, keepdim=True).to(value.dtype).clamp_min(1.)[..., None]
    return torch.where(mask, value - mean, 0.)


def _upper_target(q: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    valid = q["valid"].to(device)
    upper = q["motion"][..., list(UPPER_INDICES)].to(device)
    # Use the observed upper trajectory only for diagnostics.  This target is
    # never passed to the model and is centered independently per clip.
    return _masked_center(upper, valid), valid


def _fit_statistics(q: dict, ids: torch.Tensor, device: torch.device,
                    *, rank: int = 24, max_samples: int = 120_000):
    """Fit TRAIN-only normalization and acoustic PCA on a bounded sample."""
    f = q["audio_features"]; valid = q["valid"]
    n = f.shape[-1]
    if n != 1540:
        raise ValueError(f"Expected 1540 acoustic features, got {n}")
    s = torch.zeros(n, dtype=torch.float64); ss = torch.zeros_like(s); count = 0
    for ix in ids.split(32):
        x = f[ix][valid[ix]].double()
        s += x.sum(0); ss += x.square().sum(0); count += len(x)
    if count == 0:
        raise ValueError("No observed TRAIN audio frames")
    mean = (s / count).float(); std = (ss / count - (s / count).square()).clamp_min(1e-6).sqrt().float()
    samples = []
    for ix in ids.split(32):
        x = relative_features(f[ix].to(device), valid[ix].to(device), mean.to(device), std.to(device))[..., :1536]
        take = valid[ix].to(device)
        # Native stride reduces PCA memory but retains the frame clock.
        take = take[:, ::8]
        x = x[:, ::8][take]
        if len(x): samples.append(x.detach().cpu())
    if not samples:
        raise ValueError("No finite acoustic PCA samples")
    x = torch.cat(samples)
    if len(x) > max_samples:
        g = torch.Generator().manual_seed(20260921)
        x = x[torch.randperm(len(x), generator=g)[:max_samples]]
    # PCA on the selected execution device when possible, then persist on CPU.
    xp = x.to(device)
    qrank = min(rank, xp.shape[0], xp.shape[1])
    torch.manual_seed(20260921)
    _, _, projection = torch.pca_lowrank(xp, q=qrank, center=False, niter=3)
    if qrank < rank:
        pad = torch.zeros(1536, rank - qrank, device=projection.device, dtype=projection.dtype)
        projection = torch.cat((projection, pad), -1)
    return mean, std, projection.float().cpu()


def _make_model(stats, scales, device: torch.device):
    mean, std, projection = stats
    # TRAIN dynamic scales are fitted from the state target, not from dev.
    return RelativeAudioTiming(mean.to(device), std.to(device), projection.to(device),
                               scales.to(device), torch.ones(4, device=device) ).to(device)


def _fit_model_scales(q: dict, ids: torch.Tensor) -> torch.Tensor:
    y = q["target_centered_state"][ids]; valid = q["valid"][ids]
    return (y[valid].square().mean(0)).sqrt().clamp_min(.02)


def _fit_target_scales(q: dict, ids: torch.Tensor) -> torch.Tensor:
    """Fit physical channel scales using only the supplied TRAIN fold."""
    observed = (q["valid"][ids, :, None]
                & q["channel_mask"][ids, None]
                & q["anchor_valid"][ids, None])
    return fit_intensity_scales(q["motion"][ids], observed,
                                q["anchors"][ids], floor=.02)


def _target_state(q: dict, ids: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Rebuild the slow state with fold-local scales and no padded leakage."""
    motion = q["motion"][ids]
    valid = q["valid"][ids]
    observed = (valid[..., None] & q["channel_mask"][ids, None]
                & q["anchor_valid"][ids, None])
    raw, raw_mask = readout_slow_state(motion, observed, q["anchors"][ids], scales)
    state = masked_slow_state(raw, raw_mask, stride=16)["state"]
    return _masked_center(state, raw_mask.all(-1))


def _reverse_features(features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    out = torch.where(valid[..., None], features, 0.).clone()
    for row in range(len(out)):
        ix = valid[row].nonzero(as_tuple=True)[0]
        out[row, ix] = features[row, ix.flip(0)]
    return out


def _shuffle_features(features: torch.Tensor, valid: torch.Tensor, seed: int) -> torch.Tensor:
    out = torch.where(valid[..., None], features, 0.).clone()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    for row in range(len(out)):
        ix = valid[row].nonzero(as_tuple=True)[0]
        order = ix[torch.randperm(len(ix), generator=generator)]
        out[row, ix] = features[row, order]
    return out


def _mismatch_features(q: dict, ids: torch.Tensor, feature_index: torch.Tensor,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Transfer a donor clip's observed audio onto the query clip's frame clock."""
    dst_valid = q["valid"][ids].to(device)
    src = q["audio_features"][feature_index[ids]].to(device)
    src_valid = q["valid"][feature_index[ids]].to(device)
    out = torch.zeros_like(src)
    for row in range(len(ids)):
        di = dst_valid[row].nonzero(as_tuple=True)[0]
        si = src_valid[row].nonzero(as_tuple=True)[0]
        n = min(len(di), len(si))
        if n:
            out[row, di[:n]] = src[row, si[:n]]
    return out, dst_valid


@torch.no_grad()
def _predict(model, q: dict, ids: torch.Tensor, device: torch.device,
             mode: str = "audio", feature_index: torch.Tensor | None = None,
             input_intervention: str = "real"):
    model.eval()
    rows = []
    for ix in ids.split(32):
        b = subset(q, ix, device)
        if feature_index is None:
            feat, mask = b["audio_features"], b["valid"]
        else:
            feat, mask = _mismatch_features(q, ix, feature_index, device)
        if input_intervention == "reverse":
            feat = _reverse_features(feat, mask)
        elif input_intervention == "shuffle":
            feat = _shuffle_features(feat, mask, 20260921 + int(ix[0]))
        elif input_intervention != "real":
            raise ValueError(input_intervention)
        pred = model(feat, mask, "audio")["state"]
        if mode == "static":
            pred = pred * 0.
        elif mode != "audio":
            raise ValueError(mode)
        rows.append(pred.cpu())
    return torch.cat(rows)


def _train_epoch(model, q: dict, ids: torch.Tensor, opt, rng, device: torch.device) -> float:
    model.train(); losses = []
    target = q["target_centered_state"]
    for ix in ids[torch.randperm(len(ids), generator=rng)].split(32):
        b = trim(q, ix, device)
        pred = model(b["audio_features"], b["valid"])["state"]
        y = b["target_centered_state"]
        scale = model.dynamic_scales
        err = ((pred - y) / scale).square().mean(-1)
        loss = (torch.where(b["valid"], err, 0.).sum(1) / b["valid"].sum(1).clamp_min(1)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite signed-state objective")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step(); losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2: return None
    x = x - x.mean(); y = y - y.mean(); den = torch.sqrt(x.square().sum() * y.square().sum())
    if not torch.isfinite(den) or float(den) <= 1e-12: return None
    v = (x * y).sum() / den
    return float(v.clamp(-1., 1.)) if torch.isfinite(v) else None


def _peak_lag(x: torch.Tensor, y: torch.Tensor, max_lag: int = 15) -> int | None:
    n = int(x.numel())
    if n < 3: return None
    best = (-float("inf"), None)
    for lag in range(-min(max_lag, n - 2), min(max_lag, n - 2) + 1):
        xx, yy = (x[lag:], y[:n-lag]) if lag >= 0 else (x[:n+lag], y[-lag:])
        c = _safe_corr(xx, yy)
        if c is not None and c > best[0]: best = (c, lag)
    return best[1]


def _runs(mask: torch.Tensor):
    ids = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not ids: return []
    out=[]; left=prev=ids[0]
    for i in ids[1:]:
        if i != prev+1: out.append((left, prev+1)); left=i
        prev=i
    out.append((left, prev+1)); return out


def _envelope_diagnostics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    corr=[[] for _ in range(2)]; lag=[[] for _ in range(2)]; missing=[0,0]; denom=[0,0]
    for b in range(len(valid)):
        for left,right in _runs(valid[b]):
            for r in range(2):
                x,y=pred[b,left:right,r],target[b,left:right,r]
                c=_safe_corr(x,y)
                if c is not None: corr[r].append(c)
                l=_peak_lag(x,y)
                if l is not None: lag[r].append(l)
                th=max(1e-6,float(torch.quantile(y,.75))); sx=max(1e-6,.1*float(x.max()))
                active=y>=th; d=int(active.sum()); missing[r]+=int((active&(x<=sx)).sum()); denom[r]+=d
    def mean(v): return float(np.mean(v)) if v else None
    return {
        "correlation":{"brow":mean(corr[0]),"eye":mean(corr[1])},
        "correlation_macro":mean(corr[0]+corr[1]),
        "peak_lag_frames":{"brow":mean(lag[0]),"eye":mean(lag[1])},
        "peak_abs_lag_frames_macro":mean([abs(v) for v in lag[0]+lag[1]]),
        "prior_missing_event_coverage":{"brow":missing[0]/denom[0] if denom[0] else None,
                                         "eye":missing[1]/denom[1] if denom[1] else None},
        "prior_missing_event_coverage_macro":sum(missing)/sum(denom) if sum(denom) else None,
        "definition":"target q75; source <= max(1e-6, 0.1*source max); lag max 15 frames",
    }


@torch.no_grad()
def _state_conditions(model, q: dict, device: torch.device):
    ids = torch.arange(len(q["valid"]))
    full = _predict(model,q,ids,device,"audio")
    static = _predict(model,q,ids,device,"static")
    reverse = _predict(model,q,ids,device,input_intervention="reverse")
    shuffle = _predict(model,q,ids,device,input_intervention="shuffle")
    # Keep donor and query clocks comparable: rotate only within equal-length
    # groups, then transfer donor observed frames onto the query clock.
    perm = ids.clone()
    groups = {}
    for i, n in enumerate(q["valid"].sum(1).tolist()): groups.setdefault(int(n), []).append(i)
    for members in groups.values():
        if len(members) > 1:
            rot = members[1:] + members[:1]
            for dst, src in zip(members, rot): perm[dst] = src
    mismatch = _predict(model,q,ids,device,feature_index=perm)
    return {"full":full,"static":static,"reverse":reverse,"shuffle":shuffle,"mismatch":mismatch}


def _condition_metrics(conditions, q, model, device):
    target=q["target_centered_state"]; valid=q["valid"]
    target_upper, valid_d = _upper_target(q, device)
    # ``target_scales`` is fitted at the prepared-data level and is carried by
    # the model buffer; split dictionaries intentionally contain only clip
    # payloads.  Use the same TRAIN-fitted normalization for both target and
    # predicted envelopes so the comparison is scale invariant.
    target_env=regional_envelope(target_upper,valid_d,channel_scales=model.channel_scales[list(UPPER_INDICES)])
    result={}
    for name,state in conditions.items():
        pred=state
        sm=state_metrics(pred,target,valid)
        delta=model.lift_slow_state(pred.to(device), model.channel_scales).cpu() if hasattr(model,'lift_slow_state') else None
        # RelativeAudioTiming returns state; use its public lift helper.
        from kinetalk_b0.models.slow_state_affect import lift_slow_state
        delta=lift_slow_state(pred.to(device),model.channel_scales).cpu()[...,list(UPPER_INDICES)]
        env=regional_envelope(delta.to(device),valid_d,channel_scales=model.channel_scales[list(UPPER_INDICES)]).cpu()
        tm=_envelope_diagnostics(env,target_env.cpu(),valid)
        result[name]={"state":sm,"envelope_mse":float(((env-target_env.cpu()).square()[valid]).mean()),
                      "temporal_diagnostics":tm}
    return result


def _student_gate(metrics: dict, protection: dict, oracle_passed: bool) -> dict:
    """Conservative gate: a lower MSE alone can never declare success."""
    full = metrics["full"]; static = metrics["static"]
    better = {
        "state_mse_vs_static": full["state"]["mse"] < static["state"]["mse"],
        "envelope_mse_vs_static": full["envelope_mse"] < static["envelope_mse"],
        "state_corr_positive": full["state"]["correlation"] > 0,
        "corr_vs_static": full["temporal_diagnostics"]["correlation_macro"] is not None
                           and (static["temporal_diagnostics"]["correlation_macro"] is None
                                or full["temporal_diagnostics"]["correlation_macro"]
                                > static["temporal_diagnostics"]["correlation_macro"]),
    }
    for name in ("reverse", "shuffle", "mismatch"):
        better[f"state_mse_vs_{name}"] = full["state"]["mse"] < metrics[name]["state"]["mse"]
        better[f"envelope_mse_vs_{name}"] = full["envelope_mse"] < metrics[name]["envelope_mse"]
    diag = full["temporal_diagnostics"]
    better["lag_defined"] = diag["peak_abs_lag_frames_macro"] is not None
    better["event_coverage_defined"] = diag["prior_missing_event_coverage_macro"] is not None
    better["oracle_receiver_passed"] = bool(oracle_passed)
    better["nonupper43_exact"] = bool(protection["nonupper43_exact"])
    better["upper_mean_preserved"] = bool(protection["upper_mean_preserved"])
    better["mouth_protection"] = all(x["passed"] for x in protection["mouth_protection"].values())
    return {"checks": better, "passed": bool(all(better.values()))}


@torch.no_grad()
def _fullface(model, data, system, audio, identities, device, out: Path):
    q=data["splits"]["validation"]; valid=q["valid"]; ids=torch.arange(len(valid)); cond=_state_conditions(model,q,device)
    predictions={}; protection={}; means={}; nonupper={}
    from kinetalk_b0.models.slow_state_affect import compose_upper_face, lift_slow_state
    for seed in SEEDS:
        noise=torch.randn((*valid.shape,52),generator=torch.Generator().manual_seed(seed))
        base=[]
        for ix in ids.split(16):
            b=subset(q,ix,device); b["frozen_local"]=audio(b["audio_features"],b["valid"])["local"]
            base.append(old_prediction(system,b,identities,noise[ix].to(device)).cpu())
        base=torch.cat(base); predictions[f"{seed}/base"]=base
        mean=(base[...,list(UPPER_INDICES)]*valid[...,None]).sum(1)/valid.sum(1)[:,None]
        for name,state in cond.items():
            delta=lift_slow_state(state.to(device),model.channel_scales).cpu()[...,list(UPPER_INDICES)]
            upper=mean[:,None]+delta; pred=compose_upper_face(base,upper,valid); key=f"{seed}/{name}"; predictions[key]=pred
            nonupper[key]=bool(torch.equal(pred[...,list(NOT_UPPER)].view(torch.int32),base[...,list(NOT_UPPER)].view(torch.int32)))
            means[key]=float(((pred[...,list(UPPER_INDICES)]*valid[...,None]).sum(1)/valid.sum(1)[:,None]-mean).abs().max())
            protection[key]=protection_report(pred,base,q["motion"],valid,q["channel_mask"],q["emotion_id"])
    return {"nonupper43_exact":all(nonupper.values()),"max_upper_mean_drift":max(means.values()),
            "upper_mean_preserved":max(means.values())<1e-5,"mouth_protection":protection}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","source","output"): p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--selection-epochs",type=int,default=60)
    p.add_argument("--oracle-evidence",type=Path,default=None)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args(); device=_device(a.device)
    if a.output.exists(): raise FileExistsError("Fresh output required")
    a.output.mkdir(parents=True); started=time.monotonic(); random.seed(47);np.random.seed(47);torch.manual_seed(47)
    data,system,audio,identities,_=load_context(a.data,a.source,device,47)
    q=data["splits"]["train"]
    # In smoke mode retain the complete approved split.  Subsetting by the
    # first N clips can remove an entire held speaker/sentence group and make
    # the OOF fit fold empty.  Smoke is one epoch over the real protocol, not
    # a reduced or redefined data split.
    # Keep the approved metadata groups intact in smoke mode.  We only reduce
    # the number of optimization clips below; development remains untouched.
    speakers=sorted(set(q["speaker"])); held_s={s for i,s in enumerate(speakers) if i%5==0}
    sentences=sorted(set(q["sentence_id"])); held_sent=set(sentences[::4])
    fit=torch.tensor([i for i,(s,t) in enumerate(zip(q["speaker"],q["sentence_id"]))
                      if s not in held_s and t not in held_sent], dtype=torch.long)
    cal=torch.tensor([i for i,(s,t) in enumerate(zip(q["speaker"],q["sentence_id"]))
                      if s in held_s and t in held_sent], dtype=torch.long)
    if len(fit)==0 or len(cal)==0: raise RuntimeError("Strict speaker×sentence OOF fold is empty")
    fold_target_scales=_fit_target_scales(q,fit)
    q["target_centered_state"]=_target_state(q,torch.arange(len(q["valid"])),fold_target_scales)
    scales=_fit_model_scales(q,fit);stats=_fit_statistics(q,fit,device);model=_make_model(stats,fold_target_scales,device)
    model.dynamic_scales.copy_(scales.to(device))
    oracle_passed=False
    oracle_result=None
    if a.oracle_evidence is not None and a.oracle_evidence.exists():
        oracle_result=json.loads((a.oracle_evidence/"evaluation.json").read_text()) if a.oracle_evidence.is_dir() else json.loads(a.oracle_evidence.read_text())
        oracle_passed=bool(oracle_result.get("receiver_passed"))
    protocol={"schema":"signed_audio_state_v2","state_names":STATE_NAMES,"selection":"strict speaker×sentence OOF; fixed epoch full refit","fit_clips":len(fit),"calibration_clips":len(cal),"held_speakers":sorted(held_s),"held_sentences":sorted(held_sent),"smoke":a.smoke,"test_loaded":False,"default_replaced":False,"oracle_passed":oracle_passed,"source_sha256":sha(a.source)}
    save_json(a.output/"protocol.json",protocol)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02); rng=torch.Generator().manual_seed(47); best=float("inf");best_epoch=1; hist=[]
    for epoch in range(1,(1 if a.smoke else a.selection_epochs)+1):
        loss=_train_epoch(model,q,fit,opt,rng,device); rec={"epoch":epoch,"loss":loss}
        with torch.no_grad(): rec["oof"] = state_metrics(_predict(model,q,cal,device),q["target_centered_state"][cal],q["valid"][cal])
        hist.append(rec)
        if rec["oof"]["mse"]<best: best=rec["oof"]["mse"];best_epoch=epoch
        save_json(a.output/"selection_history.json",hist)
    save_json(a.output/"selection.json",{"chosen_epochs":best_epoch,"best_oof_mse":best,"development_used":False})
    # Full TRAIN refit with the selected epoch count.
    # Smoke changes only the epoch budget.  The refit still uses every
    # approved TRAIN clip so its outputs are representative of the protocol.
    all_ids = torch.arange(len(q["valid"]), dtype=torch.long)
    full_target_scales=_fit_target_scales(q,all_ids)
    q["target_centered_state"]=_target_state(q,torch.arange(len(q["valid"])),full_target_scales)
    stats=_fit_statistics(q,all_ids,device);model=_make_model(stats,full_target_scales,device);model.dynamic_scales.copy_(_fit_model_scales(q,all_ids).to(device));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02);rng=torch.Generator().manual_seed(47)
    model.train()
    for _ in range(best_epoch): _train_epoch(model,q,all_ids,opt,rng,device)
    model.eval()
    cond=_state_conditions(model,data["splits"]["validation"],device); metrics=_condition_metrics(cond,data["splits"]["validation"],model,device); protection=_fullface(model,data,system,audio,identities,device,a.output)
    gate=_student_gate(metrics,protection,oracle_passed)
    result={"oracle_only":False,"audio_student_started":True,"conditions":metrics,"fullface_protection":protection,"selected_epochs":best_epoch,"gate":gate,"oracle_evidence":oracle_result,"passed":bool(gate["passed"] and not a.smoke),"test_loaded":False,"default_replaced":False,"elapsed_seconds":time.monotonic()-started}
    save_json(a.output/"evaluation.json",result);save_checkpoint(a.output/"final.pt",{"model":model.state_dict(),"protocol":protocol,"protocol_sha256":canonical_hash(protocol),"selected_epochs":best_epoch,"test_loaded":False});save_json(a.output/"status.json",{"status":"complete","passed":result["passed"],"test_loaded":False})
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
