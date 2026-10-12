"""Optional counterfactual local-event execution supervision for a future run.

No GT motion is accepted.  The loss measures what the decoder actually does
when sampled events are added to otherwise fixed content/state/style conditions.
The zero-event motion is a detached reference, not a neutral target.  Incomplete,
unobserved, ambiguous-direction or out-of-range events receive zero weight.
This is a coefficient-space weak-observation loss, not physical eyelid contact
or proof that any event type is identifiable from audio.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F

SCHEMA = "counterfactual-local-event-execution-v1"
GROUPS = ("eyelid_closure", "brow_emphasis", "cheek_nose_emphasis", "lip_tension")
# The current teacher parser detects these four event families.  Mouth corners,
# jaw opening, rounding, eye look, and identity coefficients are not constrained
# by this loss.  Tension is one channel family, never all emotional lip motion.
CHANNEL_FAMILIES = (
    ("eyeBlinkLeft", "eyeBlinkRight"),
    ("browInnerUp", "browOuterUpLeft", "browOuterUpRight", "browDownLeft", "browDownRight"),
    ("cheekPuff", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight"),
    ("mouthStretchLeft", "mouthStretchRight", "mouthPressLeft", "mouthPressRight"),
)


@dataclass(frozen=True)
class EventExecutionConfig:
    # These are guarded engineering ranges, not dataset-estimated thresholds.
    # They exceed the present weak parser's .25/.055/.045/.055 active thresholds
    # at envelope floor .8. TRAIN-only calibration must replace them before a
    # paper experiment. Schedules outside them are reported and skipped.
    amplitude_min: tuple[float, ...] = (.32, .08, .065, .08)
    amplitude_max: tuple[float, ...] = (1., .40, .40, .40)
    weights: tuple[float, ...] = (1., .5, .25, .25)
    envelope_floor: float = .8
    relative_tolerance: float = .20
    absolute_tolerance: float = .01
    phase_weight: float = .10
    phase_delta_tolerance: float = .06
    peak_weight: float = .5
    off_weight: float = .10
    brow_orthogonal_weight: float = .05
    direction_min: float = .02
    direction_stability_min: float = .65
    eyelid_baseline_max: float = .70
    eyelid_min_causal_increment: float = .08
    off_halo_frames: int = 1
    # Blink and cheek events may share legitimate periocular deformation; an
    # inactive region is not forced to zero during its coupled event.
    nuisance_couplings: tuple[tuple[int, ...], ...] = ((2,), (), (0,), ())


def _observations(channels, device, dtype):
    names = [str(n).lower() for n in channels]
    if len(names) != 52 or len(set(names)) != 52:
        raise ValueError("52 unique named fixed-rig channels are required")
    index = {n: i for i, n in enumerate(names)}
    missing = [n for group in CHANNEL_FAMILIES for n in group if n.lower() not in index]
    if missing:
        raise ValueError(f"event observation channels are unavailable: {missing}")
    families = [[index[n.lower()] for n in group] for group in CHANNEL_FAMILIES]
    matrix = torch.zeros(52, 5, device=device, dtype=dtype)
    # blink mean; brow lift and down are separate directional coordinates;
    # cheek/nose mean; lip tension mean. Same scale as current anchor parser.
    for n in CHANNEL_FAMILIES[0]: matrix[index[n.lower()],0] = .5
    for n,w in (("browInnerUp",.5),("browOuterUpLeft",.25),("browOuterUpRight",.25)):
        matrix[index[n.lower()],1] = w
    for n in ("browDownLeft","browDownRight"): matrix[index[n.lower()],2] = .5
    for n in CHANNEL_FAMILIES[2]: matrix[index[n.lower()],3] = .2
    for n in CHANNEL_FAMILIES[3]: matrix[index[n.lower()],4] = .25
    return matrix, families


def _runs(valid, times):
    # Only metadata controls boundaries. No magnitude-triggered clipping of
    # genuinely fast face motion, and no compacted clock.
    connected = valid[:,1:] & valid[:,:-1] & torch.isclose(times[:,1:]-times[:,:-1],
        torch.full_like(times[:,1:], .04),rtol=1e-4,atol=1e-7)
    result = []
    for b in range(len(valid)):
        start = 0
        while start < valid.shape[1]:
            if not bool(valid[b,start]):start += 1;continue
            end = start+1
            while end < valid.shape[1] and bool(connected[b,end-1]):end += 1
            result.append((b,start,end));start=end
    return result, connected


def _masked_mean(value, mask):
    return torch.where(mask,value,torch.zeros_like(value)).sum()/mask.sum().clamp_min(1)


def event_execution_loss(motion_event: torch.Tensor, motion_zero: torch.Tensor,
                         sampled_events: torch.Tensor, expression_state: torch.Tensor,
                         base: torch.Tensor, valid: torch.Tensor, times: torch.Tensor,
                         channel_mask: torch.Tensor, channels,
                         *, config: EventExecutionConfig | None = None,
                         state_quality: torch.Tensor | None = None,
                         event_quality: torch.Tensor | None = None,
                         quality_reference_motion: torch.Tensor | None = None) -> dict[str, Any]:
    """Supervise sampled local effects without full-face GT L2.

    All tensors use the identical source clock. ``motion_event`` is decoded
    from sampled events; ``motion_zero`` is decoded with all-zero events while
    holding content/base/state/emotion/style fixed. ``expression_state`` is the
    requested 10D anchored state, not GT motion. Schedules, state, base and the
    zero-event reference are detached. Gradients enter only motion_event's
    graph; callers must detach fixed content/state/style if they want to update
    only the decoder. This loss does not train the event hazard by a surrogate GT.

    The brow event's type/amplitude lacks direction. A stable nonzero requested
    brow-state vector gives a provisional direction; cheek/lip use the stable
    sign of their requested anchored axis. Low-magnitude/unstable directions
    have zero quality. For reliable neutral-state emphasis a future calibrated
    signed event label/prototype is required; this API will not invent one.
    ``quality_reference_motion`` can pin eligibility to the frozen parent
    checkpoint's zero-event output. Formal runs should use it: otherwise a
    changing current zero-event baseline changes the eligible-event population.
    """
    cfg = config or EventExecutionConfig()
    if motion_event.shape != motion_zero.shape or motion_event.shape != base.shape or motion_event.ndim != 3 or motion_event.shape[-1] != 52:
        raise ValueError("matching motion_event/motion_zero/base [B,T,52] required")
    b,t,_ = motion_event.shape
    if valid.shape != (b,t) or valid.dtype != torch.bool or times.shape != (b,t):
        raise ValueError("bool valid and source times [B,T] required")
    if channel_mask.shape != (b,52) or channel_mask.dtype != torch.bool or expression_state.shape != (b,t,10):
        raise ValueError("observed channels [B,52] and anchored state [B,T,10] required")
    if sampled_events.shape == (b,t,12):sampled_events=sampled_events.reshape(b,t,4,3)
    if sampled_events.shape != (b,t,4,3):raise ValueError("sampled events [B,T,4,3] required")
    if any(len(v) != 4 for v in (cfg.amplitude_min,cfg.amplitude_max,cfg.weights,cfg.nuisance_couplings)):
        raise ValueError("configuration requires four event families")
    if not 0 <= cfg.envelope_floor <= 1 or cfg.off_halo_frames < 0:
        raise ValueError("invalid envelope or halo configuration")
    if any(not 0 <= low < high <= 1 for low,high in zip(cfg.amplitude_min,cfg.amplitude_max)) or any(w<0 for w in cfg.weights):
        raise ValueError("amplitude ranges must lie in [0,1] and weights be nonnegative")
    obs=valid[...,None]&channel_mask[:,None]
    event_motion=torch.where(obs,motion_event,0.)
    zero_motion=torch.where(obs,motion_zero.detach(),0.)
    base=torch.where(obs,base.detach(),0.)
    if not torch.isfinite(event_motion).all() or not torch.isfinite(zero_motion).all() or not torch.isfinite(base).all():
        raise ValueError("nonfinite observed motion")
    schedules=sampled_events.detach()
    if not torch.isfinite(schedules[valid]).all():raise ValueError("nonfinite observed event schedule")
    state=torch.where(valid[...,None],expression_state.detach(),0.)
    if state_quality is None:state_quality=torch.isfinite(state).to(state.dtype)
    if state_quality.shape != (b,t,10):raise ValueError("state_quality [B,T,10] required")
    state_quality=torch.nan_to_num(state_quality.detach(),nan=0.,posinf=0.,neginf=0.).clamp(0,1)
    state=torch.nan_to_num(state)
    if event_quality is None:event_quality=torch.ones(b,t,4,device=motion_event.device,dtype=motion_event.dtype)
    if event_quality.shape != (b,t,4):raise ValueError("event_quality [B,T,4] required")
    event_quality=torch.nan_to_num(event_quality.detach(),nan=0.,posinf=0.,neginf=0.).clamp(0,1)
    matrix,families=_observations(channels,motion_event.device,motion_event.dtype)
    event_obs=event_motion@matrix;zero_obs=zero_motion@matrix
    if quality_reference_motion is not None:
        if quality_reference_motion.shape!=motion_event.shape:raise ValueError('quality reference must share the exact source motion shape')
        reference=torch.where(obs,quality_reference_motion.detach(),0.)
        if not torch.isfinite(reference).all():raise ValueError('nonfinite observed frozen quality reference')
        quality_obs=reference@matrix
    else:quality_obs=zero_obs
    delta=event_obs-zero_obs
    family_available=torch.stack([channel_mask[:,ix].all(-1) for ix in families],-1)
    active=schedules[...,0]>.5
    phase=schedules[...,1]
    amp=schedules[...,2]
    quality=torch.zeros(b,t,4,device=motion_event.device,dtype=motion_event.dtype)
    complete=torch.zeros(b,t,4,device=motion_event.device,dtype=torch.bool)
    ids=torch.full((b,t,4),-1,device=motion_event.device,dtype=torch.long)
    direction=torch.zeros(b,t,4,2,device=motion_event.device,dtype=motion_event.dtype)
    ranges=torch.tensor(cfg.amplitude_min,device=amp.device,dtype=amp.dtype),torch.tensor(cfg.amplitude_max,device=amp.device,dtype=amp.dtype)
    amplitude_ok=(amp>=ranges[0])&(amp<=ranges[1])
    runs,connected=_runs(valid,times)
    counts={name:{'complete_events':0,'censored_events':0,'unobserved_events':0,'out_of_range_events':0,'ambiguous_direction_events':0,'insufficient_headroom_events':0,'eligible_events':0} for name in GROUPS}
    serial=0
    for bi,lo,hi in runs:
        for g,name in enumerate(GROUPS):
            frame=lo
            while frame < hi:
                if not bool(active[bi,frame,g]):frame += 1;continue
                end=frame+1
                while end<hi and bool(active[bi,end,g]) and float(phase[bi,end,g])>=float(phase[bi,end-1,g])-1e-7:end += 1
                if frame==lo or end==hi:
                    counts[name]['censored_events']+=1;frame=end;continue
                counts[name]['complete_events']+=1
                complete[bi,frame:end,g]=True;ids[bi,frame:end,g]=serial;serial+=1
                if not bool(family_available[bi,g]):
                    counts[name]['unobserved_events']+=1;frame=end;continue
                if not bool(amplitude_ok[bi,frame:end,g].all()) or not bool(((phase[bi,frame:end,g]>=0)&(phase[bi,frame:end,g]<=1)).all()):
                    counts[name]['out_of_range_events']+=1;frame=end;continue
                q=event_quality[bi,frame:end,g].amin()
                if g==0:
                    # Nearly closed baseline eyelids have little identifiable
                    # causal increment available. Do not claim execution from
                    # their standing closure, or force an impossible overshoot.
                    if (float(quality_obs[bi,frame:end,0].amax()) >= cfg.eyelid_baseline_max or
                        float((amp[bi,frame:end,0]-quality_obs[bi,frame:end,0]).amin())<cfg.eyelid_min_causal_increment):
                        counts[name]['ambiguous_direction_events']+=1;frame=end;continue
                    direction[bi,frame:end,g,0]=1.
                else:
                    axes=[2,3] if g==1 else ([5] if g==2 else [1])
                    values=state[bi,frame:end][:,axes]
                    pooled=values.mean(0)
                    strength=pooled.abs().mean()
                    stable=pooled.abs().sum()/values.abs().mean(0).sum().clamp_min(1e-6)
                    q=q*state_quality[bi,frame:end][:,axes].amin()
                    if float(strength)<cfg.direction_min or float(stable)<cfg.direction_stability_min:
                        counts[name]['ambiguous_direction_events']+=1;frame=end;continue
                    d=pooled/strength.clamp_min(1e-6)  # mean abs direction=1
                    family_zero=quality_obs[bi,frame:end,1:3] if g==1 else quality_obs[bi,frame:end,3 if g==2 else 4,None]
                    desired_peak=family_zero+amp[bi,frame:end,g,None]*d[None]
                    if not bool(((family_zero>=0)&(family_zero<=1)&(desired_peak>=0)&(desired_peak<=1)).all()):
                        counts[name]['insufficient_headroom_events']+=1;frame=end;continue
                    direction[bi,frame:end,g,:len(d)]=d
                quality[bi,frame:end,g]=q
                if float(q)>0:counts[name]['eligible_events']+=1
                frame=end
    # Projection is a signed local causal increment in coefficient-anchor
    # units. Brow orthogonal effect is kept as a nuisance diagnostic/penalty.
    projection=torch.zeros_like(amp)
    projection[...,0]=delta[...,0]
    d=direction[...,1,:]
    projection[...,1]=(delta[...,1:3]*d).sum(-1)/d.square().sum(-1).clamp_min(1e-6)
    projection[...,2]=delta[...,3]*direction[...,2,0]
    projection[...,3]=delta[...,4]*direction[...,3,0]
    shape=cfg.envelope_floor+(1-cfg.envelope_floor)*torch.sin(torch.pi*phase.clamp(0,1))
    wanted=amp*shape
    # Blink amplitude is an absolute eyeBlink peak in the current parser;
    # other family amplitudes are anchor departures. The blink increment must
    # subtract the standing eyelid state rather than inflate the raw target.
    blink_absolute_wanted=wanted[...,0].clone()
    wanted=wanted.clone();wanted[...,0]=torch.maximum((blink_absolute_wanted-zero_obs[...,0]).clamp_min(0.),cfg.eyelid_min_causal_increment*shape[...,0])
    tolerance=cfg.absolute_tolerance+cfg.relative_tolerance*amp
    low=(wanted-tolerance).clamp_min(0.);high=wanted+tolerance
    error=(F.relu(low-projection)+F.relu(projection-high))/amp.clamp_min(.03)
    # Compare actual absolute eyelid closure too: counterfactual increment alone
    # cannot hide an implausibly negative eyeBlink baseline. No physical contact
    # guarantee follows from this coefficient proxy.
    blink_absolute_error=(F.relu(blink_absolute_wanted-tolerance[...,0]-event_obs[...,0])+
                          F.relu(event_obs[...,0]-blink_absolute_wanted-tolerance[...,0])+
                          F.relu(event_obs[...,0]-1.))/amp[...,0].clamp_min(.03)
    blink_causal_error=F.relu(cfg.eyelid_min_causal_increment*shape[...,0]-projection[...,0])/.10
    active_error=F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none')
    active_error=active_error.clone();active_error[...,0]+=(F.smooth_l1_loss(blink_absolute_error,torch.zeros_like(blink_absolute_error),reduction='none')+
                                                         F.smooth_l1_loss(blink_causal_error,torch.zeros_like(blink_causal_error),reduction='none'))
    phase_mask=(ids[:,1:]>=0)&ids[:,1:].eq(ids[:,:-1])&connected[...,None]
    phase_quality=torch.minimum(quality[:,1:],quality[:,:-1])*phase_mask
    actual_delta=(projection[:,1:]-projection[:,:-1])/amp[:,1:].clamp_min(.03)
    desired_delta=(wanted[:,1:]-wanted[:,:-1])/amp[:,1:].clamp_min(.03)
    phase_error=F.relu((actual_delta-desired_delta).abs()-cfg.phase_delta_tolerance)
    phase_error=F.smooth_l1_loss(phase_error,torch.zeros_like(phase_error),reduction='none')
    peak_mask=(phase>=.25)&(phase<=.75)&complete
    peak_target=amp.clone();peak_target[...,0]=(amp[...,0]-zero_obs[...,0]).clamp_min(cfg.eyelid_min_causal_increment)
    peak_error=F.smooth_l1_loss(F.relu(peak_target*(1-cfg.relative_tolerance)-projection)/amp.clamp_min(.03),torch.zeros_like(amp),reduction='none')
    # Off support: only local family effects are restrained relative to its
    # fixed no-event output. Legitimate coupled events and a short edge halo
    # disable this penalty. No jaw/smile/rounding emotional channels are frozen.
    off_mask=torch.zeros_like(active)
    for bi,lo,hi in runs:
        for g,couplings in enumerate(cfg.nuisance_couplings):
            forbidden=active[bi,lo:hi,g].clone()
            for other in couplings:forbidden |= active[bi,lo:hi,other]
            original=forbidden.clone()
            for shift in range(1,cfg.off_halo_frames+1):
                forbidden[shift:] |= original[:-shift];forbidden[:-shift] |= original[shift:]
            off_mask[bi,lo:hi,g]=~forbidden&family_available[bi,g]
    off_values=torch.stack((delta[...,0].abs(),delta[...,1:3].abs().mean(-1),delta[...,3].abs(),delta[...,4].abs()),-1)
    off_error=F.smooth_l1_loss(F.relu(off_values-cfg.absolute_tolerance)/.10,torch.zeros_like(off_values),reduction='none')
    brow_orth=delta[...,1:3]-projection[...,1,None]*d
    nuisance_error=F.smooth_l1_loss(brow_orth,torch.zeros_like(brow_orth),reduction='none').mean(-1)
    group_losses=[];metrics={}
    zero=event_motion.sum()*0.
    for g,name in enumerate(GROUPS):
        q=quality[...,g];denom=q.sum().clamp_min(1)
        active_loss=(active_error[...,g]*q).sum()/denom
        phase_loss=(phase_error[...,g]*phase_quality[...,g]).sum()/phase_quality[...,g].sum().clamp_min(1)
        peak_q=q*peak_mask[...,g]
        peak_loss=(peak_error[...,g]*peak_q).sum()/peak_q.sum().clamp_min(1)
        off_loss=_masked_mean(off_error[...,g],off_mask[...,g]) if float(q.sum())>0 else zero
        nuisance_loss=(nuisance_error*q).sum()/denom if g==1 else zero
        loss=active_loss+cfg.phase_weight*phase_loss+cfg.peak_weight*peak_loss+cfg.off_weight*off_loss+cfg.brow_orthogonal_weight*nuisance_loss
        group_losses.append(loss*cfg.weights[g])
        metrics[name]={**counts[name], 'quality_active_frames':float(q.sum().detach()),
                       'active':active_loss,'phase':phase_loss,'peak':peak_loss,'off_support':off_loss,
                       'brow_orthogonal':nuisance_loss,'weighted_total':loss*cfg.weights[g]}
    total=torch.stack(group_losses).sum()/sum(w for w in cfg.weights if w>0) if any(w>0 for w in cfg.weights) else zero
    return {'total':total,'groups':metrics,'quality':quality.detach(),'complete_event_mask':complete,
            'observed_increment':projection,'requested_increment':wanted.detach(),
            'schema':SCHEMA,'config':asdict(cfg),
            'quality_reference_policy':'frozen_external_zero_event_motion' if quality_reference_motion is not None else 'current_detached_zero_event_motion_provisional',
            'gradient_policy':'motion_event branch only; zero-event/base/state/schedule targets detached; caller detaches conditioning for decoder-only update',
            'measurement_limit':'ARKit coefficient weak observations; no geometric contact/psychological emotion claim'}
