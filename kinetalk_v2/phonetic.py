"""Versioned rig observations and evidence-gated phonetic intervals.

The asset measures signed upper/lower *rig* point separation. It is neither a
collision solver nor a physiological lip contact label. Only externally supplied
closure/release evidence and TRAIN-fitted tolerances can activate constraints.
No phoneme is inferred from B0's jawOpen value in the production path.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import numpy as np

SCHEMA = "kinetalk_lip_intervals_v2.1"
NONE, CORE, RELEASE = 0, 1, 2
EXTERNAL_SOURCES = {"manual_closure_release", "external_closure_release"}
DIAGNOSTIC_SOURCE = "diagnostic_proxy_b0_jaw"


@dataclass(frozen=True)
class LipConstraintAsset:
    jacobian_mm: np.ndarray
    rest_gap_mm: np.ndarray
    channels: tuple
    pair_quality: np.ndarray
    digest: str
    provisional: bool = True

    @classmethod
    def load(cls, path, expected_channels=None):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"].item()) != SCHEMA:
                raise ValueError("Unsupported lip asset schema")
            j = data["jacobian_mm"].astype(np.float32)
            rest = data["rest_gap_mm"].astype(np.float32)
            channels = tuple(str(x) for x in data["channels"])
            quality = data["pair_quality"].astype(np.float32)
            provisional = bool(data["provisional"].item())
        if j.ndim != 2 or j.shape[1] != 52 or len(j) == 0:
            raise ValueError("Rig observations must be nonempty [K,52]")
        if rest.shape != (len(j),) or quality.shape != rest.shape:
            raise ValueError("Inconsistent lip asset rows")
        if len(channels) != 52 or len(set(channels)) != 52:
            raise ValueError("ARKit channel schema is incomplete/duplicated")
        if expected_channels is not None and tuple(expected_channels) != channels:
            raise ValueError("Motion channels differ from rig asset order")
        if not all(np.isfinite(x).all() for x in (j, rest, quality)):
            raise ValueError("Nonfinite rig asset")
        if (np.linalg.norm(j, axis=1) < 1e-6).any() or ((quality < 0) | (quality > 1)).any():
            raise ValueError("Degenerate observations or invalid geometry quality")
        return cls(j, rest, channels, quality,
                   hashlib.sha256(path.read_bytes()).hexdigest(), provisional)

    def observe(self, motion):
        """Signed separation in mm, including the rest geometry offset."""
        motion = np.asarray(motion)
        if motion.shape[-1] != 52:
            raise ValueError("Expected ordered ARKit52 coefficients")
        return motion @ self.jacobian_mm.T + self.rest_gap_mm


@dataclass(frozen=True)
class PhoneticEvidence:
    phase: object  # [T] or [B,T], integer NONE/CORE/RELEASE
    confidence: object
    source: str
    provenance: str


def align_bilabial_phases(times, segments, *, source, provenance,
                          min_confidence=.8):
    """Map externally identified closure/release intervals to the motion clock.

    Segments have start/end seconds, phone (p/b/m, optionally ARPAbet stress),
    phase ('core' or 'release'), and explicit confidence. Ordinary whole-phone
    forced alignment lacks closure/release boundaries and is deliberately not
    treated as contact supervision. ASR alone does not identify a lip seal.
    Invalid/overlapping equal-confidence labels fail, rather than choosing an
    arbitrary contact stage. Half-open intervals avoid double-boundary labels.
    """
    if source not in EXTERNAL_SOURCES or not provenance:
        raise ValueError("External closure/release provenance is required")
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("Strictly increasing finite frame times required")
    phases = np.zeros(len(t), dtype=np.int64)
    confidence = np.zeros(len(t), dtype=np.float32)
    for segment in segments:
        phone = re.sub(r"[012]$", "", str(segment["phone"]).lower())
        if phone not in {"p", "b", "m"}:
            continue
        phase = {"core": CORE, "release": RELEASE}.get(segment.get("phase"))
        # Whole-phone timing is not a closure label.
        if phase is None:
            continue
        start, end = float(segment["start"]), float(segment["end"])
        score = float(segment.get("confidence", 0.))
        if not np.isfinite([start, end, score]).all() or start >= end or not 0 <= score <= 1:
            raise ValueError("Invalid alignment interval/confidence")
        use = (t >= start) & (t < end) & (score >= min_confidence)
        conflict = use & (confidence == score) & (phases != phase) & (phases != NONE)
        if conflict.any():
            raise ValueError("Conflicting closure/release evidence")
        use &= score > confidence
        phases[use], confidence[use] = phase, score
    return PhoneticEvidence(phases, confidence, source, provenance)


@dataclass(frozen=True)
class PhoneticCalibration:
    lower_gap_mm: np.ndarray  # [3,K]; index 0 never activates a constraint
    upper_gap_mm: np.ndarray
    reliability: np.ndarray
    sample_count: np.ndarray
    base_mismatch_mm: np.ndarray
    asset_digest: str
    training_digest: str
    evidence_source: str
    diagnostic_only: bool

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, schema=np.array(SCHEMA), **self.__dict__)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if str(d["schema"].item()) != SCHEMA:
                raise ValueError("Unsupported interval calibration schema")
            arrays = {k: d[k].copy() for k in ("lower_gap_mm", "upper_gap_mm", "reliability", "sample_count", "base_mismatch_mm")}
            return cls(**arrays, **{k: str(d[k].item()) for k in ("asset_digest", "training_digest", "evidence_source")},
                       diagnostic_only=bool(d["diagnostic_only"].item()))


def _evidence_arrays(evidence, shape):
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    phases = array(evidence.phase)
    confidence = array(evidence.confidence).astype(np.float64)
    if phases.shape != shape or confidence.shape != shape:
        raise ValueError("Phonetic evidence must share the exact motion clock")
    if not np.issubdtype(phases.dtype, np.integer) or not np.isin(phases, [NONE, CORE, RELEASE]).all():
        raise ValueError("Unknown phonetic stage")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Invalid phonetic confidence")
    if evidence.source not in EXTERNAL_SOURCES | {DIAGNOSTIC_SOURCE} or not evidence.provenance:
        raise ValueError("Untraceable phonetic evidence")
    return phases, confidence


def calibrate_intervals(asset, motion, base, valid, evidence, *, split,
                        training_ids, fit_confidence, channel_mask,
                        min_samples=32, min_confidence=.8,
                        quantiles=(.025, .975), margin_mm=.25):
    """Robust TRAIN-only rig intervals; disagreement lowers their authority.

    fit_confidence [B,T] is independent fitting/visibility quality, not a model
    confidence head. channel_mask [B,52] prevents unobserved channels from being
    read as measured zeros. Bounds describe accepted fitted motion, including
    fitting tolerance; no literal zero-separation/contact promise is made.
    """
    if split != "train":
        raise ValueError("Interval calibration is TRAIN-only")
    motion, base = np.asarray(motion), np.asarray(base)
    valid = np.asarray(valid, dtype=bool)
    if motion.ndim != 3 or motion.shape != base.shape or motion.shape[-1] != 52 or valid.shape != motion.shape[:2]:
        raise ValueError("Aligned motion/base [B,T,52] and valid [B,T] required")
    if len(training_ids) != len(motion) or len(set(training_ids)) != len(training_ids):
        raise ValueError("Unique TRAIN recording/window identifiers required")
    phases, confidence = _evidence_arrays(evidence, valid.shape)
    fit = np.asarray(fit_confidence, dtype=np.float64)
    channels = np.asarray(channel_mask)
    if fit.shape != valid.shape or channels.shape != (len(motion), 52) or channels.dtype != np.bool_:
        raise ValueError("Independent fit confidence and observed channel mask required")
    if not np.isfinite(fit).all() or ((fit < 0) | (fit > 1)).any():
        raise ValueError("Invalid fit confidence")
    if min_samples < 2 or not 0 <= quantiles[0] < quantiles[1] <= 1 or margin_mm < 0:
        raise ValueError("Invalid interval calibration settings")
    gt, prediction = asset.observe(motion), asset.observe(base)
    k = len(asset.rest_gap_mm)
    lower = np.zeros((3,k), np.float32); upper = lower.copy()
    reliability = lower.copy(); mismatch = lower.copy()
    count = np.zeros((3,k), np.int64)
    dependency = np.abs(asset.jacobian_mm) > 1e-6
    observed = np.logical_or(channels[:,None,:], ~dependency[None,:,:]).all(-1)
    for phase in (CORE, RELEASE):
        for row in range(k):
            mask = (valid & (phases == phase) & (confidence >= min_confidence) &
                    (fit > 0) & observed[:,row,None] & np.isfinite(gt[...,row]) & np.isfinite(prediction[...,row]))
            count[phase,row] = mask.sum()
            if mask.sum() < min_samples:
                continue
            values = gt[...,row][mask]
            low, high = np.quantile(values, quantiles)
            lower[phase,row], upper[phase,row] = low-margin_mm, high+margin_mm
            error = np.median(np.abs(prediction[...,row][mask]-values))
            mismatch[phase,row] = error
            robust_scale = max(1.4826*np.median(np.abs(values-np.median(values))), margin_mm, .05)
            # A badly mismatched B0 must not become a high-authority contact target.
            authority = robust_scale/(robust_scale+error)
            reliability[phase,row] = (authority * np.mean(confidence[mask]*fit[mask]) * asset.pair_quality[row])
    provenance = json.dumps({"ids": sorted(training_ids), "source": evidence.source,
        "provenance": evidence.provenance, "quantiles": quantiles, "margin_mm": margin_mm,
        "min_samples": min_samples, "min_confidence": min_confidence}, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(provenance.encode("utf8"))
    for value in (motion, base, valid, phases, confidence, fit, channels):
        digest.update(np.ascontiguousarray(value).tobytes())
    return PhoneticCalibration(lower, upper, reliability, count, mismatch, asset.digest,
        digest.hexdigest(), evidence.source, evidence.source == DIAGNOSTIC_SOURCE)


def build_constraints(asset, calibration, evidence, valid, *, base=None,
                      allow_diagnostic=False, min_confidence=.8, release_weight=.25):
    """Build detached absolute J@y bounds for FeasibleResidualComposer.

    A missing calibration/evidence has no active rows. Weak acoustic evidence
    cannot be rescued by a learned confidence, jaw threshold, or query GT.
    At inference only audio-derived/external timing, B0 and TRAIN assets enter.
    """
    import torch
    if not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.ndim != 2:
        raise ValueError("Expected torch bool valid [B,T]")
    if not 0 <= release_weight <= 1 or not 0 <= min_confidence <= 1:
        raise ValueError("Invalid phonetic confidence/release weight")
    device = valid.device
    j = torch.as_tensor(asset.jacobian_mm, device=device).float()
    zeros = torch.zeros(*valid.shape, len(j), device=device)
    result = {"jacobian": j, "lower": zeros.clone(), "upper": zeros.clone(), "weight": zeros.clone()}
    if calibration is None or evidence is None:
        return result
    if calibration.asset_digest != asset.digest:
        raise ValueError("Calibration belongs to a different rig asset")
    if (calibration.diagnostic_only or evidence.source == DIAGNOSTIC_SOURCE) and not allow_diagnostic:
        raise ValueError("B0 diagnostic proxy is not production phonetic evidence")
    if evidence.source != calibration.evidence_source:
        raise ValueError("Evidence source differs from calibration")
    phases, confidence = _evidence_arrays(evidence, tuple(valid.shape))
    phase = torch.as_tensor(phases, device=device, dtype=torch.long)
    conf = torch.as_tensor(confidence, device=device, dtype=torch.float32)
    tables = [torch.as_tensor(x, device=device).float() for x in
              (calibration.lower_gap_mm, calibration.upper_gap_mm, calibration.reliability, calibration.base_mismatch_mm)]
    low, high, weight, mismatch = [x[phase] for x in tables]
    if any(x.shape != zeros.shape for x in (low, high, weight, mismatch)):
        raise ValueError("Calibration shape differs from rig")
    if not all(torch.isfinite(x).all() for x in tables) or (low > high).any() or ((weight < 0) | (weight > 1)).any():
        raise ValueError("Corrupt phonetic interval tables")
    active = valid & phase.ne(NONE) & conf.ge(min_confidence)
    phase_weight = torch.where(phase.eq(RELEASE), release_weight, 1.)
    weight = weight * conf[...,None].square() * active[...,None] * phase_weight[...,None]
    if base is not None:
        if base.shape != (*valid.shape,52):
            raise ValueError("Inference base must share the evidence clock")
        rest = torch.as_tensor(asset.rest_gap_mm, device=device).float()
        observation = base.detach().float() @ j.T + rest
        distance = (low-observation).clamp_min(0)+(observation-high).clamp_min(0)
        allowance = mismatch + (high-low)/2 + .25
        weight = weight * torch.exp(-distance/allowance.clamp_min(.25))
    rest = torch.as_tensor(asset.rest_gap_mm, device=device).float()
    result.update(lower=(low-rest).detach(), upper=(high-rest).detach(), weight=weight.detach())
    return result


def diagnostic_neutral_protection(base, valid, times, *, jaw_channel=17,
                                  closed_threshold=.04, min_run=4):
    """Explicit B0 low-jaw pilot; NOT bilabial/ASR/contact supervision.

    Used only to debug the composer. Runs split at clock gaps and invalid data;
    core endpoints are trimmed. Confidence .2 keeps it out of default phonetic
    calibration/building. Production callers must provide real evidence.
    """
    base, valid, times = np.asarray(base), np.asarray(valid, bool), np.asarray(times)
    if base.ndim != 3 or base.shape[-1] != 52 or valid.shape != base.shape[:2] or times.shape != valid.shape:
        raise ValueError("Aligned B0/valid/time arrays required")
    phase = np.zeros(valid.shape, np.int64); confidence = np.zeros(valid.shape, np.float32)
    flags = valid & np.isfinite(base[...,jaw_channel]) & (base[...,jaw_channel] <= closed_threshold)
    for b in range(len(base)):
        start = 0
        while start < len(valid[b]):
            if not flags[b,start]:
                start += 1; continue
            end = start+1
            while end < len(valid[b]) and flags[b,end] and np.isclose(times[b,end]-times[b,end-1],.04,atol=1e-7,rtol=1e-4):
                end += 1
            if end-start >= min_run:
                phase[b,start+1:end-1] = CORE
                confidence[b,start+1:end-1] = .2
            start = end
    return PhoneticEvidence(phase, confidence, DIAGNOSTIC_SOURCE,
        f"unvalidated B0 low-jaw diagnostic; threshold={closed_threshold}; no phonetic/contact interpretation")
