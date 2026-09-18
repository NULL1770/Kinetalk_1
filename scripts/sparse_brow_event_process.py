"""Sparse, independently controlled brow events; no audio timing or query motion.

The fitted model has at most eight joint Student-t components over 13 marks:
peak displacement from the current pose (5), return displacement from the
independent level (5), and log(onset), log1p(apex), log(release) frame
counts (3). Return is deliberately level-relative, not relative to the event
start: repeated events cannot accumulate a random-walk baseline.

Only five brow channels move. The other four upper-face channels retain the
independent level exactly. A bounded sigmoid is a parameterization, not a
naturalness guarantee. Automatic HOLD has no framewise random innovations.

``mark_space`` explicitly selects logit_delta or coefficient_delta. The latter
transfers fitted coefficient amplitude even when the predicted level is near
zero; endpoints are clipped at declared coefficient_eps, recorded, converted
to logits and interpolated with the same C2 decoder. It is an engineering
baseline; neither endpoint bounds nor smoothness certify plausible action.

``wait_hazard`` is a discrete survival table: each entry is the conditional
probability of onset within a whole ``wait_bin_frames`` bin. Inside a bin the
hazard is constant; the final bin repeats indefinitely. ``activity`` divides
sampled wait length, while ``amplitude`` scales both peak and return marks.
The two controls do not change mark RNG consumption or stage lengths.

Every event index has SHA256-keyed independent wait and mark PCG64 streams.
Sampling chunks never trigger random draws. Reference changes cannot alter an
active event or an already scheduled wait; the next mark uses the new style.
An external HOLD/RELEASE aborts the active event. RUN resumes from the actual
state, braking first if necessary, without resetting the event index.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
from scipy.special import expit


SCHEMA = "sparse_brow_event_v1"
STATE_SCHEMA = "sparse_brow_event_state_v1"
MARK_DIM = 13
BROW_GROUPS = ((2, 3, 4), (0, 1))  # raise, down
MARK_ORDER = ("peak_offset_logit5", "return_level_offset_logit5",
              "log_onset_frames", "log1p_apex_frames", "log_release_frames")


def _finite(value, shape, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return out.copy()


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _psd_sqrt(cov):
    if not np.allclose(cov, cov.T, rtol=0, atol=1e-10):
        raise ValueError("Student-t scale must be symmetric")
    eigenvalues, vectors = np.linalg.eigh((cov + cov.T) / 2)
    if eigenvalues.min() < -1e-10 * max(1., float(np.linalg.norm(cov, 2))):
        raise ValueError("Student-t scale must be positive semidefinite")
    # Principal symmetric root avoids eigenvector-sign-dependent random draws.
    # Drop roundoff eigenvalues in exactly rank-deficient joint components;
    # otherwise sqrt(machine epsilon) invents orthogonal motion directions.
    threshold = 32 * np.finfo(np.float64).eps * max(float(np.linalg.norm(cov, 2)), np.finfo(np.float64).tiny)
    eigenvalues = np.where(eigenvalues <= threshold, 0., eigenvalues)
    return (vectors * np.sqrt(eigenvalues)) @ vectors.T


def validate_model(model):
    """Validate and return an isolated JSON-serializable model copy.

    Scale is the Student-t scale matrix, not its covariance. Rank-deficient
    components are permitted (including deterministic synthetic fixtures).
    Caps are declared model parameters and their application is recorded.
    """
    if not isinstance(model, dict) or model.get("schema_version", model.get("schema")) != SCHEMA:
        raise ValueError(f"Expected {SCHEMA} model schema")
    result = copy.deepcopy(_jsonable(model))
    result["schema_version"] = SCHEMA
    result["mark_space"] = model.get("mark_space", "logit_delta")
    if result["mark_space"] not in ("logit_delta", "coefficient_delta"):
        raise ValueError("mark_space must be logit_delta or coefficient_delta")
    result["coefficient_eps"] = _positive(model.get("coefficient_eps", 1e-4), "coefficient_eps")
    if result["coefficient_eps"] >= .5:
        raise ValueError("coefficient_eps must be below .5")
    result["fps"] = _positive(model.get("fps", 25.), "fps")
    result["wait_bin_frames"] = _integer(model.get("wait_bin_frames", round(result["fps"] * .5)), "wait_bin_frames", 1)
    result["minimum_hold_frames"] = _integer(model.get("minimum_hold_frames", 4), "minimum_hold_frames", 1)
    hazard = np.asarray(model.get("wait_hazard"), dtype=np.float64)
    if hazard.ndim != 1 or not len(hazard) or not np.isfinite(hazard).all() or ((hazard < 0) | (hazard > 1)).any() or hazard[-1] <= 0:
        raise ValueError("wait_hazard must be nonempty probabilities with positive terminal hazard")
    result["wait_hazard"] = hazard.tolist()
    caps = (("max_peak_offset", .5), ("max_return_offset", .2)) if result["mark_space"] == "coefficient_delta" else (("max_peak_offset", 3.), ("max_return_offset", 1.))
    for name, default in caps:
        value = model.get(name, [default] * 5)
        array = _finite(value, (5,), name)
        if (array < 0).any():
            raise ValueError(f"{name} must be nonnegative")
        result[name] = array.tolist()
    bounds = model.get("duration_bounds", {"onset": [4, 75], "apex": [0, 50], "release": [4, 75]})
    result["duration_bounds"] = {}
    for name in ("onset", "apex", "release"):
        pair = bounds.get(name)
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("duration_bounds must supply onset, apex, release pairs")
        low = _integer(pair[0], name + " lower bound", 0 if name == "apex" else 1)
        high = _integer(pair[1], name + " upper bound", low)
        result["duration_bounds"][name] = [low, high]
    components = model.get("components")
    if not isinstance(components, list) or not 1 <= len(components) <= 8:
        raise ValueError("One to eight Student-t components required")
    normalized = []
    for component in components:
        weight = _positive(component.get("weight"), "component weight")
        loc = _finite(component.get("loc"), (MARK_DIM,), "component loc")
        scale = _finite(component.get("scale"), (MARK_DIM, MARK_DIM), "component scale")
        _psd_sqrt(scale)
        df = _positive(component.get("df", 5.), "Student-t df")
        if df != 5.:
            raise ValueError("This fixed protocol requires Student-t df=5")
        normalized.append({"weight": weight, "loc": loc.tolist(), "scale": scale.tolist(), "df": df})
    total = sum(row["weight"] for row in normalized)
    if not math.isfinite(total):
        raise ValueError("Component weight sum must be finite")
    for row in normalized:
        row["weight"] /= total
    result["components"] = normalized
    return result


def _style(style=None, previous=None):
    result = {"amplitude": np.ones(5), "activity": 1.} if previous is None else copy.deepcopy(previous)
    if style is None:
        return result
    if not isinstance(style, dict) or set(style) - {"amplitude", "activity"}:
        raise ValueError("style accepts amplitude and activity only")
    if "activity" in style:
        activity = style["activity"]
        if isinstance(activity, (bool, np.bool_)) or not isinstance(activity, (int, float, np.number)) or not math.isfinite(activity) or activity < 0:
            raise ValueError("activity must be finite and nonnegative")
        result["activity"] = float(activity)
    if "amplitude" in style:
        amplitude = np.asarray(style["amplitude"], dtype=np.float64)
        if amplitude.ndim == 0:
            amplitude = np.full(5, amplitude.item())
        elif amplitude.shape == (2,):
            amplitude = np.array([amplitude[1], amplitude[1], amplitude[0], amplitude[0], amplitude[0]])
        amplitude = _finite(amplitude, (5,), "amplitude")
        if (amplitude < 0).any() or (amplitude > 8).any():
            raise ValueError("amplitude must be in [0,8]")
        result["amplitude"] = amplitude
    return result


def _rng(seed, key, index, stream):
    payload = json.dumps([SCHEMA, seed, key, index, stream], separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))


def sample_wait_frames(model, uniform):
    """Inverse discrete survival; terminal tail is geometric, never truncated."""
    if not math.isfinite(uniform) or not 0 <= uniform < 1:
        raise ValueError("wait uniform must be in [0,1)")
    remaining = -math.log1p(-uniform)
    bin_frames = model["wait_bin_frames"]
    elapsed = 0
    hazards = model["wait_hazard"]
    for index, probability in enumerate(hazards):
        if probability == 1.:
            return model["minimum_hold_frames"] + elapsed + 1
        rate = -math.log1p(-probability) / bin_frames
        if index == len(hazards) - 1 or (rate > 0 and remaining < rate * bin_frames):
            steps = int(math.floor(remaining / rate)) + 1
            return model["minimum_hold_frames"] + elapsed + steps
        remaining -= rate * bin_frames
        elapsed += bin_frames
    raise AssertionError("Validated positive terminal hazard required")


def _quintic(start, velocity, acceleration, end, duration):
    c0 = start.copy()
    c1 = velocity * duration
    c2 = acceleration * duration**2 / 2
    rhs = np.stack((end - c0 - c1 - c2, -c1 - 2*c2, -2*c2))
    final = np.linalg.solve(np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]]), rhs)
    return np.vstack((c0, c1, c2, final))


def evaluate_quintic(coefficients, u, duration):
    """Evaluate position/velocity/acceleration; derivative units are seconds."""
    c = np.asarray(coefficients)
    x = np.power(u, np.arange(6)) @ c
    v = (np.arange(1, 6) * np.power(u, np.arange(5))) @ c[1:] / duration
    a = (np.arange(2, 6) * np.arange(1, 5) * np.power(u, np.arange(4))) @ c[2:] / duration**2
    return x, v, a


class SparseBrowEventProcess:
    """Persistent sparse-event sampler; level9 is an independent logit level.

    ``sample(n)`` advances n ticks and returns values/logits, emitted phase and
    event index. Values have nine channels; eye channels are level constants.
    State and provenance are JSON-serializable via ``state_dict``. Caller must
    select independent fit references and must not provide query motion.
    """

    def __init__(self, model, level9, seed, key="default", style=None):
        self.model = validate_model(model)
        self.model_hash = hashlib.sha256(json.dumps(self.model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.level = _finite(level9, (9,), "independent logit level")
        self.seed = _integer(seed, "seed")
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a nonempty string")
        self.key = key
        self.style = _style(style)
        self._factors = [_psd_sqrt(np.asarray(row["scale"])) for row in self.model["components"]]
        self.x = self.level[:5].copy()
        self.v = np.zeros(5)
        self.a = np.zeros(5)
        self.frame = 0
        self.next_event_index = 0
        self.mode = "RUN"
        self.phase = "HOLD"
        self.active_event = None
        self.events = []
        self.waits = []
        self.controls = []
        self.wait_remaining = 0
        self.wait_disabled = False
        self.stage_age = self.stage_frames = 0
        self.coefficients = self.endpoint = None
        self._schedule_wait()

    def _schedule_wait(self, scheduled_frame=None):
        uniform = float(_rng(self.seed, self.key, self.next_event_index, "wait").random())
        base = sample_wait_frames(self.model, uniform)
        self.wait_disabled = self.style["activity"] == 0
        scaled = 0 if self.wait_disabled else base / self.style["activity"]
        if not math.isfinite(scaled) or scaled > 2**53:
            raise ValueError("activity results in an unrepresentable waiting time")
        self.wait_remaining = 0 if self.wait_disabled else max(1, int(math.ceil(scaled)))
        self.phase = "HOLD"
        self.waits.append({"event_index": self.next_event_index, "scheduled_frame": self.frame if scheduled_frame is None else scheduled_frame,
                           "uniform": uniform, "base_wait_frames": base,
                           "wait_frames": None if self.wait_disabled else self.wait_remaining,
                           "activity": self.style["activity"]})

    def set_style(self, style):
        """Future marks/waits only; never rescale the current physical state."""
        self.style = _style(style, self.style)
        if self.wait_disabled and self.style["activity"] > 0 and self.mode == "RUN" and self.phase == "HOLD":
            self._schedule_wait()

    def _curve(self, phase, endpoint, frames):
        self.phase = phase
        self.stage_frames = int(frames)
        self.stage_age = 0
        self.endpoint = np.asarray(endpoint, dtype=np.float64).copy()
        self.coefficients = _quintic(self.x, self.v, self.a, self.endpoint,
                                      self.stage_frames / self.model["fps"])

    def _start_event(self):
        index = self.next_event_index
        rng = _rng(self.seed, self.key, index, "mark")
        weights = np.asarray([row["weight"] for row in self.model["components"]])
        component_index = int(rng.choice(len(weights), p=weights))
        component = self.model["components"][component_index]
        normal = rng.standard_normal(MARK_DIM)
        chi2 = float(rng.chisquare(component["df"]))
        mark = np.asarray(component["loc"]) + self._factors[component_index] @ normal / math.sqrt(chi2 / component["df"])
        if not np.isfinite(mark).all():
            raise FloatingPointError("Sampled mark is not finite")
        peak_cap = np.asarray(self.model["max_peak_offset"])
        return_cap = np.asarray(self.model["max_return_offset"])
        peak = np.clip(mark[:5], -peak_cap, peak_cap)
        returned = np.clip(mark[5:10], -return_cap, return_cap)
        durations = []
        duration_clipped = []
        for pos, name in enumerate(("onset", "apex", "release")):
            low, high = self.model["duration_bounds"][name]
            lower_log = math.log1p(low) if name == "apex" else math.log(low)
            upper_log = math.log1p(high) if name == "apex" else math.log(high)
            log_value = float(np.clip(mark[10+pos], lower_log, upper_log))
            value = math.expm1(log_value) if name == "apex" else math.exp(log_value)
            durations.append(int(np.clip(round(value), low, high)))
            duration_clipped.append(log_value != mark[10+pos])
        amplitude = self.style["amplitude"]
        if self.model["mark_space"] == "coefficient_delta":
            eps = self.model["coefficient_eps"]
            raw_peak = expit(self.x) + amplitude * peak
            raw_return = expit(self.level[:5]) + amplitude * returned
            bounded_peak = np.clip(raw_peak, eps, 1-eps)
            bounded_return = np.clip(raw_return, eps, 1-eps)
            endpoint_peak = np.log(bounded_peak) - np.log1p(-bounded_peak)
            endpoint_return = np.log(bounded_return) - np.log1p(-bounded_return)
            # Exactly-zero control leaves even near-boundary baseline values
            # unchanged; do not introduce logit(expit(x)) roundoff or EPS snap.
            no_peak_delta = amplitude * peak == 0
            no_return_delta = amplitude * returned == 0
            endpoint_peak[no_peak_delta] = self.x[no_peak_delta]
            endpoint_return[no_return_delta] = self.level[:5][no_return_delta]
            peak_boundary = (bounded_peak != raw_peak) & ~no_peak_delta
            return_boundary = (bounded_return != raw_return) & ~no_return_delta
        else:
            endpoint_peak = self.x + amplitude * peak
            endpoint_return = self.level[:5] + amplitude * returned
            peak_boundary = return_boundary = np.zeros(5, dtype=bool)
        event = {"event_index": index, "start_frame": self.frame, "component": component_index,
                 "raw_mark": mark.tolist(), "mark_space": self.model["mark_space"],
                 "peak_offset": peak.tolist(), "return_level_offset": returned.tolist(), "durations": durations,
                 "start_logit": self.x.tolist(), "peak_logit": endpoint_peak.tolist(),
                 "return_logit": endpoint_return.tolist(), "amplitude": amplitude.tolist(),
                 "peak_clipped": (peak != mark[:5]).tolist(),
                 "return_clipped": (returned != mark[5:10]).tolist(),
                 "peak_boundary_clipped": peak_boundary.tolist(),
                 "return_boundary_clipped": return_boundary.tolist(),
                 "duration_clipped": duration_clipped, "status": "active"}
        self.events.append(event)
        self.active_event = event
        self.next_event_index += 1
        self._curve("ONSET", endpoint_peak, durations[0])

    def _abort(self):
        if self.active_event is not None:
            self.active_event["status"] = "aborted"
            self.active_event["end_frame"] = self.frame
            self.active_event = None

    def command(self, mode, transition_frames=None):
        """C2 external stop/release, or resume from current state.

        HOLD stops at x+v*T/2. RELEASE returns exactly to independent level.
        RUN during a moving external transition first finishes an 8-frame C2
        stop before scheduling its next event. Repeating a command is a no-op.
        """
        if mode not in ("RUN", "HOLD", "RELEASE"):
            raise ValueError("mode must be RUN, HOLD, or RELEASE")
        if mode == self.mode:
            return
        frames = (16 if mode == "RELEASE" else 8) if transition_frames is None else _integer(transition_frames, "transition_frames", 1)
        self._abort()
        self.mode = mode
        self.controls.append({"frame": self.frame, "mode": mode, "transition_frames": frames})
        if mode == "RELEASE":
            self._curve("RELEASE", self.level[:5], frames)
        elif mode == "HOLD" or np.any(self.v != 0) or np.any(self.a != 0):
            self._curve("STOP", self.x + self.v * (frames / self.model["fps"]) / 2, frames)
        else:
            self.coefficients = self.endpoint = None
            self._schedule_wait()

    def _finish_stage(self):
        if self.active_event is None:
            self.phase = "HOLD"
            if self.mode == "RUN":
                self._schedule_wait(scheduled_frame=self.frame + 1)
            return
        if self.phase == "ONSET":
            apex_frames = self.active_event["durations"][1]
            if apex_frames:
                self.phase = "APEX"
                self.stage_age = 0
                self.stage_frames = apex_frames
            else:
                self._curve("RELEASE", self.active_event["return_logit"], self.active_event["durations"][2])
        elif self.phase == "APEX":
            self._curve("RELEASE", self.active_event["return_logit"], self.active_event["durations"][2])
        elif self.phase == "RELEASE":
            self.active_event["status"] = "complete"
            self.active_event["end_frame"] = self.frame + 1
            self.active_event = None
            self._schedule_wait(scheduled_frame=self.frame + 1)

    def sample(self, frames):
        frames = _integer(frames, "frames")
        values = np.empty((frames, 9), dtype=np.float64)
        logits = np.empty_like(values)
        phases = np.empty(frames, dtype="U7")
        indices = np.full(frames, -1, dtype=np.int64)
        for output_index in range(frames):
            if self.phase == "HOLD" and self.mode == "RUN" and not self.wait_disabled and self.wait_remaining == 0:
                self._start_event()
            phase = self.phase
            phases[output_index] = phase
            if self.active_event is not None:
                indices[output_index] = self.active_event["event_index"]
            if phase == "HOLD":
                if self.mode == "RUN" and not self.wait_disabled:
                    self.wait_remaining -= 1
            elif phase == "APEX":
                self.stage_age += 1
                if self.stage_age == self.stage_frames:
                    self._finish_stage()
            else:
                self.stage_age += 1
                self.x, self.v, self.a = evaluate_quintic(self.coefficients, self.stage_age / self.stage_frames,
                                                        self.stage_frames / self.model["fps"])
                if self.stage_age == self.stage_frames:
                    self.x = self.endpoint.copy()
                    self.v = np.zeros(5)
                    self.a = np.zeros(5)
                    self._finish_stage()
            if not all(np.isfinite(item).all() for item in (self.x, self.v, self.a)):
                raise FloatingPointError("Event state became nonfinite")
            logits[output_index] = self.level
            logits[output_index, :5] = self.x
            values[output_index] = expit(logits[output_index])
            self.frame += 1
        return {"values": values, "logits": logits, "phase": phases, "event_index": indices}

    def state_dict(self):
        state = {"schema": STATE_SCHEMA, "model_hash": self.model_hash,
                 "level": self.level, "seed": self.seed, "key": self.key, "style": self.style,
                 "x": self.x, "v": self.v, "a": self.a, "frame": self.frame,
                 "next_event_index": self.next_event_index, "mode": self.mode, "phase": self.phase,
                 "active_event_index": None if self.active_event is None else self.active_event["event_index"],
                 "events": self.events, "waits": self.waits, "controls": self.controls,
                 "wait_remaining": self.wait_remaining, "stage_age": self.stage_age,
                 "wait_disabled": self.wait_disabled,
                 "stage_frames": self.stage_frames, "coefficients": self.coefficients,
                 "endpoint": self.endpoint, "query_motion_used": False,
                 "reference_trajectory_used": False}
        return copy.deepcopy(_jsonable(state))

    @classmethod
    def from_state_dict(cls, model, state):
        """Restore an exact same-runtime continuation from a trusted snapshot."""
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("Incompatible state schema")
        result = cls(model, state["level"], state["seed"], state["key"], state["style"])
        if result.model_hash != state.get("model_hash"):
            raise ValueError("State model hash mismatch")
        for name in ("x", "v", "a"):
            setattr(result, name, _finite(state[name], (5,), name))
        for name in ("frame", "next_event_index", "wait_remaining", "stage_age", "stage_frames"):
            setattr(result, name, _integer(state[name], name))
        if state["mode"] not in ("RUN", "HOLD", "RELEASE") or state["phase"] not in ("HOLD", "ONSET", "APEX", "RELEASE", "STOP"):
            raise ValueError("Invalid state phase or mode")
        result.mode, result.phase = state["mode"], state["phase"]
        if not isinstance(state["wait_disabled"], bool):
            raise ValueError("wait_disabled must be Boolean")
        result.wait_disabled = state["wait_disabled"]
        result.events = copy.deepcopy(state["events"])
        result.waits = copy.deepcopy(state["waits"])
        result.controls = copy.deepcopy(state["controls"])
        active_index = state["active_event_index"]
        matches = [row for row in result.events if row["event_index"] == active_index]
        if active_index is not None and len(matches) != 1:
            raise ValueError("Active event missing or duplicated")
        result.active_event = matches[0] if active_index is not None else None
        for name, shape in (("coefficients", (6, 5)), ("endpoint", (5,))):
            setattr(result, name, None if state[name] is None else _finite(state[name], shape, name))
        if result.phase in ("ONSET", "RELEASE", "STOP") and (result.coefficients is None or result.endpoint is None or not 0 <= result.stage_age < result.stage_frames):
            raise ValueError("Invalid active curve state")
        return result
