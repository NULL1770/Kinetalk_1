"""Training-only variable-duration upper-face motion representation.

This module has no audio model and cannot establish audio predictability.  Its
teacher uses observed motion to select asynchronous piecewise-linear segments;
deployment must obtain starting levels and future segments from an audio prior.
Segment durations count native-frame transitions, so [s, s+d] has d+1 samples.
Adjacent segments share an endpoint.  Observation gaps are never compressed.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable

import numpy as np


SCHEMA = 'asynchronous_motion_process_v1'
GROUP_NAMES = ('raise', 'down', 'squint', 'wide')
UPPER_INDICES = (41, 42, 43, 44, 45, 5, 6, 12, 13)
GROUPS_52 = ((43, 44, 45), (41, 42), (5, 12), (6, 13))
GROUPS_9 = ((2, 3, 4), (0, 1), (5, 7), (6, 8))
DEFAULT_DURATIONS = (4, 8, 12, 20, 32, 48, 64)


def _values_mask(values, mask):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError('Values must be nonempty [T,G]')
    if mask.dtype != np.bool_ or mask.shape not in (values.shape, values.shape[:1]):
        raise ValueError('Boolean [T] or [T,G] observation mask required')
    if mask.ndim == 1:
        mask = np.broadcast_to(mask[:, None], values.shape)
    if not np.isfinite(values[mask]).all():
        raise ValueError('Observed values must be finite')
    return values, mask


def _scales(scales, groups):
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (groups,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('One finite positive training scale per group is required')
    return scales


def _runs(mask):
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def read_group_states(motion, observed):
    """Average available raw channels, without query centering or clipping.

    Supports a single [T,52] or ordered-upper [T,9] clip. Missing channels do
    not contribute and a group with no observations is invalid zero.
    """
    values, mask = _values_mask(motion, observed)
    if values.shape[1] not in (9, 52):
        raise ValueError('Expected [T,52] or UPPER_INDICES ordered [T,9]')
    groups = GROUPS_52 if values.shape[1] == 52 else GROUPS_9
    states = np.zeros((len(values), 4), dtype=np.float64)
    valid = np.zeros_like(states, dtype=bool)
    for g, channels in enumerate(groups):
        count = mask[:, channels].sum(-1)
        states[:, g] = np.where(mask[:, channels], values[:, channels], 0.).sum(-1) / np.maximum(count, 1)
        valid[:, g] = count > 0
    return states, valid


def fit_dynamic_scales(training_clips: Iterable, *, floor=.005):
    """Fit pooled within-contiguous-run RMS from an explicitly supplied fit set.

    Caller owns the train-only allowlist. Each item is (states[T,G], mask).
    These fixed scales normalize segment objectives; query RMS is never used
    to normalize a deployment prediction. Singleton runs add no variance.
    """
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError('Scale floor must be finite and positive')
    sums = counts = None
    for values, mask in training_clips:
        values, mask = _values_mask(values, mask)
        if sums is None:
            sums = np.zeros(values.shape[1]); counts = np.zeros(values.shape[1], dtype=np.int64)
        if values.shape[1] != len(sums):
            raise ValueError('All training clips must have the same groups')
        for g in range(values.shape[1]):
            for start, end in _runs(mask[:, g]):
                x = values[start:end, g]
                if len(x) > 1:
                    sums[g] += np.square(x-x.mean()).sum()
                    counts[g] += len(x)
    if sums is None or np.any(counts == 0):
        raise ValueError('Every group requires at least one observed training run of length two')
    return np.maximum(np.sqrt(sums/counts), floor)


def _fit_run(x, durations, penalty):
    """Dynamic programming with fixed observed endpoint values and linear SSE."""
    n = len(x)
    if n == 1:
        return [(0, 0)]
    # SSE can be evaluated in O(1) per candidate using prefix moments. Error
    # at shared endpoints is zero, so summing segment objectives is exact.
    t = np.arange(n, dtype=np.float64)
    sx = np.concatenate(([0.], np.cumsum(x)))
    sxx = np.concatenate(([0.], np.cumsum(x*x)))
    stx = np.concatenate(([0.], np.cumsum(t*x)))
    cost = np.full(n, np.inf); cost[0] = 0.
    previous = np.full(n, -1, dtype=np.int64)
    for end in range(1, n):
        candidates = np.asarray([d for d in durations if d <= end], dtype=np.int64)
        # A recording/gap can censor the final segment at any shorter length.
        # Such a segment is tagged and is not an observed duration event.
        if end == n-1:
            candidates = np.unique(np.concatenate((candidates, np.arange(1, min(end, max(durations))+1))))
        if len(candidates) == 0:
            continue
        starts = end-candidates
        reachable = np.isfinite(cost[starts])
        candidates, starts = candidates[reachable], starts[reachable]
        if len(starts) == 0:
            continue
        slope = (x[end]-x[starts])/candidates
        count = candidates+1
        # Local k = t-start avoids cancellation from absolute time offsets.
        sum_x = sx[end+1]-sx[starts]
        sum_xx = sxx[end+1]-sxx[starts]
        sum_kx = stx[end+1]-stx[starts]-starts*sum_x
        sum_k = candidates*(candidates+1)/2.
        sum_kk = candidates*(candidates+1)*(2*candidates+1)/6.
        sse = (sum_xx-2*x[starts]*sum_x-2*slope*sum_kx
               +count*x[starts]**2+2*x[starts]*slope*sum_k+slope*slope*sum_kk)
        scores = cost[starts]+np.maximum(sse, 0.)+penalty
        best = int(np.argmin(scores))
        cost[end], previous[end] = scores[best], starts[best]
    result = []; end = n-1
    while end > 0:
        start = int(previous[end])
        if start < 0:
            raise RuntimeError('Duration grid cannot cover observed run')
        result.append((start, end)); end = start
    return result[::-1]


def fit_motion_process(states, mask, scales, *, durations=DEFAULT_DURATIONS,
                       complexity_penalty=1.):
    """Fit an independent variable-duration linear plan for each action group.

    The objective is sum((target-reconstruction)/train_scale)^2 plus a fixed
    per-segment complexity penalty. Endpoints are observed teacher values;
    this is explicitly an oracle representation, never an audio condition at
    deployment. A segment is not necessarily a semantic expression event.
    """
    states, mask = _values_mask(states, mask)
    scales = _scales(scales, states.shape[1])
    durations = tuple(durations)
    if (not durations or any(type(x) is not int or x <= 0 for x in durations)
            or tuple(sorted(set(durations))) != durations):
        raise ValueError('Durations must be unique increasing positive integer transitions')
    if not np.isfinite(complexity_penalty) or complexity_penalty < 0:
        raise ValueError('Complexity penalty must be finite and nonnegative')
    plan = {'schema': SCHEMA, 'frames': len(states), 'groups': states.shape[1],
            'scales': scales.tolist(), 'durations': list(durations),
            'complexity_penalty': float(complexity_penalty), 'mask': mask.copy(),
            'segments': [[] for _ in range(states.shape[1])]}
    for g in range(states.shape[1]):
        for run_id, (start, stop) in enumerate(_runs(mask[:, g])):
            local = states[start:stop, g]
            pieces = _fit_run(local/scales[g], durations, complexity_penalty)
            for i, (left, right) in enumerate(pieces):
                plan['segments'][g].append({'start': int(start+left), 'duration': int(right-left),
                    'start_value': float(local[left]), 'end_value': float(local[right]),
                    'right_censored': bool(i == len(pieces)-1),
                    'left_censored': bool(i == 0), 'run_id': run_id})
    return plan


def render_motion_process(plan):
    """Reconstruct a teacher plan on its original clock, clearing only gaps."""
    if plan.get('schema') != SCHEMA:
        raise ValueError('Unknown motion process schema')
    frames, groups = plan['frames'], plan['groups']
    output = np.zeros((frames, groups), dtype=np.float64)
    mask = np.asarray(plan['mask'])
    if mask.shape != output.shape or mask.dtype != np.bool_ or len(plan['segments']) != groups:
        raise ValueError('Invalid plan mask or group count')
    covered = np.zeros_like(mask)
    for g, segments in enumerate(plan['segments']):
        prior = None
        for segment in segments:
            s, d = segment['start'], segment['duration']
            a, b = segment['start_value'], segment['end_value']
            if (type(s) is not int or type(d) is not int or s < 0 or d < 0 or s+d >= frames
                    or not np.isfinite([a, b]).all() or not mask[s:s+d+1, g].all()):
                raise ValueError('Invalid segment or segment crossing an observation gap')
            if d == 0 and a != b:
                raise ValueError('Singleton segment endpoints must agree')
            if prior is not None and s <= prior['start']+prior['duration']:
                if s != prior['start']+prior['duration'] or a != prior['end_value']:
                    raise ValueError('Overlapping segments or discontinuous shared endpoint')
            output[s:s+d+1, g] = np.linspace(a, b, d+1)
            covered[s:s+d+1, g] = True
            prior = segment
    if not np.array_equal(covered, mask):
        raise ValueError('Plan does not exactly cover observed frames')
    return output


def quantize_motion_process(plan, delta_grid):
    """Diagnostic: quantize normalized deltas and accumulate without GT resets.

    Initial levels at each gap-separated run remain oracle levels. Every later
    start is the previously decoded endpoint, exposing accumulated drift.
    This diagnostic is not the continuous teacher used for prior fitting.
    """
    render_motion_process(plan)
    grid = np.asarray(delta_grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) == 0 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError('A finite strictly increasing delta grid is required')
    result = copy.deepcopy(plan)
    result['quantized_delta_grid'] = grid.tolist()
    for g, segments in enumerate(result['segments']):
        current = run = None
        scale = result['scales'][g]
        for segment in segments:
            delta = (segment['end_value']-segment['start_value'])/scale
            if run != segment['run_id']:
                current = segment['start_value']; run = segment['run_id']
            qdelta = float(grid[int(np.argmin(np.abs(grid-delta)))]) if segment['duration'] else 0.
            segment['start_value'] = current
            segment['end_value'] = current+qdelta*scale
            current = segment['end_value']
    return result


def uniform_motion_process(plan, states):
    """Oracle control with exactly the same segments per group/observed run.

    Only knot timing changes to nearly uniform integer intervals. Endpoints
    remain observed teacher values. These uniform durations are a diagnostic
    control, not necessarily members of the audio prior's duration alphabet.
    """
    render_motion_process(plan)
    states, mask = _values_mask(states, plan['mask'])
    if states.shape != (plan['frames'], plan['groups']):
        raise ValueError('Teacher target shape differs from the fitted plan')
    result = copy.deepcopy(plan)
    result['control'] = 'uniform_knots_matched_per_group_run_segment_count'
    for g, segments in enumerate(result['segments']):
        for run_id, (start, stop) in enumerate(_runs(mask[:, g])):
            selected = [s for s in segments if s['run_id'] == run_id]
            if len(selected) == 1 and selected[0]['duration'] == 0:
                continue
            knots = np.rint(np.linspace(start, stop-1, len(selected)+1)).astype(np.int64)
            if np.any(np.diff(knots) <= 0):
                raise ValueError('Too many segments for uniform native clock')
            for i, segment in enumerate(selected):
                left, right = int(knots[i]), int(knots[i+1])
                segment.update(start=left, duration=right-left,
                               start_value=float(states[left, g]), end_value=float(states[right, g]))
    return result


def reconstruction_metrics(target, mask, reconstructed, scales, plan=None):
    """Raw and per-run-centered fidelity plus observed code rate (not bits)."""
    target, mask = _values_mask(target, mask)
    reconstructed, other_mask = _values_mask(reconstructed, mask)
    if reconstructed.shape != target.shape or not np.array_equal(mask, other_mask):
        raise ValueError('Reconstruction must use identical shape and mask')
    scales = _scales(scales, target.shape[1])
    rows = []
    for g in range(target.shape[1]):
        observed = mask[:, g]
        n = int(observed.sum())
        if n == 0:
            rows.append({'observed_frames': 0}); continue
        truth_parts, output_parts = [], []
        for start, end in _runs(observed):
            x, y = target[start:end, g], reconstructed[start:end, g]
            truth_parts.append(x-x.mean()); output_parts.append(y-y.mean())
        x, y = np.concatenate(truth_parts), np.concatenate(output_parts)
        err = reconstructed[observed, g]-target[observed, g]
        var_x, var_y = float(np.mean(x*x)), float(np.mean(y*y))
        cmse = float(np.mean((y-x)**2))
        row = {'observed_frames': n, 'raw_mse': float(np.mean(err*err)),
               'normalized_mse': float(np.mean(err*err)/scales[g]**2),
               'centered_mse': cmse, 'target_centered_variance': var_x,
               'centered_r2': 1-cmse/var_x if var_x > 1e-20 else None,
               'centered_corr': float(np.mean(x*y)/np.sqrt(var_x*var_y)) if min(var_x, var_y) > 1e-20 else None,
               'rms_ratio': float(np.sqrt(var_y/var_x)) if var_x > 1e-20 else None}
        if plan is not None:
            segments = plan['segments'][g]
            row.update(segments=len(segments), segments_per_100_frames=100*len(segments)/n,
                       uncensored_segments=sum(not s['right_censored'] for s in segments),
                       median_duration=float(np.median([s['duration'] for s in segments])))
        rows.append(row)
    return {'groups': rows, 'centered_policy': 'Each contiguous observed run independently centered for diagnostics only'}


class PersistentMotionDecoder:
    """Asynchronous linear action state whose time survives arbitrary chunks.

    ``current`` is the value at the current frame. ``advance(n)`` returns the
    *next* n frames, excluding current. Groups reaching their endpoint hold
    there until a caller starts another segment; elapsed time never causes a
    random resample or a reset. The caller/prior schedules segment renewals.
    """

    def __init__(self, initial):
        initial = np.asarray(initial, dtype=np.float64)
        if initial.ndim != 1 or len(initial) == 0 or not np.isfinite(initial).all():
            raise ValueError('Finite initial [G] state required')
        self.current = initial.copy()
        self.start = initial.copy(); self.target = initial.copy()
        self.duration = np.zeros(len(initial), dtype=np.int64)
        self.age = np.zeros(len(initial), dtype=np.int64)

    def start_segment(self, group, duration, end_value):
        if type(group) is not int or not 0 <= group < len(self.current):
            raise ValueError('Group index is out of range')
        if type(duration) is not int or duration <= 0 or not np.isfinite(end_value):
            raise ValueError('A positive integer duration and finite endpoint are required')
        if self.age[group] < self.duration[group]:
            raise ValueError('Cannot overwrite an unfinished group segment')
        self.start[group] = self.current[group]
        self.target[group] = end_value
        self.duration[group] = duration
        self.age[group] = 0

    def advance(self, frames):
        if type(frames) is not int or frames < 0:
            raise ValueError('Frames must be a nonnegative integer')
        output = np.empty((frames, len(self.current)), dtype=np.float64)
        for i in range(frames):
            self.age = np.minimum(self.age+1, self.duration)
            fraction = np.divide(self.age, self.duration, out=np.ones(len(self.current)), where=self.duration > 0)
            self.current = np.where(self.age == self.duration, self.target,
                                    self.start+(self.target-self.start)*fraction)
            output[i] = self.current
        return output

    def state_dict(self):
        return {key: getattr(self, key).copy() for key in ('current', 'start', 'target', 'duration', 'age')}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state['current'])
        for key in ('start', 'target', 'duration', 'age'):
            array = np.asarray(state[key])
            if array.shape != obj.current.shape or not np.isfinite(array).all():
                raise ValueError('Invalid persistent decoder checkpoint')
            if key in ('duration', 'age') and (not np.issubdtype(array.dtype, np.integer) or np.any(array < 0)):
                raise ValueError('Decoder duration and age must be nonnegative integers')
            setattr(obj, key, array.copy())
        if np.any(obj.age > obj.duration):
            raise ValueError('Decoder age exceeds duration')
        fraction = np.divide(obj.age, obj.duration, out=np.ones(len(obj.current)), where=obj.duration > 0)
        expected = np.where(obj.age == obj.duration, obj.target,
                            obj.start+(obj.target-obj.start)*fraction)
        if not np.array_equal(expected, obj.current):
            raise ValueError('Checkpoint current state differs from recorded phase')
        return obj
