"""Native-clock correspondence builder. Number-based pairs are AUDIT ONLY.

No legacy audio_triad resampling, GT-derived A/V shift or content truncation.
Explicit content-verified AND synchronization-verified pairs are required to
release framewise teachers. Geometry passing alone is never sufficient.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f

VERSION = '3.2-native-pair-audit'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def resolve_motion_path(record, manifest_path, bs_root=None):
    candidates = []
    for key in ('npz_final', 'bs'):
        if record.get(key):
            candidates.append(Path(record[key]))
    if record.get('bs_rel') and bs_root:
        candidates.append(Path(bs_root) / record['bs_rel'])
    if record.get('bs_rel'):
        candidates.append(manifest_path.parent / record['bs_rel'])
    for key in ('npz_final', 'bs'):
        if record.get(key):
            candidates.append(manifest_path.parent / record[key])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"motion BS not found for {record['clip_id']}: {candidates}")


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


@njit(cache=True)
def solve(cost, band, repeat_penalty, max_run):
    n, m = cost.shape
    states = 1 + 2 * max_run
    dp = np.full((n, m, states), np.inf, np.float64)
    prev = np.full((n, m, states), -1, np.int16)
    dp[0, 0, 0] = cost[0, 0]
    for i in range(n):
        center = i * (m - 1) / max(n - 1, 1)
        lo, hi = max(0, int(np.floor(center - band))), min(m, int(np.ceil(center + band)) + 1)
        for j in range(lo, hi):
            if i == 0 and j == 0:
                continue
            c = cost[i, j]
            if i and j:
                k = np.argmin(dp[i - 1, j - 1])
                dp[i, j, 0] = c + dp[i - 1, j - 1, k]
                prev[i, j, 0] = k
            # Vertical consumes source; horizontal consumes reference. Run
            # limits are part of the optimization, not a post-hoc clipping.
            if i:
                k = 0
                for q in range(max_run + 1, states):
                    if dp[i - 1, j, q] < dp[i - 1, j, k]:
                        k = q
                dp[i, j, 1] = c + repeat_penalty + dp[i - 1, j, k]
                prev[i, j, 1] = k
                for q in range(2, max_run + 1):
                    dp[i, j, q] = c + repeat_penalty + dp[i - 1, j, q - 1]
                    prev[i, j, q] = q - 1
            if j:
                k = np.argmin(dp[i, j - 1, :max_run + 1])
                dp[i, j, max_run + 1] = c + repeat_penalty + dp[i, j - 1, k]
                prev[i, j, max_run + 1] = k
                for q in range(max_run + 2, states):
                    dp[i, j, q] = c + repeat_penalty + dp[i, j - 1, q - 1]
                    prev[i, j, q] = q - 1
    k = np.argmin(dp[n - 1, m - 1])
    total = dp[n - 1, m - 1, k]
    if not np.isfinite(total):
        raise ValueError('no path under band/run constraints')
    path = np.empty((n + m, 2), np.int32)
    i, j, count = n - 1, m - 1, 0
    while True:
        path[count, 0], path[count, 1] = i, j
        count += 1
        if i == 0 and j == 0:
            break
        prior = prev[i, j, k]
        if prior < 0:
            raise ValueError('broken traceback')
        if k == 0:
            i, j = i - 1, j - 1
        elif k <= max_run:
            i -= 1
        else:
            j -= 1
        k = prior
    return path[:count][::-1].copy(), total / count


def align(source, reference, band_fraction=.2, repeat_penalty=.04, max_run=3):
    cost = np.clip(1 - unit(source) @ unit(reference).T, 0, 2)
    path, objective = solve(cost, max(3, int(np.ceil(len(reference) * band_fraction))), repeat_penalty, max_run)
    return path, float(objective), float(cost[path[:, 0], path[:, 1]].mean())


def mapped_times(path, source_times, reference_times):
    """Return source-time for every reference feature frame; full coverage."""
    n = len(reference_times)
    counts = np.bincount(path[:, 1], minlength=n)
    if np.any(counts == 0):
        raise ValueError('incomplete reference coverage')
    return np.bincount(path[:, 1], weights=source_times[path[:, 0]], minlength=n) / counts


def sample_at_times(values, times, query):
    times, query = np.asarray(times), np.asarray(query)
    if len(times) != len(values) or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('invalid sample clock')
    # Values outside the clock are endpoint-filled for storage only. They
    # must never contribute to supervision: valid is returned separately.
    valid = (query >= times[0]) & (query <= times[-1]) & np.isfinite(query)
    values = np.asarray(values)
    if values.ndim == 1:
        return np.interp(query, times, values), valid
    return np.stack([np.interp(query, times, col) for col in values.T], axis=1).astype(np.float32), valid


def local_slope(times, mapped, window_seconds=.2):
    w = max(1, int(round(window_seconds / np.median(np.diff(times)))))
    w = min(w, len(times) - 1)
    slopes = (mapped[w:] - mapped[:-w]) / (times[w:] - times[:-w])
    centers = (times[w:] + times[:-w]) / 2
    return np.interp(times, centers, slopes)


def path_metrics(path, st, rt):
    ds = np.diff(path, axis=0)
    expected = {(1, 0), (0, 1), (1, 1)}
    valid = all(tuple(x) in expected for x in ds)
    run = best = 0
    previous = None
    for d in ds:
        step = tuple(d)
        run = run + 1 if step == previous and 0 in step else (1 if 0 in step else 0)
        best = max(best, run)
        previous = step
    forward = mapped_times(path, st, rt)
    reverse = mapped_times(path[:, ::-1], rt, st)
    slopes = local_slope(rt, forward)
    cycle = np.abs(np.interp(forward, st, reverse) - rt)
    repeat = float(np.mean(np.any(ds == 0, axis=1))) if len(ds) else 0.
    minimum = abs(len(st) - len(rt)) / max(len(st) - 1, len(rt) - 1)
    return {
        'valid_steps': valid, 'endpoints_complete': bool(np.array_equal(path[0], [0, 0]) and np.array_equal(path[-1], [len(st)-1, len(rt)-1])),
        'source_coverage': len(np.unique(path[:, 0])) / len(st),
        'reference_coverage': len(np.unique(path[:, 1])) / len(rt),
        'repeat_ratio': repeat, 'minimum_repeat_ratio_from_lengths': minimum,
        'excess_repeat_ratio': max(0., repeat - minimum), 'max_same_direction_run': best,
        'max_hold_seconds': best * float(max(np.median(np.diff(st)), np.median(np.diff(rt)))),
        'local_slope_p01': float(np.percentile(slopes, 1)), 'local_slope_p99': float(np.percentile(slopes, 99)),
        'cycle_p95_seconds': float(np.percentile(cycle, 95)),
    }, forward, reverse, slopes, cycle


def load_feature(root, record):
    path = Path(root) / record['dataset'] / (record['clip_id'] + '.npz')
    with np.load(path, allow_pickle=False) as z:
        x, t = z['content'].astype(np.float32), z['times'].astype(np.float64)
        if 'signature' not in z or 'hop_samples' not in z:
            raise ValueError('unversioned or non-native feature cache')
        signature = str(z['signature'])
        validation = z['validation'].astype(np.float32)
        duration = float(z['audio_samples']) / float(z['sample_rate'])
    if x.ndim != 2 or x.shape[1] < 3 or len(x) != len(t) or not np.isfinite(x).all() or np.any(np.diff(t) <= 0):
        raise ValueError('invalid native feature shape, clock or values')
    return x, t, validation, duration, signature


def word_boundaries(pair, st, rt, forward):
    # Diagnostic only; timestamps are from independent ASR, not forced
    # alignment ground truth. Compare matched tokens and report coverage.
    import re
    from difflib import SequenceMatcher
    words_s, words_r = pair.get('source_words', []), pair.get('reference_words', [])
    normalize = lambda w: ''.join(re.findall('[a-z0-9]+', w['word'].lower()))
    blocks = SequenceMatcher(None, [normalize(w) for w in words_s], [normalize(w) for w in words_r], autojunk=False).get_matching_blocks()
    errors, baseline = [], []
    for block in blocks:
        for k in range(block.size):
            s, r = words_s[block.a+k], words_r[block.b+k]
            for key in ('start', 'end'):
                if rt[0] <= r[key] <= rt[-1] and st[0] <= s[key] <= st[-1]:
                    errors.append(abs(float(np.interp(r[key], rt, forward)) - s[key]))
                    baseline.append(abs(float(np.interp(r[key], [rt[0],rt[-1]], [st[0],st[-1]])) - s[key]))
    return {'matched_boundary_count': len(errors),
            'word_boundary_median_ms': float(np.median(errors)*1000) if errors else None,
            'word_boundary_p90_ms': float(np.percentile(errors,90)*1000) if errors else None,
            'linear_boundary_median_ms': float(np.median(baseline)*1000) if baseline else None}


def build_pair(pair, records, args):
    sr, rr = records[pair['source_clip_id']], records[pair['reference_clip_id']]
    if sr['speaker'] != rr['speaker'] or sr.get('split') != rr.get('split') or rr['emotion'] != 'neutral':
        raise ValueError('pair violates same-speaker/split neutral contract')
    sx, st, sv, sd, ss = load_feature(args.feature_root, sr)
    rx, rt, rv, rd, rs = load_feature(args.feature_root, rr)
    if ss != rs:
        raise ValueError('feature recipes differ')
    if sx.shape[1] != rx.shape[1]:
        raise ValueError('source/reference feature dimensions differ')
    sb = np.load(resolve_motion_path(sr, args.manifest, args.bs_root), allow_pickle=False)['coeffs']
    rb = np.load(resolve_motion_path(rr, args.manifest, args.bs_root), allow_pickle=False)['coeffs']
    if sb.shape != (sr['n_frames'], 52) or rb.shape != (rr['n_frames'], 52):
        raise ValueError('motion length/dimension mismatch')
    if not np.isfinite(sb).all() or not np.isfinite(rb).all():
        raise ValueError('nonfinite motion')
    smt, rmt = np.arange(len(sb), dtype=np.float64)/args.fps, np.arange(len(rb), dtype=np.float64)/args.fps
    identity = sr['clip_id'] == rr['clip_id']
    if identity:
        path = np.column_stack((np.arange(len(st)), np.arange(len(st)))).astype(np.int32)
        objective, cost = 0., 0.
    else:
        path, objective, cost = align(sx, rx, args.band, args.repeat_penalty, args.max_run)
    q, forward, reverse, slopes, cycle = path_metrics(path, st, rt)
    content_ok = bool(pair.get('content_verified')) or identity
    sync_ok = bool(pair.get('sync_verified'))
    # Offset is audio_time - motion_time; absent measurements stay zero and
    # cannot be silently replaced with a GT energy correlation optimum.
    soff, roff = float(pair.get('source_audio_offset_s', 0)), float(pair.get('reference_audio_offset_s', 0))
    ref_audio_times = rmt + roff
    src_times, mask = sample_at_times(forward, rt, ref_audio_times)
    # The media manifest has already checked that audio/video stream starts
    # share a clock.  We still keep offsets explicit so a later dataset can
    # carry measured offsets without changing this builder.
    motion, inside = sample_at_times(sb, smt, src_times - soff)
    content, inside_content = sample_at_times(sx, st, src_times)
    slope_at, _ = sample_at_times(slopes, rt, ref_audio_times)
    cycle_at, _ = sample_at_times(cycle, rt, ref_audio_times)
    mask &= inside & inside_content & (slope_at >= .25) & (slope_at <= 4) & (cycle_at <= .08)
    if identity:
        motion = sb.copy()  # absolutely no interpolation-induced identity shrinkage
    q['local_valid_ratio'] = float(mask.mean())
    geometry = (q['valid_steps'] and q['endpoints_complete'] and q['source_coverage']==1 and q['reference_coverage']==1 and q['local_valid_ratio'] >= .8)
    path_quality_ok = (identity or (q['repeat_ratio'] <= .35 and
                                    q['max_same_direction_run'] <= args.max_run and
                                    q['local_slope_p01'] >= .25 and
                                    q['local_slope_p99'] <= 4.0 and
                                    q['cycle_p95_seconds'] <= .08))
    duration_ok = abs(len(sb)/args.fps - sd) <= .08 and abs(len(rb)/args.fps - rd) <= .08
    verified = content_ok and sync_ok and geometry and path_quality_ok and duration_ok
    native_ref_times, native_mask = sample_at_times(reverse, st, smt + soff)
    if identity:
        # The identity teacher is already on the source motion clock.  Passing
        # it through the feature clock would introduce an avoidable boundary
        # shift and interpolation blur into the cleanest Stage1 anchor.
        neutral_teacher = rb.copy()
        inside_ref = np.ones(len(rb), dtype=bool)
    else:
        neutral_teacher, inside_ref = sample_at_times(rb, rmt, native_ref_times - roff)
    native_mask &= inside_ref
    # Local confidence mapped from the reverse correspondence as well.
    reverse_slope = local_slope(st, reverse)
    sl, _ = sample_at_times(reverse_slope, st, smt+soff)
    reverse_cycle = np.abs(np.interp(reverse, rt, forward) - st)
    cy, _ = sample_at_times(reverse_cycle, st, smt+soff)
    native_mask &= (sl >= .25) & (sl <= 4) & (cy <= .08)
    heldout = float((1 - (unit(sv)[path[:,0]] * unit(rv)[path[:,1]]).sum(1)).mean())
    reasons = []
    if not content_ok: reasons.append('content_not_verified')
    if not sync_ok: reasons.append('physical_av_sync_not_verified')
    if not duration_ok: reasons.append('duration_mismatch')
    if not geometry: reasons.append('geometry_review')
    if not path_quality_ok: reasons.append('path_quality_review')
    result = dict(pair, version=VERSION, status='accepted' if verified else 'review',
                  teacher_eligible=verified, geometry_pass=geometry, reasons=reasons,
                  source_frames=len(st), reference_frames=len(rt), feature_dim=int(sx.shape[1]), feature_recipe=ss,
                  mean_cost=cost, objective=objective, layer9_cost_diagnostic=heldout,
                  source_audio_duration=sd, reference_audio_duration=rd,
                  source_duration_difference=len(sb)/args.fps-sd,
                  reference_duration_difference=len(rb)/args.fps-rd, **q,
                  **word_boundaries(pair, st, rt, forward))
    key = sr['clip_id'] + '__to__' + rr['clip_id']
    out = args.out_root/'pairs'/(key+'.npz')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, path=path, source_feature_times=st, reference_feature_times=rt,
        source_time_on_reference=forward, reference_time_on_source=reverse,
        canonical_motion=motion, canonical_content=content, canonical_times=rmt,
        geometry_mask=mask, teacher_mask=mask & verified,
        neutral_teacher_on_source=neutral_teacher, source_motion_times=smt,
        native_geometry_mask=native_mask, native_teacher_mask=native_mask & verified,
        metadata=json.dumps(result, ensure_ascii=False))
    result['artifact'] = str(out)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--feature-root', type=Path, required=True,
                    help='Native feature cache root: <root>/<dataset>/<clip_id>.npz')
    ap.add_argument('--pairs', type=Path)
    ap.add_argument('--audit-number-pairs', action='store_true')
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--bs-root', type=Path, default=None,
                    help='Optional root containing processed/coeffs_final/... for portable manifests.')
    ap.add_argument('--fps', type=float, default=25.)
    ap.add_argument('--band', type=float, default=.2)
    ap.add_argument('--repeat-penalty', type=float, default=.04)
    ap.add_argument('--max-run', type=int, default=3)
    ap.add_argument('--limit-pairs', type=int, default=0,
                    help='Run only the first N explicit pairs for a smoke test.')
    args = ap.parse_args()
    if not args.pairs and not args.audit_number_pairs:
        ap.error('Explicit verified --pairs required; numbering alone is unsafe. Use --audit-number-pairs only for diagnostics.')
    if args.fps <= 0 or args.max_run < 1 or args.max_run > 12 or not 0 < args.band <= 1:
        ap.error('invalid fps, max-run or band')
    # Each build is immutable: no cached status can accidentally promote review.
    if args.out_root.exists() and any(args.out_root.iterdir()):
        ap.error('Output is nonempty. Choose a new revision directory.')
    args.out_root.mkdir(parents=True, exist_ok=True)
    records = {r['clip_id']:r for r in read_jsonl(args.manifest)}
    if args.pairs:
        pairs = read_jsonl(args.pairs)
        if args.limit_pairs:
            pairs = pairs[:args.limit_pairs]
    else:
        groups = defaultdict(list)
        for r in records.values():
            groups[(r['speaker'],str(r['sentence_id']))].append(r)
        pairs = []
        for group in groups.values():
            neutral = [r for r in group if r['emotion']=='neutral']
            if not neutral: continue
            ref = min(neutral, key=lambda r: (r['intensity'],r['clip_id']))
            pairs.extend({'source_clip_id':r['clip_id'],'reference_clip_id':ref['clip_id'],
                          'content_verified':False,'sync_verified':False,'kind':'number_audit'} for r in group)
    seen = set()
    for pair in pairs:
        key = (pair['source_clip_id'],pair['reference_clip_id'])
        if key in seen: raise ValueError('duplicate pair')
        seen.add(key)
    results = []
    with (args.out_root/'quality.jsonl').open('w',encoding='utf8') as f:
        for pair in pairs:
            try:
                result = build_pair(pair, records, args)
            except Exception as exc:
                result = dict(pair, status='error', teacher_eligible=False, reason=f'{type(exc).__name__}: {exc}')
            f.write(json.dumps(result, ensure_ascii=False)+'\n');f.flush()
            results.append(result)
            if len(results)%25 == 0: print('pairs',len(results),'/',len(pairs),flush=True)
    summary = {'version':VERSION,'pairs':len(results),'status':dict(Counter(r['status'] for r in results)),
               'geometry_pass':sum(r.get('geometry_pass',False) for r in results),
               'teacher_eligible':sum(r['teacher_eligible'] for r in results),
               'quality_gate_passed':bool(results) and all(r['teacher_eligible'] for r in results),
               'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               'note':'Geometry and ASR timestamps are diagnostics; neither proves correct phonemes or physical A/V synchronization.'}
    (args.out_root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    accepted = [r for r in results if r['teacher_eligible']]
    (args.out_root/'teacher_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in accepted),encoding='utf8')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
