"""Fixed-protocol brow pixel-flow evidence on 32 locked native-clock fit clips.

MediaPipe landmarks define ROIs only at each observed run's first frame. All
subsequent movement comes from image corners, forward/backward LK and stable
face similarity fitting. This is semi-independent pixel evidence with tracker
initial ROIs, not ground truth or a causal estimate of tracker error.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_tracking_reset import video_frames
from scripts.diagnose_brow_state_nuisance import sha, write_json, _verified

SCHEMA = "brow_pixel_flow_diagnostic_v1"
BROWS = {"right": [70, 63, 105, 66, 107], "left": [336, 296, 334, 293, 300]}
STABLE = {"nose_bridge": [6, 197, 195, 5], "nose_sides": [98, 327, 129, 358],
          "outer_temples": [127, 356, 162, 389]}
PROTOCOL = {
    "schema": SCHEMA, "fps": 25, "maximum_image_dimension": 720,
    "brow_landmarks": BROWS, "stable_landmarks": STABLE,
    "landmark_usage": "first frame of each valid run only; fixed initial feature tracks, no later landmark updates or feature replenishment",
    "eye_scale": "initial outer-eye-corner Euclidean distance, landmarks33 and263",
    "brow_roi_radius_eye_fraction": .035, "stable_roi_radius_eye_fraction": .045,
    "stable_exclusion": "expanded eye, brow and lip rectangles; no eyelid/lip feature participates in similarity fitting",
    "gftt": {"max_corners_per_brow": 60, "max_corners_per_stable_circle": 12,
             "quality_level": .01, "minimum_distance_px": 3., "block_size": 5},
    "lk": {"window": [21, 21], "levels": 3, "iterations": 30, "epsilon": .01,
           "maximum_forward_backward_error_px": 1.},
    "similarity": {"method": "estimateAffinePartial2D RANSAC", "threshold_px": 1.5,
                   "max_iterations": 2000, "confidence": .99, "refine_iterations": 10,
                   "minimum_tracks": 8, "minimum_inliers": 6, "minimum_inlier_fraction": .6,
                   "minimum_inlier_span_eye_fraction": [.25, .08]},
    "minimum_brow_tracks_per_side": 4,
    "movement": "median current-minus-previous corner displacement; corrected subtracts stable-face similarity prediction; positive image y is downward",
    "reported_pixel_units": "original decoded pixels, reversed fixed image resize",
    "normalization": "initial run outer-eye distance; no per-frame landmark normalization",
    "missing": "NaN with explicit reason; no zero filling, interpolation or cross-gap differences",
    "frozen_parameters": True, "threshold_search": False,
    "independent_ground_truth": False, "semi_independent_pixel_evidence": True,
    "causal_confidence": False, "used_for_training": False,
}
SIDE_NAMES = list(BROWS)
DEFAULT_FFMPEG = "C:/Users/zhh/AppData/Local/com.minimax.hub/current/resources/ffmpeg/ffmpeg.exe"


def checked_frame(rgb, expected_sha256):
    import hashlib
    actual = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    if actual != str(expected_sha256):
        raise ValueError("decoded RGB SHA differs from locked tracking-reset frame")
    return rgb


def grayscale(rgb):
    h, w = rgb.shape[:2]
    scale = min(1., PROTOCOL["maximum_image_dimension"] / max(h, w))
    if scale != 1:
        rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    # The exact rounded x/y resize is retained; eye scale and points use this grid.
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), np.array([rgb.shape[1] / w, rgb.shape[0] / h])


def initial_regions(landmarks, shape):
    h, w = shape
    lm = np.asarray(landmarks, float)
    if lm.ndim != 2 or lm.shape[1] < 2 or len(lm) < 455 or not np.isfinite(lm[:, :2]).all():
        raise ValueError("finite initial face landmarks required")
    xy = lm[:, :2] * [w, h]
    eye_distance = float(np.linalg.norm(xy[33] - xy[263]))
    if eye_distance < 10:
        raise ValueError("initial eye-corner distance too small")
    radius = max(2, round(PROTOCOL["brow_roi_radius_eye_fraction"] * eye_distance))
    brow_masks = []
    for indices in BROWS.values():
        mask = np.zeros((h, w), np.uint8)
        pts = np.rint(xy[indices]).astype(np.int32)
        cv2.polylines(mask, [pts], False, 255, 2 * radius + 1)
        for point in pts:
            cv2.circle(mask, tuple(point), radius, 255, -1)
        brow_masks.append(mask)
    excluded = np.zeros((h, w), np.uint8)
    # Fixed semantic exclusion sets; none of these points supply motion values.
    regions = [([33, 133, 160, 159, 158, 144, 145, 153], .07),
               ([362, 263, 385, 386, 387, 380, 374, 373], .07),
               ([61, 291, 0, 17, 13, 14], .10),
               (BROWS["right"], .07), (BROWS["left"], .07)]
    for indices, padding in regions:
        low = np.floor(xy[indices].min(0) - padding * eye_distance).astype(int)
        high = np.ceil(xy[indices].max(0) + padding * eye_distance).astype(int)
        cv2.rectangle(excluded, tuple(low), tuple(high), 255, -1)
    stable_masks = []
    stable_radius = max(2, round(PROTOCOL["stable_roi_radius_eye_fraction"] * eye_distance))
    for indices in STABLE.values():
        for index in indices:
            mask = np.zeros((h, w), np.uint8)
            cv2.circle(mask, tuple(np.rint(xy[index]).astype(int)), stable_radius, 255, -1)
            mask[excluded != 0] = 0
            stable_masks.append(mask)
    return brow_masks, stable_masks, eye_distance


def features(gray, masks, maximum):
    points = []
    for mask in masks:
        found = cv2.goodFeaturesToTrack(gray, maxCorners=maximum, qualityLevel=.01,
                                        minDistance=3., mask=mask, blockSize=5)
        if found is not None:
            for point in found.reshape(-1, 2):
                if not points or min(np.linalg.norm(point - p) for p in points) >= 2:
                    points.append(point)
    return np.asarray(points, np.float32).reshape(-1, 2)


def tracked_points(previous, current, points):
    points = np.asarray(points, np.float32).reshape(-1, 2)
    if not len(points):
        return points.copy(), points.copy()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)
    kwargs = dict(winSize=(21, 21), maxLevel=3, criteria=criteria)
    new, forward_status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points[:, None, :], None, **kwargs)
    if new is None:
        return points[:0], points[:0]
    back, backward_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, new, None, **kwargs)
    if back is None:
        return points[:0], points[:0]
    new, back = new.reshape(-1, 2), back.reshape(-1, 2)
    h, w = current.shape
    keep = (forward_status.ravel() != 0) & (backward_status.ravel() != 0)
    keep &= np.isfinite(new).all(1) & np.isfinite(back).all(1)
    keep &= np.linalg.norm(back - points, axis=1) <= 1.
    keep &= (new[:, 0] >= 0) & (new[:, 0] < w) & (new[:, 1] >= 0) & (new[:, 1] < h)
    return points[keep], new[keep]


def fit_similarity(old, new, eye_distance):
    if len(old) < 8:
        return None, {"reason": "insufficient_stable_tracks", "tracks": len(old), "inliers": 0}
    cv2.setRNGSeed(0)
    matrix, mask = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
                                            ransacReprojThreshold=1.5, maxIters=2000,
                                            confidence=.99, refineIters=10)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None, {"reason": "similarity_fit_failed", "tracks": len(old), "inliers": 0}
    mask = mask.ravel().astype(bool)
    count = int(mask.sum())
    report = {"reason": None, "tracks": len(old), "inliers": count}
    if count < 6 or count / len(old) < .6:
        report["reason"] = "insufficient_stable_inliers"
        return None, report
    span = np.ptp(old[mask], axis=0) / eye_distance
    if span[0] < .25 or span[1] < .08:
        report["reason"] = "stable_inliers_not_spatially_distributed"
        return None, report
    return matrix, report


class PixelTracker:
    def __init__(self, rgb, landmarks):
        self.previous, self.resize_xy = grayscale(rgb)
        masks, stable_masks, self.eye_distance = initial_regions(landmarks, self.previous.shape)
        self.brows = [features(self.previous, [mask], 60) for mask in masks]
        self.stable = features(self.previous, stable_masks, 12)
        self.initial_counts = {"stable": len(self.stable), **{name: len(points) for name, points in zip(SIDE_NAMES, self.brows)}}
        self.initial_masks = (masks, stable_masks)

    def step(self, rgb):
        current, resize = grayscale(rgb)
        if current.shape != self.previous.shape or not np.array_equal(resize, self.resize_xy):
            raise ValueError("decoded image size changed within observed run")
        old, self.stable = tracked_points(self.previous, current, self.stable)
        matrix, stable_report = fit_similarity(old, self.stable, self.eye_distance)
        raw = np.full((2, 2), np.nan)
        corrected = raw.copy()
        counts, reasons = [], []
        for index, points in enumerate(self.brows):
            old_brow, self.brows[index] = tracked_points(self.previous, current, points)
            counts.append(len(old_brow))
            if len(old_brow) < 4:
                reasons.append("insufficient_brow_tracks")
                continue
            raw[index] = np.median(self.brows[index] - old_brow, axis=0)
            if matrix is None:
                reasons.append(stable_report["reason"])
                continue
            predicted = old_brow @ matrix[:, :2].T + matrix[:, 2]
            corrected[index] = np.median(self.brows[index] - predicted, axis=0)
            reasons.append(None)
        self.previous = current
        return {"raw_px": raw / self.resize_xy, "corrected_px": corrected / self.resize_xy,
                "raw_normalized": raw / self.eye_distance, "corrected_normalized": corrected / self.eye_distance,
                "brow_tracks": counts, "stable": stable_report,
                "similarity": matrix, "missing_reason": reasons}


def diagnose_frames(frames, arrays, source_id, overlay_path=None):
    valid, times = np.asarray(arrays["valid"]), np.asarray(arrays["times"], float)
    landmarks, hashes = arrays["landmarks"], arrays["rgb_sha256"]
    n = len(valid)
    if valid.dtype != bool or times.shape != (n,) or len(landmarks) != n or len(hashes) != n:
        raise ValueError("locked frame masks, times, landmarks and hashes differ")
    if not n or not np.allclose(times, np.arange(n) / 25., rtol=0, atol=1e-8):
        raise ValueError("requires exact native 25 Hz source clock")
    outputs = {key: np.full((n, 2, 2), np.nan) for key in
               ("raw_px", "corrected_px", "raw_normalized", "corrected_normalized")}
    outputs.update(times=times.copy(), valid=valid.copy(), run_start=np.zeros(n, bool),
                   brow_tracks=np.zeros((n, 2), np.int16), stable_tracks=np.zeros(n, np.int16),
                   stable_inliers=np.zeros(n, np.int16), similarity=np.full((n, 2, 3), np.nan))
    reasons = np.full((n, 2), "invalid_tracking_frame", dtype="U64")
    tracker, run_records, decoded = None, [], 0
    for index, rgb in enumerate(frames):
        if index >= n:
            raise ValueError("source decoded more frames than locked tracking clock")
        checked_frame(rgb, hashes[index])
        decoded += 1
        if not valid[index]:
            tracker = None
            continue
        if index == 0 or not valid[index - 1]:
            outputs["run_start"][index] = True
            try:
                tracker = PixelTracker(rgb, landmarks[index])
                run_records.append({"start": index, "initial_counts": tracker.initial_counts,
                                    "initial_eye_distance_resized_px": tracker.eye_distance})
                if overlay_path is not None and len(run_records) == 1:
                    image = cv2.cvtColor(tracker.previous, cv2.COLOR_GRAY2BGR)
                    for points, color in [(tracker.stable, (0, 220, 0)), (tracker.brows[0], (0, 0, 255)), (tracker.brows[1], (255, 180, 0))]:
                        for point in points:
                            cv2.circle(image, tuple(np.rint(point).astype(int)), 2, color, -1)
                    cv2.imencode(".jpg", image)[1].tofile(str(overlay_path))
                reasons[index] = "run_initialization_no_difference"
            except ValueError as exc:
                tracker = None
                run_records.append({"start": index, "initialization_failure": str(exc)})
                reasons[index] = "initial_roi_unavailable"
            continue
        if tracker is None:
            reasons[index] = "initial_roi_unavailable"
            continue
        result = tracker.step(rgb)
        for key in ("raw_px", "corrected_px", "raw_normalized", "corrected_normalized"):
            outputs[key][index] = result[key]
        outputs["brow_tracks"][index] = result["brow_tracks"]
        outputs["stable_tracks"][index] = result["stable"]["tracks"]
        outputs["stable_inliers"][index] = result["stable"]["inliers"]
        if result["similarity"] is not None:
            outputs["similarity"][index] = result["similarity"]
        reasons[index] = [r or "observed" for r in result["missing_reason"]]
    if decoded != n:
        raise ValueError("source decoded fewer frames than locked tracking clock")
    possible = valid.copy()
    possible[0] = False
    possible[1:] &= valid[:-1]
    outputs["pair_valid"] = possible
    outputs["corrected_observed"] = np.isfinite(outputs["corrected_px"]).all(2)
    outputs["missing_reason"] = reasons
    report = {"source_id": source_id, "frames": n, "rgb_hashes_matched": n,
              "valid_pairs": int(possible.sum()), "runs": run_records, "sides": {},
              "semi_independent_pixel_evidence": True, "independent_ground_truth": False}
    for column, name in enumerate(SIDE_NAMES):
        good = outputs["corrected_observed"][:, column]
        raw_good = np.isfinite(outputs["raw_px"][:, column]).all(1)
        raw = outputs["raw_normalized"][:, column, 1]
        corrected = outputs["corrected_normalized"][:, column, 1]
        report["sides"][name] = {
            "observed_pairs": int(good.sum()), "observed_pair_fraction": float(good.sum() / possible.sum()) if possible.any() else None,
            "missing_reason_counts": dict(Counter(reasons[possible, column].tolist())),
            "raw_vertical_squared_motion": float(np.square(raw[raw_good]).sum()),
            "matched_raw_vertical_squared_motion": float(np.square(raw[good]).sum()),
            "corrected_vertical_squared_motion": float(np.square(corrected[good]).sum()),
            "raw_vertical_rms_normalized": float(np.sqrt(np.square(raw[raw_good]).mean())) if raw_good.any() else None,
            "corrected_vertical_rms_normalized": float(np.sqrt(np.square(corrected[good]).mean())) if good.any() else None,
            "corrected_abs_vertical_p95_normalized": float(np.quantile(np.abs(corrected[good]), .95)) if good.any() else None,
            "motion_detected_claim": "no amplitude threshold or semantic movement label assigned",
        }
    return report, outputs


def run(audit_dir, output, ffmpeg):
    audit_dir, output, ffmpeg = Path(audit_dir), Path(output), Path(ffmpeg)
    if output.exists() and any(output.iterdir()):
        raise ValueError("fresh output required")
    manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf8"))
    selection_path = _verified(audit_dir, manifest, "selection.json")
    provenance_path = _verified(audit_dir, manifest, "provenance.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    provenance = json.loads(provenance_path.read_text(encoding="utf8"))
    rows = selection["clips"]
    ids = [r["clip_id"] for r in rows]
    if len(ids) != 32 or len(set(ids)) != 32 or any(not isinstance(cid, str) or cid in ("", ".", "..") or any(c in cid for c in "/\\:") for cid in ids):
        raise ValueError("requires32 safe unique locked fit clips")
    if provenance["selection_sha256"] != sha(selection_path) or provenance["ffmpeg_sha256"] != sha(ffmpeg):
        raise ValueError("selection or ffmpeg differs from locked tracking reset")
    if provenance["fps"] != 25 or provenance["opencv_version"] != cv2.__version__:
        raise ValueError("native frame clock or OpenCV version differs from tracking audit")
    sources = {}
    for row in rows:
        if type(row["needs_resample"]) is not bool or sha(row["video"]) != row["video_sha256"]:
            raise ValueError("video source hash or decode protocol differs")
        sources[row["clip_id"]] = _verified(audit_dir, manifest, "arrays/" + row["clip_id"] + "_fresh_forward.npz")
    output.mkdir(parents=True, exist_ok=True)
    for subdir in ("arrays", "clips", "visual"):
        (output / subdir).mkdir()
    protocol = {**PROTOCOL, "selection_sha256": sha(selection_path), "source_manifest_sha256": sha(audit_dir / "manifest.json"),
                "ffmpeg_sha256": sha(ffmpeg), "opencv_version": cv2.__version__, "code_sha256": sha(__file__),
                "rgb_frame_sha_required": True, "scope": "all32 locked previously examined fit clips; not a held-out test"}
    write_json(output / "protocol.json", protocol)
    cv2.setNumThreads(1)
    records = []
    for row in rows:
        cid = row["clip_id"]
        with np.load(sources[cid], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("valid", "times", "landmarks", "rgb_sha256")}
        generator = video_frames(row["video"], row["needs_resample"], ffmpeg)
        try:
            report, values = diagnose_frames(generator, arrays, cid, output / "visual" / (cid + "_initial_roi.jpg"))
        finally:
            generator.close()
        report["schema"] = SCHEMA
        report["metadata"] = {k: row[k] for k in ("speaker_name", "emotion", "sentence", "video_sha256")}
        report["source_npz_sha256"] = sha(sources[cid])
        np.savez_compressed(output / "arrays" / (cid + ".npz"), **values)
        write_json(output / "clips" / (cid + ".json"), report)
        records.append(report)
        print("PIXEL_FLOW", cid, {name: report["sides"][name]["observed_pairs"] for name in SIDE_NAMES}, flush=True)
    possible = sum(r["valid_pairs"] for r in records)
    totals = {}
    for name in SIDE_NAMES:
        count = sum(r["sides"][name]["observed_pairs"] for r in records)
        misses = Counter()
        for row in records:
            misses.update(row["sides"][name]["missing_reason_counts"])
        energy = sum(r["sides"][name]["corrected_vertical_squared_motion"] for r in records)
        totals[name] = {"observed_pairs": count, "possible_pairs": possible,
                        "observed_pair_fraction": count / possible if possible else None,
                        "reason_counts": dict(misses), "corrected_vertical_squared_motion": energy,
                        "corrected_vertical_rms_normalized": (energy / count) ** .5 if count else None}
    result = {"schema": SCHEMA, "clips": records, "clip_count": len(records), "sides": totals,
              "all_rgb_hashes_matched": True, "semi_independent_pixel_evidence": True,
              "independent_ground_truth": False, "causal_confidence": False, "used_for_training": False,
              "limitations": "Initial ROIs and validity come from MediaPipe;2D similarity does not remove3D head deformation, lighting, skin movement or LK drift; no semantic brow-motion correctness claim",
              "protocol_sha256": sha(output / "protocol.json")}
    write_json(output / "summary.json", result)
    write_json(output / "manifest.json", {p.relative_to(output).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                                         for p in sorted(output.rglob("*")) if p.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(DEFAULT_FFMPEG))
    args = parser.parse_args()
    result = run(args.audit, args.output, args.ffmpeg)
    print(json.dumps({"schema": SCHEMA, "clip_count": result["clip_count"], "sides": result["sides"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
