"""Frozen EmotiEffLib VA + eight-class posterior on explicitly selected videos.

VA is the pretrained visual regression output, never synthesized from class
labels. SCRFD defines unwarped face crops; missing/multiple faces remain missing.
Outputs retain sampled25Hz frame indices and RGB hashes without interpolation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_tracking_reset import video_frames

SCHEMA = "visual_semantics_va_q_pilot_v1"
CLASS_NAMES = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TEACHER_SHA = "c43e056ad388d4a8dc911832b8291435b2af537f967e5870ebd731574ec7e812"
DETECTOR_SHA = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
COMMIT = "520a051c64cd191521e5934655314e769a319684"
DEFAULT_ASSETS = ROOT / "artifacts/visual_semantics_20260918/teacher_assets"
DEFAULT_FFMPEG = Path("C:/Users/zhh/AppData/Local/com.minimax.hub/current/resources/ffmpeg/ffmpeg.exe")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def preprocess(rgb):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("nonempty RGB crop required")
    value = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.
    value = (value - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(value.transpose(2, 0, 1))


def split_outputs(raw):
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 2 or raw.shape[1] != 10 or not np.isfinite(raw).all():
        raise ValueError("teacher must return finite B,10 (8 logits + valence,arousal)")
    logits = raw[:, :8].copy()
    centered = logits.astype(np.float64) - logits.max(1, keepdims=True)
    q = np.exp(centered)
    q /= q.sum(1, keepdims=True)
    return {"logits": logits, "q": q.astype(np.float32), "va_raw": raw[:, 8:10].copy(),
            "va": np.clip(raw[:, 8:10], -1., 1.).copy()}


def nms(boxes, scores, threshold=.4):
    if not len(boxes):
        return np.zeros(0, int)
    order = np.argsort(-scores, kind="stable")
    area = np.maximum(0, boxes[:, 2] - boxes[:, 0] + 1) * np.maximum(0, boxes[:, 3] - boxes[:, 1] + 1)
    keep = []
    while len(order):
        first = int(order[0])
        keep.append(first)
        rest = order[1:]
        low = np.maximum(boxes[first, :2], boxes[rest, :2])
        high = np.minimum(boxes[first, 2:], boxes[rest, 2:])
        intersection = np.maximum(0, high[:, 0] - low[:, 0] + 1) * np.maximum(0, high[:, 1] - low[:, 1] + 1)
        overlap = intersection / np.maximum(area[first] + area[rest] - intersection, 1e-12)
        order = rest[overlap <= threshold]
    return np.asarray(keep)


def session(path, threads=4):
    try:
        import onnxruntime as ort
    except ImportError:
        sys.path.insert(0, str(ROOT / "tmp/visual_deps"))
        import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class VisualTeacher:
    def __init__(self, assets, threads=4):
        assets = Path(assets)
        self.teacher_path = assets / "enet_b0_8_va_mtl.onnx"
        self.detector_path = assets / "scrfd_10g_bnkps.onnx"
        if sha(self.teacher_path) != TEACHER_SHA or sha(self.detector_path) != DETECTOR_SHA:
            raise ValueError("teacher/detector hash differs from frozen verified assets")
        self.teacher = session(self.teacher_path, threads)
        self.detector = session(self.detector_path, threads)
        self.teacher_input = self.teacher.get_inputs()[0].name
        self.detector_input = self.detector.get_inputs()[0].name
        self.anchors = {}

    def detect(self, rgb):
        height, width = rgb.shape[:2]
        # Match SCRFD official preserve-aspect, top-left pad preprocessing.
        ratio = height / width
        new_h, new_w = (640, int(640 / ratio)) if ratio > 1 else (int(640 * ratio), 640)
        scale = new_h / height
        canvas = np.zeros((640, 640, 3), np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(rgb, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128., (640, 640), (127.5, 127.5, 127.5), swapRB=False)
        outputs = self.detector.run(None, {self.detector_input: blob})
        boxes, scores = [], []
        for index, stride in enumerate((8, 16, 32)):
            score = outputs[index].reshape(-1)
            distances = outputs[index + 3].reshape(-1, 4) * stride
            if stride not in self.anchors:
                center = np.stack(np.mgrid[:640 // stride, :640 // stride][::-1], axis=-1).astype(np.float32)
                self.anchors[stride] = np.repeat(center.reshape(-1, 2) * stride, 2, axis=0)
            center = self.anchors[stride]
            if len(center) != len(score):
                raise ValueError("SCRFD output anchor layout changed")
            selected = score >= .6
            decoded = np.column_stack([center - distances[:, :2], center + distances[:, 2:]])
            boxes.append(decoded[selected] / scale)
            scores.append(score[selected])
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        keep = nms(boxes, scores)
        return boxes[keep], scores[keep]

    def infer(self, crops):
        inputs = np.stack([preprocess(crop) for crop in crops])
        return split_outputs(self.teacher.run(None, {self.teacher_input: inputs})[0])


def extract_clip(row, teacher, ffmpeg, stride=5, limit_samples=None):
    start = time.perf_counter()
    video = Path(row["video"])
    if sha(video) != row["video_sha256"]:
        raise ValueError("video hash differs for " + row["clip_id"])
    indices, hashes, boxes, scores, counts, reasons, crops, positions = [], [], [], [], [], [], [], []
    generator = video_frames(video, row["needs_resample"], ffmpeg)
    try:
        for index, rgb in enumerate(generator):
            if index % stride:
                continue
            position = len(indices)
            indices.append(index)
            hashes.append(hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest())
            detected, score = teacher.detect(rgb)
            counts.append(len(detected))
            if len(detected) != 1:
                boxes.append([np.nan] * 4)
                scores.append(np.nan)
                reasons.append("no_face" if not len(detected) else "multiple_faces")
            else:
                h, w = rgb.shape[:2]
                x1, y1 = np.floor(detected[0, :2]).astype(int)
                x2, y2 = np.ceil(detected[0, 2:]).astype(int)
                x1, x2 = np.clip([x1, x2], 0, w)
                y1, y2 = np.clip([y1, y2], 0, h)
                boxes.append([x1, y1, x2, y2])
                scores.append(float(score[0]))
                if x2 <= x1 or y2 <= y1:
                    reasons.append("invalid_crop")
                else:
                    reasons.append("observed")
                    crops.append(rgb[y1:y2, x1:x2].copy())
                    positions.append(position)
            if limit_samples and len(indices) >= limit_samples:
                break
    finally:
        generator.close()
    n = len(indices)
    result = {"logits": np.full((n, 8), np.nan, np.float32), "q": np.full((n, 8), np.nan, np.float32),
              "va": np.full((n, 2), np.nan, np.float32), "va_raw": np.full((n, 2), np.nan, np.float32)}
    for first in range(0, len(crops), 32):
        inferred = teacher.infer(crops[first:first + 32])
        for key, value in inferred.items():
            result[key][positions[first:first + 32]] = value
    valid = np.asarray(reasons) == "observed"
    result.update(times=np.asarray(indices, np.float64) / 25., frame_indices=np.asarray(indices, np.int64),
                  source_frame_index=np.asarray(indices, np.int64), valid=valid, bbox=np.asarray(boxes, np.float32),
                  face_score=np.asarray(scores, np.float32), face_count=np.asarray(counts, np.int16),
                  missing_reason=np.asarray(reasons), rgb_sha256=np.asarray(hashes), class_names=np.asarray(CLASS_NAMES))
    report = {"clip_id": row["clip_id"], "samples": n, "valid_samples": int(valid.sum()),
              "valid_fraction": float(valid.mean()) if n else 0., "seconds": time.perf_counter() - start,
              "va_mean": result["va"][valid].mean(0).tolist() if valid.any() else None,
              "va_std": result["va"][valid].std(0).tolist() if valid.any() else None,
              "mean_q": result["q"][valid].mean(0).tolist() if valid.any() else None,
              "mean_q_prediction": CLASS_NAMES[int(result["q"][valid].mean(0).argmax())] if valid.any() else None,
              "raw_va_out_of_range_fraction": float((np.abs(result["va_raw"][valid]) > 1).mean()) if valid.any() else None,
              "missing_counts": {name: reasons.count(name) for name in sorted(set(reasons))},
              "ground_truth": False}
    return result, report


def run(selection, output, assets=DEFAULT_ASSETS, ffmpeg=DEFAULT_FFMPEG, stride=5, max_clips=None, smoke_samples=None, threads=4):
    selection, output = Path(selection), Path(output)
    if stride < 1:
        raise ValueError("stride must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("fresh output required")
    payload = json.loads(selection.read_text(encoding="utf-8-sig"))
    rows = payload["clips"]
    ids = [r["clip_id"] for r in rows]
    if len(set(ids)) != len(ids) or any(any(c in cid for c in "/\\:") or cid in ("", ".", "..") for cid in ids):
        raise ValueError("safe unique clip ids required")
    for row in rows:
        if type(row["needs_resample"]) is not bool:
            raise ValueError("explicit source resampling required")
        if str(row.get("split", "fit_smoke")).lower() in ("test", "405"):
            raise ValueError("test extraction not permitted in pilot")
    rows = rows[:max_clips] if max_clips else rows
    output.mkdir(parents=True, exist_ok=True)
    (output / "arrays").mkdir()
    teacher = VisualTeacher(assets, threads)
    protocol = {"schema": SCHEMA, "selection_sha256": sha(selection), "teacher": "EmotiEffLib/enet_b0_8_va_mtl",
                "repository": "https://github.com/sb-ai-lab/EmotiEffLib", "commit": COMMIT,
                "model_sha256": TEACHER_SHA, "detector_sha256": DETECTOR_SHA,
                "ffmpeg_sha256": sha(ffmpeg), "class_names": CLASS_NAMES, "va_indices": [8, 9],
                "va_order": ["valence", "arousal"], "q": "softmax of actual first8 logits at temperature1",
                "preprocess": "rectangular SCRFD crop, RGB224, ImageNet mean/std; no alignment, flip or smoothing",
                "detector": "SCRFD640, threshold.6, NMS.4, exactly one face, no fallback or interpolation",
                "clock": "decoded source to native25Hz per needs_resample; sampled frame_index/25",
                "stride": stride, "sample_fps": 25 / stride,
                "invalid": "VA/logits/q NaN; false validity; no interpolation",
                "limited_smoke": bool(max_clips or smoke_samples), "semantic_ground_truth": False,
                "source": "pretrained AffectNet visual regression and classification from SAME frozen model",
                "limitations": "domain-transfer/calibration unknown; facial movement or crops may affect framewise semantics",
                "code_sha256": sha(__file__)}
    write_json(output / "protocol.json", protocol)
    records = []
    for row in rows:
        arrays, report = extract_clip(row, teacher, ffmpeg, stride, smoke_samples)
        report["metadata"] = {k: row[k] for k in ("split", "speaker", "emotion", "sentence", "video_sha256") if k in row}
        arrays["metadata_json"] = np.asarray(json.dumps({**protocol, "clip_id": row["clip_id"], "video_sha256": row["video_sha256"]}, ensure_ascii=False))
        np.savez_compressed(output / "arrays" / (row["clip_id"] + ".npz"), **arrays)
        write_json(output / (row["clip_id"] + ".json"), report)
        records.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        write_json(output / "status.json", {"state": "running", "completed": len(records), "total": len(rows)})
    total = sum(r["samples"] for r in records)
    result = {"schema": SCHEMA, "clips": records, "clip_count": len(records), "samples": total,
              "valid_samples": sum(r["valid_samples"] for r in records), "seconds": sum(r["seconds"] for r in records),
              "model_sha256": TEACHER_SHA, "class_names": CLASS_NAMES, "semantic_ground_truth": False}
    write_json(output / "summary.json", result)
    write_json(output / "status.json", {"state": "complete", "completed": len(records), "total": len(rows)})
    write_json(output / "manifest.json", {p.relative_to(output).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                                         for p in output.rglob("*") if p.is_file()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--smoke-samples", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.selection, args.output, args.assets, args.ffmpeg, args.stride, args.max_clips, args.smoke_samples, args.threads)
