"""Package fixed semantic-pilot samples into raw curves and bounded render previews.

Selection uses holdout metadata only (first ID per emotion), and draw42 only.
The target panel is target upper9 composed over baseline43, not full-face GT.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input

ARMS = ("va_oracle", "va_audio", "va_static", "posterior_oracle", "posterior_audio", "posterior_static")
VIDEO_ARMS = ("va_oracle", "va_audio", "posterior_oracle", "posterior_audio")
UPPER = list(UPPER_INDICES)
OTHER = [i for i in range(52) if i not in UPPER]
SCHEMA = "visual_semantic_pilot_review_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def safe_id(cid):
    if not isinstance(cid, str) or not cid or cid in (".", "..") or any(c in cid for c in "/\\:"):
        raise ValueError("Safe clip ID required")
    return cid


def select_examples(clips):
    selected = {}
    for cid, clip in clips.items():
        safe_id(cid)
        metadata = clip["metadata"]
        if metadata.get("split") != "holdout":
            continue
        emotion = int(metadata["emotion"])
        if emotion not in selected or cid < selected[emotion]:
            selected[emotion] = cid
    if not selected:
        raise ValueError("No explicitly marked holdout metadata")
    return [selected[key] for key in sorted(selected)]


def compose_case(clip, seeds=None):
    baseline = array(clip["baseline52"])
    valid, native_valid = array(clip["valid"]), array(clip["native_valid"])
    target, times = array(clip["target"]), array(clip["times"])
    n = len(times)
    if (baseline.shape != (n, 52) or target.shape != (n, 9)
            or valid.dtype != bool or native_valid.dtype != bool
            or valid.shape != (n,) or native_valid.shape != (n,)
            or not native_valid.any() or np.any(valid & ~native_valid)
            or not np.isfinite(times).all()
            or (n > 1 and not np.allclose(np.diff(times), .04, rtol=0, atol=1e-7))
            or not np.isfinite(baseline).all() or not np.isfinite(target[valid]).all()):
        raise ValueError("Finite native52, target9 and nested masks/25Hz clock required")
    seeds = list(clip.get("seeds", seeds if seeds is not None else []))
    if seeds.count(42) != 1 or len(set(seeds)) != len(seeds):
        raise ValueError("One explicit seed42 and unique draw seeds required")
    draw = seeds.index(42)
    composed = {"baseline": baseline.copy()}
    upper = {"reference": target}
    for arm in ARMS:
        values = array(clip["samples"][arm])
        if values.shape != (len(seeds), n, 9) or not np.isfinite(values[:, native_valid]).all():
            raise ValueError("Finite matching seeded upper9 sample arrays required")
        upper[arm] = values[draw]
    for name, values in upper.items():
        result = baseline.copy()
        # Only reference availability comes from the visual teacher. Generated
        # arms must continue on all native-valid audio, including unknown semantics.
        support = valid if name == "reference" else native_valid
        result[np.ix_(support, UPPER)] = values[support]
        if (not np.array_equal(result[:, OTHER], baseline[:, OTHER])
                or not np.array_equal(result[~native_valid], baseline[~native_valid])):
            raise RuntimeError("Protected channels or unsupported/invalid frames changed")
        composed[name] = result
    return composed, valid, native_valid, times, draw


def plot_curves(path, composed, common, native, times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(17, 10), sharex=True)
    style = {"reference": ("black", "-", 1.7), "baseline": ("#777777", "-", 1.1),
             "va_oracle": ("#3f7ddd", "-", 1.1), "va_audio": ("#3f7ddd", "--", 1.1),
             "va_static": ("#3f7ddd", ":", .9), "posterior_oracle": ("#ce654c", "-", 1.1),
             "posterior_audio": ("#ce654c", "--", 1.1), "posterior_static": ("#ce654c", ":", .9)}
    for ax, channel in zip(axes.ravel(), UPPER):
        for name, (color, linestyle, width) in style.items():
            y = composed[name][:, channel].astype(float).copy()
            y[~(common if name == "reference" else native)] = np.nan
            ax.plot(times, y, label=name, color=color, linestyle=linestyle, linewidth=width)
        ax.fill_between(times, 0, 1, where=~common, color="#aaaaaa", alpha=.14,
                        transform=ax.get_xaxis_transform())
        ax.axhline(0, color="#888888", linewidth=.4)
        ax.axhline(1, color="#888888", linewidth=.4)
        ax.set_title(ARKIT_NAMES[channel])
        ax.set_xlabel("Native seconds")
        ax.grid(alpha=.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Full native raw curves, seed42. Gray = outside common semantic support; no coefficient clamping")
    fig.tight_layout(rect=(0, .06, 1, .97))
    fig.savefig(path, dpi=135)
    plt.close(fig)


def package(args):
    if args.output.exists():
        raise FileExistsError("Fresh review output required")
    if args.preview_frames < 0:
        raise ValueError("Preview frame limit must be nonnegative")
    predictions = torch.load(args.predictions, map_location="cpu", weights_only=False)
    clips = predictions["clips"]
    chosen = select_examples(clips)
    source = json.loads((args.baseline_root / "provenance.json").read_text(encoding="utf8"))
    source_records = {r["clip_id"]: r for r in source["clips"]}
    # Resolve and verify every selected input before creating an output directory.
    prepared = []
    for cid in chosen:
        composed, common, native, times, draw = compose_case(clips[cid], predictions.get("seeds"))
        record = source_records[cid]
        full_arrays = (args.baseline_root / record["arrays"]).is_file()
        baseline_path = (args.baseline_root / record["arrays" if full_arrays else "video_npz"]).resolve()
        expected_sha = record["arrays_sha256" if full_arrays else "video_npz_sha256"]
        if not baseline_path.is_relative_to(args.baseline_root.resolve()) or sha(baseline_path) != expected_sha:
            raise ValueError("Baseline path/hash differs")
        with np.load(baseline_path, allow_pickle=False) as saved:
            if full_arrays:
                saved_baseline = saved["baseline52"]
            else:
                names = saved["mode_names"].tolist()
                if names.count("run12 frozen audio baseline") != 1:
                    raise ValueError("Unique pinned baseline render mode required")
                saved_baseline = saved["motions"][names.index("run12 frozen audio baseline")]
            if (not np.array_equal(saved_baseline, composed["baseline"])
                    or not np.array_equal(saved["valid"], native)
                    or not np.allclose(saved["times"], times, rtol=0, atol=1e-7)):
                raise ValueError("Prediction and frozen baseline clock/data differ")
            audio_path = (args.baseline_root / str(saved["audio_relative_path"].item())).resolve()
            audio_hash = str(saved["audio_sha256"].item())
            offset = float(saved["audio_offset_seconds"].item())
            channel_mask = saved["channel_mask"].copy()
        if (not audio_path.is_relative_to(args.baseline_root.resolve()) or sha(audio_path) != audio_hash
                or not np.isfinite(offset)):
            raise ValueError("Bound native audio differs")
        prepared.append((cid, composed, common, native, times, draw, audio_path, audio_hash, offset, channel_mask))
    args.output.mkdir(parents=True)
    for sub in ("video_npz", "audio", "curves", "visual"):
        (args.output / sub).mkdir()
    records, jobs, sections = [], [], []
    for cid, composed, common, native, times, draw, audio_path, audio_hash, offset, channel_mask in prepared:
        modes = ["Target upper9 + baseline43 (reference)", "Frozen run12 audio baseline",
                 "VA oracle (visual teacher)", "VA audio (predicted)",
                 "Posterior oracle (visual teacher)", "Posterior audio (predicted)"]
        ordered = ["reference", "baseline", *VIDEO_ARMS]
        npz = Path("video_npz") / (cid + ".npz")
        audio_relative = Path("audio") / (cid + audio_path.suffix)
        shutil.copy2(audio_path, args.output / audio_relative)
        np.savez_compressed(args.output / npz, channels=np.asarray(ARKIT_NAMES),
            mode_names=np.asarray(modes), motions=np.stack([composed[name] for name in ordered]),
            times=times, valid=native, semantic_valid=common, channel_mask=channel_mask,
            clip_id=np.asarray(cid), noise_seed=np.asarray(42),
            audio_relative_path=np.asarray(audio_relative.as_posix()), audio_sha256=np.asarray(audio_hash),
            audio_offset_seconds=np.asarray(offset))
        display = inspect_input(args.output / npz, 25)[-1]
        curve_path = Path("curves") / (cid + ".png")
        plot_curves(args.output / curve_path, composed, common, native, times)
        np.savez_compressed(args.output / "curves" / (cid + ".npz"), times=times,
                            valid=native, semantic_valid=common, **composed)
        frames = len(times) if args.preview_frames == 0 else min(len(times), args.preview_frames)
        output_relative = Path("visual") / cid
        job = {"input": npz.as_posix(), "output": output_relative.as_posix(), "fps": 25,
               "columns": 3, "tile_size": args.tile_size, "samples": args.samples,
               "max_frames": args.preview_frames, "audio": audio_relative.as_posix(),
               "audio_offset_seconds": offset, "audio_sha256": audio_hash,
               "expected_video": (output_relative / "comparison.mp4").as_posix()}
        jobs.append(job)
        if args.render:
            command = [sys.executable, str(Path(__file__).with_name("render_dynamic_rig_comparison.py")),
                "--input", str(args.output / npz), "--output", str(args.output / output_relative),
                "--audio", str(args.output / audio_relative), "--audio-offset-seconds", str(offset),
                "--columns", "3", "--tile-size", str(args.tile_size), "--samples", str(args.samples),
                "--max-frames", str(args.preview_frames), "--fps", "25",
                "--blend", str(args.blend), "--blender", str(args.blender), "--ffmpeg", str(args.ffmpeg)]
            subprocess.run(command, check=True)
        metadata = clips[cid]["metadata"]
        records.append({"clip_id": cid, "metadata": metadata, "seed": 42, "draw_index": draw,
            "native_frames": len(times), "common_frames": int(common.sum()),
            "preview_frames": frames, "truncated_preview": frames < len(times),
            "protected43_exact": True, "generated_support": "native_valid, independent of semantic teacher availability",
            "reference_support": "common semantic mask only; baseline outside reference support",
            "native_invalid_exact": True, "display_report": display,
            "video_npz_sha256": sha(args.output / npz), "audio_sha256": audio_hash})
        video = (output_relative / "comparison.mp4").as_posix()
        sections.append(f'<section><h2>{html.escape(cid)}</h2><p>原生 {len(times)} 帧；共同监督支持 {int(common.sum())} 帧。'
            f'视频仅前 {frames} 帧 / {frames / 25:.2f} 秒；曲线覆盖完整片段。</p>'
            f'<video controls preload="metadata" src="{video}"></video>'
            f'<p><a href="{npz.as_posix()}">完整原始系数 NPZ</a></p>'
            f'<img src="{curve_path.as_posix()}" alt="完整九通道曲线"></section>')
        print("SEMANTIC_REVIEW_CASE", cid, "preview", frames, "of", len(times), flush=True)
    write(args.output / "render_jobs.json", {"schema": SCHEMA, "jobs": jobs,
        "driver": "scripts/render_dynamic_rig_comparison.py", "rendered": bool(args.render),
        "paths_relative_to": "directory containing this JSON"})
    write(args.output / "report.json", {"schema": SCHEMA, "records": records,
        "predictions_sha256": sha(args.predictions), "baseline_provenance_sha256": sha(args.baseline_root / "provenance.json"),
        "selection": "First lexicographic holdout clip ID per emotion; seed42; no output-based choice",
        "reference": "Target upper9 composed over frozen baseline43, not full-face ground truth",
        "raw_clamping": False, "display_clamping": "[0,1] by renderer; statistics saved per mode/channel",
        "curves_full_native": True, "preview_frame_limit": args.preview_frames,
        "static_controls_in_curves": True, "identity_mouth_emotion_certified": False,
        "source_split": "Previously exposed historical fit clips; sentence holdout applies only to this pilot",
        "script_sha256": sha(__file__)})
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>视觉情感条件试验</title>'
    page += '<style>body{font-family:system-ui;max-width:1450px;margin:30px auto;padding:0 20px;color:#222}video,img{width:100%}section{margin:35px 0;border-top:1px solid #ddd}p{line-height:1.6}.note{background:#fff4d9;padding:18px}</style>'
    page += '<h1>视觉情感条件试验：固定留句样例，seed42</h1><p class="note">'
    page += '这页是实际模型输出的诊断预览。每情感按 clip ID 固定选择首个 holdout 样例。'
    page += 'oracle 使用当前视频的视觉教师条件，仅检验条件接收能力；audio 使用音频预测条件。'
    page += '左上参考仅将目标眉眼 9 通道放到基座，其余 43 通道仍为基座，不是完整 GT。'
    page += '模型在全部音频有效帧持续生成，视觉教师缺测不遮盖生成结果；参考上脸只在共同监督支持内可用，其余帧显示基座。无效帧仅由显示器就近填充。视频显示会截到 [0,1]，原始曲线和 NPZ 不截幅。'
    page += 'static 对照见完整曲线；这些历史训练片不能作为最终泛化验证，也未独立证明身份、情感、口型全部合格。</p>'
    page += ''.join(sections) + '<p><a href="report.json">来源、保护与截幅报告</a> · <a href="render_jobs.json">渲染任务</a></p></html>'
    (args.output / "index.html").write_text(page, encoding="utf8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-frames", type=int, default=96)
    parser.add_argument("--tile-size", type=int, default=320)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--blend", type=Path, default=Path("D:/实验室项目/新实验/arkit2.blend"))
    parser.add_argument("--blender", type=Path, default=Path("D:/3d_engine/blender-4.3.0-windows-x64/blender.exe"))
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    package(parser.parse_args())


if __name__ == "__main__":
    main()
