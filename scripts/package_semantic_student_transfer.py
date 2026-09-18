"""Review old/new audio students through identical frozen 300-epoch receivers.

This is an input-transfer/distribution-shift diagnostic, not matched retraining.
Fixed metadata-selected holdout examples and seed42 are used without selection
on prediction quality. The reference contains target upper9 plus baseline43.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_visual_semantic_experiment import array, safe_id, select_examples, sha, write, UPPER, OTHER
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input

SOURCE_SCHEMA = "semantic_student_transfer_v1"
SCHEMA = "semantic_student_transfer_review_v1"
ARMS = ("va_old", "va_new", "va_static", "va_reverse", "posterior_old", "posterior_new", "posterior_static", "posterior_reverse")
OPTIONAL_ARMS = ("va_constant", "posterior_constant")
VIDEO_ARMS = ("va_old", "va_new", "posterior_old", "posterior_new")
MODES = ("Target upper9 + baseline43 (reference)", "Frozen run12 audio baseline",
         "VA old student / frozen receiver", "VA new student / frozen receiver",
         "Posterior old student / frozen receiver", "Posterior new student / frozen receiver")


def compose_case(clip, seeds=None):
    base, target = array(clip["baseline52"]), array(clip["target"])
    common, native, times = array(clip["valid"]), array(clip["native_valid"]), array(clip["times"])
    if times.ndim != 1 or not len(times):
        raise ValueError("Nonempty native clock required")
    n = len(times)
    if (base.shape != (n, 52) or target.shape != (n, 9)
            or common.dtype != bool or native.dtype != bool or common.shape != (n,) or native.shape != (n,)
            or not native.any() or np.any(common & ~native)
            or not np.isfinite(times).all() or not np.isfinite(base).all()
            or not np.isfinite(target[common]).all()
            or (n > 1 and not np.allclose(np.diff(times), .04, rtol=0, atol=1e-7))):
        raise ValueError("Finite native52/target9 and nested Boolean masks on 25Hz clock required")
    seeds = list(clip.get("seeds", seeds if seeds is not None else []))
    if len(set(seeds)) != len(seeds) or seeds.count(42) != 1:
        raise ValueError("Unique explicit draw seeds including seed42 required")
    draw = seeds.index(42)
    upper = {"reference": target}
    for arm in (*ARMS, *(name for name in OPTIONAL_ARMS if name in clip["samples"])):
        samples = array(clip["samples"][arm])
        if samples.shape != (len(seeds), n, 9) or not np.isfinite(samples[:, native]).all():
            raise ValueError("Every arm must have finite [draw,native_frame,9] samples")
        upper[arm] = samples[draw]
    composed = {"baseline": base.copy()}
    for name, motion in upper.items():
        supported = common if name == "reference" else native
        result = base.copy()
        result[np.ix_(supported, UPPER)] = motion[supported]
        if not np.array_equal(result[:, OTHER], base[:, OTHER]) or not np.array_equal(result[~native], base[~native]):
            raise RuntimeError("Protected43 or native invalid values changed")
        composed[name] = result
    return composed, common, native, times, draw


def bind_baseline(root, record, composed, native, times):
    root = root.resolve()
    array_path = (root / record["arrays"]).resolve()
    if not array_path.is_relative_to(root):
        raise ValueError("Baseline path leaves source root")
    full = array_path.is_file()
    path = array_path if full else (root / record["video_npz"]).resolve()
    expected = record["arrays_sha256" if full else "video_npz_sha256"]
    if not path.is_relative_to(root) or sha(path) != expected:
        raise ValueError("Baseline source path/hash differs")
    with np.load(path, allow_pickle=False) as saved:
        if full:
            baseline = saved["baseline52"]
        else:
            names = saved["mode_names"].tolist()
            if names.count("run12 frozen audio baseline") != 1:
                raise ValueError("Unique frozen baseline mode required")
            baseline = saved["motions"][names.index("run12 frozen audio baseline")]
        if (not np.array_equal(baseline, composed["baseline"])
                or not np.array_equal(saved["valid"], native)
                or not np.array_equal(saved["times"], times)):
            raise ValueError("Transfer and frozen baseline arrays or native clock differ")
        waveform = (root / str(saved["audio_relative_path"].item())).resolve()
        waveform_sha = str(saved["audio_sha256"].item())
        offset = float(saved["audio_offset_seconds"].item())
        channel_mask = saved["channel_mask"].copy()
    if (not waveform.is_relative_to(root) or sha(waveform) != waveform_sha
            or not np.isfinite(offset) or channel_mask.shape != (52,) or channel_mask.dtype != bool):
        raise ValueError("Native audio hash/offset or channel mask differs")
    return waveform, waveform_sha, offset, channel_mask


def plot_curves(path, values, common, native, times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(17, 10), sharex=True)
    styles = {"reference": ("#111111", "-", 1.7), "baseline": ("#888888", "-", 1.0)}
    for prefix, color in (("va", "#3f7ddd"), ("posterior", "#ce654c")):
        for suffix, line in (("old", "--"), ("new", "-"), ("static", ":"), ("reverse", "-.")):
            styles[prefix + "_" + suffix] = (color, line, 1.1 if suffix == "new" else .9)
        if prefix + "_constant" in values:
            styles[prefix + "_constant"] = ("#17419c" if prefix == "va" else "#813222", (0, (3, 1, 1, 1, 1, 1)), 1.0)
    for ax, channel in zip(axes.ravel(), UPPER):
        for name, (color, style, width) in styles.items():
            y = values[name][:, channel].astype(float).copy()
            y[~(common if name == "reference" else native)] = np.nan
            ax.plot(times, y, color=color, linestyle=style, linewidth=width, label=name)
        ax.fill_between(times, 0, 1, where=~common, transform=ax.get_xaxis_transform(), color="#aaaaaa", alpha=.14)
        ax.axhline(0, color="#888888", linewidth=.4)
        ax.axhline(1, color="#888888", linewidth=.4)
        ax.set_title(ARKIT_NAMES[channel]); ax.set_xlabel("Native seconds"); ax.grid(alpha=.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("Frozen receiver input transfer, seed42. Full raw curves; gray = unavailable visual reference")
    fig.tight_layout(rect=(0, .06, 1, .97)); fig.savefig(path, dpi=135); plt.close(fig)


def package(args):
    if args.output.exists():
        raise FileExistsError("Fresh transfer review directory required")
    if not 1 <= args.preview_frames <= 96 or args.tile_size < 128 or args.samples < 1:
        raise ValueError("Preview requires 1..96 frames and positive render dimensions")
    predictions = torch.load(args.predictions, map_location="cpu", weights_only=False)
    if predictions.get("schema") != SOURCE_SCHEMA:
        raise ValueError("Require explicit semantic_student_transfer_v1 predictions")
    clips = predictions["clips"]
    chosen = select_examples(clips)
    source = json.loads((args.baseline_root / "provenance.json").read_text(encoding="utf8"))
    source_records = {row["clip_id"]: row for row in source["clips"]}
    if len(source_records) != len(source["clips"]):
        raise ValueError("Duplicate baseline source clips")
    prepared = []
    for cid in chosen:
        values, common, native, times, draw = compose_case(clips[cid], predictions.get("seeds"))
        record = source_records[cid]
        if int(record["emotion"]) != int(clips[cid]["metadata"]["emotion"]):
            raise ValueError("Transfer and baseline emotion metadata differ")
        bound = bind_baseline(args.baseline_root, record, values, native, times)
        prepared.append((cid, values, common, native, times, draw, *bound))
    args.output.mkdir(parents=True)
    for name in ("video_npz", "audio", "curves", "visual"):
        (args.output / name).mkdir()
    records, jobs, sections = [], [], []
    for cid, values, common, native, times, draw, waveform, audio_sha, offset, channels in prepared:
        audio_path = Path("audio") / (cid + waveform.suffix)
        shutil.copy2(waveform, args.output / audio_path)
        npz_path = Path("video_npz") / (cid + ".npz")
        np.savez_compressed(args.output / npz_path, channels=np.asarray(ARKIT_NAMES), mode_names=np.asarray(MODES),
            motions=np.stack([values[name] for name in ("reference", "baseline", *VIDEO_ARMS)]),
            times=times, valid=native, semantic_valid=common, channel_mask=channels,
            clip_id=np.asarray(cid), noise_seed=np.asarray(42), audio_relative_path=np.asarray(audio_path.as_posix()),
            audio_sha256=np.asarray(audio_sha), audio_offset_seconds=np.asarray(offset))
        display = inspect_input(args.output / npz_path, 25)[-1]
        curve_path = Path("curves") / (cid + ".png")
        plot_curves(args.output / curve_path, values, common, native, times)
        np.savez_compressed(args.output / "curves" / (cid + ".npz"), times=times,
                            valid=native, semantic_valid=common, **values)
        visual = Path("visual") / cid
        frames = min(len(times), args.preview_frames)
        job = {"input": npz_path.as_posix(), "output": visual.as_posix(), "fps": 25,
               "columns": 3, "tile_size": args.tile_size, "samples": args.samples,
               "max_frames": args.preview_frames, "audio": audio_path.as_posix(),
               "audio_offset_seconds": offset, "audio_sha256": audio_sha,
               "expected_video": (visual / "comparison.mp4").as_posix()}
        jobs.append(job)
        render_status = None
        if args.render:
            command = [sys.executable, str(Path(__file__).with_name("render_dynamic_rig_comparison.py")),
                "--input", str(args.output / npz_path), "--output", str(args.output / visual),
                "--audio", str(args.output / audio_path), "--audio-offset-seconds", str(offset),
                "--columns", "3", "--tile-size", str(args.tile_size), "--samples", str(args.samples),
                "--max-frames", str(args.preview_frames), "--fps", "25", "--blend", str(args.blend),
                "--blender", str(args.blender), "--ffmpeg", str(args.ffmpeg)]
            subprocess.run(command, check=True)
            render_status = json.loads((args.output / visual / "display_report.json").read_text(encoding="utf8"))
            if (render_status.get("status") != "complete" or render_status.get("audio_sha256") != audio_sha
                    or render_status.get("original_blend_unchanged") is not True
                    or render_status.get("output_video_sha256") != sha(args.output / visual / "comparison.mp4")):
                raise RuntimeError("Rendered transfer video/audio/source binding differs")
        records.append({"clip_id": cid, "metadata": clips[cid]["metadata"], "seed": 42, "draw_index": draw,
            "native_frames": len(times), "common_frames": int(common.sum()), "preview_frames": frames,
            "truncated_preview": frames < len(times), "protected43_exact": True, "native_invalid_exact": True,
            "generated_support": "all native_valid, independent of visual teacher availability",
            "display_report": display, "video_npz_sha256": sha(args.output / npz_path),
            "curve_arms": [name for name in (*ARMS, *OPTIONAL_ARMS) if name in values],
            "video_sha256": render_status.get("output_video_sha256") if render_status else None,
            "audio_sha256": audio_sha})
        video = (visual / "comparison.mp4").as_posix()
        sections.append(f'<section><h2>{html.escape(cid)}</h2><p>原生 {len(times)} 帧；共同参考 {int(common.sum())} 帧。'
            f'视频为前 {frames} 帧 / {frames / 25:.2f} 秒；曲线为完整片段。</p>'
            f'<video controls preload="metadata" src="{video}"></video><p><a href="{npz_path.as_posix()}">完整系数</a></p>'
            f'<img src="{curve_path.as_posix()}" alt="固定receiver的旧新学生与static/reverse完整曲线"></section>')
        print("STUDENT_TRANSFER_REVIEW", cid, frames, "of", len(times), flush=True)
    write(args.output / "render_jobs.json", {"schema": SCHEMA, "jobs": jobs, "rendered": bool(args.render),
        "driver": "scripts/render_dynamic_rig_comparison.py", "paths_relative_to": "directory containing this JSON"})
    write(args.output / "report.json", {"schema": SCHEMA, "source_schema": SOURCE_SCHEMA, "records": records,
        "predictions_sha256": sha(args.predictions), "baseline_provenance_sha256": sha(args.baseline_root / "provenance.json"),
        "selection": "First lexicographic holdout clip ID per emotion; seed42; no outcome choice",
        "receiver_policy": "Existing audio300 receiver frozen; only semantic student input replaced",
        "distribution_shift_diagnostic": True, "matched_receiver_retraining": False,
        "reference": "Target upper9 over baseline43; not full-face GT",
        "raw_clamping": False, "display_clamping": "[0,1], display only; per-channel counts recorded",
        "static_and_reverse_in_full_curves": True, "preview_frame_limit": args.preview_frames,
        "constant_acoustic_intervention_in_full_curves": any(name in clips[cid]["samples"] for cid in chosen for name in OPTIONAL_ARMS),
        "historical_upstream_exposure": True, "identity_mouth_emotion_certified": False,
        "script_sha256": sha(__file__)})
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>语义学生替换试验</title>'
    page += '<style>body{font-family:system-ui;max-width:1450px;margin:30px auto;padding:0 20px;color:#222}video,img{width:100%}section{margin:35px 0;border-top:1px solid #ddd}p{line-height:1.6}.note{background:#fff4d9;padding:18px}</style>'
    page += '<h1>语义学生替换：相同冻结接收器，固定样例与 seed42</h1><p class="note">'
    page += '旧学生和新学生的音频预测均送入已有 300 轮 audio 接收器。本轮没有重新训练接收器，'
    page += '因此是输入替换及分布偏移诊断，不是新学生与接收器匹配重训后的结果。'
    page += '每情感固定选择首个 holdout clip；static 与时间反转 reverse 对照使用同接收器，见完整曲线。'
    if any(name in clips[cid]["samples"] for cid in chosen for name in OPTIONAL_ARMS):
        page += 'constant 是同一 TCN 学生的恒定声学输入干预，与 static 条件对照分别显示。'
    page += '左上仅为目标眉眼 9 通道与基座其余 43 通道的组合，不是完整 GT。模型在所有音频有效帧持续生成；'
    page += '视觉参考缺测处仅参考面板回到基座，原始无效帧保持基座并由显示器就近填充。'
    page += '视频显示截到 [0,1]，原始曲线不截幅。历史训练片的留句结果仍不能作为最终泛化或身份口型验收。</p>'
    page += ''.join(sections) + '<p><a href="report.json">来源、保护与显示统计</a> · <a href="render_jobs.json">渲染任务</a></p></html>'
    (args.output / "index.html").write_text(page, encoding="utf8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "baseline-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
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
