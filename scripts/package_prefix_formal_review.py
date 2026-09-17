"""Package the fixed prefix12 review from hash-bound saved trajectories.

No inference, fitting, sample selection, curve modification or oracle display.
Only this review's visual directory and index.html are written.
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
from scripts.audit_temporal_repair import metadata_equal, read, sha, validate_curves
from scripts.export_temporal_repair_examples import plot
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES

MODES = ('Tracked reference', 'Previous audio mean + aligned',
         'Previous audio mean + flow', 'Formal no prefix',
         'Formal generated prefix', 'Formal empty prefix (ablation)')
SEEDS = (42, 123, 2026)


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def save_arrays(path, **arrays):
    """Preserve a previously rendered NPZ's byte hash on unchanged reruns."""
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            if set(previous.files) != set(arrays) or any(
                    previous[k].dtype != v.dtype or not np.array_equal(previous[k], v)
                    for k, v in arrays.items()):
                raise ValueError('Existing visual data differs; use a fresh review root: ' + str(path))
    else:
        np.savez_compressed(path, **arrays)


def export(root, old, audio_root):
    complete = read(old / 'complete.json')
    if complete.get('status') != 'complete':
        raise ValueError('Historical review is incomplete')
    for name, digest in complete['files'].items():
        if sha(old / name) != digest:
            raise ValueError('Historical complete binding differs: ' + name)
    # Both hashes must pass before either pickle is deserialized.
    bindings, reports = {}, {}
    for arm in ('no_prefix', 'scheduled_prefix'):
        folder = root / arm
        manifest = read(folder / 'complete.json')
        digest = sha(folder / 'curves.pt')
        if digest != manifest['curves_sha256'] or manifest['completed_epochs'] != 12:
            raise ValueError('Formal curve SHA or completed budget differs: ' + arm)
        reports[arm] = read(folder / 'evaluation.json')
        report = reports[arm]
        if (report['recipe_sha256'] != manifest['recipe_sha256']
                or report['arm'] != arm or report['smoke'] is not False
                or report['test_loaded'] is not False or report['default_replaced'] is not False
                or report['noise_seeds'] != list(SEEDS) or report['clips'] != 405):
            raise ValueError('Formal evaluation protocol differs: ' + arm)
        bindings[arm] = {'curves_path': str((folder / 'curves.pt').resolve()),
                         'curves_sha256': digest, 'complete_sha256': sha(folder / 'complete.json'),
                         'evaluation_sha256': sha(folder / 'evaluation.json'),
                         'final_sha256_from_complete': manifest['final_sha256'],
                         'recipe_sha256': manifest['recipe_sha256']}
    curves = {arm: torch.load(root / arm / 'curves.pt', map_location='cpu',
                              weights_only=False, mmap=True) for arm in bindings}
    reference = curves['no_prefix']
    for arm, value in curves.items():
        validate_curves(value)
        metadata_equal(reference, value)
        if value['recipe_sha256'] != bindings[arm]['recipe_sha256']:
            raise ValueError('Curve recipe differs: ' + arm)
    previous = read(old / 'visual/provenance.json')
    picks = previous['nine_plot_clips']
    if len(picks) != 9 or previous.get('noise_seed') != 42 or previous.get('oracle_in_main_visual') is not False:
        raise ValueError('Historical fixed selection protocol differs')
    with np.load(old / 'visual/nine_clip_curves.npz', allow_pickle=False) as z:
        prior = {key: z[key].copy() for key in z.files}
    if prior['mode_names'].tolist()[:3] != list(MODES[:3]) or prior['channels'].tolist() != ARKIT_NAMES:
        raise ValueError('Historical control names or channel ordering differs')
    indices = [pick['index'] for pick in picks]
    if prior['clip_id'].tolist() != [pick['clip_id'] for pick in picks]:
        raise ValueError('Historical selection and arrays disagree')
    for pick in picks:
        index = pick['index']
        if (reference['clip_id'][index] != pick['clip_id']
                or int(reference['speaker_id'][index]) != pick['speaker_id']
                or int(reference['emotion_id'][index]) != pick['emotion_id']):
            raise ValueError('Fixed clip metadata differs: ' + pick['clip_id'])
    for name in ('times', 'valid', 'channel_mask'):
        current = reference[name][indices].numpy()
        if current.dtype != prior[name].dtype or not np.array_equal(current, prior[name]):
            raise ValueError('Historical target timing/masks differ: ' + name)
    target = reference['target'][indices].numpy()
    if target.dtype != prior['motions'].dtype or not np.array_equal(target, prior['motions'][0]):
        raise ValueError('Historical target coefficients differ')
    data = np.stack([prior['motions'][0], prior['motions'][1], prior['motions'][2],
                     reference['predictions']['42/full'][indices].numpy(),
                     curves['scheduled_prefix']['predictions']['42/full'][indices].numpy(),
                     curves['scheduled_prefix']['predictions']['42/empty'][indices].numpy()])
    if data.shape != (6, 9, 96, 52):
        raise ValueError('Expected six raw modes, nine fixed clips, 96 frames and 52 channels')
    visual = root / 'visual'
    for folder in (visual, visual / 'video_npz', visual / 'audio'):
        folder.mkdir(exist_ok=True, parents=True)
    save_arrays(visual / 'nine_clip_curves.npz', motions=data, mode_names=np.asarray(MODES),
                **{key: prior[key] for key in ('times', 'valid', 'channel_mask', 'channels', 'clip_id')})
    jobs = []
    for previous_job in previous['jobs']:
        speaker = previous_job['speaker']
        j = next(i for i, pick in enumerate(picks) if pick['speaker'] == speaker)
        with np.load(old / 'visual' / previous_job['input'], allow_pickle=False) as video:
            for key in ('times', 'valid', 'channel_mask'):
                if not np.array_equal(video[key], prior[key][j]):
                    raise ValueError('Historical video selection differs: ' + speaker + '/' + key)
            if not np.array_equal(video['motions'][:3], prior['motions'][:3, j]):
                raise ValueError('Historical video curves differ: ' + speaker)
            if video['clip_id'].item() != picks[j]['clip_id']:
                raise ValueError('Historical video clip differs: ' + speaker)
        source_report = old / 'visual' / speaker / 'display_report.json'
        rendered = read(source_report)
        if rendered['input_sha256'] != sha(old / 'visual' / previous_job['input']):
            raise ValueError('Historical render input differs: ' + speaker)
        audio = audio_root / (speaker + '.wav')
        if sha(audio) != rendered['audio_sha256']:
            raise ValueError('Previously rendered audio differs: ' + speaker)
        destination = visual / 'audio' / audio.name
        if not destination.exists():
            shutil.copy2(audio, destination)
        if sha(destination) != rendered['audio_sha256']:
            raise ValueError('Copied audio differs: ' + speaker)
        filename = visual / 'video_npz' / (speaker + '.npz')
        save_arrays(filename, motions=data[:, j], mode_names=np.asarray(MODES),
                    times=prior['times'][j], valid=prior['valid'][j], channel_mask=prior['channel_mask'][j],
                    channels=prior['channels'], clip_id=np.asarray(picks[j]['clip_id']), noise_seed=np.asarray(42))
        jobs.append({'speaker': speaker, 'clip_id': picks[j]['clip_id'],
                     'input': f'video_npz/{speaker}.npz', 'input_sha256': sha(filename),
                     'audio': f'audio/{speaker}.wav', 'audio_sha256': sha(destination),
                     'audio_offset_seconds': previous_job['audio_offset_seconds'],
                     'source_audio_path': str(audio.resolve()),
                     'historical_display_report_sha256': sha(source_report)})
    provenance = {'schema': 'prefix_formal_review_v1', 'nine_plot_clips': picks, 'jobs': jobs,
                  'mode_names': list(MODES), 'bindings': bindings,
                  'historical_complete_sha256': sha(old / 'complete.json'),
                  'historical_complete_files': complete['files'],
                  'historical_source': str(old.resolve()),
                  'historical_control_bindings': previous['control_bindings'],
                  'noise_seed': 42, 'selection_unchanged': True, 'selection_uses_metadata_only': True,
                  'outcome_based_selection': False, 'target_time_masks_equal_exactly': True,
                  'old_aligned_white_52_coefficients_equal_exactly': True,
                  'raw_clamped': False, 'gain_or_lag_fitted': False, 'oracle_in_main_visual': False,
                  'test_loaded': False, 'default_replaced': False,
                  'nine_clip_curves_sha256': sha(visual / 'nine_clip_curves.npz')}
    json_write(visual / 'provenance.json', provenance)
    return read(old / 'audit.json'), reports, provenance


def plots(visual, reports):
    plot(visual)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), squeeze=False)
    entries = [('no_prefix', 'full', 'Formal no prefix'),
               ('scheduled_prefix', 'full', 'Formal generated prefix'),
               ('scheduled_prefix', 'empty', 'Formal empty prefix')]
    common = None
    for arm, mode, label in entries:
        profile = reports[arm]['modes']['42/' + mode]['chunk_diagnostics']['common_clip_coverage']
        if common is None:
            common = profile['clip_indices']
        elif common != profile['clip_indices']:
            raise ValueError('Profile populations differ')
        for row, region in enumerate(('brows', 'eyes_expression')):
            values = profile['profiles'][region]
            for col, key in enumerate(('raw_mse', 'mean_error_rms', 'outside_fraction', 'prediction_raw_mean')):
                axis = axes[row, col]
                axis.plot(range(1, len(values) + 1), [v[key] for v in values], marker='o', label=label)
                if key == 'prediction_raw_mean' and arm == 'no_prefix':
                    axis.plot(range(1, len(values) + 1), [v['reference_raw_mean'] for v in values],
                              color='black', linestyle='--', label='Tracked reference mean')
                axis.set(title=region + ' / ' + key, xlabel='Chunk (16 native frames)')
                axis.grid(alpha=.2)
    fig.legend(*axes[0, 3].get_legend_handles_labels(), loc='upper center', ncol=4)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(visual / 'rollout_profiles.png', dpi=120)
    plt.close(fig)


def table(rows):
    return '<table>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(cell)) + '</td>'
                              for cell in row) + '</tr>' for row in rows) + '</table>'


def verify_videos(visual, provenance, required):
    videos, checks = [], []
    for job in provenance['jobs']:
        speaker = job['speaker']
        folder = visual / speaker
        if not (folder / 'display_report.json').exists():
            if required:
                raise FileNotFoundError('Video not ready: ' + speaker)
            videos.append('<p>' + speaker + '：固定片段视频待生成。</p>')
            continue
        report = read(folder / 'display_report.json')
        video = folder / 'comparison.mp4'
        if (report['status'] != 'complete' or not report['original_blend_unchanged']
                or report['input_sha256'] != job['input_sha256']
                or report['audio_sha256'] != job['audio_sha256']
                or report['rendered_modes'] != list(MODES)
                or report['metadata']['clip_id'] != job['clip_id']
                or report['metadata']['noise_seed'] != 42
                or report['frames'] != 96 or report['fps'] != 25
                or report['native_coefficients_modified'] is not False
                or report['output_video_sha256'] != sha(video)):
            raise ValueError('Rendered video binding differs: ' + speaker)
        with np.load(visual / job['input'], allow_pickle=False) as source:
            trim_start = float(source['times'][0]) + job['audio_offset_seconds']
        if abs(report['audio_trim_start_seconds'] - trim_start) > 1e-7:
            raise ValueError('Rendered audio trim differs from native clock: ' + speaker)
        probe = shutil.which('ffprobe')
        if not probe:
            raise FileNotFoundError('ffprobe is required to validate videos')
        streams = json.loads(subprocess.check_output([probe, '-v', 'error', '-show_streams',
                             '-of', 'json', str(video)], encoding='utf8'))['streams']
        stream = next(s for s in streams if s['codec_type'] == 'video')
        if (stream['avg_frame_rate'] != '25/1' or int(stream['nb_frames']) != 96
                or not any(s['codec_type'] == 'audio' for s in streams)):
            raise ValueError('Video native clock, frame count or audio differs: ' + speaker)
        checks.append({'speaker': speaker, 'input_sha256': job['input_sha256'],
                       'audio_sha256': job['audio_sha256'], 'video_sha256': sha(video),
                       'display_report_sha256': sha(folder / 'display_report.json'),
                       'frames': 96, 'fps': 25, 'audio': True})
        videos.append(f'<h2>{speaker}</h2><video controls preload="metadata" '
                      f'poster="visual/{speaker}/preview.png" src="visual/{speaker}/comparison.mp4"></video>')
    json_write(visual / 'video_verification.json', checks)
    return ''.join(videos)


def page(root, old_audit, reports, provenance, require_videos):
    rows = [['方案（405 内部开发片段；三 seed 均值）', '眉 raw MSE', '眉时序相关', '眉 RMS/参考',
             '眉越界', '眼 raw MSE', '眼时序相关', '口 raw MSE']]
    entries = [(label, old_audit['deployment'][key]) for key, label in (
        ('state_aligned', '旧：音频均值 + 对齐动态'), ('state_white', '旧：音频均值 + flow'),
        ('no_history', '上一轮：无历史'), ('scheduled_history', '上一轮：池化生成历史'))]
    entries += [('本轮正式：无前缀', reports['no_prefix']['distribution']),
                ('本轮正式：生成前缀', reports['scheduled_prefix']['distribution'])]
    for label, scores in entries:
        groups = scores['populations']['all']['groups']
        brow, eye, mouth = [groups[k]['mean_over_three_seeds'] for k in ('brows', 'eyes_expression', 'mouth')]
        rows.append([label, f"{brow['raw_mse']:.6f}", f"{brow['centered_correlation']:.3f}",
                     f"{brow['rms_ratio']:.3f}", f"{100*brow['outside_fraction']:.2f}%",
                     f"{eye['raw_mse']:.6f}", f"{eye['centered_correlation']:.3f}", f"{mouth['raw_mse']:.6f}"])
    single = [['本轮干预（仅 seed42）', '眉 raw MSE', '眉时序相关', '眉 RMS/参考', '眉接缝 RMS', '眼接缝 RMS']]
    for arm, mode, label in [('no_prefix', 'full', '无前缀'), ('scheduled_prefix', 'full', '生成前缀'),
                             ('scheduled_prefix', 'empty', '同模型清空前缀')]:
        record = reports[arm]['modes']['42/' + mode]
        brow = record['populations']['all']['brows']
        seam = [record['boundaries'][k]['chunk_boundary']['pred_rms'] for k in ('brows', 'eyes_expression')]
        single.append([label, f"{brow['raw_mse']:.6f}", f"{brow['centered_correlation']:.3f}",
                       f"{brow['rms_ratio']:.3f}", f'{seam[0]:.5f}', f'{seam[1]:.5f}'])
    status = read(root / 'status.json')
    if status['status'] != 'complete' or status['smoke'] is not False or status['epochs_per_arm'] != 12:
        raise ValueError('Formal completion status differs')
    minutes = status['seconds'] / 60
    videos = verify_videos(root / 'visual', provenance, require_videos)
    output = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>正式 prefix12 结果复核</title><style>body{{font:16px/1.7 system-ui;max-width:1400px;margin:24px auto;padding:0 20px;color:#20262b;background:#f6f8fa}}h1{{font-size:28px}}h2{{font-size:21px}}table{{border-collapse:collapse;background:white;width:100%;white-space:nowrap}}td{{border-bottom:1px solid #ddd;padding:8px}}tr:first-child{{font-weight:bold;background:#eef2f5}}.scroll{{overflow-x:auto}}video,img{{width:100%}}a{{color:#17639f}}.callout{{border-left:4px solid #aa352e;padding:10px 18px;background:#fff0ed}}</style>
<h1>正式前缀上下文实验 · 12 轮</h1>
<div class="callout">本轮正式实验未达到目标，不能据此宣布动态方案成功。两组各 12 轮，训练与评估已完成，用时约 {minutes:.1f} 分钟。
生成前缀组的眉时序相关只有 0.034、动态 RMS 约为参考的 2.00 倍；接缝略有下降仍明显偏大。小集 pilot 的改善没有在这次完整内部开发集上复现。</div>
<p>405 条内部开发片段，固定 seed 42 / 123 / 2026，固定第 12 轮权重，无测试集和结果择优。
新旧候选还存在目标、分段、训练等差异，跨轮比较不能归因于单个部件。本轮配对比较改变的是整个前缀上下文：无前缀组同时屏蔽过去动作和过去声学 token，不能把差异只归因于动作值。</p>
<div class="scroll">{table(rows)}</div>
<p>raw MSE 和时序相关与固定跟踪参考逐条比较；一对多生成仍需结合分布和感知评估。RMS 较大只代表幅度，不能作为同步或情感动态正确的证据。这里的 mouth 指 27 个口部系数，不等价于已验证口型同步或身份。</p>
<h2>同模型清空前缀（单 seed 干预）</h2><div class="scroll">{table(single)}</div>
<p>该表仅 seed42，不与三 seed 均值混用；接缝为相邻生成段边界的系数位移 RMS。</p>
<p><a href="RESULTS.md">正式实验结论</a> · <a href="audit.json">独立审计</a> · <a href="no_prefix/evaluation.json">无前缀原始评估</a> · <a href="scheduled_prefix/evaluation.json">生成前缀原始评估</a> · <a href="visual/provenance.json">选片与来源绑定</a> · <a href="visual/video_verification.json">视频核验</a></p>
<h2>固定三人物片段 · 六模式对照</h2>
<p>上排：跟踪参考 / 旧对齐动态 / 旧 flow；下排：正式无前缀 / 正式生成前缀 / 同模型清空前缀。
沿用上一轮九片段选择和三个视频片段，不按本轮结果重选。原始 52 维系数、96 帧、25Hz、seed42、同一音轨；不做增幅、平滑或时间偏移拟合。
视频仅为显示把系数截断到 [0,1]，所有评分与曲线保留原值；统一网格不代表人物几何身份。主图和视频均不含真实历史 oracle。</p>
{videos}
<h2>逐段诊断（seed42，同一覆盖规则下的共同片段）</h2>
<p>误差、偏置与越界均在原始系数上计算。后段变化也可能来自内容差异，不能只凭曲线上升认定递归漂移。</p>
<img src="visual/rollout_profiles.png" alt="逐段原始误差、均值偏差、越界与预测均值">
<h2>九条固定片段：中心动态</h2><p>每条曲线减去自身有效帧均值，仅用于显示时序变化；raw 主评分未作此处理。</p>
<img src="visual/centered.png" alt="固定九片段中心动态六模式">
<h2>九条固定片段：原始系数</h2><img src="visual/raw.png" alt="固定九片段原始系数六模式"></html>'''
    (root / 'index.html').write_text(output, encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('artifacts/prefix_context_20260917/prefix12'))
    parser.add_argument('--previous-root', type=Path, default=Path('artifacts/history_context_20260917/history12'))
    parser.add_argument('--audio-root', type=Path, default=Path('artifacts/temporal_repair_20260917/visual/audio'))
    parser.add_argument('--require-videos', action='store_true')
    args = parser.parse_args()
    old, reports, provenance = export(args.root, args.previous_root, args.audio_root)
    plots(args.root / 'visual', reports)
    page(args.root, old, reports, provenance, args.require_videos)
    print(json.dumps({'page': str((args.root / 'index.html').resolve()),
                      'jobs': provenance['jobs'], 'raw_six_modes_ready': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
