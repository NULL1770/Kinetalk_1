"""Package fixed context12 trajectories and review without model inference.

Only this review's visual directory and index.html are written. All main visual
modes are deployment outputs; target-history oracle is a separately labeled
numeric diagnostic. Historical sample selection is preserved exactly.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_temporal_repair import read, sha
from scripts.export_temporal_repair_examples import plot
from scripts.package_prefix_formal_review import json_write, save_arrays, table
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input

ARMS = ('chunk_empty', 'chunk_teacher', 'whole')
SEEDS = (42, 123, 2026)
MODES = ('Tracked reference', 'Previous audio mean + flow', 'Context empty chunks',
         'Context generated prefix', 'Context whole window', 'Teacher empty prefix (ablation)')


def load_sources(root, old):
    manifest = read(root / 'new_nine_clips_manifest.json')
    if sha(root / 'new_nine_clips.npz') != manifest['output_sha256']:
        raise ValueError('Downloaded nine-clip subset hash differs')
    if sha(root / 'fixed_visual_selection.json') != manifest['selection_sha256']:
        raise ValueError('Fixed selection binding differs')
    if (manifest['noise_seed'] != 42 or manifest['oracle_included'] is not False
            or manifest['raw_clamped'] is not False or manifest['test_loaded'] is not False):
        raise ValueError('Subset seed or generation scope differs')
    reports, bindings = {}, {}
    for arm in ARMS:
        complete = read(root / arm / 'complete.json')
        report = read(root / arm / 'evaluation.json')
        source = {'curves_sha256': manifest['source_curves'][arm],
                  'recipe_sha256': complete['recipe_sha256']}
        if (source['curves_sha256'] != complete['curves_sha256']
                or report['recipe_sha256'] != complete['recipe_sha256']
                or complete['completed_epochs'] != 12 or report['clips'] != 405
                or report['arm'] != arm or report['noise_seeds'] != list(SEEDS)
                or report['smoke'] is not False or report['test_loaded'] is not False
                or report['default_replaced'] is not False):
            raise ValueError('Subset/evaluation/source binding differs: ' + arm)
        reports[arm] = report
        bindings[arm] = {**source, 'complete_sha256': sha(root / arm / 'complete.json'),
                         'evaluation_sha256': sha(root / arm / 'evaluation.json')}
    historical = read(old / 'complete.json')
    if historical['status'] != 'complete':
        raise ValueError('Historical source is incomplete')
    for name, digest in historical['files'].items():
        if sha(old / name) != digest:
            raise ValueError('Historical source binding differs: ' + name)
    old_provenance = read(old / 'visual/provenance.json')
    selection = read(root / 'fixed_visual_selection.json')
    if selection['nine_plot_clips'] != old_provenance['nine_plot_clips']:
        raise ValueError('Fixed metadata-selected historical clips changed')
    return manifest, reports, bindings, historical, old_provenance


def export(root, old, audio_root):
    manifest, reports, bindings, historical, previous = load_sources(root, old)
    picks = previous['nine_plot_clips']
    with np.load(root / 'new_nine_clips.npz', allow_pickle=False) as z:
        new = {key: z[key].copy() for key in z.files}
    with np.load(old / 'visual/nine_clip_curves.npz', allow_pickle=False) as z:
        prior = {key: z[key].copy() for key in z.files}
    if new['arms'].tolist() != list(ARMS) or new['motions'].shape != (3, 9, 96, 52):
        raise ValueError('Expected the three complete deployment modes')
    if prior['mode_names'][2] != 'Previous audio mean + flow' or prior['channels'].tolist() != ARKIT_NAMES:
        raise ValueError('Historical white mode or channel order differs')
    ids = [pick['clip_id'] for pick in picks]
    if new['clip_id'].tolist() != ids or prior['clip_id'].tolist() != ids:
        raise ValueError('Nine-clip ID order differs')
    for key in ('times', 'valid', 'channel_mask'):
        if new[key].dtype != prior[key].dtype or not np.array_equal(new[key], prior[key]):
            raise ValueError('Native times or masks differ: ' + key)
    if new['target'].dtype != prior['motions'].dtype or not np.array_equal(new['target'], prior['motions'][0]):
        raise ValueError('Historical/new target coefficients differ')
    if new['teacher_empty'].shape != (9, 96, 52):
        raise ValueError('Teacher empty intervention dimensions differ')
    motions = np.stack([new['target'], prior['motions'][2], *new['motions'], new['teacher_empty']])
    if motions.dtype != np.float32 or not np.isfinite(motions).all():
        raise ValueError('Expected finite original float32 coefficients')
    visual = root / 'visual'
    for folder in (visual, visual / 'video_npz', visual / 'audio'):
        folder.mkdir(exist_ok=True, parents=True)
    save_arrays(visual / 'nine_clip_curves.npz', motions=motions, mode_names=np.asarray(MODES),
                channels=prior['channels'], **{key: new[key] for key in ('times', 'valid', 'channel_mask', 'clip_id')})
    jobs = []
    for old_job in previous['jobs']:
        speaker = old_job['speaker']
        j = next(i for i, pick in enumerate(picks) if pick['speaker'] == speaker)
        input_path = old / 'visual' / old_job['input']
        report_path = old / 'visual' / speaker / 'display_report.json'
        report = read(report_path)
        if report['input_sha256'] != sha(input_path):
            raise ValueError('Historical video input differs: ' + speaker)
        with np.load(input_path, allow_pickle=False) as z:
            if (z['clip_id'].item() != ids[j] or not np.array_equal(z['motions'][2], motions[1, j])
                    or not np.array_equal(z['motions'][0], motions[0, j])):
                raise ValueError('Historical video selection differs: ' + speaker)
        audio = audio_root / (speaker + '.wav')
        if sha(audio) != report['audio_sha256']:
            raise ValueError('Previously bound audio differs: ' + speaker)
        copied = visual / 'audio' / audio.name
        if not copied.exists():
            shutil.copy2(audio, copied)
        if sha(copied) != report['audio_sha256']:
            raise ValueError('Copied audio differs: ' + speaker)
        filename = visual / 'video_npz' / (speaker + '.npz')
        save_arrays(filename, motions=motions[:, j], mode_names=np.asarray(MODES), channels=prior['channels'],
                    times=new['times'][j], valid=new['valid'][j], channel_mask=new['channel_mask'][j],
                    clip_id=np.asarray(ids[j]), noise_seed=np.asarray(42))
        inspect_input(filename, 25)
        jobs.append({'speaker': speaker, 'clip_id': ids[j], 'input': f'video_npz/{speaker}.npz',
                     'input_sha256': sha(filename), 'audio': f'audio/{speaker}.wav', 'audio_sha256': sha(copied),
                     'audio_offset_seconds': old_job['audio_offset_seconds'],
                     'historical_display_report_sha256': sha(report_path)})
    provenance = {'schema': 'context_mechanism_review_v1', 'mode_names': list(MODES), 'nine_plot_clips': picks,
                  'jobs': jobs, 'bindings': bindings, 'noise_seed': 42,
                  'remote_subset_manifest_sha256': sha(root / 'new_nine_clips_manifest.json'),
                  'remote_subset_sha256': sha(root / 'new_nine_clips.npz'),
                  'fixed_selection_sha256': sha(root / 'fixed_visual_selection.json'),
                  'historical_complete_sha256': sha(old / 'complete.json'),
                  'historical_complete_files': historical['files'], 'historical_root': str(old.resolve()),
                  'historical_white_binding': previous['control_bindings']['state_white'],
                  'selection_unchanged': True, 'target_time_masks_equal_exactly': True,
                  'raw_clamped': False, 'gain_or_lag_fitted': False, 'oracle_in_main_visual': False,
                  'old_white_original_52_coefficients_preserved': True, 'test_loaded': False,
                  'nine_clip_curves_sha256': sha(visual / 'nine_clip_curves.npz')}
    json_write(visual / 'provenance.json', provenance)
    return reports, read(old / 'audit.json'), provenance


def plots(visual, reports):
    plot(visual)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), squeeze=False)
    entries = [('chunk_empty', 'full', 'Empty chunks'), ('chunk_teacher', 'full', 'Generated prefix'),
               ('whole', 'full', 'Whole window'), ('chunk_teacher', 'empty', 'Teacher empty prefix')]
    common = None
    for arm, mode, label in entries:
        profile = reports[arm]['modes']['42/' + mode]['chunk_diagnostics']['common_clip_coverage']
        if common is None:
            common = profile['clip_indices']
        elif common != profile['clip_indices']:
            raise ValueError('Profile coverage differs')
        for row, region in enumerate(('brows', 'eyes_expression')):
            values = profile['profiles'][region]
            for col, key in enumerate(('raw_mse', 'mean_error_rms', 'outside_fraction', 'prediction_raw_mean')):
                axis = axes[row, col]
                axis.plot(range(1, len(values) + 1), [value[key] for value in values], marker='o', label=label)
                if key == 'prediction_raw_mean' and arm == 'chunk_empty':
                    axis.plot(range(1, len(values) + 1), [value['reference_raw_mean'] for value in values],
                              color='black', linestyle='--', label='Tracked reference mean')
                axis.set(title=region + ' / ' + key, xlabel='Fixed 16-frame grid')
                axis.grid(alpha=.2)
    fig.legend(*axes[0, 3].get_legend_handles_labels(), loc='upper center', ncol=5)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(visual / 'rollout_profiles.png', dpi=120)
    plt.close(fig)


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
                or report['input_sha256'] != job['input_sha256'] or report['audio_sha256'] != job['audio_sha256']
                or report['rendered_modes'] != list(MODES) or report['metadata']['clip_id'] != job['clip_id']
                or report['metadata']['noise_seed'] != 42 or report['frames'] != 96 or report['fps'] != 25
                or report['native_coefficients_modified'] is not False or report['output_video_sha256'] != sha(video)):
            raise ValueError('Rendered video binding differs: ' + speaker)
        with np.load(visual / job['input'], allow_pickle=False) as source:
            trim_start = float(source['times'][0]) + job['audio_offset_seconds']
        if abs(report['audio_trim_start_seconds'] - trim_start) > 1e-7:
            raise ValueError('Rendered audio trim differs: ' + speaker)
        ffprobe = shutil.which('ffprobe')
        if not ffprobe:
            raise FileNotFoundError('ffprobe is required for video verification')
        streams = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-show_streams',
                                                     '-of', 'json', str(video)], encoding='utf8'))['streams']
        stream = next(s for s in streams if s['codec_type'] == 'video')
        if (stream['avg_frame_rate'] != '25/1' or int(stream['nb_frames']) != 96
                or not any(s['codec_type'] == 'audio' for s in streams)):
            raise ValueError('Video native clock, frames or audio differs: ' + speaker)
        checks.append({'speaker': speaker, 'input_sha256': job['input_sha256'], 'audio_sha256': job['audio_sha256'],
                       'video_sha256': sha(video), 'report_sha256': sha(folder / 'display_report.json'),
                       'frames': 96, 'fps': 25, 'audio': True})
        videos.append(f'<h2>{speaker}</h2><video controls preload="metadata" '
                      f'poster="visual/{speaker}/preview.png" src="visual/{speaker}/comparison.mp4"></video>')
    json_write(visual / 'video_verification.json', checks)
    return ''.join(videos)


def ratio(record, group):
    values = record['boundaries'][group]['chunk_boundary']
    return values['pred_rms'] / values['gt_rms']


def main_table(reports, old_audit):
    rows = [['部署结果：405 dev，三 seed 均值', '眉 raw MSE', '眉相关', '眉 RMS/参考', '眉越界',
             '眼 raw MSE', '眼相关', '口 raw MSE']]
    entries = [('旧：音频均值 + flow', old_audit['deployment']['state_white'])]
    entries += [(label, reports[arm]['distribution']) for arm, label in (
        ('chunk_empty', '本轮：空前缀分块'), ('chunk_teacher', '本轮：teacher训练 → 生成前缀'),
        ('whole', '本轮：整窗生成'))]
    for label, entry in entries:
        groups = entry['populations']['all']['groups']
        brow, eye, mouth = [groups[g]['mean_over_three_seeds'] for g in ('brows', 'eyes_expression', 'mouth')]
        rows.append([label, f"{brow['raw_mse']:.6f}", f"{brow['centered_correlation']:.3f}",
                     f"{brow['rms_ratio']:.3f}", f"{100*brow['outside_fraction']:.2f}%",
                     f"{eye['raw_mse']:.6f}", f"{eye['centered_correlation']:.3f}", f"{mouth['raw_mse']:.6f}"])
    return table(rows)


def continuation_table(reports):
    rows = [['部署连续性：三 seed 均值', '眉位移 RMS/GT', '眼位移 RMS/GT', '实际含义']]
    for arm, label in [('chunk_empty', '空前缀分块'), ('chunk_teacher', '生成前缀'), ('whole', '整窗')]:
        values = [np.mean([ratio(reports[arm]['modes'][f'{seed}/full'], group) for seed in SEEDS])
                  for group in ('brows', 'eyes_expression')]
        rows.append([label, f'{values[0]:.3f}', f'{values[1]:.3f}',
                     '同16帧网格；不是解码接缝' if arm == 'whole' else '实际分块解码接缝'])
    return table(rows)


def intervention_table(reports):
    rows = [['teacher模型：仅 seed42', '眉 raw MSE', '眉相关', '眉 RMS/参考', '眉接缝 RMS/GT', '眼接缝 RMS/GT']]
    for mode, label in [('full', '生成前缀'), ('empty', '同权重清空前缀')]:
        record = reports['chunk_teacher']['modes']['42/' + mode]
        brow = record['populations']['all']['brows']
        rows.append([label, f"{brow['raw_mse']:.6f}", f"{brow['centered_correlation']:.3f}",
                     f"{brow['rms_ratio']:.3f}", f"{ratio(record, 'brows'):.3f}", f"{ratio(record, 'eyes_expression'):.3f}"])
    return table(rows)


def oracle_table(root, reports):
    rows = [['GT条件诊断：仅 seed42', '眉 raw MSE', '眉相关', '眼 raw MSE', '眼相关', '眉实际续接 RMS/GT']]
    fit = read(root / 'chunk_teacher/fit_evaluation.json')
    for mode, name in [('oracle_history', '128 fit / GT过去'), ('oracle_reverse_history', '128 fit / GT过去逆序')]:
        record = fit['metrics'][mode]
        brow, eye, actual = record['brows'], record['eyes_expression'], record['actual_supplied_prefix']['brows']
        rows.append([name, f"{brow['raw_mse']:.6f}", f"{brow['centered_correlation']:.3f}",
                     f"{eye['raw_mse']:.6f}", f"{eye['centered_correlation']:.3f}", f"{actual['rms']/actual['reference_rms']:.3f}"])
    for mode, name in [('oracle_history', '405 dev / GT过去'), ('oracle_reverse_history', '405 dev / GT过去逆序')]:
        record = reports['chunk_teacher']['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['modes']['42/' + mode]
        brow, eye = [record['populations']['all'][g] for g in ('brows', 'eyes_expression')]
        actual = record['actual_prefix_continuation']['metrics']['brows']
        rows.append([name, f"{brow['raw_mse']:.6f}", f"{brow['centered_correlation']:.3f}",
                     f"{eye['raw_mse']:.6f}", f"{eye['centered_correlation']:.3f}", f"{actual['rms']/actual['reference_rms']:.3f}"])
    return table(rows)


def page(root, reports, old_audit, provenance, required):
    status = read(root / 'status.json')
    if status['status'] != 'complete' or status['smoke'] is not False or status['epochs_per_arm'] != 12:
        raise ValueError('Fixed training completion differs')
    minutes = status['seconds'] / 60
    empty, teacher, whole = reports['chunk_empty'], reports['chunk_teacher'], reports['whole']
    get_ratio = lambda report, group: np.mean([ratio(report['modes'][f'{seed}/full'], group) for seed in SEEDS])
    values = [get_ratio(report, group) for report in (empty, teacher, whole) for group in ('brows', 'eyes_expression')]
    brow = teacher['distribution']['populations']['all']['groups']['brows']['mean_over_three_seeds']
    eye = teacher['distribution']['populations']['all']['groups']['eyes_expression']['mean_over_three_seeds']
    videos = verify_videos(root / 'visual', provenance, required)
    probe = read(root / 'chunk_empty/step0_fit.json')['position0_vs8_same_weights_same_noise']['observed_upper_rms_difference']
    output = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>context12：连续性与动态对应复核</title><style>body{{font:16px/1.7 system-ui;max-width:1400px;margin:24px auto;padding:0 20px;color:#20262b;background:#f6f8fa}}h1{{font-size:28px}}h2{{font-size:21px}}table{{border-collapse:collapse;background:white;width:100%;white-space:nowrap}}td{{border-bottom:1px solid #ddd;padding:8px}}tr:first-child{{font-weight:bold;background:#eef2f5}}.scroll{{overflow-x:auto}}video,img{{width:100%}}a{{color:#17639f}}.callout{{border-left:4px solid #3c7187;padding:12px 18px;background:#eaf4f8}}.warning{{border-left:4px solid #b47b20;padding:12px 18px;background:#fff8e8}}details{{background:white;padding:12px;margin:18px 0}}</style>
<h1>context12 · 连续性显著改善，音频动态对应仍弱</h1>
<div class="callout">三臂各12轮、1740次更新已完成，训练与评估合计约 {minutes:.1f} 分钟。
teacher训练后使用自己生成的历史，眉/眼接缝位移从空前缀对照的 {values[0]:.2f}/{values[1]:.2f} 倍参考，降至 {values[2]:.2f}/{values[3]:.2f} 倍。
这次部署条件下的续接机制有明显进展，但不能称为目标动态已经重建成功。</div>
<div class="warning">teacher部署眉/眼时序相关仍只有 {brow['centered_correlation']:.3f}/{eye['centered_correlation']:.3f}；眼 raw MSE 没有改善，绝对状态和动作时机仍有问题。
接缝顺滑不等于情感动态正确。whole 的同位置位移为 {values[4]:.3f}/{values[5]:.3f} 倍参考；它一次生成整窗，没有分块解码接缝。</div>
<p>405条反复使用的内部开发片段、三固定seed42/123/2026，固定第12轮；128条fit仅作已曝光训练片段的接收诊断。未读封存test，默认模型未替换。
三臂共享原history12/no_history warmstart与冻结system/audio/local；whole输入为8个invalid槽+96帧，共104槽，前16帧位置匹配。原始目标不做GT均值中心化。</p>
<h2>可部署结果</h2><div class="scroll">{main_table(reports, old_audit)}</div>
<p>全部为原始系数评分，越界未经clamp。单GT误差不是一对多自然度充分判据；应结合原始分布指标与感知评价。其余43通道保持的工程检查不等于身份或唇音同步已独立通过。
旧候选与本轮存在多项训练和参数化差异，不能作为单部件因果消融。</p>
<h2>连续性</h2><div class="scroll">{continuation_table(reports)}</div>
<p>表内为每个seed的预测位移RMS/GT位移RMS再求均值，单位一致。两个chunk臂为实际解码接缝；whole只在同一16帧网格统计。whole同时改变位置、上下文范围和求解重启，不能把改善只归因于某一项。</p>
<h2>同teacher权重清空前缀</h2><div class="scroll">{intervention_table(reports)}</div>
<p>仅seed42，与三seed主表分开。清空前缀同时删除过去动作与过去声学token，不能把差异只归因于动作数值。</p>
<p><a href="RESULTS.md">结论与限制</a> · <a href="metrics_review.md">逐项指标复核</a> · <a href="metrics_review.json">复核数值</a> · <a href="new_nine_clips_manifest.json">远端导出绑定</a> · <a href="visual/provenance.json">本地可视化绑定</a> · <a href="visual/video_verification.json">视频核验</a></p>
<h2>固定片段六模式对照</h2><p>上排：跟踪参考 / 旧音频均值+flow / 本轮空前缀分块；下排：本轮生成前缀 / 本轮整窗 / 同teacher权重清空前缀。
沿用旧九片段选择及三个视频片段，均为96帧、25Hz、seed42、同音轨。六模式原始52维系数，不按本轮效果重选、不增幅、不平滑、不拟合时间偏移。
显示时如clamp到[0,1]只影响渲染，评分与曲线仍用原值；统一rig不代表人物几何身份。主视频和曲线均不含GT-history oracle。</p>
{videos}
<h2>逐段原始状态诊断（seed42）</h2><p>沿同一mask覆盖规则选共同片段；whole的“段”只是评分网格。误差随段变化也可能受内容变化影响，不能单凭斜率认定递归漂移。</p>
<img src="visual/rollout_profiles.png" alt="逐段raw误差、均值误差、越界及原始均值">
<h2>九片段中心动态</h2><p>每条曲线仅为显示减自身有效帧均值；主评分仍保留绝对状态。</p><img src="visual/centered.png" alt="固定九片段六模式中心动态">
<h2>九片段原始系数</h2><img src="visual/raw.png" alt="固定九片段六模式原始系数">
<details><summary>独立诊断：GT历史接收与位置偏移（不是部署表现）</summary>
<p>GT过去已携带真实动作信息，以下结果只检验接收，不能证明音频可以预测。逆序只改变历史动作值，保留声学、mask与位置。
实际续接用真正提供的known末token；逆序末token不等于时间上紧邻的GT端点。</p><div class="scroll">{oracle_table(root, reports)}</div>
<p>step0相同权重、噪声与当前条件的position0–15对比8–23，观测上脸输出RMS差为 {probe:.6f}，提示空前缀padding/位置布局本身有影响；这是只读诊断，不是新增训练臂。</p>
<p>fit文件的其余43通道为显式占位，仅用于上脸诊断，不用于本页视频。</p></details></html>'''
    (root / 'index.html').write_text(output, encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('artifacts/context_mechanism_20260917/context12'))
    parser.add_argument('--previous-root', type=Path, default=Path('artifacts/history_context_20260917/history12'))
    parser.add_argument('--audio-root', type=Path, default=Path('artifacts/temporal_repair_20260917/visual/audio'))
    parser.add_argument('--require-videos', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    reports, old_audit, provenance = export(args.root, args.previous_root, args.audio_root)
    if not args.export_only:
        plots(args.root / 'visual', reports)
        page(args.root, reports, old_audit, provenance, args.require_videos)
    print(json.dumps({'page': str((args.root / 'index.html').resolve()), 'jobs': provenance['jobs'],
                      'raw_six_modes_ready': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
