"""Package the fixed audio-prefix adaptation review without inference or fitting.

Saved coefficients, native clocks, audio, and historical selection are hash-bound.
Only this review's visual directory and index.html are written.
"""
from __future__ import annotations

import argparse
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

ARMS = ('frozen_local', 'adapt_local')
SEEDS = (42, 123, 2026)
MODES = ('Tracked reference', 'Old audio mean + flow',
         'Previous prefix + audio mean', 'New frozen local + audio mean',
         'New adapted local + audio mean', 'New adapted local raw')


def load_reports(root, old, context, origin):
    historical = read(old / 'complete.json')
    if historical['status'] != 'complete':
        raise ValueError('Historical source is incomplete')
    for name, digest in historical['files'].items():
        if sha(old / name) != digest:
            raise ValueError('Historical source binding differs: ' + name)
    previous = read(old / 'visual/provenance.json')
    context_complete = read(context / 'chunk_teacher/complete.json')
    context_report = read(context / 'chunk_teacher/evaluation.json')
    origin_complete = read(origin / 'complete.json')
    origin_report = read(origin / 'report.json')
    if (sha(origin / 'report.json') != origin_complete['report_sha256']
            or origin_report['test_loaded'] is not False
            or origin_report['source_sha256']['context'] != context_complete['curves_sha256']
            or context_report['clips'] != 405 or context_report['noise_seeds'] != list(SEEDS)
            or context_report['recipe_sha256'] != context_complete['recipe_sha256']):
        raise ValueError('Previous context/DC report binding differs')
    expected = {'old_white': previous['control_bindings']['state_white'],
                'previous_context_dc': origin_complete['curves_sha256']}
    reports, bindings = {}, {}
    for arm in ARMS:
        folder = root / arm
        complete = read(folder / 'complete.json')
        raw, dc = read(folder / 'evaluation.json'), read(folder / 'dc_evaluation.json')
        if (complete['completed_epochs'] != 12 or complete['total_steps'] != 1740
                or complete['local_protected_unchanged'] is not True
                or complete['local_changed'] != (arm == 'adapt_local')):
            raise ValueError('Adaptation completion differs: ' + arm)
        for report in (raw, dc):
            if (report['recipe_sha256'] != complete['recipe_sha256']
                    or report['adaptation_arm'] != arm or report['clips'] != 405
                    or report['noise_seeds'] != list(SEEDS) or report['smoke'] is not False
                    or report['test_loaded'] is not False or report['default_replaced'] is not False
                    or report['nonupper_exact'] is not True or report['invalid_baseline_exact'] is not True):
                raise ValueError('Adaptation evaluation protocol differs: ' + arm)
        if (raw['schema'] != 'audio_prefix_adaptation_v1'
                or dc['schema'] != 'audio_prefix_dc_composition_v1'
                or dc['oracle_composition_performed'] is not False
                or dc['actual_prefix_scoring_performed'] is not False
                or dc['dc_invariants_passed'] is not True):
            raise ValueError('DC composition protocol differs: ' + arm)
        expected[arm + '/raw'] = complete['curves_sha256']
        expected[arm + '/dc'] = complete['dc_curves_sha256']
        reports[arm] = {'raw': raw, 'dc': dc, 'fit': read(folder / 'fit_evaluation.json')}
        bindings[arm] = {**complete, 'complete_sha256': sha(folder / 'complete.json'),
                         'evaluation_sha256': sha(folder / 'evaluation.json'),
                         'dc_evaluation_sha256': sha(folder / 'dc_evaluation.json'),
                         'fit_evaluation_sha256': sha(folder / 'fit_evaluation.json')}
    if origin_report['source_sha256']['state_white'] != expected['old_white']:
        raise ValueError('Previous DC white source differs')
    reports['context'] = {'raw': context_report, 'dc_distribution': origin_report['results']['audio_mean']}
    bindings['context'] = {'raw_curves_sha256': context_complete['curves_sha256'],
                           'raw_evaluation_sha256': sha(context / 'chunk_teacher/evaluation.json'),
                           'dc_curves_sha256': origin_complete['curves_sha256'],
                           'dc_report_sha256': sha(origin / 'report.json')}
    return reports, bindings, historical, previous, expected


def load_sources(root, old, context, origin):
    reports, bindings, historical, previous, expected = load_reports(root, old, context, origin)
    manifest = read(root / 'new_nine_clips_manifest.json')
    if sha(root / 'new_nine_clips.npz') != manifest['output_sha256']:
        raise ValueError('Downloaded nine-clip subset hash differs')
    if sha(root / 'fixed_visual_selection.json') != manifest['selection_sha256']:
        raise ValueError('Fixed selection binding differs')
    if (manifest['noise_seed'] != 42 or manifest['oracle_included'] is not False
            or manifest['raw_clamped'] is not False or manifest['test_loaded'] is not False
            or manifest['mode_names'] != list(MODES[1:])):
        raise ValueError('Subset seed, modes, or generation scope differs')
    if manifest['source_curves'] != expected:
        raise ValueError('Remote subset source bindings differ')
    selection = read(root / 'fixed_visual_selection.json')
    if selection['nine_plot_clips'] != previous['nine_plot_clips']:
        raise ValueError('Fixed historical clips changed')
    audit = read(root / 'integrity_audit.json')
    if (audit['status'] != 'passed' or audit['test_loaded'] is not False
            or audit['matched_rng_and_updates'] is not True
            or audit['all_405_metadata_equal'] is not True):
        raise ValueError('Remote independent integrity audit is incomplete')
    bindings['remote_integrity_audit_sha256'] = sha(root / 'integrity_audit.json')
    return manifest, reports, bindings, historical, previous


def export(root, old, context, origin, audio_root):
    manifest, reports, bindings, historical, previous = load_sources(root, old, context, origin)
    picks = previous['nine_plot_clips']
    with np.load(root / 'new_nine_clips.npz', allow_pickle=False) as z:
        new = {key: z[key].copy() for key in z.files}
    with np.load(old / 'visual/nine_clip_curves.npz', allow_pickle=False) as z:
        prior = {key: z[key].copy() for key in z.files}
    if new['mode_names'].tolist() != list(MODES[1:]) or new['motions'].shape != (5, 9, 96, 52):
        raise ValueError('Expected five fixed deployment modes in declared order')
    if prior['mode_names'][2] != 'Previous audio mean + flow' or prior['channels'].tolist() != ARKIT_NAMES:
        raise ValueError('Historical white mode or channel order differs')
    ids = [pick['clip_id'] for pick in picks]
    if new['clip_id'].tolist() != ids or prior['clip_id'].tolist() != ids:
        raise ValueError('Nine-clip ID order differs')
    for key in ('times', 'valid', 'channel_mask'):
        if new[key].dtype != prior[key].dtype or not np.array_equal(new[key], prior[key]):
            raise ValueError('Native times or masks differ: ' + key)
    if (new['target'].dtype != prior['motions'].dtype
            or not np.array_equal(new['target'], prior['motions'][0])
            or not np.array_equal(new['motions'][0], prior['motions'][2])):
        raise ValueError('Historical/new target or old-white coefficients differ')
    motions = np.concatenate([new['target'][None], new['motions']], axis=0)
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
    provenance = {'schema': 'audio_prefix_adaptation_review_v1', 'mode_names': list(MODES),
                  'nine_plot_clips': picks, 'jobs': jobs, 'bindings': bindings, 'noise_seed': 42,
                  'remote_subset_manifest_sha256': sha(root / 'new_nine_clips_manifest.json'),
                  'remote_subset_sha256': sha(root / 'new_nine_clips.npz'),
                  'remote_source_curves': manifest['source_curves'],
                  'fixed_selection_sha256': sha(root / 'fixed_visual_selection.json'),
                  'historical_complete_sha256': sha(old / 'complete.json'),
                  'historical_complete_files': historical['files'], 'historical_root': str(old.resolve()),
                  'historical_white_binding': previous['control_bindings']['state_white'],
                  'selection_unchanged': True, 'target_time_masks_equal_exactly': True,
                  'raw_clamped': False, 'gain_or_lag_fitted': False, 'oracle_in_main_visual': False,
                  'old_white_original_52_coefficients_preserved': True, 'test_loaded': False,
                  'dc_is_offline': True, 'dc_fed_back_as_prefix': False,
                  'nine_clip_curves_sha256': sha(visual / 'nine_clip_curves.npz')}
    json_write(visual / 'provenance.json', provenance)
    return reports, read(old / 'audit.json'), provenance


def plots(visual, reports):
    plot(visual)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), squeeze=False)
    entries = [('context', 'raw', 'Previous context raw'),
               ('frozen_local', 'raw', 'Frozen local raw'), ('adapt_local', 'raw', 'Adapted local raw'),
               ('frozen_local', 'dc', 'Frozen local DC'), ('adapt_local', 'dc', 'Adapted local DC')]
    common = None
    for arm, mode, label in entries:
        profile = reports[arm][mode]['modes']['42/full']['chunk_diagnostics']['common_clip_coverage']
        if common is None:
            common = profile['clip_indices']
        elif common != profile['clip_indices']:
            raise ValueError('Rollout profile coverage differs')
        for row, region in enumerate(('brows', 'eyes_expression')):
            values = profile['profiles'][region]
            for col, key in enumerate(('raw_mse', 'mean_error_rms', 'outside_fraction')):
                axis = axes[row, col]
                axis.plot(range(1, len(values) + 1), [value[key] for value in values], marker='o', label=label)
                axis.set(title=region + ' / ' + key, xlabel='Fixed 16-frame grid')
                axis.grid(alpha=.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .91))
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
            videos.append('<p>' + speaker + ': fixed-clip video pending.</p>')
            continue
        report = read(folder / 'display_report.json')
        video = folder / 'comparison.mp4'
        if (report['status'] != 'complete' or not report['original_blend_unchanged']
                or report['input_sha256'] != job['input_sha256'] or report['audio_sha256'] != job['audio_sha256']
                or report['rendered_modes'] != list(MODES) or report['metadata']['clip_id'] != job['clip_id']
                or report['metadata']['noise_seed'] != 42 or report['frames'] != 96 or report['fps'] != 25
                or report['native_coefficients_modified'] is not False or report['output_video_sha256'] != sha(video)):
            raise ValueError('Rendered video binding differs: ' + speaker)
        if sha(visual / job['input']) != job['input_sha256'] or sha(visual / job['audio']) != job['audio_sha256']:
            raise ValueError('Rendered source file changed: ' + speaker)
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
        videos.append(f'<h3>{speaker}</h3><video controls preload="metadata" '
                      f'poster="visual/{speaker}/preview.png" src="visual/{speaker}/comparison.mp4"></video>')
    json_write(visual / 'video_verification.json', checks)
    return ''.join(videos)


def means(distribution, region):
    return distribution['populations']['all']['groups'][region]['mean_over_three_seeds']


def ratio(record, region):
    values = record['boundaries'][region]['chunk_boundary']
    return values['pred_rms'] / values['gt_rms']


def tables(reports, old_audit):
    old = old_audit['deployment']['state_white']
    rows = [['405 dev, three seeds', 'Brow raw MSE', 'Eye raw MSE', 'Brow outside', 'Eye outside', 'Mouth raw MSE']]
    entries = [('Old audio mean + flow', old),
               ('Previous context raw', reports['context']['raw']['distribution']),
               ('Previous context DC', reports['context']['dc_distribution'])]
    entries += [(label, reports[arm][mode]['distribution']) for arm, mode, label in (
        ('frozen_local', 'raw', 'Frozen local raw'), ('frozen_local', 'dc', 'Frozen local DC'),
        ('adapt_local', 'raw', 'Adapted local raw'), ('adapt_local', 'dc', 'Adapted local DC'))]
    for label, entry in entries:
        brow, eye, mouth = [means(entry, g) for g in ('brows', 'eyes_expression', 'mouth')]
        rows.append([label, f"{brow['raw_mse']:.6f}", f"{eye['raw_mse']:.6f}",
                     f"{100*brow['outside_fraction']:.2f}%", f"{100*eye['outside_fraction']:.2f}%",
                     f"{mouth['raw_mse']:.6f}"])
    raw = table(rows)
    rows = [['Temporal dynamics, three seeds', 'Brow centered MSE', 'Brow corr.', 'Brow RMS/ref.',
             'Eye centered MSE', 'Eye corr.', 'Eye RMS/ref.', 'Brow seam/ref.', 'Eye seam/ref.']]
    for label, entry, report in [('Old audio mean + flow', old, None),
                                ('Previous context', reports['context']['raw']['distribution'], reports['context']['raw']),
                                ('Frozen local', reports['frozen_local']['raw']['distribution'], reports['frozen_local']['raw']),
                                ('Adapted local', reports['adapt_local']['raw']['distribution'], reports['adapt_local']['raw'])]:
        values = [label]
        for region in ('brows', 'eyes_expression'):
            group = means(entry, region)
            values += [f"{group['centered_mse']:.6f}", f"{group['centered_correlation']:.3f}", f"{group['rms_ratio']:.3f}"]
        values += ['n/a', 'n/a'] if report is None else [
            f"{np.mean([ratio(report['modes'][f'{seed}/full'], region) for seed in SEEDS]):.3f}"
            for region in ('brows', 'eyes_expression')]
        rows.append(values)
    temporal = table(rows)
    rows = [['Local-only intervention, seed42', 'Brow corr.', 'Eye corr.', 'Brow RMS/ref.', 'Eye RMS/ref.',
             'Brow raw MSE', 'Brow DC MSE']]
    for arm, label in [('frozen_local', 'Frozen'), ('adapt_local', 'Adapted')]:
        for mode in ('full', 'local_static', 'local_reverse'):
            raw_groups = reports[arm]['raw']['modes']['42/' + mode]['populations']['all']
            dc_groups = reports[arm]['dc']['modes']['42/' + mode]['populations']['all']
            brow, eye = raw_groups['brows'], raw_groups['eyes_expression']
            rows.append([label + ' / ' + mode, f"{brow['centered_correlation']:.3f}",
                         f"{eye['centered_correlation']:.3f}", f"{brow['rms_ratio']:.3f}",
                         f"{eye['rms_ratio']:.3f}", f"{brow['raw_mse']:.6f}", f"{dc_groups['brows']['raw_mse']:.6f}"])
    return raw, temporal, table(rows)


def oracle_table(reports):
    rows = [['GT-history diagnostic only, seed42', 'Brow raw MSE', 'Brow corr.', 'Eye raw MSE',
             'Eye corr.', 'Actual brow continuation/ref.']]
    for arm, label in [('frozen_local', 'Frozen'), ('adapt_local', 'Adapted')]:
        fit = reports[arm]['fit']
        if fit['clips'] != 128 or fit['test_loaded'] is not False or fit['nonupper_scored'] is not False:
            raise ValueError('Fit diagnostic protocol differs')
        for population in ('fit', 'dev'):
            for mode in ('oracle_history', 'oracle_reverse_history'):
                if population == 'fit':
                    record = fit['metrics'][mode]
                    brow, eye = record['brows'], record['eyes_expression']
                    actual = record['actual_supplied_prefix']['brows']
                else:
                    record = reports[arm]['raw']['ORACLE_GT_HISTORY_DIAGNOSTIC_ONLY']['modes']['42/' + mode]
                    brow, eye = [record['populations']['all'][g] for g in ('brows', 'eyes_expression')]
                    actual = record['actual_prefix_continuation']['metrics']['brows']
                rows.append([f'{label} / {population} / {mode}', f"{brow['raw_mse']:.6f}",
                             f"{brow['centered_correlation']:.3f}", f"{eye['raw_mse']:.6f}",
                             f"{eye['centered_correlation']:.3f}", f"{actual['rms']/actual['reference_rms']:.3f}"])
    return table(rows)


def page(root, reports, old_audit, provenance, required, reports_only=False):
    status = read(root / 'status.json')
    if status['status'] != 'complete' or status['smoke'] is not False or status['epochs_per_arm'] != 12:
        raise ValueError('Fixed training completion differs')
    raw_table, temporal_table, intervention_table = tables(reports, old_audit)
    if reports_only:
        videos = '<p>新九片段曲线和视频尚未取回，本页暂不展示本轮媒体。</p>'
        verification = '<div class="warning"><strong>仅下载的JSON指标复核：</strong>远端独立权重/曲线审计尚未完成，新曲线与视频尚未取回。下述数字来自训练器报告及本地JSON交叉检查，不表示独立张量重算已通过。</div>'
        source_links = '<a href="metrics_review.md">JSON指标复核</a> · <a href="metrics_review.json">指标与报告哈希</a>'
        images = ''
    else:
        videos = verify_videos(root / 'visual', provenance, required)
        verification = '<p>远端独立权重/曲线审计与固定样本绑定已核对。视频验收状态单独记录；完整性不代表生成质量已通过。</p>'
        source_links = '<a href="metrics_review.md">指标复核</a> · <a href="new_nine_clips_manifest.json">远端导出绑定</a> · <a href="visual/provenance.json">本地可视化绑定</a> · <a href="visual/video_verification.json">视频核验</a>'
        images = '''<h2>逐段原始状态诊断（seed42）</h2><p>同一mask覆盖规则与共同片段。内容变化也影响分段误差，单凭斜率不能认定递归漂移。</p>
<img src="visual/rollout_profiles.png" alt="逐段原始误差、均值误差与越界比例">
<h2>九片段中心动态</h2><p>仅显示时每条曲线减自身有效帧均值，未作幅度或时间调整。</p><img src="visual/centered.png" alt="固定九片段六模式中心动态">
<h2>九片段原始系数</h2><img src="visual/raw.png" alt="固定九片段六模式原始系数">'''
    adapted = reports['adapt_local']['raw']['distribution']
    frozen = reports['frozen_local']['raw']['distribution']
    brow, eye = means(adapted, 'brows'), means(adapted, 'eyes_expression')
    old_brow, old_eye = [means(old_audit['deployment']['state_white'], g) for g in ('brows', 'eyes_expression')]
    f_brow, f_eye = means(frozen, 'brows'), means(frozen, 'eyes_expression')
    output = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>audio-prefix12: local adaptation review</title><style>body{{font:16px/1.7 system-ui;max-width:1400px;margin:24px auto;padding:0 20px;color:#20262b;background:#f6f8fa;letter-spacing:0}}h1{{font-size:28px}}h2{{font-size:21px}}h3{{font-size:18px}}table{{border-collapse:collapse;background:white;width:100%;white-space:nowrap}}td{{border-bottom:1px solid #ddd;padding:8px}}tr:first-child{{font-weight:bold;background:#eef2f5}}.scroll{{overflow-x:auto}}video,img{{width:100%;display:block}}a{{color:#17639f}}.callout{{border-left:4px solid #3c7187;padding:12px 18px;background:#eaf4f8}}.warning{{border-left:4px solid #b47b20;padding:12px 18px;background:#fff8e8}}details{{background:white;padding:12px;margin:18px 0}}@media(max-width:600px){{body{{padding:0 12px}}h1{{font-size:24px}}}}</style>
<h1>audio-prefix12 · 局部音频适配有小幅进展，仍未超过旧方案</h1>
{verification}
<div class="callout">两臂在相同 context12/chunk_teacher 上各继续训练12轮、1740次更新，训练与评估共约 {status['seconds']/60:.1f} 分钟。
解冻 local 输入层、时序块和局部输出头后，眉/眼时序相关为 {brow['centered_correlation']:.3f}/{eye['centered_correlation']:.3f}；冻结对照为 {f_brow['centered_correlation']:.3f}/{f_eye['centered_correlation']:.3f}。
时序对应有小幅改善；接缝表现并非眉眼两个区域都优于冻结对照，尚不能称为目标动态已重建。</div>
<div class="warning">旧 state_white 的眉/眼相关为 {old_brow['centered_correlation']:.3f}/{old_eye['centered_correlation']:.3f}，仍高于本轮适配方案。
本轮 raw 绝对状态误差仍明显；DC 可纠正全局均值，不能改善动作时机。旧方案与本轮训练和参数化不同，仅作历史参照。</div>
<p>405条反复使用的内部开发片段，固定第12轮和 seed42/123/2026。两臂 upper 均可训练；adapt_local 额外训练 local.input/blocks/local_head。
system、audio、全局情感/状态头、身份与底座冻结，motion→audio全局教导沿用既有权重；本轮没有重新优化该教导。训练使用严格过去8帧GT，部署使用生成历史，每次生成16帧。
这属于匹配接收器的继续适配，并非首次端到端音频训练。封存test未读取，默认模型未替换。</p>
<h2>原始状态与离线均值组合</h2><div class="scroll">{raw_table}</div>
<p>DC = 冻结音频预测的 static_upper + raw生成轨迹减自身有效帧均值。只做整片常数平移，无GT输入、增益、平滑或时间对齐；DC结果不反馈给前缀。
该组合依赖整片输出，属于离线部署，不是因果流式。越界比例和误差均在未经clamp的原值上计算。</p>
<h2>中心动态与实际分块接缝</h2><div class="scroll">{temporal_table}</div>
<p>DC保持中心波形和相邻帧位移，数值检查容差2e-6，因此此表不重复列DC。接缝统计每个seed的位移RMS/GT再取三seed均值。
旧方案一次生成整窗，不把同位置网格称为分块接缝。幅度、续接平滑和时序相关各自反映不同问题；单GT误差也不能充分评价一对多生成自然度。</p>
<h2>仅局部音频的时间干预</h2><div class="scroll">{intervention_table}</div>
<p>仅seed42。local_static/local_reverse只改变局部音频条件，保留h0、全局/静态状态和身份；与会同时改变local和h0的旧static/reverse干预区别开。
干预敏感说明模型使用了局部信息，不能单独证明正确恢复了语义或眉眼事件。其余43通道保持相同不等于唇音同步、人物身份和情感已经通过独立评价。</p>
<p>{source_links}</p>
<h2>固定片段六模式</h2><p>上排：跟踪参考 / 旧state_white / 前轮context + 音频均值；下排：本轮冻结local + 音频均值 / 本轮适配local + 音频均值 / 本轮适配local原值。
沿用既有九片段及三个视频片段，96帧、25Hz、seed42、同音轨，保留原始52维系数。未按本轮结果重选，未放大动态或拟合时间偏移。显示clamp仅用于rig渲染，曲线与评分仍使用原值。
统一rig不代表人物几何身份；主视频和曲线均不含GT-history oracle。</p>
{videos}
{images}
<details><summary>独立诊断：GT过去动作接收（不是部署结果）</summary>
<p>GT历史含真实动作，成功只说明接收通路可用，不能证明音频能预测。fit为128条已曝光训练片段；dev为405条内部开发片段。
以下仅为组合前raw输出，未对oracle施加DC。实际续接以真正提供的known末token计算；逆序末token不是时间上紧邻的GT端点。</p>
<div class="scroll">{oracle_table(reports)}</div><p>fit的其余43通道是占位，只用于上脸诊断，不用于本页视频。未展示GT均值oracle。</p></details></html>'''
    (root / 'index.html').write_text(output, encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('artifacts/audio_prefix_adaptation_20260917/audio_prefix12'))
    parser.add_argument('--previous-root', type=Path, default=Path('artifacts/history_context_20260917/history12'))
    parser.add_argument('--context-root', type=Path, default=Path('artifacts/context_mechanism_20260917/context12'))
    parser.add_argument('--origin-root', type=Path, default=Path('artifacts/audio_prefix_adaptation_20260917/origin_diagnostic'))
    parser.add_argument('--audio-root', type=Path, default=Path('artifacts/context_mechanism_20260917/context12/visual/audio'))
    parser.add_argument('--require-videos', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--reports-only', action='store_true')
    args = parser.parse_args()
    if args.reports_only:
        if args.require_videos or args.export_only:
            parser.error('--reports-only cannot be combined with --require-videos or --export-only')
        reports, bindings, _, _, _ = load_reports(args.root, args.previous_root, args.context_root, args.origin_root)
        page(args.root, reports, read(args.previous_root / 'audit.json'), None, False, reports_only=True)
        print(json.dumps({'page': str((args.root / 'index.html').resolve()), 'reports_only': True,
                          'remote_integrity_pending': True, 'new_media_available': False}, ensure_ascii=False))
        return
    if args.require_videos and args.export_only:
        parser.error('--require-videos cannot be combined with --export-only')
    reports, old_audit, provenance = export(args.root, args.previous_root, args.context_root, args.origin_root, args.audio_root)
    if not args.export_only:
        plots(args.root / 'visual', reports)
        page(args.root, reports, old_audit, provenance, args.require_videos)
    print(json.dumps({'page': str((args.root / 'index.html').resolve()), 'jobs': provenance['jobs'],
                      'six_modes_ready': True, 'videos_required': args.require_videos}, ensure_ascii=False))


if __name__ == '__main__':
    main()
