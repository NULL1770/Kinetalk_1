"""Package fixed temporal-adapter review; no inference, sample selection or fitting.

Only the specified run's index.html and visual directory are written. Source
reports and trajectories are immutable inputs. Transfer608 and development405
remain separate populations throughout the page.
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
from scripts.package_audio_prefix_review import means, ratio
from scripts.package_prefix_formal_review import json_write, save_arrays, table
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input

ARMS = ('frozen_local', 'rank8_adapter', 'full_local')
LABELS = {'frozen_local': 'Frozen local', 'rank8_adapter': 'Rank8 adapter', 'full_local': 'Full local'}
SEEDS = (42, 123, 2026)
SCHEMA = 'temporal_adapter_incremental_transfer_v1'
EXPORTED_MODES = ('Old audio mean + flow', 'Previous adapted local + audio mean',
                  'New frozen local + audio mean', 'New rank8 adapter + audio mean',
                  'New full local + audio mean')


def transfer_contract(report, holdout):
    if (report['schema'] != SCHEMA or report['clips'] != 608 or report['sentence_count'] != 13
            or report['noise_seeds'] != list(SEEDS) or report['decode_steps'] != 12
            or report['smoke'] is not False or report['GT_was_input'] is not False
            or report['oracle_modes'] != [] or report['nonupper_scored'] is not False
            or report['fullface_deployment_evaluation'] is not False
            or report['test_loaded'] is not False or report['default_replaced'] is not False
            or report['per_clip_order'] != holdout):
        raise ValueError('608 incremental transfer contract or membership differs')


def load_reports(root, old, previous):
    status = read(root / 'status.json')
    if status['status'] != 'complete' or status['smoke'] is not False or status['epochs_per_arm'] != 12:
        raise ValueError('Twelve-epoch training completion differs')
    historical = read(old / 'complete.json')
    if historical['status'] != 'complete':
        raise ValueError('Historical source incomplete')
    for filename, digest in historical['files'].items():
        if sha(old / filename) != digest:
            raise ValueError('Historical source binding differs: ' + filename)
    old_provenance = read(old / 'visual/provenance.json')
    split = read(root / 'sentence_split.json')
    fit, hold = split['split']['fit'], split['split']['internal_sentence_holdout']
    if (len(fit['clips']) != 1707 or len(hold['clips']) != 608
            or set(fit['sentences']) & set(hold['sentences'])
            or set(fit['clips']) & set(hold['clips']) or split['test_loaded'] is not False):
        raise ValueError('New-stage sentence split differs')
    source = read(root / 'step0_transfer.json')
    transfer_contract(source, hold['clips'])
    previous_complete = read(previous / 'adapt_local/complete.json')
    previous_provenance = read(previous / 'visual/provenance.json')
    previous_bound = previous_provenance['bindings']['adapt_local']
    if (sha(previous / 'adapt_local/evaluation.json') != previous_bound['evaluation_sha256']
            or sha(previous / 'adapt_local/dc_evaluation.json') != previous_bound['dc_evaluation_sha256']
            or sha(previous / 'visual/nine_clip_curves.npz') != previous_provenance['nine_clip_curves_sha256']
            or previous_complete['dc_curves_sha256'] != previous_bound['dc_curves_sha256']):
        raise ValueError('Previous audio-prefix report or subset binding differs')
    reports = {'source': source, 'previous': {'raw': read(previous / 'adapt_local/evaluation.json'),
                                            'dc': read(previous / 'adapt_local/dc_evaluation.json')}}
    bindings = {'step0_transfer_sha256': sha(root / 'step0_transfer.json'),
                'sentence_split_sha256': sha(root / 'sentence_split.json'),
                'fit_selection_sha256': sha(root / 'fit_selection.json'),
                'previous_provenance_sha256': sha(previous / 'visual/provenance.json'),
                'previous_complete_sha256': sha(previous / 'adapt_local/complete.json'),
                'previous_raw_evaluation_sha256': sha(previous / 'adapt_local/evaluation.json'),
                'previous_dc_evaluation_sha256': sha(previous / 'adapt_local/dc_evaluation.json')}
    expected_hashes = {'old_white': old_provenance['control_bindings']['state_white'],
                       'previous_adapt_local_dc': previous_complete['dc_curves_sha256']}
    for arm in ARMS:
        folder = root / arm
        complete = read(folder / 'complete.json')
        provenance = read(folder / 'provenance.json')
        recipe = provenance['recipe']
        if (complete['status'] != 'complete' or complete['completed_epochs'] != 12
                or complete['total_steps'] != 1284 or complete['test_loaded'] is not False
                or complete['nonupper_invalid_exact'] is not True or complete['compact_roundtrip_exact'] is not True
                or recipe['arm'] != arm or recipe['fit_update_clips'] != 1707 or recipe['transfer_clips'] != 608
                or recipe['sentence_split_sha256'] != bindings['sentence_split_sha256']
                or recipe['fit_selection_sha256'] != bindings['fit_selection_sha256']
                or complete['recipe_sha256'] != provenance['recipe_sha256']):
            raise ValueError('Arm completion/recipe contract differs: ' + arm)
        for filename, digest in complete['files'].items():
            if filename.endswith('.json') and sha(folder / filename) != digest:
                raise ValueError('Downloaded report hash differs: ' + arm + '/' + filename)
        raw, dc = read(folder / 'evaluation.json'), read(folder / 'dc_evaluation.json')
        transfer, diagnostic = read(folder / 'transfer_evaluation.json'), read(folder / 'fit_evaluation.json')
        transfer_contract(transfer, hold['clips'])
        for report in (raw, dc):
            if (report['clips'] != 405 or report['noise_seeds'] != list(SEEDS)
                    or report['adaptation_arm'] != arm or report['recipe_sha256'] != complete['recipe_sha256']
                    or report['test_loaded'] is not False or report['default_replaced'] is not False
                    or report['smoke'] is not False or report['nonupper_exact'] is not True
                    or report['invalid_baseline_exact'] is not True):
                raise ValueError('405 development report contract differs: ' + arm)
        if (dc['oracle_composition_performed'] is not False or dc['actual_prefix_scoring_performed'] is not False
                or dc['dc_invariants_passed'] is not True or diagnostic['clips'] != 128
                or diagnostic['nonupper_scored'] is not False or diagnostic['test_loaded'] is not False):
            raise ValueError('DC/oracle diagnostic scope differs: ' + arm)
        reports[arm] = {'raw': raw, 'dc': dc, 'transfer': transfer, 'fit': diagnostic,
                        'drift': read(folder / 'feature_drift.json')}
        bindings[arm] = {'complete_sha256': sha(folder / 'complete.json'),
                         'provenance_sha256': sha(folder / 'provenance.json'),
                         'recipe_sha256': complete['recipe_sha256'], 'complete_files': complete['files'],
                         'baseline_curves_sha256': recipe['baseline_curves_sha256']}
        expected_hashes[arm + '/raw'] = complete['files']['curves.pt']
        expected_hashes[arm + '/dc'] = complete['files']['dc_curves.pt']
    matched = read(root / 'matched_audit.json')
    if matched['equal'] is not True or any(matched['arms'][arm] != matched['arms'][ARMS[0]] for arm in ARMS):
        raise ValueError('Matched new-stage update records differ')
    bindings['matched_audit_sha256'] = sha(root / 'matched_audit.json')
    return reports, bindings, historical, old_provenance, expected_hashes


def export(root, old, previous, audio_root):
    reports, bindings, historical, old_provenance, expected_hashes = load_reports(root, old, previous)
    manifest = read(root / 'new_nine_clips_manifest.json')
    if (sha(root / 'new_nine_clips.npz') != manifest['output_sha256']
            or sha(root / 'fixed_visual_selection.json') != manifest['selection_sha256']
            or manifest['noise_seed'] != 42 or manifest['oracle_included'] is not False
            or manifest['raw_clamped'] is not False or manifest['test_loaded'] is not False):
        raise ValueError('Fixed subset binding or scope differs')
    # The exporter owns display names; provenance and renderer keep them exactly.
    exported_modes = manifest['mode_names']
    if exported_modes != list(EXPORTED_MODES):
        raise ValueError('The five saved deployment modes differ')
    modes = ('Tracked reference', *exported_modes)
    if manifest['source_curves'] != expected_hashes:
        raise ValueError('Source raw/DC curve SHA mapping differs')
    if any(manifest['baseline_sha256'] != bindings[arm]['baseline_curves_sha256'] for arm in ARMS):
        raise ValueError('Reconstructed full-face baseline binding differs')
    audit = read(root / 'integrity_audit.json')
    if audit['status'] != 'passed' or audit['test_loaded'] is not False:
        raise ValueError('Remote independent integrity audit incomplete')
    bindings['remote_integrity_audit_sha256'] = sha(root / 'integrity_audit.json')
    picks = old_provenance['nine_plot_clips']
    if read(root / 'fixed_visual_selection.json')['nine_plot_clips'] != picks:
        raise ValueError('Historical fixed selection changed')
    with np.load(root / 'new_nine_clips.npz', allow_pickle=False) as z:
        new = {key: z[key].copy() for key in z.files}
    with np.load(old / 'visual/nine_clip_curves.npz', allow_pickle=False) as z:
        historical_arrays = {key: z[key].copy() for key in z.files}
    with np.load(previous / 'visual/nine_clip_curves.npz', allow_pickle=False) as z:
        previous_arrays = {key: z[key].copy() for key in z.files}
    ids = [pick['clip_id'] for pick in picks]
    if (new['mode_names'].tolist() != exported_modes or new['motions'].shape != (5, 9, 96, 52)
            or new['target'].shape != (9, 96, 52) or new['clip_id'].tolist() != ids
            or historical_arrays['clip_id'].tolist() != ids or previous_arrays['clip_id'].tolist() != ids
            or historical_arrays['channels'].tolist() != ARKIT_NAMES
            or previous_arrays['mode_names'][4] != 'New adapted local + audio mean'):
        raise ValueError('Source fixed sample/mode/channel contract differs')
    for key in ('times', 'valid', 'channel_mask'):
        for source in (historical_arrays, previous_arrays):
            if new[key].dtype != source[key].dtype or not np.array_equal(new[key], source[key]):
                raise ValueError('Native times/masks differ: ' + key)
    for actual, expected in ((new['target'], historical_arrays['motions'][0]),
                             (new['target'], previous_arrays['motions'][0]),
                             (new['motions'][0], historical_arrays['motions'][2]),
                             (new['motions'][1], previous_arrays['motions'][4])):
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError('Historical target/white/previous-adapted DC differs')
    motions = np.concatenate([new['target'][None], new['motions']])
    if motions.dtype != np.float32 or not np.isfinite(motions).all():
        raise ValueError('Finite saved float32 coefficients required')
    visual = root / 'visual'
    for folder in (visual, visual / 'video_npz', visual / 'audio'):
        folder.mkdir(exist_ok=True, parents=True)
    save_arrays(visual / 'nine_clip_curves.npz', motions=motions, mode_names=np.asarray(modes),
                channels=np.asarray(ARKIT_NAMES), **{key: new[key] for key in ('times', 'valid', 'channel_mask', 'clip_id')})
    jobs = []
    for old_job in old_provenance['jobs']:
        speaker = old_job['speaker']
        index = next(i for i, pick in enumerate(picks) if pick['speaker'] == speaker)
        display_path = old / 'visual' / speaker / 'display_report.json'
        display = read(display_path)
        audio = audio_root / (speaker + '.wav')
        if sha(audio) != display['audio_sha256'] or display['metadata']['clip_id'] != ids[index]:
            raise ValueError('Historical video/audio selection differs: ' + speaker)
        copied = visual / 'audio' / audio.name
        if not copied.exists():
            shutil.copy2(audio, copied)
        if sha(copied) != display['audio_sha256']:
            raise ValueError('Copied audio differs: ' + speaker)
        filename = visual / 'video_npz' / (speaker + '.npz')
        save_arrays(filename, motions=motions[:, index], mode_names=np.asarray(modes), channels=np.asarray(ARKIT_NAMES),
                    times=new['times'][index], valid=new['valid'][index], channel_mask=new['channel_mask'][index],
                    clip_id=np.asarray(ids[index]), noise_seed=np.asarray(42))
        inspect_input(filename, 25)
        jobs.append({'speaker': speaker, 'clip_id': ids[index], 'input': f'video_npz/{speaker}.npz',
                     'input_sha256': sha(filename), 'audio': f'audio/{speaker}.wav', 'audio_sha256': sha(copied),
                     'audio_offset_seconds': old_job['audio_offset_seconds'],
                     'historical_display_report_sha256': sha(display_path)})
    provenance = {'schema': 'temporal_adapter_review_v1', 'mode_names': list(modes), 'nine_plot_clips': picks,
                  'jobs': jobs, 'bindings': bindings, 'noise_seed': 42,
                  'remote_subset_sha256': sha(root / 'new_nine_clips.npz'),
                  'remote_subset_manifest_sha256': sha(root / 'new_nine_clips_manifest.json'),
                  'remote_source_curves': manifest['source_curves'],
                  'fixed_selection_sha256': sha(root / 'fixed_visual_selection.json'),
                  'historical_complete_sha256': sha(old / 'complete.json'), 'historical_complete_files': historical['files'],
                  'selection_unchanged': True, 'target_time_masks_equal_exactly': True,
                  'historical_white_and_previous_adapted_exact': True, 'raw_clamped': False,
                  'gain_or_lag_fitted': False, 'oracle_in_main_visual': False, 'test_loaded': False,
                  'dc_is_offline': True, 'dc_fed_back_as_prefix': False, 'visual_population': '405 development',
                  'nine_clip_curves_sha256': sha(visual / 'nine_clip_curves.npz')}
    json_write(visual / 'provenance.json', provenance)
    return reports, read(old / 'audit.json'), provenance


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
        if (report['status'] != 'complete' or report['original_blend_unchanged'] is not True
                or report['input_sha256'] != job['input_sha256'] or report['audio_sha256'] != job['audio_sha256']
                or report['rendered_modes'] != provenance['mode_names'] or report['metadata']['clip_id'] != job['clip_id']
                or report['metadata']['noise_seed'] != 42 or report['frames'] != 96 or report['fps'] != 25
                or report['native_coefficients_modified'] is not False or report['output_video_sha256'] != sha(video)
                or sha(visual / job['input']) != job['input_sha256'] or sha(visual / job['audio']) != job['audio_sha256']):
            raise ValueError('Rendered video/source binding differs: ' + speaker)
        with np.load(visual / job['input'], allow_pickle=False) as source:
            trim_start = float(source['times'][0]) + job['audio_offset_seconds']
        if abs(report['audio_trim_start_seconds'] - trim_start) > 1e-7:
            raise ValueError('Audio trim differs: ' + speaker)
        ffprobe = shutil.which('ffprobe')
        if not ffprobe:
            raise FileNotFoundError('ffprobe required for video verification')
        streams = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-show_streams',
                                                     '-of', 'json', str(video)], encoding='utf8'))['streams']
        stream = next(s for s in streams if s['codec_type'] == 'video')
        if stream['avg_frame_rate'] != '25/1' or int(stream['nb_frames']) != 96 or not any(s['codec_type'] == 'audio' for s in streams):
            raise ValueError('Video clock, frames, or audio differs: ' + speaker)
        checks.append({'speaker': speaker, 'input_sha256': job['input_sha256'], 'audio_sha256': job['audio_sha256'],
                       'video_sha256': sha(video), 'report_sha256': sha(folder / 'display_report.json'),
                       'frames': 96, 'fps': 25, 'audio': True})
        videos.append(f'<h3>{speaker}</h3><video controls preload="metadata" '
                      f'poster="visual/{speaker}/preview.png" src="visual/{speaker}/comparison.mp4"></video>')
    json_write(visual / 'video_verification.json', checks)
    return ''.join(videos)


def score_table(entries):
    rows = [['Model / composition', 'Brow raw MSE', 'Brow centered MSE', 'Brow corr.', 'Brow RMS/ref.',
             'Eye raw MSE', 'Eye centered MSE', 'Eye corr.', 'Eye RMS/ref.', 'Brow/eye outside']]
    for label, distribution in entries:
        brow, eye = [means(distribution, group) for group in ('brows', 'eyes_expression')]
        values = [label]
        for group in (brow, eye):
            values += [f"{group['raw_mse']:.6f}", f"{group['centered_mse']:.6f}",
                       f"{group['centered_correlation']:.3f}", f"{group['rms_ratio']:.3f}"]
        rows.append(values + [f"{100*brow['outside_fraction']:.2f}% / {100*eye['outside_fraction']:.2f}%"])
    return table(rows)


def distribution_table(entries):
    rows = [['608, three full seeds', 'Brow raw energy', 'Brow centered energy', 'Brow adjacent variogram',
             'Eye raw energy', 'Eye centered energy', 'Eye adjacent variogram']]
    for label, distribution in entries:
        values = [label]
        for region in ('brows', 'eyes_expression'):
            group = distribution['populations']['all']['groups'][region]['distribution']
            values += [f"{group[k][m]['fair']['mean']:.6f}" for k, m in (
                ('raw', 'trajectory_energy_score'), ('centered_coefficients', 'trajectory_energy_score'),
                ('raw', 'adjacent_variogram_score'))]
        rows.append(values)
    return table(rows)


def intervention_table(reports):
    rows = [['608, seed42 local-only intervention', 'Brow corr.', 'Eye corr.', 'Brow RMS/ref.', 'Eye RMS/ref.']]
    for arm in ('source', *ARMS):
        transfer = reports['source'] if arm == 'source' else reports[arm]['transfer']
        for mode in ('full', 'local_static', 'local_reverse'):
            group = transfer['compositions']['raw']['modes']['42/' + mode]['paired']
            brow, eye = group['brows'], group['eyes_expression']
            rows.append([('Source' if arm == 'source' else LABELS[arm]) + ' / ' + mode,
                         f"{brow['centered_correlation']:.3f}", f"{eye['centered_correlation']:.3f}",
                         f"{brow['rms_ratio']:.3f}", f"{eye['rms_ratio']:.3f}"])
    return table(rows)


def drift_table(reports):
    rows = [['Feature changes, all valid clips', '608 correction RMS mean', '608 same-forward mean max',
             '405 correction RMS mean', '405 same-forward mean max']]
    for arm in ARMS:
        values = [LABELS[arm]]
        for population in ('transfer', 'development'):
            records = reports[arm]['drift'][population]
            values += [f"{np.mean([r['correction_rms'] for r in records]):.6f}",
                       f"{max(r['same_forward_mean_correction_max_abs'] for r in records):.3g}"]
        rows.append(values)
    return table(rows)


def oracle_table(reports):
    rows = [['New fit128 diagnostic, seed42', 'Brow raw MSE', 'Brow corr.', 'Eye raw MSE', 'Eye corr.']]
    for arm in ARMS:
        for mode in ('full', 'oracle_history', 'oracle_reverse_history'):
            group = reports[arm]['fit']['metrics'][mode]
            brow, eye = group['brows'], group['eyes_expression']
            rows.append([LABELS[arm] + ' / ' + mode, f"{brow['raw_mse']:.6f}",
                         f"{brow['centered_correlation']:.3f}", f"{eye['raw_mse']:.6f}", f"{eye['centered_correlation']:.3f}"])
    return table(rows)


def page(root, reports, old_audit, provenance, required):
    transfer_entries = [('Source / ' + mode, reports['source']['compositions'][mode]['distribution']) for mode in ('raw', 'dc')]
    transfer_entries += [(LABELS[arm] + ' / ' + mode, reports[arm]['transfer']['compositions'][mode]['distribution'])
                         for arm in ARMS for mode in ('raw', 'dc')]
    dev_entries = [('Old state_white', old_audit['deployment']['state_white'])]
    dev_entries += [('Previous adapted / ' + mode, reports['previous'][mode]['distribution']) for mode in ('raw', 'dc')]
    dev_entries += [(LABELS[arm] + ' / ' + mode, reports[arm][mode]['distribution']) for arm in ARMS for mode in ('raw', 'dc')]
    videos = verify_videos(root / 'visual', provenance, required)
    status = read(root / 'status.json')
    result_links = '<a href="metrics_review.md">独立指标复核</a> · <a href="metrics_review.json">复核数值</a>'
    if (root / 'RESULTS.md').exists():
        result_links += ' · <a href="RESULTS.md">结论记录</a>'
    output = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Temporal adapter: incremental transfer review</title><style>body{{font:16px/1.7 system-ui;max-width:1500px;margin:24px auto;padding:0 20px;color:#20262b;background:#f6f8fa}}h1{{font-size:28px}}h2{{font-size:21px}}h3{{font-size:18px}}table{{border-collapse:collapse;background:white;width:100%;white-space:nowrap}}td{{border-bottom:1px solid #ddd;padding:8px}}tr:first-child{{font-weight:bold;background:#eef2f5}}.scroll{{overflow-x:auto}}video,img{{width:100%;display:block}}a{{color:#17639f}}.callout{{border-left:4px solid #3c7187;padding:12px 18px;background:#eaf4f8}}.warning{{border-left:4px solid #b47b20;padding:12px 18px;background:#fff8e8}}details{{background:white;padding:12px;margin:18px 0}}</style>
<h1>temporal adapter · rank8未带来明确动态收益</h1>
<div class="callout">三臂各12轮、1284次更新已完成，共约 {status['seconds']/60:.1f} 分钟。零均值rank8适配器维持局部特征均值，但表现接近冻结对照；
完整局部适配在608片段的部分中心动态指标上更好，眉相关仍未超过共享source。本轮不足以证明动态目标已解决。</div>
<div class="warning">608片段只从本轮新更新中排除，包含13种句子；共享预训练source曾用完整原fit池学习。
因此这里评估的是增量适配对留出句子的迁移，不能称为全模型未见句子泛化或封存test成绩。405 dev是另一组反复使用的开发数据，下文分开列出。</div>
<p>三臂从相同context12/chunk_teacher上脸权重和既有local开始，在1707片段上继续更新upper。Frozen冻结local；Rank8只额外训练64→8→64、1024参数的零均值校正；Full额外训练local输入、时序块、输出头。
身份、全局情感与底座保持既有权重，全局motion→audio教导未重新训练。本轮是匹配三臂实验，旧audio-prefix方案训练集合及更新次数不同，只作历史参照。默认模型未替换，封存test未读。</p>
<h2>608片段：source与三臂，raw/DC分别评分</h2><div class="scroll">{score_table(transfer_entries)}</div>
<p>DC用冻结音频static_upper替换生成轨迹的整片均值，不使用GT、不改变中心波形或相邻位移，也不反馈到前缀。
这里DC使raw MSE变差，不能把它笼统称为有益修正；它仅在部分405开发指标上改善绝对偏移。DC依赖整片输出，属于离线组合。</p>
<h2>608片段：分布评分与局部时序干预</h2><div class="scroll">{distribution_table(transfer_entries)}</div>
<p>Energy/variogram均为三样本fair评分，数值越低越好；只有三固定推理seed，不能据此宣称概率校准、自然度或显著性已验证。</p>
<div class="scroll">{intervention_table(reports)}</div><p>干预仅seed42，固定噪声，只静态化或逆序局部音频条件，保留h0、全局/静态状态和身份。
响应说明使用了局部信息，不能单独证明动作语义及时机正确。608评估仅含上脸9通道，其他通道占位，不用于视频，也不验证身份或唇音同步。</p>
<h2>405内部开发片段：历史参照与三臂</h2><div class="scroll">{score_table(dev_entries)}</div>
<p>本表与608表不能按绝对数字作泛化差距推断，样本总体及历史曝光不同。完整局部适配眉相关下降、眼相关改善，结果仍混合。
其余43通道保持底座输出是工程完整性检查，不等于身份、口型与情感已通过独立评价。单GT误差也不足以评价一对多自然度。</p>
<h2>局部特征变化</h2><div class="scroll">{drift_table(reports)}</div>
<p>“same-forward mean”在同一次前向中比较校正前后均值，避免缓存/批处理数值差异。Rank8零均值约束成立，但约束本身未带来明确时序收益；完整适配可改变特征均值。</p>
<p>{result_links} · <a href="integrity_audit.json">远端完整性审计</a> · <a href="new_nine_clips_manifest.json">子集导出绑定</a> · <a href="visual/provenance.json">可视化绑定</a> · <a href="visual/video_verification.json">视频验收</a></p>
<h2>固定405开发样本的六模式对照</h2><p>上排：跟踪参考 / 旧state_white / 前轮audio-prefix完整适配DC；下排：本轮冻结DC / 本轮rank8 DC / 本轮完整适配DC。
沿用九个固定样本和三个angry视频，96帧、25Hz、seed42、同音轨。视频展示DC仅为跨轮固定比较，不表示DC在608有效。
所有原始52维系数、native时间及mask保留，未按本轮表现重选、放大、平滑或调整时间。显示clamp只影响rig，评分与曲线用未裁剪系数；统一rig不代表人物身份。主视图无GT-history oracle。</p>
{videos}
<h2>九片段中心动态</h2><p>仅画图时各曲线减自身有效帧均值，无增益或时间调整。</p><img src="visual/centered.png" alt="固定九样本六模式中心动态">
<h2>九片段原始系数</h2><img src="visual/raw.png" alt="固定九样本六模式原始系数">
<details><summary>本轮128 fit诊断，GT-history单独标注</summary><p>这128片段从本轮1707更新集合中固定选取，不是旧轮的同一128样本，不能直接作跨轮指标对比。
full使用生成历史；oracle_history及逆序oracle提供真实过去，只检验接收通路，不证明音频可预测。表内均为组合前raw，未对oracle作DC。占位的其他43通道不评分、不渲染。</p>
<div class="scroll">{oracle_table(reports)}</div></details></html>'''
    (root / 'index.html').write_text(output, encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('artifacts/temporal_adapter_20260917/adapter_transfer12_v2'))
    parser.add_argument('--old-root', type=Path, default=Path('artifacts/history_context_20260917/history12'))
    parser.add_argument('--previous-root', type=Path, default=Path('artifacts/audio_prefix_adaptation_20260917/audio_prefix12'))
    parser.add_argument('--audio-root', type=Path, default=Path('artifacts/audio_prefix_adaptation_20260917/audio_prefix12/visual/audio'))
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--require-videos', action='store_true')
    args = parser.parse_args()
    if args.export_only and args.require_videos:
        parser.error('--export-only and --require-videos cannot be combined')
    reports, old_audit, provenance = export(args.root, args.old_root, args.previous_root, args.audio_root)
    if not args.export_only:
        plot(args.root / 'visual')
        page(args.root, reports, old_audit, provenance, args.require_videos)
    print(json.dumps({'page': str((args.root / 'index.html').resolve()), 'jobs': provenance['jobs'],
                      'six_modes_ready': True, 'videos_required': args.require_videos}, ensure_ascii=False))


if __name__ == '__main__':
    main()
