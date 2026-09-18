"""Package the fixed native-context review from saved, hash-bound outputs.

This script performs no inference, fitting, selection or coefficient correction.
Raw predictions are the main display; the separately saved DC outputs are marked
as an offline composition. Historical nine-clip selection and audio are retained.
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
from scripts.package_audio_prefix_review import means
from scripts.package_prefix_formal_review import json_write, save_arrays, table
from scripts.package_temporal_adapter_review import score_table, distribution_table, verify_videos
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input

ARMS = ('center96', 'full_native')
LABELS = {'center96': 'Center96', 'full_native': 'Full native'}
SAVED_MODES = ('center96/raw', 'full_native/raw', 'center96/dc', 'full_native/dc')
DISPLAY_MODES = ('Tracked reference', 'Center96 raw', 'Full native raw',
                 'Old audio mean + flow', 'Center96 DC', 'Full native DC')


def load_reports(root):
    status = read(root / 'status.json')
    if (status['status'] != 'complete' or status['smoke'] is not False
            or status['epochs_per_arm'] != 30 or status['test_loaded'] is not False
            or status['default_replaced'] is not False):
        raise ValueError('Formal thirty-epoch completion differs')
    split = read(root / 'sentence_split.json')['split']
    fit, hold = split['fit'], split['internal_sentence_holdout']
    if (len(fit['clips']) != 1707 or len(hold['clips']) != 608
            or set(fit['clips']) & set(hold['clips'])
            or set(fit['sentences']) & set(hold['sentences'])):
        raise ValueError('Incremental sentence split differs')
    matched = read(root / 'matched_audit.json')
    if matched['equal'] is not True or matched['arms'][ARMS[0]] != matched['arms'][ARMS[1]]:
        raise ValueError('Matched updates and draws differ')
    common = read(root / 'provenance.json')
    reports, bindings = {}, {'status': sha(root / 'status.json'),
                             'provenance': sha(root / 'provenance.json'),
                             'sentence_split': sha(root / 'sentence_split.json'),
                             'coverage': sha(root / 'coverage.json'),
                             'matched_audit': sha(root / 'matched_audit.json')}
    for arm in ARMS:
        folder = root / arm
        complete = read(folder / 'complete.json')
        provenance = read(folder / 'provenance.json')
        if (complete['status'] != 'complete' or complete['completed_epochs'] != 30
                or complete['total_steps'] != 3210 or complete['test_loaded'] is not False
                or complete['recipe_sha256'] != provenance['recipe_sha256']
                or complete['frozen'] != common['recipe']['frozen']
                or provenance['recipe']['arm'] != arm):
            raise ValueError('Arm recipe/completion differs: ' + arm)
        reports[arm] = {}
        for label, filename, count in (('source', 'step0_holdout.json', 608),
                                       ('hold', 'hold_evaluation.json', 608),
                                       ('dev', 'dev_evaluation.json', 405)):
            if sha(folder / filename) != complete['files'][filename]:
                raise ValueError('Evaluation hash differs: ' + arm + '/' + filename)
            report = read(folder / filename)
            if (report['schema'] != 'full_native_context_common_center_evaluation_v1'
                    or report['clips'] != count or report['decode_steps'] != 12
                    or report['noise_seeds'] != [42, 123, 2026]
                    or report['GT_was_input'] is not False or report['oracle_modes'] != []
                    or report['test_loaded'] is not False or report['default_replaced'] is not False
                    or report['nonupper_scored'] is not False
                    or report['primary_composition'] != 'raw' or report['secondary_composition'] != 'dc'
                    or report['mode'] != ('center' if arm == 'center96' else 'full')):
                raise ValueError('Evaluation scope differs: ' + arm + '/' + filename)
            if count == 608 and report['per_clip_order'] != hold['clips']:
                raise ValueError('608 membership differs')
            if label == 'dev':
                if report['nonupper_protection_checked'] is not True:
                    raise ValueError('Development base protection was not checked')
                if not all(all(check.values()) for check in report['protection_checks'].values()):
                    raise ValueError('Nonupper/invalid protection failed')
            reports[arm][label] = report
        bindings[arm] = {'complete': sha(folder / 'complete.json'),
                         'provenance': sha(folder / 'provenance.json'),
                         'files': complete['files']}
    return reports, bindings


def export(root, old, audio_root):
    reports, bindings = load_reports(root)
    audit = root / 'result_audit'
    manifest = read(audit / 'manifest.json')
    audited = read(audit / 'complete.json')
    if audited['status'] != 'passed':
        raise ValueError('Independent result audit incomplete')
    for name, digest in audited['files'].items():
        if sha(audit / name) != digest:
            raise ValueError('Independent result audit binding differs: ' + name)
    if (sha(audit / 'nine_clips.npz') != manifest['export_sha256']
            or manifest['noise_seed'] != 42 or manifest['oracle_included'] is not False
            or manifest['raw_clamped'] is not False or manifest['test_loaded'] is not False
            or manifest['modes'] != list(SAVED_MODES)
            or manifest['source_curves'] != {a: bindings[a]['files']['dev_upper9.pt'] for a in ARMS}):
        raise ValueError('Nine-clip export hash differs')
    historical = read(old / 'complete.json')
    if historical['status'] != 'complete':
        raise ValueError('Historical source incomplete')
    for filename in ('visual/nine_clip_curves.npz', 'visual/provenance.json'):
        if sha(old / filename) != historical['files'][filename]:
            raise ValueError('Historical source hash differs: ' + filename)
    previous = read(old / 'visual/provenance.json')
    picks = previous['nine_plot_clips']
    if manifest['selection']['nine_plot_clips'] != picks:
        raise ValueError('Historical fixed selection changed')
    with np.load(audit / 'nine_clips.npz', allow_pickle=False) as z:
        current = {key: z[key].copy() for key in z.files}
    with np.load(old / 'visual/nine_clip_curves.npz', allow_pickle=False) as z:
        history = {key: z[key].copy() for key in z.files}
    ids = [pick['clip_id'] for pick in picks]
    if (current['modes'].tolist() != list(SAVED_MODES)
            or current['noise_seed'].item() != 42
            or current['predictions'].shape != (4, 9, 96, 52)
            or current['target'].shape != (9, 96, 52)
            or current['clip_id'].tolist() != ids or history['clip_id'].tolist() != ids
            or history['channels'].tolist() != ARKIT_NAMES
            or history['mode_names'][2] != 'Previous audio mean + flow'):
        raise ValueError('Fixed sample/mode/channel contract differs')
    for key in ('times', 'valid', 'channel_mask'):
        if current[key].dtype != history[key].dtype or not np.array_equal(current[key], history[key]):
            raise ValueError('Historical native time/mask differs: ' + key)
    if current['target'].dtype != history['motions'].dtype or not np.array_equal(current['target'], history['motions'][0]):
        raise ValueError('Historical target differs')
    motions = np.stack((current['target'], current['predictions'][0], current['predictions'][1],
                        history['motions'][2], current['predictions'][2], current['predictions'][3]))
    if motions.dtype != np.float32 or not np.isfinite(motions).all():
        raise ValueError('Finite unchanged float32 coefficients required')
    nonupper = sorted(set(range(52)) - {41, 42, 43, 44, 45, 5, 6, 12, 13})
    for prediction in current['predictions']:
        if not np.array_equal(np.take(prediction, nonupper, axis=-1), np.take(history['motions'][2], nonupper, axis=-1)):
            raise ValueError('Fixed new/old nonupper channels differ')
    visual = root / 'visual'
    for folder in (visual, visual / 'video_npz', visual / 'audio'):
        folder.mkdir(parents=True, exist_ok=True)
    save_arrays(visual / 'nine_clip_curves.npz', motions=motions, mode_names=np.asarray(DISPLAY_MODES),
                channels=np.asarray(ARKIT_NAMES), **{k: current[k] for k in ('times', 'valid', 'channel_mask', 'clip_id')})
    jobs = []
    for old_job in previous['jobs']:
        speaker = old_job['speaker']
        index = next(i for i, pick in enumerate(picks) if pick['speaker'] == speaker)
        display_path = old / 'visual' / speaker / 'display_report.json'
        display = read(display_path)
        audio = audio_root / (speaker + '.wav')
        if sha(audio) != display['audio_sha256'] or display['metadata']['clip_id'] != ids[index]:
            raise ValueError('Historical video/audio differs: ' + speaker)
        copied = visual / 'audio' / audio.name
        if not copied.exists():
            shutil.copy2(audio, copied)
        if sha(copied) != display['audio_sha256']:
            raise ValueError('Copied audio differs: ' + speaker)
        filename = visual / 'video_npz' / (speaker + '.npz')
        save_arrays(filename, motions=motions[:, index], mode_names=np.asarray(DISPLAY_MODES),
                    channels=np.asarray(ARKIT_NAMES), times=current['times'][index], valid=current['valid'][index],
                    channel_mask=current['channel_mask'][index], clip_id=np.asarray(ids[index]), noise_seed=np.asarray(42))
        inspect_input(filename, 25)
        jobs.append({'speaker': speaker, 'clip_id': ids[index], 'input': f'video_npz/{speaker}.npz',
                     'input_sha256': sha(filename), 'audio': f'audio/{speaker}.wav', 'audio_sha256': sha(copied),
                     'audio_offset_seconds': old_job['audio_offset_seconds'],
                     'historical_display_report_sha256': sha(display_path)})
    provenance = {'schema': 'native_context_review_v1', 'mode_names': list(DISPLAY_MODES),
                  'nine_plot_clips': picks, 'jobs': jobs, 'bindings': bindings, 'noise_seed': 42,
                  'remote_subset_sha256': sha(audit / 'nine_clips.npz'),
                  'remote_manifest_sha256': sha(audit / 'manifest.json'),
                  'historical_complete_sha256': sha(old / 'complete.json'),
                  'historical_visual_sha256': sha(old / 'visual/nine_clip_curves.npz'),
                  'selection_unchanged': True, 'target_time_masks_equal_exactly': True,
                  'nonupper43_equal_between_displayed_predictions': True,
                  'raw_clamped': False, 'gain_or_lag_fitted': False,
                  'oracle_in_main_visual': False, 'test_loaded': False,
                  'dc_is_offline': True, 'dc_fed_back_as_prefix': False,
                  'visual_population': '405 development; fixed historical common center96',
                  'nine_clip_curves_sha256': sha(visual / 'nine_clip_curves.npz')}
    json_write(visual / 'provenance.json', provenance)
    return reports, provenance


def intervention_table(reports, population):
    rows = [['Seed42 局部条件', '眉相关', '眼相关', '眉 RMS / GT', '眼 RMS / GT']]
    for arm in ARMS:
        for mode in ('full', 'local_static', 'local_reverse'):
            groups = reports[arm][population]['compositions']['raw']['modes']['42/' + mode]['paired']
            brow, eye = groups['brows'], groups['eyes_expression']
            rows.append([LABELS[arm] + ' / ' + mode, f"{brow['centered_correlation']:.3f}",
                         f"{eye['centered_correlation']:.3f}", f"{brow['rms_ratio']:.3f}", f"{eye['rms_ratio']:.3f}"])
    return table(rows)


def seam_table(reports):
    rows = [['608, seed42', '实际接缝对数', '眉接缝位移 MSE', '眉非接缝位移 MSE',
             '眼接缝位移 MSE', '眼非接缝位移 MSE']]
    for arm in ARMS:
        seam = reports[arm]['hold']['compositions']['raw']['modes']['42/full']['actual_decoder_seams']
        values = [LABELS[arm], str(seam['observed_seam_pairs'])]
        for region in ('brows', 'eyes_expression'):
            values += [f"{seam['groups'][region][kind]['displacement_mse']:.6f}" for kind in ('seam', 'nonseam')]
        rows.append(values)
    return table(rows)


def page(root, reports, provenance, required):
    videos = verify_videos(root / 'visual', provenance, required)
    hold_entries = [(LABELS[a] + ' step0 / raw', reports[a]['source']['compositions']['raw']['distribution']) for a in ARMS]
    hold_entries += [(LABELS[a] + ' epoch30 / raw', reports[a]['hold']['compositions']['raw']['distribution']) for a in ARMS]
    dev_entries = [(LABELS[a] + ' epoch30 / raw', reports[a]['dev']['compositions']['raw']['distribution']) for a in ARMS]
    dc_entries = [(LABELS[a] + ' ' + pop + ' / DC', reports[a][pop]['compositions']['dc']['distribution'])
                  for pop in ('hold', 'dev') for a in ARMS]
    status, coverage = read(root / 'status.json'), read(root / 'coverage.json')
    c, f = [means(reports[a]['hold']['compositions']['raw']['distribution'], 'brows') for a in ARMS]
    ce, fe = [means(reports[a]['hold']['compositions']['raw']['distribution'], 'eyes_expression') for a in ARMS]
    cd, fd = [means(reports[a]['dev']['compositions']['raw']['distribution'], 'brows') for a in ARMS]
    coverage_gain = 100 * (coverage['fit']['full_valid'] / coverage['fit']['center_valid'] - 1)
    clamp_note = ''
    if (root / 'display_clamp_diagnostic.json').exists():
        clamp = read(root / 'display_clamp_diagnostic.json')['full_native']['raw']
        clamp_note = (f'<p>另查405片段、seed42，完整上下文raw经过显示裁剪后保留'
                      f"{100*clamp['brows']['amplitude_retained']:.2f}%眉中心幅度、"
                      f"{100*clamp['eyes_expression']['amplitude_retained']:.2f}%眼中心幅度；"
                      '显示裁剪不是这轮raw时序不足的主要原因。'
                      '<a href="display_clamp_diagnostic.json">显示裁剪诊断</a></p>')
    output = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native context30 · 眉眼时序复核</title><style>
body{{font:16px/1.7 system-ui;max-width:1500px;margin:24px auto;padding:0 20px;color:#20262b;background:#f6f8fa}}
h1{{font-size:28px}}h2{{font-size:21px}}h3{{font-size:18px}}table{{border-collapse:collapse;background:white;width:100%;white-space:nowrap}}
td{{border-bottom:1px solid #ddd;padding:8px}}tr:first-child{{font-weight:bold;background:#eef2f5}}.scroll{{overflow-x:auto}}
video,img{{width:100%;display:block}}a{{color:#17639f}}.callout{{border-left:4px solid #3c7187;padding:12px 18px;background:#eaf4f8}}
.warning{{border-left:4px solid #b47b20;padding:12px 18px;background:#fff8e8}}details{{background:white;padding:12px;margin:18px 0}}</style>
<h1>完整连续上下文 · 30轮结果复核</h1>
<div class="callout">两臂各30轮、3210次更新完成，共 {status['seconds_this_invocation']/60:.1f} 分钟。
608片段中，完整上下文的眉相关为 {f['centered_correlation']:.3f}（96帧对照 {c['centered_correlation']:.3f}），
眼相关为 {fe['centered_correlation']:.3f}（对照 {ce['centered_correlation']:.3f}）。
405开发片段眉相关为 {fd['centered_correlation']:.3f}（对照 {cd['centered_correlation']:.3f}）。
完整上下文有局部收益，但没有形成跨人群一致的眉眼动态提升，本轮不能视为动态问题已解决。</div>
<div class="warning">608片段包含13种本轮更新未用的句子；共享source曾学习完整2315 fit池，所以不是全模型未见句子泛化。
405是反复使用的内部开发集。两组分别报告，封存test未读。所有指标只覆盖共同的历史96帧，完整序列额外前后段的质量没有在这里认证。</div>
<p>从同一source出发，对照继续使用96帧；实验臂使用完整native序列，训练有效动作帧增加 {coverage_gain:.1f}%。
两臂训练upper及local分支，身份、全局情感与口型基座冻结；推理只用音频条件和生成历史。未新增loss。
两臂更新数、初始权重和随机抽样匹配，处理帧数与算力不相等；完整方案同时改变下游上下文、动作覆盖及静态统计，不能把效果单独归因于某一项。
emotion2vec原本已读取完整音频，这次新增的是下游完整帧覆盖。</p>
<h2>608增量句子留出 · raw主结果</h2><div class="scroll">{score_table(hold_entries)}</div>
<p>两个step0使用同一权重、各自输入上下文，因此数值可不同。相关更高、中心MSE更低才支持更好的配对时序；仅幅度接近GT并不等于动作时机正确。
RMS/GT是运动幅度比，超出[0,1]比例来自未经裁剪的系数，可能影响显示时的运动。</p>
<h2>405内部开发 · raw主结果</h2><div class="scroll">{score_table(dev_entries)}</div>
<p>开发集未复现一致收益，且眉相关仍很低。不能只看608或选取有利视频宣布成功。
这轮其他43通道及无效帧与旧基座逐位一致，说明工程上没有改动这些输出，不等于身份、唇音同步、全局情感已通过独立质量评价。统一rig也不展示身份质量。</p>
<h2>固定九片、三段视频 · raw置于上排</h2>
<p>上排：跟踪参考 / Center96 raw / Full native raw。下排：旧audio mean + flow / Center96 DC / Full native DC。
沿用旧九个样本与三个angry视频，96帧、25Hz、seed42、同音轨；未重选、放大、平滑或调整时间。
DC是使用预测静态状态的离线均值替换，单独展示、不反馈到生成历史；它在608绝对误差上变差，不能当作默认修复。
rig显示裁剪到[0,1]，无效帧仅为显示取最近有效native帧（同距离取较早帧）；时间线保留，无效帧不评分。
评分与曲线使用保存的原始系数，原blend不变。</p>
{clamp_note}
{videos}
<h2>九片段中心动态</h2><p>仅画图时减各自有效帧均值，不拟合增益或时滞。raw与DC中心曲线应重合；原始曲线在下方。</p>
<img src="visual/centered.png" alt="九片段中心曲线：眉内抬、眉下压、眼眯、下颌">
<details><summary>九片段原始系数</summary><img src="visual/raw.png" alt="九片段原始系数"></details>
<details><summary>局部音频时序干预与接缝</summary><div class="scroll">{intervention_table(reports, 'hold')}</div>
<p>固定seed42，仅将local条件静态化或倒序，其他输入保持不变。对干预有响应只说明用了局部信息，不能单独证明动态语义正确。</p>
<div class="scroll">{seam_table(reports)}</div><p>实际解码接缝按各自native索引定义，两臂接缝位置不完全相同，表内不能按逐接缝配对差解释。</p></details>
<details><summary>608三种子分布评分与DC次结果</summary><div class="scroll">{distribution_table(hold_entries)}</div>
<p>Energy/variogram为三样本fair评分，越低越好；仅三个固定推理seed，未验证概率校准、显著性或感知自然度。</p>
<div class="scroll">{score_table(dc_entries)}</div><p>DC仅改整体均值，中心波形和相邻位移保持不变；hold与dev是不同样本总体。</p></details>
<p><a href="RESULTS.md">本轮结论</a> · <a href="metrics_review.md">独立指标复核</a> ·
<a href="result_audit/manifest.json">结果与固定子集审计</a> · <a href="visual/provenance.json">显示绑定</a> ·
<a href="visual/video_verification.json">视频验收</a> · <a href="matched_audit.json">匹配更新审计</a> ·
<a href="center96/hold_evaluation.json">96帧完整报告</a> · <a href="full_native/hold_evaluation.json">完整上下文报告</a></p></html>'''
    (root / 'index.html').write_text(output, encoding='utf8')


def render(visual, provenance):
    for job in provenance['jobs']:
        folder = visual / job['speaker']
        if folder.exists():
            if not (folder / 'display_report.json').exists():
                raise FileExistsError('Incomplete render directory requires inspection: ' + str(folder))
            continue
        subprocess.run([sys.executable, str(Path(__file__).with_name('render_dynamic_rig_comparison.py')),
                        '--input', str(visual / job['input']), '--output', str(folder),
                        '--audio', str(visual / job['audio']), '--audio-offset-seconds', str(job['audio_offset_seconds']),
                        '--tile-size', '360', '--samples', '8', '--columns', '3'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/native_context_20260918/native_context30'))
    parser.add_argument('--old-root', type=Path, default=Path('artifacts/history_context_20260917/history12'))
    parser.add_argument('--audio-root', type=Path, default=Path('artifacts/audio_prefix_adaptation_20260917/audio_prefix12/visual/audio'))
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--require-videos', action='store_true')
    args = parser.parse_args()
    if args.export_only and (args.require_videos or args.render):
        parser.error('--export-only cannot combine with video arguments')
    reports, provenance = export(args.root, args.old_root, args.audio_root)
    if not args.export_only:
        plot(args.root / 'visual')
        if args.render:
            render(args.root / 'visual', provenance)
        page(args.root, reports, provenance, args.require_videos or args.render)
    print(json.dumps({'page': str((args.root / 'index.html').resolve()), 'jobs': provenance['jobs'],
                      'six_modes_ready': True, 'videos_required': args.require_videos or args.render}, ensure_ascii=True))


if __name__ == '__main__':
    main()
