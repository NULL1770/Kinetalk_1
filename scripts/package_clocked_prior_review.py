"""Package saved fixed-clock upper-face probes without inference or fitting.

Raw-coordinate paired, bounded residual, and prosody residual schemas are accepted. Example
selection reads protocol metadata only: first lexicographic clip per emotion
in the consumed 353 development cell. Seed42 is displayed unchanged; all eight
draws remain included in scored reports and optional uncertainty ranges.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch


SCHEMAS = {'clocked_motion_shape_development_v1', 'clocked_bounded_static_residual_v1',
           'clocked_prosody_static_residual_v1'}
SEEDS = (42, 123, 2026, 77, 91, 301, 509, 997)
GROUPS = ((2,), (2, 3, 4), (0, 1), (5, 7), (6, 8))
PLOT_LABELS = ('Inner brow up', 'Brow up mean', 'Brow down mean', 'Eye squint mean', 'Eye wide mean')
EMOTIONS = {0: '中性', 1: '愤怒', 2: '蔑视', 3: '厌恶', 4: '恐惧', 5: '开心', 6: '悲伤', 7: '惊讶'}
ARM_LABELS = {'static_trained': '独立训练的静态音频基线', 'temporal': '真实时序音频',
              'static': '时序模型 / 静态音频', 'reverse': '时序模型 / 倒放音频',
              'mismatch': '时序模型 / 错配音频', 'uniform': '均匀抽取动作'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def fmt(value, digits=5):
    if value is None:
        return '—'
    number = float(value)
    return f'{number:.{digits}f}' if np.isfinite(number) else '非有限值'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+\
        '</tbody></table></div>'


def load_run(root, *, curves=True):
    root = Path(root).resolve()
    status, protocol = read(root/'status.json'), read(root/'protocol.json')
    if (status.get('schema') not in SCHEMAS or protocol.get('schema') != status['schema']
            or status.get('status') != 'complete' or status.get('smoke') is not False
            or status.get('test_loaded') is not False or status.get('default_replaced') is not False
            or status.get('generator_integrated') is not False):
        raise ValueError('Completed formal stand-alone fixed-clock development run required')
    metadata = protocol.get('split', {}).get('confirmation', [])
    ids = [row['clip_id'] for row in metadata]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Unique nonempty development protocol IDs required')
    assessment = read(root/'confirmation_assessment.json')
    if (status.get('timing_passed') != assessment.get('timing_passed')
            or status.get('quality_passed') != assessment.get('quality_passed')):
        raise ValueError('Completion and saved assessment disagree')
    reports = {}
    for arm in ARM_LABELS:
        path = root/('confirmation_'+arm+'.json')
        if path.exists():
            reports[arm] = read(path)
            actual = [r['clip_id'] for r in reports[arm]['rows']]
            if len(actual) != len(set(actual)) or not set(actual) <= set(ids):
                raise ValueError('Report membership differs from protocol: '+arm)
            if arm != 'mismatch' and set(actual) != set(ids):
                raise ValueError('Primary report omits development clips: '+arm)
    if not {'static_trained', 'temporal', 'static', 'reverse', 'mismatch'} <= reports.keys():
        raise ValueError('All paired baseline/intervention reports are required')
    payload = {}
    if curves:
        for arm in ('static_trained', 'temporal'):
            payload[arm] = torch.load(root/('confirmation_'+arm+'.pt'), map_location='cpu', weights_only=False)
            if set(payload[arm]) != set(ids):
                raise ValueError('Saved trajectory membership differs: '+arm)
    return {'root': root, 'status': status, 'protocol': protocol, 'assessment': assessment,
            'reports': reports, 'curves': payload, 'metadata': metadata}


def fixed_selection(metadata):
    chosen, seen = [], set()
    for row in sorted(metadata, key=lambda value: value['clip_id']):
        if row['emotion'] not in seen:
            chosen.append(row)
            seen.add(row['emotion'])
    return chosen


def validate_example(real, baseline, cid):
    target, valid = np.asarray(real['target']), np.asarray(real['valid'])
    if valid.dtype != np.bool_ or valid.ndim != 1 or not valid.any() or target.shape != (len(valid), 9):
        raise ValueError('Invalid raw upper9 target/mask: '+cid)
    if not np.array_equal(target, np.asarray(baseline['target'])) or not np.array_equal(valid, np.asarray(baseline['valid'])):
        raise ValueError('Paired target/mask differs: '+cid)
    for item in (real, baseline):
        samples = np.asarray(item['samples'])
        if (samples.shape != (len(SEEDS), len(valid), 9) or not np.isfinite(samples[:, valid]).all()
                or not np.isfinite(target[valid]).all()):
            raise ValueError('Eight finite native upper9 draws required: '+cid)


def plot_example(path, row, real, baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    samples = np.asarray(real['samples'], dtype=np.float64)
    static = np.asarray(baseline['samples'], dtype=np.float64)
    target, valid = np.asarray(real['target']), np.asarray(real['valid'])
    time = np.arange(len(valid), dtype=np.float64)/25.
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True, layout='constrained')
    mask = lambda value: np.where(valid, value, np.nan)
    for ax, group, label in zip(axes, GROUPS, PLOT_LABELS):
        dynamic_group = samples[..., list(group)].mean(-1)
        static_group = static[..., list(group)].mean(-1)
        truth = target[..., list(group)].mean(-1)
        ax.fill_between(time, mask(dynamic_group.min(0)), mask(dynamic_group.max(0)),
                        color='#176fb0', alpha=.13, label='Temporal: range of all 8 draws')
        ax.plot(time, mask(truth), color='#142434', linewidth=1.8, label='Tracked GT')
        ax.plot(time, mask(dynamic_group[0]), color='#176fb0', linewidth=1.25, label='Temporal: seed42')
        ax.plot(time, mask(static_group[0]), color='#bd5745', linewidth=1.15, linestyle='--', label='Static baseline: seed42')
        ax.set_ylabel(label+'\nraw coefficient')
        ax.grid(alpha=.18)
        limits = ax.get_ylim()
        for boundary in (0., 1.):
            if limits[0] <= boundary <= limits[1]:
                ax.axhline(boundary, color='#929aa2', linewidth=.65, linestyle=':')
        ax.set_ylim(limits)
    axes[0].set_title(row['clip_id']+' | no gain, lag, or target-mean fitting', fontsize=11, pad=42)
    axes[0].legend(loc='lower center', bbox_to_anchor=(.5, 1.035), ncol=2, fontsize=8, frameon=False)
    axes[-1].set_xlabel('Native time from clip start (seconds), 25 Hz; gaps remain blank')
    fig.savefig(path, format='svg', metadata={'Date': None})
    plt.close(fig)


def arm_rows(run):
    result = []
    for arm, label in ARM_LABELS.items():
        if arm not in run['reports']:
            continue
        report = run['reports'][arm]
        s = report['summary']
        if s is None:
            result.append([label, 0, '无共同支持', '—', '—', '—'])
        else:
            result.append([label, len(report['rows']), fmt(s['joint_fair_es']['centered']),
                           fmt(s['joint_fair_es']['raw']), fmt(s['variogram']['aggregate']),
                           fmt(s['covariance_distance']['velocity'])])
    return result


def package(run_path, output_path, previous_path=None, initial_path=None):
    run = load_run(run_path)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError('Fresh review directory required')
    picks = fixed_selection(run['metadata'])
    if len(picks) < 3:
        raise ValueError('At least three metadata-selected emotions required for this review')
    for row in picks:
        cid = row['clip_id']
        validate_example(run['curves']['temporal'][cid], run['curves']['static_trained'][cid], cid)
    output.mkdir(parents=True)
    plots = output/'plots'; plots.mkdir()
    sections = []
    for index, row in enumerate(picks, 1):
        relative = f'plots/example_{index:02d}.svg'
        plot_example(output/relative, row, run['curves']['temporal'][row['clip_id']], run['curves']['static_trained'][row['clip_id']])
        sections.append('<section><h2>'+html.escape(EMOTIONS.get(row['emotion'], str(row['emotion'])))+
                        ' · '+html.escape(row['clip_id'])+'</h2><p>身份编号 '+str(row['speaker'])+
                        '；句子 '+html.escape(row['sentence'])+'</p><img alt="五组眉眼原始系数曲线" src="'+relative+'"></section>')
    assessment, status = run['assessment'], run['status']
    timing, quality = bool(assessment['timing_passed']), bool(assessment['quality_passed'])
    headline = '时序门槛'+('通过' if timing else '未通过')+'；系数质量工程门槛'+('通过' if quality else '未通过')+'。'
    scope = '本页展示独立眉眼先验的真实系数输出，尚未接入完整面部生成器。工程门槛检查幅度、速度、加速度和系数范围，不等于感知自然度通过；不能据此验收身份、口型或整体情感，也不把曲线幅度增大等同于动态成功。'
    timing_rows = []
    for key, gain in assessment.get('gains_baseline_minus_real', {}).items():
        timing_rows.append([ARM_LABELS.get(key, key), fmt(gain['gain']),
                            '['+', '.join(fmt(v) for v in gain['ci95'])+']'])
    quality_rows = [[name, '通过' if value else '未通过'] for name, value in assessment.get('quality_checks', {}).items()]
    s = run['reports']['temporal']['summary']
    rms = s['rms_ratio']
    quality_rows += [['抬眉 / 压眉 / 眯眼 / 睁大 RMS÷GT', ' / '.join(fmt(x, 3) for x in rms)],
                     ['速度 RMS÷GT', fmt(assessment.get('speed_ratio'), 3)],
                     ['加速度 RMS÷GT', fmt(run['reports']['temporal']['acceleration']['rms_ratio'], 3)],
                     ['最大通道越界比例', fmt(max(s['raw_oob'])*100, 3)+'%']]
    comparisons, bindings = [], {}
    if previous_path is not None:
        prior = load_run(previous_path, curves=False)
        if {x['clip_id'] for x in prior['metadata']} != {x['clip_id'] for x in run['metadata']}:
            raise ValueError('Previous clocked comparison has different development membership')
        prior_label = {'clocked_motion_shape_development_v1': '前一版：原始系数独立双分支',
                       'clocked_bounded_static_residual_v1': '前一版：冻结静态基线与时序残差',
                       'clocked_prosody_static_residual_v1': '前一版：韵律时序残差'}[prior['status']['schema']]
        for label, item in ((prior_label, prior), ('本版', run)):
            b, r = item['reports']['static_trained']['summary'], item['reports']['temporal']['summary']
            comparisons.append([label, fmt(b['joint_fair_es']['centered']), fmt(r['joint_fair_es']['centered']),
                                fmt(r['joint_fair_es']['raw']), '通过' if item['assessment']['timing_passed'] else '未通过',
                                '通过' if item['assessment']['quality_passed'] else '未通过'])
        bindings['previous'] = {'path': str(prior['root']), 'status_sha256': sha(prior['root']/'status.json'),
                                'assessment_sha256': sha(prior['root']/'confirmation_assessment.json')}
    initial_note = ''
    if initial_path is not None:
        initial = Path(initial_path).resolve()
        protocol = read(initial/'protocol.json')
        if {r['clip_id'] for r in protocol['split']['confirmation']} != {r['clip_id'] for r in run['metadata']}:
            raise ValueError('Initial joint-prior comparison membership differs')
        old_global, old_local = (read(initial/name)['summary'] for name in ('confirmation_global.json', 'confirmation_local.json'))
        initial_note = '<p>迁移后第一轮 joint prior：global centered ES '+fmt(old_global['joint_fair_es']['centered'])+'，local '+\
            fmt(old_local['joint_fair_es']['centered'])+'。这些是同一开发单元的历史参照；基线结构不同，不能把跨版差值全部归因于音频。</p>'
        bindings['initial'] = {'path': str(initial), 'protocol_sha256': sha(initial/'protocol.json')}
    selected_scale = status.get('selected_scale')
    scale_note = '' if selected_scale is None else '<p>仅在 199 调整单元选择的残差尺度：<b>'+fmt(selected_scale, 2)+'</b>。尺度为 0 时不能称时序有效。</p>'
    reason = assessment.get('reason')
    reason_note = '' if not reason else '<p>保存的判定原因：<code>'+html.escape(reason)+'</code>。</p>'
    summary = '<h2>整组结果</h2>'+table(['条件', '样本数', 'Centered ES ↓', 'Raw ES ↓', 'Variogram ↓', '速度协方差误差 ↓'], arm_rows(run))
    gains = '<h2>共同样本上的配对增益</h2><p>数值为对照 ES 减去真实时序 ES，正数表示真实时序更好；按句子聚类 bootstrap。支持比例 '+fmt(assessment.get('support_fraction', 0)*100, 1)+'%。</p>'+\
        (table(['对照', 'Centered ES 增益', '95% 区间'], timing_rows) if timing_rows else '<p>共同支持不足，未计算有效配对结论。</p>')
    earlier = '' if not comparisons else '<h2>与前一版对照</h2>'+table(['版本', '静态 ES', '时序 ES', '时序 Raw ES', '时序门槛', '系数质量门槛'], comparisons)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+\
        '<title>眉眼时序先验：固定开发样本评审</title><style>body{font:16px/1.65 system-ui,sans-serif;margin:32px auto;max-width:1200px;padding:0 20px;color:#173047;background:#f7fafc}h1{font-size:28px}h2{font-size:21px;margin-top:32px}.lead{padding:18px;background:#fff3d5;border-left:5px solid #d79422}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;background:white}td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #dce5ed;white-space:nowrap}th{background:#eaf1f6}img{max-width:100%;background:white;border:1px solid #dce5ed}code{font-size:13px}section{margin-top:35px}.small{font-size:14px;color:#536b7d}</style>'+\
        '<h1>眉眼时序先验：固定开发样本评审</h1><div class="lead"><b>'+headline+'</b><p>'+scope+'</p></div>'+\
        '<p>训练完成：'+str(status.get('epochs_per_stage', status.get('epochs_per_arm', status.get('epochs', '—'))))+' 轮/阶段；总计 '+fmt(status.get('seconds', 0)/60, 2)+' 分钟。'+\
        '此处的 353 条已被前序实验查看，是开发回归数据；不是未见测试集。上游全局音频表示有历史训练曝光。</p>'+scale_note+reason_note+summary+gains+\
        '<h2>系数质量检查</h2>'+table(['检查项', '结果'], quality_rows)+earlier+initial_note+\
        '<h2>固定示例与阅读方式</h2><p>先按 clip_id 排序，每种实际存在的情感取第一条，只按元数据选择。黑线为 GT，蓝线为 seed42 的一次真实生成，红虚线为静态音频基线的同 seed；淡蓝带是全部 8 次生成的最小—最大范围，不是置信区间。'+\
        '没有选择最佳 seed，没有对齐时延、乘增益或用 GT 校准均值。曲线显示未裁剪的原始系数；缺失帧留空。</p>'+\
        '<p class="small">上排 inner brow 与下一排 brow-up 部分重叠，用来分别观察内眉通道和抬眉整体。范围带很宽只代表模型随机性大，不自动代表合理多样性。图中没有眨眼、视线、头动或口型。</p>'+''.join(sections)+\
        '<p class="small">本页只读取保存结果，不加载训练模型、不做拟合。<a href="review_manifest.json">样本选择、来源与文件校验</a> · <a href="assessment.json">原始判定</a> · <a href="summary.json">数值摘要</a></p></html>'
    (output/'index.html').write_text(page, encoding='utf8')
    write(output/'assessment.json', assessment)
    write(output/'summary.json', {'status': status, 'reports': {key: value['summary'] for key, value in run['reports'].items()}})
    source_files = ['status.json', 'protocol.json', 'confirmation_assessment.json',
                    'confirmation_temporal.pt', 'confirmation_static_trained.pt']
    source_files += ['confirmation_'+name+'.json' for name in run['reports']]
    manifest = {'schema': 'clocked_upper_prior_review_v1', 'run': str(run['root']),
                'selection_rule': 'First lexicographic development clip per emotion from protocol metadata',
                'selection_uses_metadata_only': True, 'outcome_based_selection': False, 'selected': picks,
                'display_seed': 42, 'all_scored_seeds': list(SEEDS), 'native_fps': 25,
                'source_sha256': {name: sha(run['root']/name) for name in source_files},
                'comparisons': bindings, 'script_sha256': sha(__file__), 'model_loaded': False,
                'fullface_generated': False, 'target_used_to_correct_generation': False,
                'development_only': True, 'test_loaded': False,
                'output_sha256': {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}}
    write(output/'review_manifest.json', manifest)
    return {'output': str(output), 'examples': len(picks), 'timing_passed': timing, 'quality_passed': quality}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--initial', type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.run, args.output, args.previous, args.initial), ensure_ascii=False))


if __name__ == '__main__':
    main()
