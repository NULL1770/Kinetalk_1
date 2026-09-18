"""Package audited saved activity probabilities; no model or native-data load.

Examples are the first lexicographic protocol clip per emotion in the consumed
confirmation cell. Curves average every saved training seed's probabilities;
neither examples nor seeds are selected using prediction quality.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch


SCHEMA = 'four_group_activity_condition_v1'
INPUTS = ('status.json', 'protocol.json', 'reports.json', 'thresholds.json',
          'target_diagnostics.json', 'predictions.pt', 'audit.json')
ARMS = ('static_trained', 'real', 'reverse', 'shift', 'mismatch')
ARM_NAMES = {'static_trained': '静态基线', 'real': '真实音频', 'reverse': '倒序音频',
             'shift': '半程循环移位', 'mismatch': '错配音频'}
GROUP_NAMES = ('抬眉组', '压眉组', '眯眼组', '睁眼组')
GROUP_PLOT = ('Brow up', 'Brow down', 'Eye squint', 'Eye wide')
EMOTIONS = {0: '中性', 1: '愤怒', 2: '蔑视', 3: '厌恶', 4: '恐惧', 5: '开心', 6: '悲伤', 7: '惊讶'}
CELLS = ('calibration', 'confirmation')
COLORS = {'real': '#1d67b1', 'static_trained': '#8b552d', 'reverse': '#9c38a7',
          'shift': '#dc8232', 'mismatch': '#267d63'}
ENGLISH = {'real': 'Real audio', 'static_trained': 'Static baseline', 'reverse': 'Reversed audio',
           'shift': 'Half-run shift', 'mismatch': 'Mismatched audio'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fmt(value, digits=6):
    if value is None:
        return '—'
    value = float(value)
    return f'{value:.{digits}f}' if np.isfinite(value) else '非有限值'


def ci(value):
    return '['+', '.join(fmt(v) for v in value)+']'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+\
        '</tbody></table></div>'


def fixed_selection(metadata):
    """Select by recorded metadata only; never replace an unavailable example."""
    selected, seen = [], set()
    for row in sorted(metadata, key=lambda item: item['clip_id']):
        if row['emotion'] not in seen:
            selected.append(row); seen.add(row['emotion'])
    return selected


def load_run(root):
    root = Path(root).resolve()
    hashes = {name: sha(root/name) for name in INPUTS}
    status, protocol, audit = read(root/'status.json'), read(root/'protocol.json'), read(root/'audit.json')
    if (status.get('schema') != SCHEMA or protocol.get('schema') != SCHEMA
            or status.get('status') != 'complete' or status.get('smoke') is not False
            or status.get('test_loaded') is not False or protocol.get('test_loaded') is not False
            or status.get('generator_integrated') is not False or status.get('default_replaced') is not False):
        raise ValueError('A completed formal standalone activity probe with no test/default replacement required')
    if audit.get('all_checks_passed') is not True:
        raise ValueError('Saved prediction audit must pass before packaging')
    audit_inputs = audit.get('input_sha256', {})
    if set(audit_inputs) != set(INPUTS)-{'audit.json'}:
        raise ValueError('Audit must bind all six current run inputs')
    for name, digest in audit_inputs.items():
        if hashes[name] != digest:
            raise ValueError('Run file changed after saved-probability audit: '+name)
    reports = read(root/'reports.json')
    thresholds = read(root/'thresholds.json')
    diagnostics = read(root/'target_diagnostics.json')
    predictions = torch.load(root/'predictions.pt', map_location='cpu', weights_only=False)
    if (predictions.get('schema') != SCHEMA or predictions.get('seeds') != protocol.get('seeds')
            or len(predictions['seeds']) != 3):
        raise ValueError('Saved predictions must bind all three formal training seeds')
    if status.get('activity_passed') != all(reports[cell]['passed'] for cell in CELLS):
        raise ValueError('Final activity result differs from development reports')
    for cell in CELLS:
        reference = {row['clip_id']: row for row in protocol['split'][cell]}
        if len(reference) != len(protocol['split'][cell]):
            raise ValueError('Duplicate protocol membership: '+cell)
        rows = predictions['cells'][cell]
        if len({row['clip_id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate saved predictions: '+cell)
        for row in rows:
            if row['clip_id'] not in reference or any(row[key] != reference[row['clip_id']][key] for key in ('sentence', 'speaker', 'emotion')):
                raise ValueError('Saved example metadata differs from protocol')
    return {'root': root, 'status': status, 'protocol': protocol, 'audit': audit, 'reports': reports,
        'thresholds': thresholds, 'diagnostics': diagnostics, 'predictions': predictions, 'hashes': hashes}


def plot_example(path, row, thresholds, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    starts = np.asarray(row['starts'], dtype=np.int64)
    time = (starts+15)/25.
    target, energy = np.asarray(row['target']), np.asarray(row['energy'])
    thresholds = np.asarray(thresholds)
    if (target.shape != (len(starts), 4) or target.dtype != np.bool_
            or energy.shape != target.shape or not np.isfinite(energy).all()
            or not np.array_equal(target, energy > thresholds)):
        raise ValueError('Saved example energy/target/threshold binding differs')
    probabilities = {}
    for arm in ARMS:
        if arm not in row['probabilities']:
            continue
        p = np.asarray(row['probabilities'][arm], dtype=np.float64)
        if p.shape != (len(seeds), len(starts), 4) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('Invalid saved example probability: '+arm)
        probabilities[arm] = p.mean(0)
    # A gap in the full-window clock means a new observed run. Separate lines
    # explicitly instead of drawing an apparent transition through missing data.
    cuts = np.r_[0, np.flatnonzero(np.diff(starts) > 8)+1, len(starts)]
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    for group, axis in enumerate(axes):
        energy_axis = axis.twinx()
        for segment, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
            label = lambda text: text if segment == 0 else '_nolegend_'
            axis.step(time[left:right], target[left:right, group].astype(float), where='mid',
                color='#20252b', linewidth=1.3, alpha=.8, label=label('Observed activity label'), zorder=4)
            for arm, p in probabilities.items():
                axis.plot(time[left:right], p[left:right, group], color=COLORS[arm], linewidth=1.25,
                    linestyle='--' if arm == 'static_trained' else '-', label=label(ENGLISH[arm]))
            energy_axis.plot(time[left:right], energy[left:right, group], color='#a5adb5', linewidth=1.,
                alpha=.5, label=label('Smoothed raw speed'), zorder=0)
        energy_axis.axhline(thresholds[group], color='#a5adb5', linestyle=':', linewidth=.9,
                           label='Fit threshold')
        energy_axis.set_ylim(bottom=0); energy_axis.set_ylabel('Raw RMS speed / s', color='#7b848c', fontsize=8)
        axis.set_ylabel(GROUP_PLOT[group]+'\nactivity probability / label')
        axis.set_ylim(-.05, 1.05); axis.grid(alpha=.14)
        if group == 0:
            h1, l1 = axis.get_legend_handles_labels(); h2, l2 = energy_axis.get_legend_handles_labels()
            axis.legend(h1+h2, l1+l2, loc='lower center', bbox_to_anchor=(.5, 1.06),
                        ncol=3, fontsize=8, frameon=False)
    axes[0].set_title(row['clip_id']+' | mean probabilities of all 3 training seeds', fontsize=10, pad=63)
    axes[-1].set_xlabel('Original clip time (s): (window start + 15) / 25; target covers central 8 transitions')
    fig.savefig(path, format='svg', metadata={'Date': None})
    plt.close(fig)


def package(run_path, output_path):
    run = load_run(run_path)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError('Fresh output directory required')
    selected = fixed_selection(run['protocol']['split']['confirmation'])
    if not selected:
        raise ValueError('No confirmation metadata for fixed example selection')
    by_id = {row['clip_id']: row for row in run['predictions']['cells']['confirmation']}
    output.mkdir(parents=True); (output/'plots').mkdir()
    main_rows, group_rows, seed_rows, support_rows = [], [], [], []
    for cell in CELLS:
        report = run['reports'][cell]
        count = len(run['protocol']['split'][cell])
        label = f'{count}条已用开发集（'+('校准' if cell == 'calibration' else '回归')+'）'
        support_rows.append([label, report['support'], fmt(report['support_fraction']*100, 2)+'%',
            report['support_sentences'], '通过' if report['passed'] else '未通过'])
        for arm in ARMS:
            score = report.get('paired', {}).get(arm)
            gain = report.get('gains', {}).get(arm)
            main_rows.append([label, ARM_NAMES[arm], fmt(score['brier'] if score else None),
                fmt(gain['gain']) if gain else '参考真实音频', ci(gain['ci95']) if gain else '—'])
            group_rows.append([label, ARM_NAMES[arm]]+[fmt(value) for value in (score['group_brier'] if score else [None]*4)])
        for row in report.get('per_seed', []):
            seed_rows.append([label, run['predictions']['seeds'][row['seed_index']], fmt(row['static_brier']),
                fmt(row['real_brier']), fmt(row['gain'])])
    excerpts = []
    output_examples = []
    for index, metadata in enumerate(selected):
        cid = metadata['clip_id']; row = by_id.get(cid)
        heading = EMOTIONS.get(metadata['emotion'], '情感'+str(metadata['emotion']))+' · '+cid
        if row is None:
            excerpts.append('<article><h3>'+html.escape(heading)+'</h3><p>此元数据固定样例没有完整32帧支持；不另挑效果更好的替代样例。</p></article>')
            output_examples.append({'clip_id': cid, 'emotion': metadata['emotion'], 'available': False})
            continue
        filename = f'plots/example_{index:02d}.svg'
        plot_example(output/filename, row, run['thresholds']['thresholds'], run['predictions']['seeds'])
        missing = '；本样例没有符合规则的错配供体，因此未画错配曲线' if 'mismatch' not in row['probabilities'] else ''
        excerpts.append('<article><h3>'+html.escape(heading)+'</h3><p>原记录句子：'+html.escape(str(metadata['sentence']))+
            '；身份编号：'+html.escape(str(metadata['speaker']))+html.escape(missing)+'。</p><img src="'+filename+'" alt="四组活动标签与多种声学条件的概率曲线"></article>')
        output_examples.append({'clip_id': cid, 'emotion': metadata['emotion'], 'available': True, 'plot': filename})
    thresholds = table(['分组', '拟合速度阈值', '阈值触底'], [[name, fmt(value), '是' if value <= 1e-6 else '否']
        for name, value in zip(GROUP_NAMES, run['thresholds']['thresholds'])])
    diagnostics = table(['数据范围', '来源片段', '有完整窗口片段', '窗口数', '四组活动比例'], [[cell,
        run['diagnostics'][cell]['source_clips'], run['diagnostics'][cell]['scored_clips'],
        run['diagnostics'][cell]['windows'], ' / '.join(fmt(value, 3) for value in run['diagnostics'][cell]['prevalence'])]
        for cell in ('fit',)+CELLS])
    passed = bool(run['status']['activity_passed'])
    body = '<h1>眉眼活动概率：已用开发集审阅</h1><p class="result">本轮活动判据：'+('通过' if passed else '未通过')+'</p>'
    body += '<p>这里只检查音频能否预测眉眼组在一个短窗口内是否活跃，<strong>不是完整面部生成效果</strong>。标签只有活动与否，不含动作方向、形状或持续轨迹；概率也不是条件发生风险率（hazard），没有直接产生原生动作。</p>'
    body += '<p>199条校准集和353条回归集都已在先前多轮实验中使用，上游全局模型也见过历史来源。以下收益与句级bootstrap区间属于开发证据，不能当独立测试、整体泛化或论文结论。情感、身份和口型未在本次概率探针中重新验证。</p>'
    body += '<h2>共同支持与结论</h2>'+table(['数据范围', '共同片段', '覆盖率', '句子数', '活动判据'], support_rows)
    body += '<p>所有主表采用同一组具有全部干预支持的片段。每个窗口先平均三个训练seed的预测概率，再计算Brier；先在片段内平均窗口，再让片段等权。Brier越低越好；“对照−真实”收益为正表示真实音频更好。</p>'
    body += table(['数据范围', '输入条件', '集成Brier ↓', '对照−真实收益', '句级95%区间'], main_rows)
    body += '<h2>各训练seed</h2>'+table(['数据范围', '训练seed', '静态Brier', '真实Brier', '静态−真实'], seed_rows)
    body += '<h2>四组Brier</h2>'+table(['数据范围', '输入条件', *GROUP_NAMES], group_rows)
    body += '<h2>标签与覆盖</h2><p>速度由原始9维眉眼系数的3点中心平滑后计算：32帧上下文的第11至19帧形成8个相邻差分，乘以25Hz后按各组通道取RMS。阈值只由拟合集逐片均衡的65%分位数确定，查询片段不调整阈值。</p>'+thresholds+diagnostics
    body += '<h2>固定样例</h2><p>按353条回归集元数据的clip_id排序，每种情感只取第一条，不依据预测结果挑选；全部曲线使用三个训练seed的概率均值。横轴为(start+15)/25秒，表示中央8个目标转移的中心。黑色阶梯是活动标签，浅灰曲线及右轴是速度能量；缺失时钟不连接。图例用英文避免服务器字体缺失。</p>'+''.join(excerpts)
    body += '<h2>记录与审计</h2><p>保存概率审计已通过。审计复算集成评分和记录绑定，不重新运行模型；bootstrap与原始motion到标签的处理是否重算，以审计字段为准。</p>'+table(['项目', '值'], [
        ['审计检查数', run['audit'].get('checks', '—')], ['bootstrap独立重算', run['audit'].get('bootstrap_recomputed', False)],
        ['原motion到目标独立重算', run['audit'].get('target_from_raw_motion_recomputed', False)],
        ['训练seed', ', '.join(map(str, run['predictions']['seeds']))], ['源run', str(run['root'])]])
    body += '<p>页面与图片由已保存概率及JSON记录生成，没有加载模型、原音频或原动作数据。输入与输出哈希记录在<a href="review_manifest.json">review_manifest.json</a>。</p>'
    style = 'body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#243244;background:#f3f6fa;margin:0;padding:32px}main{max-width:1280px;margin:auto}h1{font-size:30px}h2{margin-top:38px}p{max-width:1100px}.result{font-size:23px;font-weight:700;color:'+('#246741' if passed else '#93452e')+'}.scroll{overflow:auto;background:white;border-radius:8px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e1e7ef;white-space:nowrap}th{background:#e8eef6}article{background:white;padding:20px;border-radius:10px;margin:22px 0}article img{width:100%;height:auto}h3{overflow-wrap:anywhere;font-size:17px}'
    (output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>眉眼活动概率审阅</title><style>'+style+'</style><main>'+body+'</main></html>', encoding='utf8')
    outputs = {str(path.relative_to(output)).replace('\\', '/'): sha(path) for path in sorted(output.rglob('*')) if path.is_file()}
    manifest = {'schema': 'activity_condition_review_v1', 'source_run': str(run['root']),
        'input_sha256': run['hashes'], 'output_sha256': outputs, 'selected_examples': output_examples,
        'selection_rule': 'First lexicographic protocol clip_id per emotion, confirmation metadata only',
        'probability_ensemble': 'Mean over all three saved training seeds before scoring or plotting',
        'plot_clock': '(start+15)/25 seconds; center of8 target transitions within H32',
        'script_sha256': sha(__file__), 'model_loaded': False, 'native_data_loaded': False,
        'development_only': True, 'activity_passed': passed, 'full_face_verified': False}
    (output/'review_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = package(args.run, args.output)
    print(json.dumps({'status': 'packaged', 'output': str(args.output.resolve()),
        'examples': len(result['selected_examples']), 'activity_passed': result['activity_passed']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
