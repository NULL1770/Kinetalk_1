"""Chinese review of saved motion-process probes, with no fitting or inference.

The four-state teacher is an oracle compression diagnostic. Audio curves are
free rollouts of a stand-alone prior, not final facial animations. Selection is
the first three sorted clip identifiers in each of the three holdout cells.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch


CELLS = ('sentence', 'speaker', 'joint')
ARMS = ('global_only', 'local_audio')
GROUPS = ('brow_up', 'brow_down', 'eye_squint', 'eye_wide')
GROUP_LABELS = ('抬眉 / brow up', '压眉 / brow down', '眯眼 / squint', '睁大 / wide')
CELL_LABELS = {'sentence': '留句', 'speaker': '留身份', 'joint': '身份与句子同时留出'}
SEEDS = (42, 123, 2026, 77, 91, 301, 509, 997)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def number(value, digits=4):
    if value is None:
        return '—'
    value = float(value)
    return f'{value:.{digits}f}' if np.isfinite(value) else '非有限值'


def load_inputs(root):
    status, representation, protocol, gate = (read(root/name) for name in
        ('status.json', 'representation.json', 'protocol.json', 'audio_gate.json'))
    if (status.get('status') != 'complete' or status.get('smoke') is not False
            or protocol.get('smoke') is not False or protocol.get('epochs') != 30
            or protocol.get('test_loaded') is not False or protocol.get('dev_indexed') is not False
            or status.get('generator_started') is not False or status.get('default_replaced') is not False):
        raise ValueError('A completed formal stand-alone 30-epoch probe is required')
    if status.get('representation_passed') != representation.get('passed') or status.get('audio_passed') != gate.get('passed'):
        raise ValueError('Completion and saved gate results differ')
    matching = read(root/'matching.json')
    if matching['global_only'] != matching['local_audio']:
        raise ValueError('Paired training update/order record differs')
    reports, examples, selection = {}, {}, {}
    for arm in ARMS:
        reports[arm], examples[arm] = {}, {}
        for cell in CELLS:
            reports[arm][cell] = read(root/arm/(cell+'.json'))
            # These are trusted, locally produced experiment artifacts and
            # contain NumPy arrays; weights_only=True rejects their schema.
            examples[arm][cell] = torch.load(root/arm/(cell+'.pt'), map_location='cpu', weights_only=False)
            expected = {row['clip_id'] for row in protocol['split'][cell]}
            actual = [row['clip_id'] for row in reports[arm][cell]['rows']]
            if (len(actual) != len(set(actual)) or set(actual) != expected
                    or set(examples[arm][cell]) != expected):
                raise ValueError('Saved holdout membership differs: '+arm+'/'+cell)
    for cell in CELLS:
        left, right = reports['global_only'][cell], reports['local_audio'][cell]
        if [r['clip_id'] for r in left['rows']] != [r['clip_id'] for r in right['rows']]:
            raise ValueError('Paired output order differs')
        selection[cell] = sorted(examples['local_audio'][cell])[:3]
        if len(selection[cell]) != 3:
            raise ValueError('Three fixed examples per holdout cell required')
        for cid in selection[cell]:
            a, b = examples['global_only'][cell][cid], examples['local_audio'][cell][cid]
            for key in ('target', 'teacher', 'valid', 'anchor'):
                if not np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
                    raise ValueError('Paired target/teacher/mask/anchor differs: '+cid)
            n = len(b['valid'])
            for name, item in (('global', a), ('local', b)):
                samples, valid = np.asarray(item['samples']), np.asarray(item['valid'])
                if (samples.shape != (8, n, 4) or valid.shape != (n,) or valid.dtype != np.bool_
                        or not np.isfinite(samples[:, valid]).all()
                        or np.asarray(item['target']).shape != (n, 4)
                        or not np.isfinite(np.asarray(item['target'])[valid]).all()):
                    raise ValueError('Saved free-rollout shape/observations differ: '+name+'/'+cid)
    return status, representation, protocol, gate, matching, reports, examples, selection


def gate_failures(gate, reports):
    failures = []
    cells = gate.get('cells', {})
    if gate.get('reason'):
        failures.append(str(gate['reason']))
    if 'sentence' in cells:
        cell = cells['sentence']
        if cell['nll_gain'] < .01:
            failures.append('留句 completed-event NLL 增益小于 0.01。')
        if cell['sentence_bootstrap_ci95'][0] <= 0:
            failures.append('留句 NLL 增益的句子聚类 bootstrap 95% 区间未完全高于零。')
    if 'speaker' in cells and cells['speaker']['nll_gain'] <= 0:
        failures.append('留身份 NLL 增益未高于零。')
    for cell in ('sentence', 'speaker'):
        if cell not in cells:
            continue
        for metric, values in cells[cell]['distribution_ratios'].items():
            bad = [GROUP_LABELS[g] for g, v in enumerate(values) if not np.isfinite(v) or v > 1.05]
            if bad:
                failures.append(f'{CELL_LABELS[cell]} {metric} 超过对照 1.05 倍：'+ '、'.join(bad)+'。')
        summary = reports['local_audio'][cell]['summary']
        bad_rms = [GROUP_LABELS[g] for g, v in enumerate(summary['rms_ratio']) if not .25 <= v <= 2]
        if bad_rms:
            failures.append(f'{CELL_LABELS[cell]} 动态 RMS 比超出 [0.25, 2]：'+'、'.join(bad_rms)+'。')
        bad_domain = [GROUP_LABELS[g] for g, v in enumerate(summary['out_of_domain']) if v > .05]
        if bad_domain:
            failures.append(f'{CELL_LABELS[cell]} 原幅值越界率超过 5%：'+'、'.join(bad_domain)+'。')
    return failures


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def html_table(headers, rows):
    return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+\
        '</tbody></table></div>'


def tables(representation, gate, reports):
    reconstruction = []
    for cell in CELLS:
        row = representation['cells'][cell]['reconstructed']
        uniform = representation['cells'][cell]['uniform']
        for g, name in enumerate(GROUPS):
            values = row['groups'][name]
            reconstruction.append([CELL_LABELS[cell], GROUP_LABELS[g], number(values['centered_r2']),
                number(uniform['groups'][name]['centered_r2']), number(values['correlation']),
                number(values['rms_ratio']), number(row['segments_per_second'], 2)])
    gains = []
    for cell in CELLS:
        values = gate.get('cells', {}).get(cell, {})
        a, b = reports['global_only'][cell]['summary'], reports['local_audio'][cell]['summary']
        ci = values.get('sentence_bootstrap_ci95')
        gains.append([CELL_LABELS[cell], number(a.get('event_nll')), number(b.get('event_nll')),
                      number(values.get('nll_gain')), '—' if ci is None else '['+', '.join(number(v) for v in ci)+']'])
    free = []
    for cell in CELLS:
        a, b = reports['global_only'][cell]['summary'], reports['local_audio'][cell]['summary']
        for g, label in enumerate(GROUP_LABELS):
            pair = lambda metric: number(a[metric][g])+' → '+number(b[metric][g])
            free.append([CELL_LABELS[cell], label, pair('raw_es'), pair('centered_es'), pair('variogram'),
                         pair('correlation'), pair('rms_ratio'), number(b['out_of_domain'][g]*100, 2)+'%'])
    return [(['划分', '动作组', 'Teacher R²', '均匀同段数 R²', 'Teacher corr', 'Teacher RMS/GT', '总段/秒'], reconstruction),
            (['划分', 'Global NLL', 'Local NLL', '增益 Global−Local', '句子 bootstrap 95% 区间'], gains),
            (['划分', '动作组', 'Raw ES ↓', 'Centered ES ↓', 'Variogram ↓', '单样本相关均值', '动态 RMS/GT', 'Local 越界率'], free)]


def optional_failure_diagnosis(root):
    """Present saved post-hoc forward diagnostics without altering the gate."""
    folder = root/'failure_diagnosis_v2'
    if not (folder/'report.json').is_file() or not (folder/'complete.json').is_file():
        return None
    report, complete = read(folder/'report.json'), read(folder/'complete.json')
    if complete['report_sha256'] != sha(folder/'report.json') or any(complete.get(k) is not False for k in
        ('training_started', 'normalization_refitted', 'segmentation_refitted', 'checkpoint_selected')):
        raise ValueError('Failure-diagnosis completion binding or read-only scope differs')
    replay = report['replay']
    if set(replay) != {a+'/'+c for a in ARMS for c in CELLS}:
        raise ValueError('Failure diagnosis must replay the six held-out evaluation sets')
    max_event = max(v['max_event_nll_abs_error'] for v in replay.values())
    max_initial = max(v['max_initial_nll_abs_error'] for v in replay.values())
    score = report['scores']
    delta = score['local_audio']['real']['sentence']['event_weighted']['squared_error']/\
        score['global_only']['real']['sentence']['event_weighted']['squared_error']-1
    paragraphs = [f'保存的最终模型经真实前向重算六组留出评价：event NLL 最大绝对差 {number(max_event, 8)}，initial NLL 最大绝对差 {number(max_initial, 8)}。本诊断未重新训练、拟合尺度、拟合分段或选择 checkpoint。',
        f'局部先验在训练集拟合更好，但留句条件位移 MSE 比 global-only 增加 {100*delta:.2f}%，预测不确定度偏小，存在过度自信。校准不确定度可能改善似然，但不能据此称时序已成功。',
        '下表全部使用真实 teacher 边界和历史状态，delta 还条件于真实 duration；按完整控制段汇聚，和主表逐 clip 平均的权重不同。z² 是 (真实 delta − 预测均值)² / 预测方差 的逐段均值；±2σ 覆盖率用于校准诊断。方向准确率即使约 70%，也包含真实运动历史提供的信息，不能称为纯音频预测正确率。']
    rows = []
    for cell in ('fit', 'sentence'):
        for arm in ARMS:
            r = score[arm]['real'][cell]['event_weighted']
            rows.append(['训练' if cell == 'fit' else '留句', 'Global' if arm == 'global_only' else 'Local',
                number(r['squared_error']), number(r['predicted_std']), number(r['standardized_squared_error']),
                number(100*r['coverage_2std'], 2)+'%', number(r['duration_ce'])])
    return paragraphs, (['划分', '先验', '条件 delta MSE', '平均预测 σ', '平均 z²', '±2σ 覆盖', 'Duration CE'], rows)


def plot_clip(destination, cid, cell, metadata, global_example, local_example):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    # Plot labels are English to keep SVG portable without a bundled CJK font;
    # the containing Chinese HTML supplies group names and interpretation.
    samples = np.asarray(local_example['samples'], dtype=np.float64)
    global_samples = np.asarray(global_example['samples'], dtype=np.float64)
    valid = np.asarray(local_example['valid'], bool)
    time = np.arange(len(valid), dtype=np.float64)/25.
    def mask(x):
        return np.where(valid, x, np.nan)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8.8), sharex=True, layout='constrained')
    for g, ax in enumerate(axes):
        lo, hi = samples[:, :, g].min(0), samples[:, :, g].max(0)
        ax.fill_between(time, mask(lo), mask(hi), color='#4c92d5', alpha=.16,
                        label='Local: range of 8 samples')
        ax.plot(time, mask(np.asarray(local_example['target'])[:, g]), color='#172431', linewidth=1.65, label='Tracked GT')
        ax.plot(time, mask(np.asarray(local_example['teacher'])[:, g]), color='#dd941e', linewidth=1.35,
                linestyle='--', label='Motion teacher (oracle)')
        ax.plot(time, mask(samples[:, :, g].mean(0)), color='#126bbb', linewidth=1.7, label='Local audio: sample mean')
        ax.plot(time, mask(global_samples[:, :, g].mean(0)), color='#bf5265', linewidth=1.3,
                label='Global only: sample mean')
        ax.set_ylabel(GROUPS[g]+'\nraw coefficient')
        ax.grid(alpha=.17)
        # Use each group's observed display range. Forcing a 0..1 axis hides
        # subtle motion; physical bound markers must not expand that range.
        limits = ax.get_ylim()
        for bound in (0., 1.):
            if limits[0] <= bound <= limits[1]:
                ax.axhline(bound, color='#8a929a', linewidth=.5, linestyle=':')
        ax.set_ylim(limits)
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, 1.35), ncol=3, fontsize=8, frameon=False)
    axes[0].set_title(f'{cell} | {cid} | speaker {metadata["speaker"]} | sentence {metadata["sentence"]} | emotion {metadata["emotion"]}',
                      fontsize=10, pad=44)
    axes[-1].set_xlabel('Native time (seconds), 25 Hz; gaps are blank; raw values are not clipped')
    file_format = Path(destination).suffix.lower().lstrip('.')
    fig.savefig(destination, format=file_format, metadata={'Date': None} if file_format == 'svg' else None)
    plt.close(fig)


def package(root):
    root = Path(root).resolve()
    status, representation, protocol, gate, matching, reports, examples, selection = load_inputs(root)
    plots = root/'plots'; plots.mkdir(exist_ok=True)
    selected_rows = []
    for cell in CELLS:
        lookup = {row['clip_id']: row for row in protocol['split'][cell]}
        for position, cid in enumerate(selection[cell], 1):
            relative = f'plots/{cell}_{position:02d}.svg'
            plot_clip(root/relative, cid, cell, lookup[cid], examples['global_only'][cell][cid], examples['local_audio'][cell][cid])
            selected_rows.append({'cell': cell, 'clip_id': cid, 'file': relative, **lookup[cid]})
    failures = gate_failures(gate, reports)
    main_point = ('持续动作表示通过，音频先验通过预注册准入；尚未接入最终生成器。' if gate['passed'] else
                  '持续动作表示通过，但音频先验未通过预注册准入；尚未接入最终生成器。') if representation['passed'] else \
                 '持续动作表示未通过预注册准入；本页不构成动态生成成功。'
    counts = '，'.join(('训练' if k == 'fit' else CELL_LABELS[k])+str(len(v))+'条' for k, v in protocol['split'].items())
    scope = '只读出抬眉、压眉、眯眼、睁大四组控制量，不覆盖眨眼、视线或头部。旧口型、身份和全局情感模型本轮未更新；没有进行最终面部质量评估。'
    exposure = '本轮新模块只用 fit 划分训练；冻结上游表示存在历史源数据曝光，因此留身份／留句是新模块的留出验证，不能称整个系统从未见过这些来源。405 开发集与封存 test 未索引。'
    sampling = '九条固定曲线取每个留出划分按 clip_id 排序的前三条。蓝色阴影是 8 个固定采样结果的最小—最大范围，不是置信区间；蓝／红实线是采样均值，不是单次可部署轨迹。橙色 teacher 使用真实动作，只衡量表示重构。'
    metric_scope = '自由生成仅用音频、独立参考锚点和自己生成的历史；NLL 使用 teacher 边界／状态，且只评价完整可观察控制段，不能代替自由生成结果。ES／variogram 在固定训练尺度中计算，表内箭头为 Global → Local，越低越好。动态指标按每个连续有效片段中心化；原曲线不做 DC 校准，不裁到 [0,1]。'
    rms_note = '上表 RMS/GT 是先逐 clip 求比值再平均的原准入指标，接近静止的 GT 会产生很小的分母。因此 118.54 不能解释为整体动作幅度扩大 118 倍；原指标和 gate 保留，不用事后诊断替换。'
    rms_table = None
    if (root/'rms_denominator_diagnostic.json').is_file():
        diagnostic = read(root/'rms_denominator_diagnostic.json')
        pooled = []
        for cell in CELLS:
            a, b = diagnostic['global_only/'+cell], diagnostic['local_audio/'+cell]
            for g, label in enumerate(GROUP_LABELS):
                pooled.append([CELL_LABELS[cell], label, number(a['pooled_rms_ratio'][g]),
                    number(b['pooled_rms_ratio'][g]),
                    str(int(b['gt_dynamic_rms_below_point001_clips'][g]))+'/'+str(b['clips'])])
        rms_table = (['划分', '动作组', 'Global 汇聚 RMS/GT', 'Local 汇聚 RMS/GT', 'GT 动态 RMS < 0.001 的 clip'], pooled)
    table_items = tables(representation, gate, reports)
    failure_diagnosis = optional_failure_diagnosis(root)
    hours = float(status.get('seconds', 0))/3600
    markdown = ['# 持续动作过程与音频先验：30 轮评审', '', main_point, '',
                f'两臂各 30 轮，匹配 {matching["local_audio"]["updates"]} 次更新和样本顺序；本次总耗时 {hours:.2f} 小时。{counts}。', '',
                scope, '', exposure, '', '## 准入结果', '',
                f'- 动作表示：{"通过" if representation["passed"] else "未通过"}。',
                f'- 音频先验：{"通过" if gate["passed"] else "未通过"}。',
                '- 最终生成器：未启动；默认模型未替换。', '']
    if failures:
        markdown += ['未通过的具体条件：', '']+['- '+v for v in failures]+['']
    markdown += ['## 表示重构（Oracle）', '', markdown_table(*table_items[0]), '',
        'Teacher 和均匀线段对照都读取真实动作。总段率为四组相加；实值 delta 的参数率不是编码 bit rate。', '',
        '## 音频预测', '', metric_scope, '', markdown_table(*table_items[1]), '', markdown_table(*table_items[2]), '', rms_note, '']
    if rms_table is not None:
        markdown += ['以下是汇聚动态能量后再求 RMS 比值的独立分母诊断：', '', markdown_table(*rms_table), '']
    if failure_diagnosis is not None:
        paragraphs, diagnostic_table = failure_diagnosis
        markdown += ['## 失败诊断：最终模型真实前向重算', '']
        for paragraph in paragraphs:
            markdown += [paragraph, '']
        markdown += [markdown_table(*diagnostic_table), '']
    markdown += ['## 固定示例', '', sampling, '', f'固定 seeds：{list(SEEDS)}。', '']
    for row in selected_rows:
        markdown += [f'- {CELL_LABELS[row["cell"]]} `{row["clip_id"]}`：[四组曲线]({quote(row["file"])})。']
    links = ['status.json', 'protocol.json', 'representation.json', 'audio_gate.json', 'matching.json']
    links += [f'{arm}/{cell}{suffix}' for arm in ARMS for cell in CELLS for suffix in ('.json', '.pt')]
    for name in ('local_audio/common_interventions.json', 'local_audio/static.json', 'local_audio/reverse.json',
                 'local_audio/mismatch.json', 'global_only/losses.json', 'local_audio/losses.json',
                 'rms_denominator_diagnostic.json', 'failure_diagnosis_v2/report.json',
                 'failure_diagnosis_v2/complete.json', 'result_audit/audit.json', 'result_audit/manifest.json'):
        if (root/name).is_file():
            links.append(name)
    markdown += ['', '## 原始记录', '']+['- ['+name+']('+quote(name)+')' for name in links]+['']
    (root/'RESULTS.md').write_text('\n'.join(markdown), encoding='utf8')
    panels = ''
    for row in selected_rows:
        panels += '<article><h3>'+html.escape(CELL_LABELS[row['cell']]+' · '+row['clip_id'])+'</h3><p>抬眉 / 压眉 / 眯眼 / 睁大；原始幅值，25 Hz。</p><img loading="lazy" src="'+quote(row['file'])+'" alt="'+html.escape(row['clip_id'])+' 四组动作曲线"></article>'
    failure_html = '<ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in failures)+'</ul>' if failures else ''
    diagnosis_html = ''
    if failure_diagnosis is not None:
        paragraphs, diagnostic_table = failure_diagnosis
        diagnosis_html = '<h2>失败诊断：最终模型真实前向重算</h2>'+''.join('<p>'+html.escape(x)+'</p>' for x in paragraphs)+html_table(*diagnostic_table)
    link_html = '<ul class="links">'+''.join('<li><a href="'+quote(p)+'">'+html.escape(p)+'</a></li>' for p in ['RESULTS.md']+links)+'</ul>'
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>持续动作先验 · 30 轮评审</title><style>'+\
        'body{margin:0;background:#f1f4f6;color:#223142;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1420px;margin:auto;padding:28px}h1{font-size:29px;line-height:1.35}h2{margin-top:36px}h3{font-size:17px;overflow-wrap:anywhere}p{max-width:1160px}.lead{font-size:20px;font-weight:650}.card,article{background:#fff;border:1px solid #dde4eb;border-radius:12px;padding:22px;margin:18px 0}.badge{display:inline-block;padding:5px 12px;border-radius:15px;background:#e9eef3;margin:0 8px 8px 0}.pass{background:#e0f2e8}.fail{background:#ffeadb}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{border-bottom:1px solid #dfe5eb;padding:9px 11px;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{background:#edf2f7}img{display:block;width:100%;height:auto}a{color:#1465a7}.links{column-count:2;overflow-wrap:anywhere}.note{color:#596b7e}@media(max-width:700px){main{padding:14px}.card,article{padding:14px}.links{column-count:1}}'+\
        '</style><main><h1>持续动作过程与音频先验 · 30 轮评审</h1><p class="lead">'+html.escape(main_point)+'</p>'+\
        '<span class="badge '+('pass' if representation['passed'] else 'fail')+'">表示 '+('通过' if representation['passed'] else '未通过')+'</span>'+\
        '<span class="badge '+('pass' if gate['passed'] else 'fail')+'">音频准入 '+('通过' if gate['passed'] else '未通过')+'</span><span class="badge">未接最终生成器</span>'+\
        '<p>'+html.escape(f'两臂各 30 轮，{matching["local_audio"]["updates"]} 次更新／臂；总耗时 {hours:.2f} 小时。{counts}。')+'</p>'+\
        '<div class="card"><p>'+html.escape(scope)+'</p><p>'+html.escape(exposure)+'</p>'+failure_html+'</div>'+\
        '<h2>表示重构：真实动作构造的 Oracle</h2><p>四组持久线段可重叠，具有变时长状态。此表说明表示是否保留动态，不证明音频能预测。</p>'+html_table(*table_items[0])+\
        '<h2>音频预测与自由生成</h2><p>'+html.escape(metric_scope)+'</p>'+html_table(*table_items[1])+'<br>'+html_table(*table_items[2])+\
        '<p>'+html.escape(rms_note)+'</p>'+('' if rms_table is None else '<p>补充：汇聚动态能量后计算的 RMS 比值，仅作分母诊断。</p>'+html_table(*rms_table))+\
        diagnosis_html+\
        '<h2>九条固定曲线</h2><p>'+html.escape(sampling)+'</p><p class="note">固定 seeds：'+html.escape(str(list(SEEDS)))+'。这不是最终面部视频，也没有重新测量嘴型、身份或全局情感。</p>'+panels+\
        '<h2>原始记录</h2>'+link_html+'</main></html>'
    (root/'index.html').write_text(document, encoding='utf8')
    manifest = {'schema': 'motion_process_review_v1', 'selection_policy': 'Three lexicographically first clip IDs per holdout cell',
        'examples': selected_rows, 'fps': 25, 'seeds_from_training_recipe': list(SEEDS),
        'display': 'raw group values; eight-sample range is not confidence interval; no clipping or DC',
        'generator_integrated': False, 'input_sha256': {p: sha(root/p) for p in links},
        'output_sha256': {p: sha(root/p) for p in ['RESULTS.md', 'index.html']+[r['file'] for r in selected_rows]}}
    (root/'review_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return {'root': str(root), 'examples': len(selected_rows), 'representation_passed': representation['passed'],
            'audio_passed': gate['passed'], 'index': str(root/'index.html')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.root), ensure_ascii=False))


if __name__ == '__main__':
    main()
