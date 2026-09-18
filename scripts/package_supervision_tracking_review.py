"""Package completed reset/order diagnostic reports without model inference.

Only saved summary, provenance, selection, arrays and fixed visual artifacts
are read. New output consists of index.html, review_summary.json and
review_manifest.json; original audit outputs are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np


SCHEMA = 'tracking_reset_order_diagnostic_v1'
COMPARISONS = {
    'reset_forward': '正序：新实例 vs 复用实例',
    'reset_reverse': '逆序：新实例 vs 复用实例',
    'fresh_order_timestamp_control': '新实例：两种片段顺序／时间戳',
    'reused_order': '复用实例：两种片段顺序',
    'historical_raw_vs_fresh': '历史 raw vs 新实例 raw',
    'historical_final_vs_fresh_raw_processing_differs': '历史 final vs 新实例 raw（处理不同）',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def fmt(value, digits=5):
    return '无支持' if value is None else f'{float(value):.{digits}f}'


def percent(value):
    return '无支持' if value is None else f'{100*float(value):.2f}%'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+\
        '</tbody></table></div>'


def aggregate(records, comparison_name):
    """Clip-equal descriptive aggregates; no pooled-percentile substitution."""
    comps = [r['comparisons'][comparison_name] for r in records.values()]
    supported = [r for r in comps if r['overall'] is not None]
    if not supported:
        return {'clips': len(comps), 'supported_clips': 0}
    values = lambda key: np.asarray([r['overall'][key] for r in supported], dtype=np.float64)
    activity = [r['activity_label_disagreement'] for r in comps if r['activity_label_disagreement'] is not None]
    return {'clips': len(comps), 'supported_clips': len(supported),
            'common_valid_frames': sum(r['common_valid'] for r in comps),
            'median_clip_rms': float(np.median(values('rms'))),
            'median_clip_p95_abs': float(np.median(values('p95_abs'))),
            'max_clip_p95_abs': float(values('p95_abs').max()),
            'max_abs': float(values('max_abs').max()),
            'clips_p95_over_002': int((values('p95_abs') > .02).sum()),
            'clips_max_over_01': int((values('max_abs') > .1).sum()),
            'activity_supported_clips': len(activity),
            'mean_clip_activity_disagreement': float(np.mean(activity)) if activity else None,
            'max_clip_activity_disagreement': float(np.max(activity)) if activity else None,
            'clips_activity_disagreement_over_005': sum(value > .05 for value in activity)}


def package(root):
    root = Path(root).resolve()
    for name in ('index.html', 'review_summary.json', 'review_manifest.json'):
        if (root/name).exists():
            raise FileExistsError('New presentation outputs required; existing file will not be replaced: '+name)
    status, summary, provenance, selection = [read(root/name) for name in
        ('status.json', 'summary.json', 'provenance.json', 'selection.json')]
    if (status.get('status') != 'complete' or any(v.get('schema') != SCHEMA for v in (status, summary, provenance))
            or summary.get('independent_ground_truth') is not False
            or provenance.get('same_model_landmark_proxy_is_independent_ground_truth') is not False
            or provenance.get('source_files_modified') is not False
            or provenance.get('fresh_vs_reused_same_absolute_timestamps') is not True):
        raise ValueError('Completed, source-preserving same-clock reset audit required')
    if summary['provenance_sha256'] != sha(root/'provenance.json') or provenance['selection_sha256'] != sha(root/'selection.json'):
        raise ValueError('Provenance or selection binding differs')
    records = summary['clips']; ids = [row['clip_id'] for row in selection['clips']]
    if len(ids) != len(set(ids)) or set(ids) != set(records) or len(ids) != status['clips'] or len(ids) != summary['clip_count']:
        raise ValueError('Selected/completed clip membership differs')
    original_manifest = read(root/'manifest.json')
    checked_sources = {}
    needed = ['status.json', 'summary.json', 'provenance.json', 'selection.json', 'activity_thresholds.json']
    needed += ['visual/'+cid+suffix for cid in ids for suffix in ('_contact.jpg', '_curves.png')]
    for name in needed:
        path = root/name
        if not path.is_file() or path.stat().st_size == 0 or name not in original_manifest:
            raise ValueError('Missing completed report/visual: '+name)
        digest = sha(path)
        if digest != original_manifest[name]['sha256'] or path.stat().st_size != original_manifest[name]['bytes']:
            raise ValueError('Saved report/visual differs from completed manifest: '+name)
        checked_sources[name] = digest
    flags, unsupported, summaries, detail_rows, sections = [], [], {}, [], []
    for name in COMPARISONS:
        summaries[name] = aggregate(records, name)
    for index, cid in enumerate(ids, 1):
        row = records[cid]
        if row['same_decoded_frames_all_arms'] is not True:
            raise ValueError('Frame identity was not verified: '+cid)
        comps = row['comparisons']
        reset = [comps[name] for name in ('reset_forward', 'reset_reverse')]
        flagged = any((c['overall'] is not None and (c['overall']['p95_abs'] > .02 or c['overall']['max_abs'] > .1))
                      or (c['activity_label_disagreement'] is not None and c['activity_label_disagreement'] > .05) for c in reset)
        missing = any(c['overall'] is None for c in reset)
        if flagged != row['reset_warrants_further_investigation'] or missing != row['no_shared_valid_support']:
            raise ValueError('Saved reset flag differs from fixed thresholds: '+cid)
        if flagged: flags.append(cid)
        if missing: unsupported.append(cid)
        max_p95 = max((c['overall']['p95_abs'] for c in reset if c['overall']), default=None)
        max_abs = max((c['overall']['max_abs'] for c in reset if c['overall']), default=None)
        max_disagreement = max((c['activity_label_disagreement'] for c in reset if c['activity_label_disagreement'] is not None), default=None)
        detail_rows.append([index, cid, row['frames'], percent(row['fresh_detection_rate']), fmt(max_p95), fmt(max_abs),
                            percent(max_disagreement), '需进一步核验' if flagged else '本次未越阈值'])
        individual = []
        for name, label in COMPARISONS.items():
            c = comps[name]; whole, first, late = c['overall'], c['first8'], c['after8']
            individual.append([label, c['common_valid'], fmt(whole['rms'] if whole else None),
                fmt(whole['p95_abs'] if whole else None), fmt(whole['max_abs'] if whole else None),
                fmt(first['rms'] if first else None), fmt(late['rms'] if late else None), percent(c['activity_label_disagreement'])])
        coupling_rows = []
        for name, label in [('coupling_historical_final_to_fresh_geometry', '历史 final 与新几何代理'),
                            ('coupling_fresh_coefficients_to_same_geometry', '新 raw 与同模型几何代理')]:
            c = row[name]
            for j, group in enumerate(('抬眉', '压眉', '眯眼', '睁大')):
                coupling_rows.append([label, group, fmt(c['group_geometry_level_correlation'][j], 3),
                    fmt(c['group_geometry_signed_speed_correlation'][j], 3),
                    fmt(c['group_blink_absolute_speed_correlation'][j], 3), fmt(c['group_head_speed_correlation'][j], 3)])
        metadata = row['metadata']; safe = html.escape(cid)
        sections.append('<section id="clip'+str(index)+'" class="card"><h2>'+str(index)+' · '+safe+'</h2><p class="small">'+
            html.escape(str(metadata.get('speaker_name', metadata.get('speaker'))))+' · '+str(row['frames'])+' 帧 · '+fmt(row['frames']/25, 2)+
            ' 秒 · '+('达到进一步核验阈值' if flagged else '此次未超过重置阈值')+'</p>'+
            '<p>16 个均匀时刻的固定原视频裁剪。没有按动作峰或预测好坏挑帧。</p><a href="visual/'+safe+'_contact.jpg"><img loading="lazy" src="visual/'+safe+'_contact.jpg" alt="固定原视频眉眼接触表"></a>'+
            '<p>上两排：历史 final 与 fresh raw 系数；第三排：同模型几何比值；末排：同模型头旋转。两种系数的处理链不同，不能只凭差异判定正确的一方。</p>'+
            '<a href="visual/'+safe+'_curves.png"><img loading="lazy" src="visual/'+safe+'_curves.png" alt="固定眉眼和几何曲线"></a>'+
            table(['对比','共同帧','RMS','p95绝对差','最大绝对差','首8帧RMS','后续RMS','活动标签分歧'],individual)+
            '<details><summary>几何／眨眼／头姿同现关系</summary><p class="small">零时延相关，不拟合增益；高度相关可能来自真实协同，也可能来自同模型耦合，均不能单独判断标签正确。抬眉/压眉/眯眼/睁大对几何代理的预期符号分别为 +/−/−/+。</p>'+
            table(['信号','组','几何水平相关','几何带符号速度相关','眨眼速度相关','头动速度相关'],coupling_rows)+'</details></section>')
    if set(flags) != set(summary['clips_warranting_reset_investigation']) or set(unsupported) != set(summary['clips_without_shared_support']):
        raise ValueError('Aggregate reset flags differ from completed audit')
    report = {'schema': 'tracking_reset_review_summary_v1', 'clips': len(ids), 'frames': sum(records[c]['frames'] for c in ids),
        'seconds': summary['seconds'], 'thresholds': {'clip_p95_abs': .02, 'clip_max_abs': .1, 'clip_activity_label_disagreement': .05},
        'activity_energy_thresholds': provenance['activity_thresholds'], 'clip_equal_descriptive_aggregates': summaries,
        'flagged_clip_ids': flags, 'unsupported_clip_ids': unsupported,
        'independent_ground_truth': False, 'no_inference_or_training': True,
        'qualitative_review_completed_by_packager': False}
    headers = ['比较','可评分片段','片级p95中位数','片级p95最大','最大绝对差','片均标签分歧','分歧>5%片段']
    aggregate_rows = [[label, s['supported_clips'], fmt(s.get('median_clip_p95_abs')), fmt(s.get('max_clip_p95_abs')),
        fmt(s.get('max_abs')), percent(s.get('mean_clip_activity_disagreement')), s.get('clips_activity_disagreement_over_005', 0)]
        for name,label in COMPARISONS.items() for s in [summaries[name]]]
    outcome = ('发现 '+str(len(flags))+' / '+str(len(ids))+' 段超过至少一项预设重置影响阈值。' if flags else
               '这 '+str(len(ids))+' 段未超过预设重置影响阈值。')
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定32段 · 眉眼监督与跟踪重置诊断</title><style>
*{box-sizing:border-box}body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f2f5fa;color:#24364a;margin:0}main{max-width:1280px;margin:auto;padding:30px 22px 60px}h1{font-size:28px;line-height:1.4}h2{font-size:21px;overflow-wrap:anywhere}.lead{background:#fff1cf;border-left:5px solid #b67a17;padding:18px 22px;border-radius:6px}.card,.notes{background:white;padding:20px;border:1px solid #dce5ee;border-radius:10px;margin-top:22px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px;border-bottom:1px solid #e1e8ef;text-align:left;white-space:nowrap}th{background:#edf3f8}.small{font-size:13px;color:#52677c}img{display:block;max-width:100%;height:auto;margin:14px auto}a{color:#126298}details{margin-top:16px}summary{cursor:pointer;font-weight:600}nav{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}nav a{background:white;border:1px solid #dce5ee;border-radius:14px;padding:3px 10px}li{margin:6px 0}</style></head><body><main><h1>固定片段 · 眉眼监督与跟踪重置诊断</h1>'''+\
        '<div class="lead"><b>'+outcome+'</b><p>这项结果仅检验 MediaPipe VIDEO 跨片段状态和顺序敏感性，不能证明哪一条轨迹是真值，也不能断言它解释全部训练失败。</p><p>眉眼几何代理与 blendshape 来自同一个模型；独立人工标注或第二提取器的监督质量证据仍未完成。</p></div>'+\
        '<div class="notes"><h2>实验边界</h2><ul><li>固定 '+str(len(ids))+' 段、'+str(len({r.get('sentence') for r in selection['clips']}))+' 句，来自4个拟合身份、4类情感；非中性覆盖 L1/L3。仅按元数据选取，不按模型效果选片。</li>'+\
        '<li>四臂为 fresh_forward、reused_forward、fresh_reverse、reused_reverse。同一顺序内新实例和复用实例收到完全相同的像素与绝对时间戳；逆序指片段处理顺序，不是倒放每段视频。</li>'+\
        '<li>预设进一步核验阈值：同片 p95 绝对系数差 >0.02，或最大差 >0.1，或活动标签分歧 >5%。每片任一正序/逆序配对超过即标记；未超过不等于标签正确。</li>'+\
        '<li>活动定义沿用3帧平滑、中心8次帧差的 RMS；四组阈值固定为 '+', '.join(fmt(x,5) for x in provenance['activity_thresholds'])+' 系数/秒，使用原拟合阈值，不从本次32段重新拟合。</li>'+\
        '<li>原始检测无效帧保持掩码。历史 final 文件没有mask，显式继承历史 raw 检测mask；插值填帧不计为新观测。历史 final 已做 SG5 与裁剪，不能把它与新 raw 的差异直接归因于重置。</li>'+\
        '<li>所有原始数据和默认训练权重保持不变。本次完成耗时 '+fmt(summary['seconds']/60,2)+' 分钟。</li></ul></div>'+\
        '<section class="notes"><h2>整组描述统计</h2><p class="small">p95 中位数是“每片 p95 的中位数”，不是混合全部帧后计算的 p95；活动分歧按片均分，缺少窗口的片段不计分歧均值。</p>'+table(headers,aggregate_rows)+\
        '<h2>逐片重置影响</h2><p class="small">下表p95、最大差、活动分歧各自取正序/逆序配对中的较大值。</p>'+\
        table(['编号','clip_id','帧数','fresh检测率','p95绝对差','最大差','活动标签分歧','判定'],detail_rows)+'</section>'+\
        '<nav aria-label="固定片段索引">'+''.join('<a href="#clip'+str(i)+'">'+str(i)+'</a>' for i in range(1,len(ids)+1))+'</nav>'+\
        ''.join(sections)+'<footer class="notes small">此页从保存结果生成，没有模型推理、重新选择样本或拟合。'+\
        '<a href="review_summary.json">本页汇总</a> · <a href="summary.json">原始逐片报告</a> · <a href="provenance.json">来源与选项</a> · <a href="review_manifest.json">本页校验</a></footer></main></body></html>'
    write(root/'review_summary.json', report)
    (root/'index.html').write_text(page, encoding='utf8')
    write(root/'review_manifest.json', {'schema': 'tracking_reset_review_manifest_v1', 'input_sha256': checked_sources,
        'original_manifest_sha256': sha(root/'manifest.json'), 'script_sha256': sha(__file__),
        'no_model_or_dataset_loaded': True, 'outcome_based_selection': False, 'all_selected_clips_displayed': True,
        'output_sha256': {name: sha(root/name) for name in ('index.html', 'review_summary.json')}})
    return {'root': str(root), 'clips': len(ids), 'flagged': len(flags), 'unsupported': len(unsupported)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.root), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
