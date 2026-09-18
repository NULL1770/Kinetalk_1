"""Render all prelocked reference-controlled queries without fitting or selection.

The page displays seed42 (chosen by protocol before generation), while score
tables contain all saved seeds. Group curves average channels, never draws.
Invalid observations break lines instead of connecting through a tracking gap.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch


SCHEMA = 'independent_reference_continuous_prior_v1'
INPUTS = ('protocol.json', 'status.json', 'query_selection.json', 'references.json',
          'reports.json', 'long_controls.json', 'predictions.pt', 'fit_dynamics_reference.json')
ARMS = ('medoid', 'process', 'reference_swap')
NAMES = {'medoid': '有界 medoid 先验', 'process': '连续随机先验', 'reference_swap': '更换独立表达参考'}
PLOT_NAMES = {'medoid': 'Bounded medoid, seed42', 'process': 'Continuous process, seed42',
              'reference_swap': 'Reference swap, seed42'}
COLORS = {'medoid': '#a66a28', 'process': '#176db8', 'reference_swap': '#963fa8'}
GROUPS = ((2, 3, 4), (0, 1), (5, 7), (6, 8))
GROUP_NAMES = ('抬眉', '压眉', '眯眼', '睁眼')
GROUP_PLOT = ('Brow raise', 'Brow down', 'Eye squint', 'Eye wide')
GAIN_KEYS = ('0.0', '0.5', '1.0', '1.5')
GAIN_COLORS = ('#263444', '#64a385', '#176db8', '#c25445')
EMOTIONS = {0: '中性', 1: '愤怒', 5: '开心', 6: '悲伤'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def fmt(value, digits=4):
    if value is None:
        return '—'
    return f'{float(value):.{digits}f}' if np.isfinite(float(value)) else '非有限值'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+\
        '</tbody></table></div>'


def safe_id(cid):
    if not isinstance(cid, str) or not cid or '/' in cid or '\\' in cid or cid in ('.', '..'):
        raise ValueError('Clip ID must be a safe basename')
    return cid


def validate_predictions(protocol, status, selection, references, reports, predictions, long_rows):
    """Bind membership and conditions; this is not an independent score audit."""
    if (protocol.get('schema') != SCHEMA or status.get('schema') != SCHEMA
            or predictions.get('schema') != SCHEMA or status.get('status') != 'complete'
            or status.get('smoke') is not False or protocol.get('smoke') is not False
            or status.get('test_loaded') is not False or protocol.get('test_loaded') is not False
            or status.get('default_replaced') is not False
            or protocol.get('extra_expression_reference') is not True
            or protocol.get('audio_timing_claim') is not False
            or status.get('audio_timing_proven') is not False):
        raise ValueError('Completed formal extra-reference run with no test/audio timing claim required')
    ids = protocol['query_clip_ids']; picks = selection['queries']; seeds = protocol['seeds']
    if (len(ids) != 32 or len(set(ids)) != len(ids) or [row['clip_id'] for row in picks] != ids
            or set(predictions['curves']) != set(ids) or set(references) != set(ids)
            or status.get('queries') != len(ids) or len(seeds) != 8 or len(set(seeds)) != 8 or seeds[0] != 42
            or protocol.get('activity_gains') != [0., .5, 1., 1.5]):
        raise ValueError('All32 fixed queries, all8 seeds and four preset gains must be present')
    expected_long = {}
    fit_ids = set(protocol['fit_clip_ids'])
    if fit_ids & set(ids):
        raise ValueError('Fit/query ID overlap')
    for metadata in picks:
        cid = safe_id(metadata['clip_id']); row = predictions['curves'][cid]
        if any(row['metadata'][key] != metadata[key] for key in ('clip_id', 'speaker', 'sentence', 'emotion')):
            raise ValueError('Saved query metadata differs from locked selection')
        valid = array(row['valid']); target = array(row['target'])
        if (valid.dtype != np.bool_ or valid.ndim != 1 or not valid.any()
                or target.shape != (len(valid), 9) or not np.isfinite(target[valid]).all()):
            raise ValueError('Finite observed target and Boolean native mask required')
        for arm in ARMS:
            values = array(row['samples'][arm])
            if values.shape != (len(seeds), len(valid), 9) or not np.isfinite(values[:, valid]).all():
                raise ValueError('Missing/invalid all-seed draws: '+arm)
        if set(row['gain_controls_seed42']) != set(GAIN_KEYS):
            raise ValueError('Four predeclared gain controls required')
        for values in row['gain_controls_seed42'].values():
            values = array(values)
            if values.shape != target.shape or not np.isfinite(values[valid]).all():
                raise ValueError('Invalid seed42 gain control')
        reference = references[cid]
        refmeta = reference['reference_metadata']
        if (len(refmeta) != 2 or [r['clip_id'] for r in refmeta] != reference['reference_ids']
                or len(reference['references']) != 2
                or len({r['sentence'] for r in refmeta}) != 2):
            raise ValueError('Two independently bound reference sentences required')
        for ref, encoded in zip(refmeta, reference['references']):
            if (ref['clip_id'] not in fit_ids or ref['clip_id'] == cid
                    or ref['speaker'] != metadata['speaker'] or ref['sentence'] == metadata['sentence']
                    or encoded['source_clip_id'] != ref['clip_id']
                    or array(encoded['style']).shape != (5,) or not np.isfinite(encoded['style']).all()):
                raise ValueError('Independent fit reference/style binding differs')
        expected_long.setdefault(str(metadata['speaker']), cid)
    for arm in ARMS:
        rows = reports[arm]['rows']
        if ([row['clip_id'] for row in rows] != ids or reports[arm]['summary']['clips'] != len(ids)
                or any(row['sample_count'] != len(seeds) for row in rows)):
            raise ValueError('Summary rows must contain all fixed queries and all seeds')
    if set(long_rows) != set(expected_long) or set(predictions['long']) != set(expected_long.values()):
        raise ValueError('First metadata query per person must supply long controls')
    for speaker, cid in expected_long.items():
        long = predictions['long'][cid]
        if set(long) != {'gain_'+key for key in GAIN_KEYS} | {'controls'}:
            raise ValueError('All long seed42 control arrays required')
        for name, values in long.items():
            values = array(values)
            if values.shape != (512 if name == 'controls' else 1500, 9) or not np.isfinite(values).all():
                raise ValueError('Native25Hz long control shape differs')
        expected = {(seed, 'stationary_60s', gain) for seed in seeds for gain in protocol['activity_gains']}
        expected |= {(seed, 'run_hold_release_resume_swap', None) for seed in seeds}
        keys = [(r['seed'], r['kind'], r.get('gain')) for r in long_rows[speaker]]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('All8 seeds and four gains must be reported for long controls')
    return picks


def load_run(root):
    root = Path(root).resolve()
    hashes = {name: sha(root/name) for name in INPUTS}
    data = {name[:-5]: read(root/name) for name in INPUTS if name.endswith('.json')}
    predictions = torch.load(root/'predictions.pt', map_location='cpu', weights_only=False)
    picks = validate_predictions(data['protocol'], data['status'], data['query_selection'],
                                 data['references'], data['reports'], predictions, data['long_controls'])
    return {'root': root, 'hashes': hashes, 'predictions': predictions, 'picks': picks, **data}


def group_curves(values, valid=None):
    """Channel means, preserving native time and explicit gaps."""
    values = array(values)
    if values.ndim != 2 or values.shape[-1] != 9:
        raise ValueError('Expected one raw[T,9] trajectory')
    result = np.stack([values[:, list(group)].mean(-1) for group in GROUPS], axis=-1).astype(float)
    if valid is not None:
        valid = array(valid)
        if valid.dtype != np.bool_ or valid.shape != values.shape[:1]:
            raise ValueError('Boolean native mask required')
        result[~valid] = np.nan
    return result


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    return plt


def plot_query(path, cid, row):
    plt = plotting(); valid = array(row['valid'])
    target = group_curves(row['target'], valid)
    # Seed42 is the first protocol draw; never use a favorable sampled draw or
    # the ensemble mean, which would hide generated motion in this experiment.
    curves = {arm: group_curves(row['samples'][arm][0], valid) for arm in ARMS}
    gains = {key: group_curves(row['gain_controls_seed42'][key], valid) for key in GAIN_KEYS}
    time = np.arange(len(valid))/25.
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True, constrained_layout=True)
    for group in range(4):
        left, right = axes[group]
        left.plot(time, target[:, group], color='#222222', lw=1.6, label='Tracked GT')
        for arm, values in curves.items():
            left.plot(time, values[:, group], color=COLORS[arm], lw=1.2, label=PLOT_NAMES[arm])
        for (key, values), color in zip(gains.items(), GAIN_COLORS):
            right.plot(time, values[:, group], color=color, lw=1.2, label='Gain '+key+', seed42')
        combined = np.concatenate([target[:, group]]+[v[:, group] for v in curves.values()]+[v[:, group] for v in gains.values()])
        low, high = np.nanmin(combined), np.nanmax(combined); pad = max((high-low)*.08, .005)
        for axis in (left, right):
            axis.set_ylim(max(0., low-pad), min(1., high+pad)); axis.grid(alpha=.18)
            axis.set_ylabel(GROUP_PLOT[group]+'\nraw channel mean')
        if group == 0:
            left.legend(ncol=2, fontsize=8, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.02))
            right.legend(ncol=2, fontsize=8, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.02))
    axes[-1, 0].set_xlabel('Native clip time (seconds; 25 Hz)')
    axes[-1, 1].set_xlabel('Native clip time (seconds; 25 Hz)')
    fig.suptitle(cid+' | fixed seed42; channel means, no sample averaging', fontsize=11)
    fig.savefig(path, format='svg', metadata={'Date': None}); plt.close(fig)


def plot_long(path, cid, row):
    plt = plotting()
    gains = {key: group_curves(row['gain_'+key]) for key in GAIN_KEYS}
    control = group_curves(row['controls'])
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), constrained_layout=True)
    phases = ((0, 128, 'RUN', '#e2edf8'), (128, 192, 'HOLD', '#ece7cc'),
              (192, 256, 'RELEASE', '#dfece1'), (256, 384, 'RUN', '#e2edf8'),
              (384, 512, 'REF SWAP', '#edddf0'))
    for group in range(4):
        left, right = axes[group]
        for (key, values), color in zip(gains.items(), GAIN_COLORS):
            left.plot(np.arange(len(values))/25., values[:, group], color=color, lw=1., label='Gain '+key)
        right.plot(np.arange(len(control))/25., control[:, group], color=COLORS['process'], lw=1.1)
        for start, end, label, color in phases:
            right.axvspan(start/25., end/25., color=color, alpha=.45, zorder=-1)
            if group == 0:
                right.text((start+end)/50., 1.02, label, transform=right.get_xaxis_transform(),
                           ha='center', fontsize=8)
        for axis in (left, right):
            axis.set_ylabel(GROUP_PLOT[group]+'\nraw channel mean'); axis.grid(alpha=.16)
        if group == 0:
            left.legend(ncol=4, fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02), frameon=False)
    axes[-1, 0].set_xlabel('60 second rollout (25 Hz); seed42')
    axes[-1, 1].set_xlabel('Continuous control schedule (25 Hz); seed42')
    fig.suptitle(cid+' | fixed seed42 long/control diagnostics', fontsize=11)
    fig.savefig(path, format='svg', metadata={'Date': None}); plt.close(fig)


def video_link(output, cid, fullface_root):
    path = Path(fullface_root)/safe_id(cid)/'comparison.mp4'
    if not path.is_file():
        return '<p class="pending">此片全脸对比视频尚未生成。</p>', None
    relative = os.path.relpath(path.resolve(), Path(output).resolve()).replace('\\', '/')
    href = quote(relative, safe='/.:')
    poster = path.with_name('preview.png')
    poster_record = None
    if poster.is_file():
        poster_href = quote(os.path.relpath(poster.resolve(), Path(output).resolve()).replace('\\', '/'), safe='/.:')
        poster_record = {'path': str(poster.resolve()), 'sha256': sha(poster), 'href': poster_href}
    player = ('<video controls preload="none"'+(
        ' poster="'+html.escape(poster_record['href'], quote=True)+'"' if poster_record else '')+
        '><source src="'+html.escape(href, quote=True)+'" type="video/mp4">当前浏览器无法播放此视频。</video>')
    return player+'<p><a href="'+html.escape(href, quote=True)+'">单独打开此片全脸对比视频</a></p>', {
        'path': str(path.resolve()), 'sha256': sha(path), 'href': href, 'poster': poster_record}


def package(run_path, output_path, fullface_root=None):
    run = load_run(run_path); output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError('Fresh output directory required')
    output.mkdir(parents=True); (output/'plots').mkdir()
    fullface_root = Path(fullface_root) if fullface_root is not None else output.parent/'fullface'
    protocol, status = run['protocol'], run['status']
    metrics = []
    for arm in ARMS:
        report = run['reports'][arm]; summary = report['summary']
        metrics.append([NAMES[arm], summary['clips'], fmt(summary['joint_fair_es']['raw']),
            fmt(summary['joint_fair_es']['centered']), ' / '.join(fmt(v, 3) for v in summary['rms_ratio']),
            fmt(report['acceleration']['rms_ratio'], 3)])
    body = '<h1>独立表达参考：连续动态审阅</h1>'
    body += '<p class="result">固定 32 段开发样例已生成；本页检查可控性和连续性，不宣称音频动态预测成功。</p>'
    body += '<!-- VIDEO_INDEX -->'
    body += '<p>生成器额外接受<strong>同一人物、不同句子的一段独立表达参考</strong>，参考只压缩为四组活动幅度和一个时间尺度，不输入查询片段的真实动作轨迹。中性身份锚点和冻结音频全局特征提供原有均衡状态；参考与随机过程提供运动。更换参考不等于让音频决定动作发生时刻。</p>'
    body += '<p><strong>自然度仍需原视频与全脸视频人工核验。</strong>系数有界、曲线会动、控制指令生效都不能单独证明自然表达、情绪正确或音频时序正确。43 个非眉眼通道是否逐值保留，需要全脸 exporter 的检查；本页面没有复算该项，也没有重新认证口型、身份和整体情感。</p>'
    body += '<p>32 段按预先记录的元数据选取，全部展示；曲线固定 seed 42，不挑 seed、不平均多次随机生成。四组曲线是组内通道均值，可能掩盖左右眼差异，不能替代九通道指标。GT 是跟踪器提取值，监督精度核验是另一条工作线。199/353 开发数据已被反复使用，本结果不能作为独立测试或论文级泛化结论。</p>'
    body += table(['已保存状态', '值'], [['拟合片段', status['fit_clips']], ['固定查询片段', status['queries']],
        ['控制保持/释放数值检查', str(status['control_exact'])], ['长序列有限且有界', str(status['finite_bounded'])],
        ['自然度验证', '尚未完成'], ['音频时序已证明', '否'], ['默认模型已替换', '否'],
        ['评分采样 seed', ', '.join(map(str, protocol['seeds']))], ['曲线显示 seed', '42（预先固定）']])
    body += '<h2>全部样例、全部采样的已保存评分</h2><p>ES 是整段生成分布相对单条跟踪轨迹的诊断，越低越好；中心化 ES 去掉每段水平偏置。RMS 比值和加速度比值只描述幅度及变化量，接近 1 不等于自然或与音频同步。这里读取原报告，不将页面整理称为独立评分审计。</p>'
    body += table(['版本', '片段数', '原始 ES ↓', '中心化 ES ↓', 'RMS 比：抬/压/眯/睁', '加速度 RMS 比'], metrics)
    manifest_examples = []; articles = []; links = []; video_index = []
    for index, metadata in enumerate(run['picks']):
        cid = metadata['clip_id']; reference = run['references'][cid]
        filename = f'plots/query_{index:02d}.svg'
        plot_query(output/filename, cid, run['predictions']['curves'][cid])
        video, video_record = video_link(output, cid, fullface_root)
        refs = []
        for ref, encoded in zip(reference['reference_metadata'], reference['references']):
            refs.append([ref['clip_id'], ref['sentence'], EMOTIONS.get(ref['emotion'], ref['emotion']),
                         ' / '.join(fmt(v, 3) for v in encoded['style'][:4]), fmt(encoded['style'][4], 3)])
        title = f'{index+1:02d} · {cid} · '+EMOTIONS.get(metadata['emotion'], str(metadata['emotion']))
        links.append('<a href="#query_'+str(index)+'">'+html.escape(title)+'</a>')
        if video_record:
            video_index.append('<a href="#query_'+str(index)+'">'+html.escape(title)+'</a>')
        articles.append('<article id="query_'+str(index)+'"><h3>'+html.escape(title)+'</h3><p>句子 '+html.escape(str(metadata['sentence']))+
            '；身份编号 '+html.escape(str(metadata['speaker']))+'；冻结音频预测情感 '+html.escape(EMOTIONS.get(reference['predicted_emotion'], str(reference['predicted_emotion'])))+
            '。表达参考按音频情感匹配：'+('是' if reference['reference_emotion_matched'] else '否，按预定规则回退')+'。</p>'+table(
                ['独立参考', '句子', '参考标签', '四组 logit RMS', '时间尺度（秒）'], refs)+
            '<img loading="lazy" src="'+filename+'" alt="GT与三种生成轨迹及四档强度控制；固定seed42">'+video+'</article>')
        manifest_examples.append({'clip_id': cid, 'plot': filename, 'seed': 42, 'video': video_record})
    video_count = len(video_index)
    body = body.replace('<!-- VIDEO_INDEX -->', '<section class="video-index"><h2>全脸视频索引</h2><p><strong>'+
        str(video_count)+' / '+str(len(manifest_examples))+' 段已渲染</strong>；另外 '+str(len(manifest_examples)-video_count)+
        ' 段尚未渲染。点击样例跳转到播放器；视频不自动播放。视频存在不代表自然度已经验收。</p>'+
        ('<nav>'+''.join(video_index)+'</nav>' if video_index else '<p>目前没有可播放的全脸视频。</p>')+'</section>')
    body += '<h2>固定样例索引</h2><nav>'+''.join(links)+'</nav><h2>原生 25 Hz 轨迹与强度控制</h2><p>左侧：跟踪 GT、有界 medoid、连续随机先验、更换独立参考。右侧：相同参考与随机噪声下的 0 / 0.5 / 1 / 1.5 强度。缺失跟踪帧留空，不跨缺口连线；同一组左右两图使用同一纵轴范围，未做结果归一化或曲线平滑。</p>'+''.join(articles)
    body += '<h2>长序列与控制切换</h2><p>每个人只取预先排序的第一段元数据查询，展示固定 seed 42；所有 8 个 seed 的数值记录完整列在下方。左图是连续 60 秒、四档强度。右图依次为 RUN 128 帧、HOLD 64 帧、RELEASE 64 帧、RUN 128 帧、更换参考 128 帧；HOLD 用 8 帧过渡，RELEASE 用 16 帧回到均衡。阴影只标示外部指令，不是由音频预测的动作事件。</p>'
    long_manifest = []
    for index, (cid, row) in enumerate(run['predictions']['long'].items()):
        filename = f'plots/long_{index:02d}.svg'; plot_long(output/filename, cid, row)
        body += '<article><h3>'+html.escape(cid)+'</h3><img loading="lazy" src="'+filename+'" alt="60秒四档强度及连续控制切换"></article>'
        long_manifest.append({'clip_id': cid, 'seed': 42, 'plot': filename})
    stationary, switches = [], []
    for speaker, rows in run['long_controls'].items():
        for row in rows:
            if row['kind'] == 'stationary_60s':
                stationary.append([speaker, row['seed'], row['gain'], ' / '.join(fmt(v, 3) for v in row['group_rms']),
                    fmt(row['velocity_p95']), fmt(row['acceleration_p95']), fmt(row['above_3hz_energy_fraction']),
                    fmt(row['activity_fraction_speed_norm_gt_005'])])
            else:
                ratios = row.get('boundary_to_fit_median_ratio', {})
                switches.append([speaker, row['seed'], fmt(row['hold_speed_max'], 9), fmt(row['release_equilibrium_max_error'], 9),
                    fmt(row['boundary_velocity_p95']), fmt(row['boundary_acceleration_p95']),
                    fmt(ratios.get('velocity_p95')), fmt(ratios.get('acceleration_p95'))])
    body += '<details><summary>全部 8 seed × 4 档强度的 60 秒数值</summary>'+table(
        ['身份', 'seed', 'gain', '四组 RMS', '速度 P95', '加速度 P95', '>3Hz 能量占比', '活动占比'], stationary)+'</details>'
    body += '<details><summary>全部 8 seed 的切换检查</summary>'+table(
        ['身份', 'seed', 'HOLD 最大逐帧差', '释放到均衡最大误差', '边界速度 P95', '边界加速度 P95', '速度/fit中位数', '加速度/fit中位数'], switches)+'</details>'
    fit = run['fit_dynamics_reference']
    body += '<p>边界统计取切换前后各 8 帧；与拟合集参考的比值仅用于工程检查。拟合参考先计算 32 帧窗口统计，再片段内平均、片段等权取分位数；长序列 P95 的时间范围不同，不能把这些比值直接解释为统计通过阈值。四个 15 秒区间的 RMS 等原始诊断保留在源记录。</p>'
    body += table(['拟合集参考指标', 'q10', '中位数', 'q90', 'q95'], [[key]+[fmt(values[str(q)]) for q in (.1, .5, .9, .95)] for key, values in fit['quantiles'].items()])
    body += '<h2>来源与可追溯记录</h2><p>本页面读取已保存的结果，未加载生成模型、原始音频或原始动作数据；结果文件和图片的 SHA256 记录在 <a href="review_manifest.json">review_manifest.json</a>。页面链接到视频只说明文件存在，不代表已完成人工自然度验收。</p><p>来源：<code>'+html.escape(str(run['root']))+'</code></p>'
    css = 'body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#263448;background:#f2f5f9;margin:0;padding:30px}main{max-width:1480px;margin:auto}h1{font-size:30px}h2{margin-top:38px}h3{font-size:17px;overflow-wrap:anywhere}.result{font-size:21px;font-weight:700;color:#8c4b26}article{background:white;padding:20px;border-radius:10px;margin:24px 0}img,video{width:100%;height:auto}.video-index{background:#e5edf7;padding:10px 20px 20px;border-radius:10px}.scroll{overflow:auto;background:white;border-radius:8px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #dfe6ee;white-space:nowrap}th{background:#e5ecf5}nav{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:8px;font-size:13px}a{color:#1268ae}.pending{color:#846750}details{padding:12px;background:#fff;margin:16px 0}summary{cursor:pointer}code{overflow-wrap:anywhere}'
    (output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>独立表达参考连续动态审阅</title><style>'+css+'</style><main>'+body+'</main></html>', encoding='utf8')
    manifest = {'schema': 'controlled_prior_review_v1', 'source_run': str(run['root']), 'input_sha256': run['hashes'],
        'output_sha256': {str(p.relative_to(output)).replace('\\', '/'): sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'query_examples': manifest_examples, 'long_examples': long_manifest, 'plot_seed': 42,
        'selection': 'Every prelocked protocol query in original order; first query per person for long controls',
        'group_curve': 'Raw channel mean; no seed averaging; native25Hz with invalid gaps',
        'all_seed_scores': True, 'extra_expression_reference': True, 'audio_timing_proven': False,
        'perceptual_naturalness_verified': False, 'full_face_verified': False,
        'model_loaded': False, 'native_data_loaded': False, 'script_sha256': sha(__file__)}
    (output/'review_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fullface', type=Path)
    args = parser.parse_args(); result = package(args.run, args.output, args.fullface)
    print(json.dumps({'status': 'packaged', 'output': str(args.output.resolve()),
        'query_examples': len(result['query_examples']), 'long_examples': len(result['long_examples'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
