"""Build a local, source-backed review page and CSVs; never modify training."""
from __future__ import annotations

import argparse
import csv
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


def read(path):
    return json.loads(path.read_text(encoding='utf8')) if path.is_file() else {}


def get(value, *keys):
    for key in keys:
        try:
            value = value[key]
        except (KeyError, TypeError, IndexError):
            return None
    return value


def number(value):
    if value is None:
        return 'pending'
    if isinstance(value, bool):
        return '通过' if value else '未通过'
    if isinstance(value, (int, float)):
        return f'{value:.6f}' if math.isfinite(value) else 'pending'
    return str(value)


def interval(value):
    return '[' + ', '.join(number(v) for v in value) + ']' if isinstance(value, list) and len(value) == 2 else 'pending'


def table(rows, columns):
    if not rows:
        return '<p class="pending">pending：源数据缺失，未填入任何推测成绩。</p>'
    headers = ''.join('<th>' + html.escape(title) + '</th>' for _, title in columns)
    body = ''.join('<tr>' + ''.join('<td>' + html.escape(number(row.get(key))) + '</td>' for key, _ in columns) + '</tr>' for row in rows)
    return '<div class="scroll"><table><thead><tr>' + headers + '</tr></thead><tbody>' + body + '</tbody></table></div>'


def csv_file(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows or [{'status': 'pending'}])


def link(root, relative, label):
    path = root / relative
    return ('<a href="' + quote(str(relative).replace('\\', '/'), safe='/') + '">' + html.escape(label) + '</a>') if path.is_file() else '<span class="pending">' + html.escape(label) + '（pending）</span>'


def event_rows(seeds):
    rows = []
    for seed in ('42', '123', '2026'):
        item = seeds.get(seed, {})
        if not item:
            rows.append({'seed': seed, 'status': 'pending'})
            continue
        rows.append({'seed': seed, 'status': item.get('status', 'computed'),
            'audio_brier': get(item, 'summary', 'audio', 'brier'),
            'static_brier': get(item, 'summary', 'matched_static', 'brier'),
            'audio_joint_nll': get(item, 'summary', 'audio', 'joint_nll'),
            'static_joint_nll': get(item, 'summary', 'matched_static', 'joint_nll'),
            'audio_duration_nll': get(item, 'summary', 'audio', 'duration_nll'),
            'static_duration_nll': get(item, 'summary', 'matched_static', 'duration_nll'),
            'paired_brier_delta': get(item, 'audio_vs_static', 'audio_minus_control'),
            'paired_brier_ci95': interval(get(item, 'audio_vs_static', 'ci95')),
            'paired_joint_nll_delta': get(item, 'audio_vs_static_joint_nll', 'audio_minus_control'),
            'paired_joint_nll_ci95': interval(get(item, 'audio_vs_static_joint_nll', 'ci95')),
            'audio_reverse_nll_delta': get(item, 'audio_vs_reverse', 'audio_minus_control')})
    return rows


def prosody_rows(seeds):
    rows = []
    for seed in ('42', '123', '2026'):
        item = seeds.get(seed, {})
        rows.append({'seed': seed, 'status': 'computed' if item else 'pending',
            'audio_brier': get(item, 'prosody_residual', 'brier'), 'static_brier': get(item, 'static', 'brier'),
            'audio_joint_nll': get(item, 'prosody_residual', 'joint_nll'), 'static_joint_nll': get(item, 'static', 'joint_nll'),
            'audio_duration_nll': get(item, 'prosody_residual', 'duration_nll'), 'static_duration_nll': get(item, 'static', 'duration_nll'),
            'paired_brier_delta': get(item, 'paired_brier', 'audio_minus_control'),
            'paired_brier_ci95': interval(get(item, 'paired_brier', 'ci95')),
            'paired_joint_nll_delta': get(item, 'paired_joint_nll', 'audio_minus_control'),
            'paired_joint_nll_ci95': interval(get(item, 'paired_joint_nll', 'ci95')),
            'audio_reverse_nll_delta': get(item, 'paired_reverse_nll', 'audio_minus_control'),
            'fit_audio_joint_nll': get(item, 'fit_prosody_residual', 'joint_nll'),
            'fit_static_joint_nll': get(item, 'fit_static', 'joint_nll')})
    return rows


def mouth_row(seed, arm, item):
    result = {'seed': seed, 'arm': arm, 'clips': get(item, 'arkit', 'arkit_mbe', 'clips_total')}
    for key in ('arkit_mbe', 'arkit_lbe', 'arkit_fdd_absolute', 'supp_lip23_lbe', 'supp_upper9_fdd_absolute'):
        result[key] = get(item, 'arkit', key, 'value')
    for key in ('lip23_mae', 'lip23_velocity_mae_per_second', 'lip23_temporal_std_pred',
                'lip23_temporal_std_gt', 'lip23_temporal_std_absolute_gap', 'lip23_raw_oob_fraction',
                'mouth27_mae', 'mouth27_velocity_mae_per_second', 'mouth27_temporal_std_pred',
                'mouth27_temporal_std_gt', 'mouth27_temporal_std_absolute_gap', 'mouth27_raw_oob_fraction'):
        result[key] = get(item, 'diagnostics', key)
    return result


def mouth_rows(data):
    base = mouth_row('baseline', 'baseline', data.get('baseline', {}))
    singles = [mouth_row(seed, arm, get(data, 'seeds', seed, arm) or {})
               for seed in ('42', '123', '2026') for arm in ('audio', 'matched_static', 'audio_reverse')]
    aggregate = [base]
    for arm in ('audio', 'matched_static', 'audio_reverse'):
        group = [row for row in singles if row['arm'] == arm]
        row = {'seed': 'mean_of_3_training_seeds', 'arm': arm, 'clips': group[0]['clips']}
        for key in base:
            if key not in ('seed', 'arm', 'clips'):
                values = [r[key] for r in group]
                row[key] = sum(values) / len(values) if all(isinstance(v, (int, float)) for v in values) else None
        aggregate.append(row)
    return singles, aggregate


def generation_rows(root, run, event):
    result = []
    for category, arms in (('control', ('oracle', 'empty', 'shifted', 'null')), ('generation', ('audio', 'matched_static', 'reverse'))):
        for arm in arms:
            raw = read(root / run / 'event' / category / arm / 'result.json')
            fallback = get(event, 'controls' if category == 'control' else 'generation', arm) or {}
            summary = raw.get('summary') or fallback.get('motion_summary', {})
            draws = sorted({row['sample_count'] for row in raw.get('per_clip_scores', []) if 'sample_count' in row})
            row = {'arm': arm, 'condition': '真实动作事件；oracle' if arm == 'oracle' else 'audio inference' if category == 'generation' else 'receiver control',
                   'clips': summary.get('clips'), 'draws_per_clip': draws[0] if len(draws) == 1 else None,
                   'up_rms_ratio': get(summary, 'rms_ratio', 0), 'down_rms_ratio': get(summary, 'rms_ratio', 1),
                   'squint_rms_ratio': get(summary, 'rms_ratio', 2), 'wide_rms_ratio': get(summary, 'rms_ratio', 3),
                   'up_raw_oob_fraction': get(summary, 'raw_oob', 0),
                   'centered_energy_score': get(summary, 'joint_fair_es', 'centered'),
                   'numerical_gate': get(raw, 'numerical_gate', 'passed')}
            counts = get(summary, 'pooled', 'raw_count')
            oob = get(summary, 'pooled', 'oob_count')
            row['all_upper_raw_oob_fraction'] = sum(oob) / sum(counts) if counts and oob and sum(counts) else None
            result.append(row)
    return result


class LocalReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        self.references.extend(value for key, value in attrs if key in ('href', 'src', 'poster') and value)


def validate_links(path):
    parser = LocalReferences()
    parser.feed(path.read_text(encoding='utf8'))
    checked, missing = [], []
    for reference in parser.references:
        parsed = urlsplit(reference)
        if parsed.scheme or not parsed.path:
            continue
        target = (path.parent / unquote(parsed.path)).resolve()
        checked.append(str(target))
        if not target.is_file():
            missing.append(reference)
    if missing:
        raise FileNotFoundError('Missing local page references: ' + ', '.join(missing))
    return {'local_references_checked': len(checked), 'missing': missing}


def build(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = Path('runs/event_mouth_formal_20260919_v1')
    prosody_run = Path('runs/prosody_residual_v2_formal_20260919')
    summary_path = run / 'posthoc_full/summary.json'
    summary = read(root / summary_path)
    event = summary.get('event', {})
    mouth = summary.get('mouth', {})
    e_rows = event_rows(event.get('predictor_seeds', {}))
    p_rows = prosody_rows(read(root / prosody_run / 'results.json'))
    p_decision = read(root / prosody_run / 'decision.json')
    p_status = read(root / prosody_run / 'status.json')
    singles, means = mouth_rows(mouth)
    g_rows = generation_rows(root, run, event)
    outputs = {'event_three_seeds.csv': e_rows, 'prosody_three_seeds.csv': p_rows,
               'mouth_single_seeds.csv': singles, 'mouth_seed_means.csv': means,
               'event_generation_diagnostics.csv': g_rows}
    for filename, rows in outputs.items():
        csv_file(root / filename, rows)
    event_done = get(event, 'status', 'state') in ('complete', 'completed')
    prosody_done = p_status.get('state') in ('complete', 'completed')
    failed = get(event, 'decision', 'audio_predictability_passed') is False and p_decision.get('prosody_predictability_passed') is False
    title = '两轮训练已完成，音频动态仍未达标' if event_done and prosody_done and failed else '事件动态与口部候选结果复核'
    sections = [f'<header><div class="eyebrow">KineTalk · 2026-09-19 · inner development</div><h1>{title}</h1><p>事件日程与口部修复已分开核验。完成训练不代表方法成功；本页直接读取已保存的实验 JSON，不填补缺失指标，不替换默认模型。</p></header>',
        '<div class="notice"><strong>当前结论：</strong>全维音频事件模型与后续韵律残差模型均未通过跨句预测门槛。控制接收通路能响应事件，但真实事件驱动下的抬眉幅度仍低且越界明显，因此不能只归因于音频预测器。</div>',
        '<section><h2>评估范围与训练状态</h2><p>事件生成使用固定的 24 条 inner-validation 片段、每片 4 个随机生成；事件概率预测和口部候选使用 206 条 inner-validation 片段。此前历史 64 条结果属于另一组开发样本，三个样本集合不能直接混表比较。上述结果均不是 sealed test，冻结基座与特征提取器继承历史训练暴露。</p>',
        table([{'stage': '事件接收器与全维音频预测', 'state': get(event, 'status', 'state'), 'gate': get(event, 'decision', 'audio_predictability_passed')},
               {'stage': '韵律残差预测', 'state': p_status.get('state'), 'gate': p_decision.get('prosody_predictability_passed')},
               {'stage': '口部候选', 'state': get(mouth, 'status', 'state'), 'gate': '口型同步尚未验收'}], [('stage', '阶段'), ('state', '执行状态'), ('gate', '科学/质量验收')]), '</section>']
    columns = [('seed', '训练种子'), ('audio_brier', '音频 Brier ↓'), ('static_brier', '静态 Brier ↓'),
               ('audio_joint_nll', '音频 joint NLL ↓'), ('static_joint_nll', '静态 joint NLL ↓'),
               ('paired_brier_delta', '配对 ΔBrier'), ('paired_brier_ci95', 'ΔBrier 95% CI'),
               ('paired_joint_nll_delta', '配对 ΔNLL'), ('paired_joint_nll_ci95', 'ΔNLL 95% CI')]
    sections += ['<section><h2>第一轮：全维音频事件预测</h2><p>Brier / NLL 越低越好；配对差值为音频减静态，负值才支持音频更好。表中单模型指标是片段平均，配对统计采用原报告的句子聚合与 bootstrap，因此差值不一定等于两个显示均值直接相减。</p>', table(e_rows, columns),
        link(root, 'event_three_seeds.csv', '下载事件三种子 CSV') + ' · ' + link(root, run/'event/predictor_results.json', '事件原始 JSON'), '</section>',
        '<section><h2>第二轮：冻结静态基座 + 韵律残差</h2><p>三种子训练集有所改善，但验证集相对静态的 Brier / joint NLL 配对差值仍为正。部分正序优于反序，只证明输入顺序有影响，尚不足以证明泛化到新的句子。</p>', table(p_rows, columns),
        link(root, 'prosody_three_seeds.csv', '下载韵律三种子 CSV') + ' · ' + link(root, prosody_run/'results.json', '韵律原始 JSON') + ' · ' + link(root, prosody_run/'decision.json', '门槛判定'), '</section>',
        '<section><h2>事件接收器与最终生成：24 条 × 4 draws</h2><p>以下为同一组固定片段。RMS 比例由 pooled 动态能量计算，目标为 1；它描述幅度而非时序对齐。up 对应抬眉组。OOB 是原始系数越过 [0,1] 的比例，未通过 clamp 隐藏。Oracle 使用真实动作提取的事件，只验证接收通路，不能作为音频推理成绩。</p>',
        table(g_rows, [('arm', '条件'), ('clips', '片段数'), ('draws_per_clip', 'draws'), ('up_rms_ratio', '抬眉 RMS 比'),
                       ('down_rms_ratio', '压眉 RMS 比'), ('up_raw_oob_fraction', '抬眉 OOB 比例'),
                       ('all_upper_raw_oob_fraction', 'Upper9 OOB 比例'), ('centered_energy_score', 'Centered ES ↓')]),
        '<p class="warning">工程控制门槛通过不等于动态质量通过。即使输入真实事件，抬眉幅度和越界仍未达到可接受水平；更换音频预测器不能单独解决接收端问题。</p>',
        link(root, 'event_generation_diagnostics.csv', '下载生成诊断 CSV') + ' · ' + link(root, run/'event/control/oracle/result.json', 'Oracle 原始结果') + ' · ' + link(root, run/'event/generation/audio/result.json', '音频生成原始结果'), '</section>']
    m_columns = [('arm', '方案'), ('clips', '片段数'), ('arkit_mbe', 'MBE ↓'), ('arkit_lbe', 'LBE ↓'),
                 ('lip23_mae', 'Lip23 MAE ↓'), ('mouth27_mae', 'Mouth27 MAE ↓'),
                 ('lip23_velocity_mae_per_second', 'Lip23 速度 MAE/s ↓'),
                 ('lip23_temporal_std_absolute_gap', 'Lip23 std 差 ↓'), ('lip23_raw_oob_fraction', 'Lip23 OOB 比例')]
    sections += ['<section><h2>口部：206 条验证集，基线及三个训练种子的均值</h2><p>各训练种子内部先等权平均片段，表中再平均三个种子；不是最佳种子成绩，也不是同一生成模型的 Multimodality。只有 mouth27 更新，其他 25 个通道精确保留。静态对照仍读取动态的音频基座口部轨迹，只将额外声学特征改为 run 均值。</p>', table(means, m_columns),
        '<p class="warning">MBE/LBE 的局部改善不能掩盖 MAE、时间标准差差值和越界的退化。口部不是全部指标都更好；不能凭 LBE 下降宣布口型或音画同步通过。</p>',
        link(root, 'mouth_seed_means.csv', '三种子均值 CSV') + ' · ' + link(root, 'mouth_single_seeds.csv', '全部单种子 CSV') + ' · ' + link(root, run/'mouth/results.json', '口部原始 JSON'), '</section>',
        '<section><h2>固定样本可视化（无声视频）</h2><p>固定诊断名单前两条，noise seed 42，未按效果选择。视频无音轨，仅观察动画和对照；不能据此验收 AV 同步。显示缺失帧的填补仅用于展示，数值评分仍用原始观测 mask。</p>']
    for suffix in ('007', '019'):
        movie = Path('review') / ('render_' + suffix) / 'comparison.mp4'
        preview = movie.with_name('preview.png')
        if (root / movie).is_file():
            poster = f' poster="{quote(preview.as_posix(), safe="/")}"' if (root / preview).is_file() else ''
            sections.append(f'<figure><figcaption>mead_M003_angry_L1_{suffix} · silent</figcaption><video controls preload="metadata"{poster} src="{movie.as_posix()}"></video></figure>')
        else:
            sections.append(f'<p class="pending">样本 {suffix} 视频 pending。</p>')
    sections += [link(root, 'review/manifest.json', '固定样本来源') + '</section>', '<section><h2>接收端瓶颈定位</h2>']
    audit_candidates = [root/'receiver_bottleneck_audit/summary.json', root/run/'receiver_bottleneck_audit/summary.json']
    audit_path = next((path for path in audit_candidates if path.is_file()), None)
    if audit_path:
        sections.append('<p>已取得同24例的瓶颈诊断。它们偏向少数身份和情感，不能用24例的RMS比值概括全部206例。</p>' + link(root, audit_path.relative_to(root), '24例瓶颈审计 JSON'))
    else:
        sections.append('<p class="pending">pending：SSH 中断，接收端瓶颈诊断尚未完成或尚未取回。当前没有诊断结果，不能据此宣称已定位或已修复。</p>')
    scope_path = Path('audits/event_validation_scope_20260919/summary.json')
    scope = read(root/scope_path)
    if scope:
        scope_rows = [dict(arm=row['arm'], up=row['rms_ratio'][0], down=row['rms_ratio'][1],
                     squint=row['rms_ratio'][2], wide=row['rms_ratio'][3],
                     es=get(row,'joint_fair_es','centered'), mbe=get(row,'arkit','arkit_mbe','value'))
                     for row in scope['table']]
        csv_file(root/'receiver_all206.csv', scope_rows)
        sections += ['<h3>全部206例：AE、原始先验与事件微调</h3>',
                     table(scope_rows,[('arm','模型/条件'),('up','抬眉 RMS 比'),('down','压眉 RMS 比'),
                       ('squint','眯眼 RMS 比'),('wide','睁眼 RMS 比'),('es','Centered ES ↓'),('mbe','ARKit-MBE ↓')]),
                     '<p>AE能保留约98%的抬眉动态RMS；源先验约79%，事件oracle约63%。事件微调在全验证集也降低动态幅度。AE是读取真实动作的重建上限，不是音频生成成绩。</p>',
                     link(root,scope_path,'全206原始审计')+' · '+link(root,'receiver_all206.csv','全206 CSV')]
    support_path = Path('audits/event_training_support_v2_final_20260919/report.json')
    support = read(root/support_path)
    if support:
        fit = support['populations']['fit']['all']
        sections += ['<h3>监督裁断问题</h3><p>源先验训练保留 '+str(fit['source_clips_retained'])+' 条、'+str(fit['frames']['source_kept'])+
            ' 帧；事件known4共同有效区间与最短段筛选后，只有 '+str(fit['receiver_clips_retained'])+' 条、'+str(fit['frames']['receiver_kept'])+
            ' 帧。保留片段本身的动态也更弱。这是明确的训练分布变化，因果影响仍需恢复完整监督的配对训练确认。</p>',
            link(root,support_path,'监督覆盖原始审计')]
    sampling_path = Path('audits/prior_sampling_resolution_20260919/summary.json')
    sampling = read(root/sampling_path)
    if sampling:
        rows=[dict(arm=r['arm'],nfe=r['nfe'],up=r['rms_ratio'][0],mbe=get(r,'arkit','arkit_mbe','value')) for r in sampling['table']]
        sections += ['<h3>采样精度对照（同24例）</h3>',table(rows,[('arm','积分器'),('nfe','向量场计算次数'),('up','抬眉 RMS 比'),('mbe','MBE ↓')]),
                     '<p>增加步数与Heun积分只小幅改变幅度，不能解释或修复主要损失。没有据此挑选最佳采样器或替换默认模型。</p>',link(root,sampling_path,'采样对照 JSON')]
    repair_path=Path('audits/receiver_support_repair_20260919/summary.json')
    repair=read(root/repair_path)
    if repair:
        rows=[dict(arm=r['arm'],up=r['rms_ratio'][0],down=r['rms_ratio'][1],squint=r['rms_ratio'][2],
                   wide=r['rms_ratio'][3],es=get(r,'joint_fair_es','centered'),
                   mbe=get(r,'arkit','arkit_mbe','value'),lbe=get(r,'arkit','arkit_lbe','value')) for r in repair['table']]
        csv_file(root/'support_repair_all206.csv',rows)
        sections+=['<h3>已完成：恢复完整监督的2000步修复对照</h3>',
                   table(rows,[('arm','模型'),('up','抬眉 RMS 比'),('down','压眉 RMS 比'),('squint','眯眼 RMS 比'),
                     ('wide','睁眼 RMS 比'),('es','Centered ES ↓'),('mbe','MBE ↓'),('lbe','LBE ↓')]),
                   '<p>从相同源prior初始化，完整监督恢复了旧null的部分动态退化，MBE改善。但抬眉幅度和ES仍未胜源prior，局部音频条件始终为零，不能当作音频时序成功。固定单训练种子，无默认模型替换。</p>',
                   link(root,repair_path,'修复原始结果')+' · '+link(root,'support_repair_all206.csv','修复CSV')]
        movie=Path('support_review/render/comparison.mp4')
        if (root/movie).is_file():
            sections+=['<p>固定原先第二个样例，seed42、全长、无声；六面板包含GT、基线、AE重建、源prior、旧null与完整监督null。</p>',
                       '<video controls preload="metadata" src="'+movie.as_posix()+'"></video>']
    sections += ['</section><section><h2>仍待完成的论文主表项目</h2><p>AV offset / confidence、正式 Multimodality、FD / WInD 仍为 pending：缺少完成并验证的公共音画或运动特征评价流程。FDD 是时序能量标准差差异，对帧顺序不敏感，不能替代动作时机评价。</p><p>FaceFormer、CodeTalker、FaceDiffuser 尚未按本项目统一 ARKit52 协议重训，因此当前不能排名或宣称超越论文方法。身份和情感支路冻结只说明此次没有更新其参数，并不构成生成身份/情感质量通过的证据。</p></section>',
        '<section><h2>来源与已有报告</h2><p>' + ' · '.join([link(root, summary_path, '完整 posthoc JSON'), link(root, run/'posthoc_full/report.md', '已有 posthoc 报告'), link(root, run/'protocol.json', '顺序训练协议'), link(root, prosody_run/'protocol.json', '韵律协议'), link(root, run/'event/receiver_gate.json', '接收器工程门槛')]) + '</p></section>']
    stylesheet = 'body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f5f7fa;color:#17223b;margin:0;line-height:1.65}main{max-width:1420px;margin:auto;padding:36px 24px}header{padding:18px 0}h1{font-size:32px;line-height:1.3}h2{font-size:23px;margin-top:0}.eyebrow{font-size:13px;color:#476078;letter-spacing:.08em}section{background:white;border:1px solid #dce4ec;border-radius:14px;padding:24px;margin:22px 0}.notice{background:#fff0df;border-left:5px solid #bf6b21;padding:20px;border-radius:5px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{text-align:right;padding:11px;border-bottom:1px solid #e3e9ef}th{background:#f0f4f8;color:#34506d}th:first-child,td:first-child{text-align:left}a{color:#165f9e}.warning{color:#9c4916}.pending{color:#805b1f}video{width:100%;max-height:600px;background:#131820}figure{margin:22px 0}figcaption{font-weight:600;margin-bottom:10px}pre{white-space:pre-wrap;max-height:500px;overflow:auto;font-size:12px}p{max-width:1200px}'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KineTalk 事件动态与口部结果</title><style>' + stylesheet + '</style><main>' + ''.join(sections) + '</main></html>'
    path = root / 'index.html'
    path.write_text(page, encoding='utf8')
    return {'page': str(path), 'csv_files': list(outputs), **validate_links(path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/event_schedule_20260919'))
    print(json.dumps(build(parser.parse_args().root), ensure_ascii=False))
