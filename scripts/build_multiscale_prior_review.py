"""Build a review from the unmodified V2 audit and predeclared examples."""
import argparse
import csv
import html
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def build(run, examples):
    status = read(run/'status.json'); point = run/f'point_{status["last_endpoint"]}'
    if status.get('smoke'): raise ValueError('Smoke is not a quality review')
    result = read(point/'acceptance.json')
    rows = []
    arms = ('prior', 'audio', 'matched_static', 'static', 'reverse', 'mismatch')
    for arm in arms:
        native = read(point/arm/'result.json'); s = native['summary']
        arkit = read(point/arm/'arkit_benchmark.json')['summary']
        rows.append({'arm':arm, 'clips':native['clips'],
            'centered_ES':s['joint_fair_es']['centered'], 'raw_ES':s['joint_fair_es']['raw'],
            'variogram':s['variogram']['aggregate'],
            **{name: s['rms_ratio'][i] for i,name in enumerate(('up_RMS','down_RMS','squint_RMS','wide_RMS'))},
            **{name: arkit[name]['value'] for name in ('arkit_mbe','arkit_lbe','arkit_fdd_absolute')}})
    with (run/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dest = run/'review'; dest.mkdir(exist_ok=True)
    sections = []; manifest = []
    for entry in read(examples):
        cid = entry['clip_id']; loaded = {}; hashes = {}; missing = []
        for arm in arms:
            source = point/arm/'npz'/(cid+'.npz')
            if arm == 'mismatch' and not source.exists() and cid in result['mismatch_excluded_no_donor']:
                missing.append(arm); continue
            hashes[arm] = hashlib.sha256(source.read_bytes()).hexdigest()
            with np.load(source,allow_pickle=False) as z:
                loaded[arm] = {k:z[k].copy() for k in z.files}
        first = loaded['prior']
        for arm, other in loaded.items():
            for field in ('times','valid','channels','clip_id','native_valid','reference_valid','reference_display_baseline_filled'):
                if not np.array_equal(first[field],other[field]): raise ValueError('Different native clock')
            if not np.array_equal(first['motions'][:2],other['motions'][:2]): raise ValueError('References differ')
            if int(other['noise_seed']) != 42 or not str(other['mode_names'][2]).endswith(' 42'):
                raise ValueError('Expected fixed seed42 in first draw')
        visible = [a for a in arms if a in loaded]
        names = ['GT (missing reference baseline-filled)','Frozen baseline']+['Seed42 '+arm for arm in visible]
        first.update(motions=np.stack([first['motions'][0],first['motions'][1]]+
                                      [loaded[a]['motions'][2] for a in visible]),mode_names=np.array(names))
        np.savez_compressed(dest/(cid+'.npz'),**first)
        manifest.append({'clip_id':cid,'frames':len(first['times']),'seed':42,
                         'selection':'same predeclared two examples as event schedule review',
                         'source_manifest':str(examples), 'examples_sha256':hashlib.sha256(examples.read_bytes()).hexdigest(),
                         'source_npz_sha256':hashes, 'unavailable_panels':missing,
                         'audio':'not attached; no AV evidence'})
        video = cid+'/comparison.mp4'
        sections.append('<section><h2>'+html.escape(cid)+'</h2><p>固定种子42，完整原生时长；GT缺失处明确用基线填充；面板：'+
                        html.escape(' / '.join(names))+'</p><video controls loop preload="metadata" src="review/'+video+'"></video>'+
                        '<p>无合法供体而省略：'+html.escape(str(missing))+'</p><p><a href="review/'+cid+'.npz">未缩放系数</a></p></section>')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    table = '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in rows[0])+'</tr>'
    for row in rows:
        table += '<tr>'+''.join('<td>'+ (f'{v:.6f}' if isinstance(v,float) else html.escape(str(v)))+'</td>' for v in row.values())+'</tr>'
    table += '</table>'
    comparisons = '<ul>'
    for arm,c in result['comparisons'].items():
        b=c['centered']; comparisons += f'<li>audio − {arm}: {b["mean_delta"]:.6f}, 95%句簇区间 {b["ci95"]}, n={b["clip_count"]}</li>'
    comparisons += '</ul>'
    page = '<!doctype html><meta charset="utf-8"><title>多尺度音频残差 V2</title><style>body{font:16px/1.6 sans-serif;margin:24px;background:#f2f4f7}section{background:white;padding:20px;margin:20px 0}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccc;padding:8px}video{max-width:100%;width:1100px}</style>'
    page += '<h1>多尺度音频残差 V2 · 固定端点 '+str(status['last_endpoint'])+'</h1>'
    page += '<p>音频增量验收：'+('通过内部统计门槛，仍需感知与独立验证' if result['accepted'] else '未通过；不能宣称音频时序成功')+'</p>'
    page += '<p>真实音频、独立静态训练、同模型静态/反向、冻结先验使用相同全量样本。错配仅使用合法供体子集，表中样本量不同；配对差值另按相同子集计算。历史内部开发集，非论文最终测试。</p>'
    page += '<p><a href="metrics.csv">指标 CSV</a> · <a href="'+point.name+'/acceptance.json">完整审计</a> · <a href="protocol.json">固定协议</a></p>'
    page += table+'<h2>配对 Centered ES 差值（负数较好）</h2>'+comparisons
    page += '<h2>未通过项</h2><pre>'+html.escape('\n'.join(result['failures']))+'</pre>'
    page += '<p>其余43通道与原基线一致仅证明未修改，不认证原口型、身份或情感质量。AV/FD/WInD等仍见各arm报告中的pending依赖。视频无音频，非AV证据；显示裁剪不影响原始评分。</p>'
    page += ''.join(sections)
    (run/'index.html').write_text(page,encoding='utf8')
    return rows


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--examples',type=Path,required=True)
    args=parser.parse_args(); build(args.run,args.examples)
