"""Create local review figures/page from already audited history outputs."""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_temporal_repair import read, sha
from scripts.export_temporal_repair_examples import plot


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--require-videos', action='store_true'); a = p.parse_args()
    root = a.root; audit = read(root/'audit.json'); visual = root/'visual'
    for name, digest in read(root/'complete.json')['files'].items():
        if sha(root/name) != digest: raise ValueError('Downloaded audit artifact differs: '+name)
    plot(visual)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    for row, region in enumerate(('brows', 'eyes_expression')):
        for arm, mode, label in [('no_history', 'full', 'No history'), ('scheduled_history', 'full', 'Generated history'),
                                  ('scheduled_history', 'empty', 'Empty history'), ('scheduled_history', 'oracle_history', 'ORACLE past GT')]:
            values = audit['common_clip_profiles_all_modes'][arm]['42/'+mode]['profiles'][region]
            for col, key in enumerate(('raw_mse', 'mean_error_rms', 'outside_fraction')):
                axes[row, col].plot(range(1, len(values)+1), [v[key] for v in values], marker='o', label=label)
                axes[row, col].set(title=region+' / '+key, xlabel='Chunk (16 frames each)')
                axes[row, col].grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4); fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(visual/'rollout_profiles.png', dpi=120); plt.close(fig)
    labels = {'state_aligned': '上一轮音频均值 + 对齐动态', 'state_white': '上一轮音频均值 + flow',
              'no_history': '本轮无历史', 'scheduled_history': '本轮生成历史'}
    headers = ['方案', '眉raw MSE', '眉时序相关', '眉RMS/参考', '眼raw MSE', '眼时序相关', '口raw MSE']
    rows = []
    for key, label in labels.items():
        groups = audit['deployment'][key]['populations']['all']['groups']
        brow, eye, mouth = [groups[region]['mean_over_three_seeds'] for region in ('brows', 'eyes_expression', 'mouth')]
        rows.append([label, f"{brow['raw_mse']:.6f}", f"{brow['centered_correlation']:.3f}", f"{brow['rms_ratio']:.3f}",
            f"{eye['raw_mse']:.6f}", f"{eye['centered_correlation']:.3f}", f"{mouth['raw_mse']:.6f}"])
    table = '<table>'+''.join('<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in row)+'</tr>' for row in [headers, *rows])+'</table>'
    videos = []; verification = []
    for job in read(visual/'provenance.json')['jobs']:
        person = job['speaker']; folder = visual/person
        if not (folder/'display_report.json').exists():
            if a.require_videos: raise FileNotFoundError('Video not ready: '+person)
            videos.append('<p>'+html.escape(person)+'：视频正在生成。</p>'); continue
        report = read(folder/'display_report.json'); video = folder/'comparison.mp4'
        if report['status'] != 'complete' or not report['original_blend_unchanged'] or sha(video) != report['output_video_sha256']:
            raise ValueError('Video report or hash differs')
        streams = json.loads(subprocess.check_output([shutil.which('ffprobe'), '-v', 'error', '-show_streams', '-of', 'json', str(video)], encoding='utf8'))['streams']
        v = next(s for s in streams if s['codec_type'] == 'video')
        if v['avg_frame_rate'] != '25/1' or int(v['nb_frames']) != 96 or not any(s['codec_type'] == 'audio' for s in streams):
            raise ValueError('Video native clock, frames or audio differs')
        verification.append({'person': person, 'sha256': sha(video), 'frames': 96, 'fps': 25, 'audio': True})
        videos.append(f'<h2>{html.escape(person)}</h2><video controls preload="metadata" poster="visual/{person}/preview.png" src="visual/{person}/comparison.mp4"></video>')
    (root/'video_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf8')
    summary = (root/'summary.txt').read_text(encoding='utf8') if (root/'summary.txt').exists() else '两组训练和完整审计已完成，结论正在核查。'
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>前序动作上下文实验</title>
<style>body{font:16px/1.7 system-ui;max-width:1250px;margin:24px auto;padding:0 20px;color:#20262b;background:#f6f8fa}h1{font-size:28px}h2{font-size:21px}table{border-collapse:collapse;background:white;width:100%}td{border-bottom:1px solid #ddd;padding:8px}.scroll{overflow-x:auto}video,img{width:100%}a{color:#17639f}.callout{border-left:4px solid #b2731e;padding:10px 18px;background:#fff8e8;white-space:pre-line}</style>
<h1>前序动作上下文 · 12轮配对实验</h1><div class="callout">'''+html.escape(summary)+'''</div>
<p>两组各12轮；405内部开发片段、三固定seed。主表只含可部署结果。只有本轮两组的对比能隔离历史条件作用；旧候选还存在目标中心化、分段和训练方式等差异。</p>
<div class="scroll">'''+table+'''</div><p><a href="RESULTS.md">结论与限制</a> · <a href="audit.json">全部原始评分与诊断</a> · <a href="../tracking/review.html">原始视频/跟踪系数核验</a></p>
<p>视频上排：跟踪参考 / 上一轮对齐动态 / 上一轮flow；下排：本轮无历史 / 本轮生成历史 / 清空历史消融。固定片段、种子42、原生25Hz同音轨；不调动作倍率或时间偏移。视频只为显示而截断系数到[0,1]，评分使用原值。统一网格不代表人物几何身份。</p>
'''+''.join(videos)+'''<h2>逐段诊断（seed42，固定共同片段）</h2><p>ORACLE只在此诊断图中出现：使用真实前序动作，不是部署效果。误差随段变化也受内容变化影响，不单凭曲线上升认定漂移。</p><img src="visual/rollout_profiles.png" alt="逐段误差与越界"><h2>九条固定片段：中心动态</h2><img src="visual/centered.png" alt="中心动态"><h2>原始系数</h2><img src="visual/raw.png" alt="原始系数"></html>'''
    (root/'index.html').write_text(page, encoding='utf8'); print(root/'index.html')


if __name__ == '__main__': main()
