"""Render a fixed four-emotion, full-length view of an existing latent run.

No inference, seed selection, amplitude adjustment or checkpoint selection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


UPPER = [41, 42, 43, 44, 45, 5, 6, 12, 13]
GROUPS = ((2, 3, 4), (0, 1), (5, 7), (6, 8))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, audio_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main = json.loads((root/'audio_generation/holdout/result.json').read_text(encoding='utf8'))
    selected = main['selected_clip_ids'][:4]
    out = root/'review'; out.mkdir(exist_ok=True)
    jobs = []
    for cid in selected:
        loaded = {}
        for stage in ('audio_generation', 'matched_global_generation', 'ae_reconstruction', 'audio_static'):
            with np.load(root/stage/'holdout/npz'/f'{cid}.npz', allow_pickle=False) as data:
                loaded[stage] = {k: data[k] for k in data.files}
        a = loaded['audio_generation']; base = a['motions'][1].copy()
        for record in loaded.values():
            np.testing.assert_array_equal(record['times'], a['times'])
            np.testing.assert_array_equal(record['valid'], a['valid'])
            np.testing.assert_array_equal(record['motions'][1], base)
        # Reference combines target upper face with the protected base mouth.
        reference = base.copy(); reference[:, UPPER] = a['motions'][0][:, UPPER]
        names = ['GT upper + base43', 'Frozen base', 'AE oracle', 'Global prior s42', 'Audio s42', 'Static audio s42']
        motion = np.stack((reference, base, loaded['ae_reconstruction']['motions'][2],
                           loaded['matched_global_generation']['motions'][2], a['motions'][2],
                           loaded['audio_static']['motions'][2]))
        other = [i for i in range(52) if i not in UPPER]
        np.testing.assert_array_equal(motion[:, :, other], np.broadcast_to(base[:, other], (6, len(base), 43)))
        path = out/f'{cid}.npz'
        np.savez_compressed(path, channels=a['channels'], mode_names=np.asarray(names),
                            motions=motion, times=a['times'], valid=a['valid'],
                            clip_id=np.asarray(cid), noise_seed=np.asarray(42))
        audio = audio_root/f'{cid}.wav'
        if not audio.is_file() or sha(audio) != str(a['audio_sha256']):
            raise ValueError(f'Original audio hash mismatch or missing: {audio}')
        fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
        for ax, channels, label in zip(axes, GROUPS, ('Brow raise', 'Brow down', 'Squint', 'Wide')):
            for index, color in ((0, '#183f6d'), (1, '#888888'), (3, '#267d62'), (4, '#df681d')):
                values = motion[index][:, UPPER][:, channels].mean(-1)
                values = np.where(a['reference_valid'] if index == 0 else a['valid'], values, np.nan)
                ax.plot(a['times'], values, color=color, label=names[index], lw=1.3)
            ax.set_ylabel(label); ax.grid(alpha=.2)
        axes[0].legend(ncol=4, fontsize=8); axes[-1].set_xlabel('Original audio time (seconds)')
        fig.suptitle(cid+' | raw coefficients, fixed seed 42, no gain or lag fitting')
        fig.tight_layout(); fig.savefig(out/f'{cid}.png', dpi=140); plt.close(fig)
        jobs.append({'clip_id': cid, 'input': str(path.resolve()), 'output': str((out/cid).resolve()),
                     'audio': str(audio.resolve()), 'audio_offset_seconds': float(a['audio_offset_seconds']),
                     'frames': len(base), 'seed': 42, 'audio_sha256': sha(audio)})
    (out/'jobs.json').write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding='utf8')
    return jobs


def render(job):
    destination = Path(job['output'])
    if (destination/'display_report.json').is_file():
        record = json.loads((destination/'display_report.json').read_text(encoding='utf8'))
        if record.get('status') == 'complete' and (destination/'comparison.mp4').is_file(): return
    command = [sys.executable, '-m', 'scripts.render_dynamic_rig_comparison', '--input', job['input'],
               '--output', job['output'], '--audio', job['audio'], '--audio-offset-seconds', str(job['audio_offset_seconds']),
               '--tile-size', '320', '--columns', '3', '--samples', '8']
    subprocess.run(command, check=True)


def page(root, jobs):
    stages = [('matched_global_generation', '全局条件先验（配对训练）'), ('audio_generation', '逐帧音频'),
              ('audio_static', '同模型静态音频'), ('audio_reverse', '同模型反序音频'), ('audio_mismatch', '同模型异句音频')]
    rows = []
    for stage, label in stages:
        s = json.loads((root/stage/'holdout/result.json').read_text(encoding='utf8'))['summary']
        speed = s['speed']['all']['rms']/s['speed']['reference_all']['rms']
        rows.append(f'<tr><td>{label}</td><td>{s["joint_fair_es"]["centered"]:.6f}</td><td>{s["variogram"]["aggregate"]:.6f}</td><td>{speed:.3f}</td><td>'+', '.join(f'{x:.3f}' for x in s['rms_ratio'])+'</td></tr>')
    sections = []
    for job in jobs:
        cid = html.escape(job['clip_id'])
        sections.append(f'<section><h2>{cid}</h2><p>完整{job["frames"]}帧 / {job["frames"]/25:.2f}秒，固定seed42，不挑样本。</p><video controls preload="metadata" src="{cid}/comparison.mp4" poster="{cid}/preview.png"></video><img src="{cid}.png"></section>')
    links = ''.join(f'<li><a href="../{stage}/holdout/index.html">{label}：全部固定种子原生曲线</a></li>' for stage, label in stages)
    content = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>连续动作latent结果复核</title>
<style>body{max-width:1180px;margin:30px auto;font:16px/1.65 system-ui;background:#eef2f6;color:#182c3e}section,header{background:white;padding:24px;border-radius:12px;margin:24px 0}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border:1px solid #ccd6df;text-align:left}video,img{width:100%;display:block;margin-top:18px}h1{font-size:30px}</style>
<header><h1>连续动作latent：运动已生成，音频时序尚未通过</h1><p>24,000次更新完整结束。64条历史来源内部留句诊断，不是独立最终测试。生成幅度/速度接近参考，但逐帧音频在留句集的中心化分布评分差于配对全局条件先验；训练内改善不能证明泛化。</p><p>视频六格：GT upper + base43（合成参考）、原基座、AE oracle、全局条件先验、逐帧音频、静态音频。前三/后三从左至右。全部共享原音频和原时钟，seed42预先固定；其他seed保留于诊断链接。显示将越界系数裁剪至[0,1]，raw评分和下方曲线未裁剪、未放大。此通用rig只检验动作，不认证身份相似性。</p></header>
<table><tr><th>方案</th><th>中心化fair ES ↓</th><th>时差变化评分 ↓</th><th>速度/GT</th><th>幅度/GT：抬眉、压眉、眯眼、睁眼</th></tr>'''+''.join(rows)+'</table><ul>'+links+'</ul>'+''.join(sections)+'</html>'
    (root/'review/index.html').write_text(content, encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(); jobs = prepare(args.root, args.audio_root)
    if args.render:
        with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(render, jobs))
    page(args.root, jobs)
    print(json.dumps({'review': str((args.root/'review/index.html').resolve()), 'clips': len(jobs)}))


if __name__ == '__main__':
    main()
