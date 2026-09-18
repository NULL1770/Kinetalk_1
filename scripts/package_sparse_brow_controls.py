"""Verify already-rendered engineering controls and publish a local review.

Does not generate motion, train or select outcomes. Exactly the four fixed
metadata records, every native frame, seed42, and both modes are required.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_dynamic_rig_comparison as rig


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def package(root):
    root = Path(root).resolve()
    if (root/'review.html').exists():
        raise FileExistsError('Review exists; preserve the earlier verified artifact')
    manifest = read(root/'manifest.json')
    assert len(manifest['records']) == 4 and not manifest['trained'] and not manifest['teacher_fit']
    assert manifest['all_engineering_checks_pass']
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>眉部事件机制演示</title>',
        '<style>body{max-width:1100px;margin:36px auto;font:17px/1.7 system-ui;padding:0 18px;background:#f7f8fb;color:#182234}video,img{max-width:100%}section{background:white;padding:20px;margin:24px 0;border-radius:12px}.note{background:#fff2d8;padding:18px}code{font-size:14px}</style>',
        '<h1>有保持、起势与回落的眉部事件</h1>',
        '<p class="note"><b>手工参数的工程演示，不是训练成功结果。</b>固定两种抬眉/压眉参数和等待概率，无音频时机输入。其他47通道固定，不用于口型同步评价。机制通过不等于自然度通过。</p>',
        '<p>每段10秒，前4秒自动运行，4–5秒外部保持，5–6秒释放回基础表情，6–10秒恢复。自动运行内部也有真正静止的HOLD。固定seed42，四人全部展示。</p>',
        '<p>四人×8seed×60秒检查通过；0/.5/1/1.5幅度不改变事件时机，.5/1/2活动控制保持同序号的动作参数。画面变化较细微，建议放大全屏观看。</p>',
        '<p><a href="../teacher_review/index.html">原视频动作教师审阅</a> · <a href="../index.html">本轮结论</a></p>']
    summaries = []
    for row in manifest['records']:
        cid = row['clip_id']; source = root/row['npz']; dest = root/cid
        assert source.is_file() and sha(source) == row['npz_sha256']
        channels, times, valid, modes, expected, inspected = rig.inspect_input(source, 25)
        report = read(dest/'display_report.json')
        assert report['status'] == 'complete' and report['input_sha256'] == sha(source)
        assert report['rendered_frames'] == len(times) == 250 and report['original_blend_unchanged']
        assert report['audio_sha256'] is None and report['rendered_modes'] == modes
        assert sha(dest/'comparison.mp4') == report['output_video_sha256']
        assert all((dest/'frames'/f'{i:06d}.png').is_file() for i in range(1,251))
        with np.load(dest/'display_curves.npz', allow_pickle=False) as d:
            assert np.array_equal(d['motions'], expected) and np.array_equal(d['times'], times)
        with np.load(source, allow_pickle=False) as d:
            raw = d['motions'].copy(); phase = d['phase'].copy()
            assert bool(d['engineering_only']) and not bool(d['trained']) and int(d['noise_seed']) == 42
        brow = [channels.index(n) for n in rig.BROWS]
        other = [i for i in range(52) if i not in brow]
        assert np.array_equal(raw[0,:,other], raw[1,:,other])
        assert np.array_equal(raw[0,:,other], np.broadcast_to(raw[0,0,other,None],raw[0,:,other].shape))
        fig, axes = plt.subplots(2,1,figsize=(11,4.2),sharex=True,gridspec_kw={'height_ratios':[3,1]})
        for i, name in enumerate(rig.BROWS):
            axes[0].plot(times, raw[1,:,brow[i]]-raw[0,:,brow[i]],label=name)
        axes[0].set_ylim(-.025,.15);axes[0].set_ylabel('Raw coefficient delta');axes[0].legend(ncol=3,fontsize=8)
        names=['HOLD','ONSET','APEX','RELEASE','STOP']
        axes[1].step(times,[names.index(p) for p in phase],where='post')
        axes[1].set_yticks(range(5),names,fontsize=8);axes[1].set_xlabel('Seconds')
        for ax in axes:
            ax.axvline(4,color='black',alpha=.3);ax.axvline(5,color='black',alpha=.3);ax.axvline(6,color='black',alpha=.3)
        fig.tight_layout();fig.savefig(dest/'phase_curve.png',dpi=140);plt.close(fig)
        body.append(f'<section><h2>{html.escape(cid)}</h2><video controls preload="metadata" poster="{cid}/preview.png" src="{cid}/comparison.mp4"></video><img src="{cid}/phase_curve.png" loading="lazy"><p>完整250帧，47通道逐值保护。两侧基础表情相同，右侧仅增加5个眉通道动作。</p></section>')
        summaries.append({'clip_id':cid,'frames':250,'video_sha256':sha(dest/'comparison.mp4'),
            'render_report_sha256':sha(dest/'display_report.json'),'input_sha256':sha(source),
            'other47_static_exact':True,'display_arrays_exact':True})
    body.append('</html>')
    (root/'review.html').write_text('\n'.join(body),encoding='utf8')
    summary={'status':'complete','engineering_only':True,'learned':False,'naturalness_verified':False,
        'source_manifest_sha256':sha(root/'manifest.json'),'packager_sha256':sha(__file__), 'records':summaries}
    (root/'render_verification.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(package(parser.parse_args().root)))
