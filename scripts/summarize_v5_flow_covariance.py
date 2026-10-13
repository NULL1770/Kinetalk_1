"""Matched covariance pilot tables, native-coefficient comparisons and renders."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES
from kinetalk_b0.protocol import EMOTIONS


def main(a):
    root = Path(a.results)
    report = json.loads((root/'report.json').read_text(encoding='utf8'))
    arms = list(report['results'])
    table = []
    for arm, roles in report['results'].items():
        for role, r in roles.items():
            table.append(dict(arm=arm, role=role, mouth_mean=r['mean_mouth_mae'],
                mouth_sample=r['sample_mouth_mae'],
                **{f'emotion_mean_{i}':p['mean'] for i,p in r['independent_probes'].items()},
                **{f'emotion_sample_{i}':p['sample'] for i,p in r['independent_probes'].items()},
                acceleration_ratio=r['sample_acceleration']/r['gt_dynamics']['acceleration'],
                seed_diversity=r['seed_diversity'], same_person_delta=r['reference_motion_delta'],
                cross_person_delta=r['cross_style_motion_delta']))
    with (root/'comparison.csv').open('w',encoding='utf8',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    lines = ['# 同起点 flow 基分布对照', '',
        f"总运行 {report['elapsed_seconds']:.1f}s；每臂 {report['steps_per_arm']} 步。",
        '除 flow 外所有权重和 buffer 经 digest 验证一致，个人响应场统一关闭。',
        '情感探针为准确率；加速度接近 GT 不能单独证明自然动态。', '',
        '|模型|集合|均值嘴部 MAE|采样嘴部 MAE|4个采样情感探针|采样加速度/GT|',
        '|---|---|---:|---:|---|---:|']
    for row in table:
        probes = '/'.join(f"{row['emotion_sample_'+str(i)]:.3f}" for i in range(4))
        lines.append(f"|{row['arm']}|{row['role']}|{row['mouth_mean']:.4f}|{row['mouth_sample']:.4f}|{probes}|{row['acceleration_ratio']:.2f}|")
    lines += ['', '均值路径不经过随机 flow，因此保持一致是冻结检查，不是情感提升证据。',
        '本轮仍未解决参考码的内容/情绪泄漏；不能把噪声协方差修复当作独立创新。',
        '未晋级长训。小样本、多指标，尚无统计显著性结论。']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for emotion in range(8):
        examples = {arm:dict(np.load(root/arm/f'held_sentence_emotion_{emotion}.npz',allow_pickle=False))
                    for arm in arms}
        source = examples[arms[0]]
        for x in examples.values():
            assert int(x['row']) == int(source['row'])
            np.testing.assert_array_equal(x['motions'][0],source['motions'][0])
            np.testing.assert_array_equal(x['times'],source['times'])
            np.testing.assert_array_equal(x['valid'],source['valid'])
            np.testing.assert_array_equal(x['channel_mask'],source['channel_mask'])
        end = np.flatnonzero(source['valid'])[-1]+1
        arrays = [source['motions'][0]]
        names = ['GT']
        if a.baseline:
            old = dict(np.load(Path(a.baseline)/'retained_v5_cross_ref'/f'held_sentence_emotion_{emotion}.npz',allow_pickle=False))
            assert int(old['row']) == int(source['row'])
            np.testing.assert_array_equal(old['motions'][0],source['motions'][0])
            arrays.append(old['motions'][3]); names.append('retained_v5')
        for arm in arms:
            if arm == 'frozen_start':
                continue
            arrays.append(examples[arm]['motions'][3]); names.append(arm+'_flow')
        arrays = np.stack(arrays)[:, :end]
        np.savez_compressed(root/f'render_{EMOTIONS[emotion]}.npz', motions=arrays,
            channels=np.array(ARKIT_NAMES),mode_names=np.array(names),
            times=source['times'][:end],valid=source['valid'][:end],channel_mask=source['channel_mask'],
            row=source['row'],emotion=source['emotion'])
        fig, axes = plt.subplots(4,1,figsize=(10,8),sharex=True)
        for ax, channel in zip(axes,['jawOpen','mouthSmileLeft','browDownLeft','browInnerUp']):
            for i, name in enumerate(names):
                curve=arrays[i,:,ARKIT_NAMES.index(channel)].copy()
                curve[~source['valid'][:end]]=np.nan
                ax.plot(source['times'][:end],curve,label=name,linewidth=1.3 if i==0 else 1.)
            ax.set_ylabel(channel); ax.grid(alpha=.2)
        axes[0].legend(ncol=3,fontsize=8); axes[-1].set_xlabel('Native seconds')
        fig.suptitle(EMOTIONS[emotion]+': same query, unmodified native coefficients')
        fig.tight_layout(); fig.savefig(root/f'curves_{EMOTIONS[emotion]}.png',dpi=150); plt.close(fig)
    print(json.dumps(dict(output=str(root.resolve()),rows=len(table))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--results',required=True)
    p.add_argument('--baseline'); main(p.parse_args())
