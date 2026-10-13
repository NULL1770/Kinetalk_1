"""Export auditable short-pilot tables and same-query native renders."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES
from kinetalk_b0.protocol import EMOTIONS


def main(a):
    root = Path(a.results)
    report = json.loads((root/'report.json').read_text(encoding='utf8'))
    records = json.loads((root.parent/'data_provenance.json').read_text(encoding='utf8'))['records']
    table = []
    for name, roles in report['results'].items():
        for role, m in roles.items():
            p = m['independent_probes']
            table.append(dict(model=name, role=role, mouth_mae=m['mean_mouth_mae'],
                sample_mouth_mae=m['sample_mouth_mae'], full_mae=m['mean_mae'],
                teacher_full_mae=m['teacher_mae'],
                **{f'probe{i}': p[str(i)]['mean'] for i in range(4)},
                **{f'sample_probe{i}': p[str(i)]['sample'] for i in range(4)},
                style_code_ratio=m['style_distance_ratio'],
                same_person_reference_delta=m['reference_motion_delta'],
                cross_person_delta=m['cross_style_motion_delta'],
                seed_diversity=m['seed_diversity'],
                acceleration_ratio=m['sample_acceleration']/m['gt_dynamics']['acceleration']))
    with (root/'comparison.csv').open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    lines = ['# v5 恢复架构短训记录', '',
        f"TRAIN: 256 MEAD + 32 HDTF windows. 总运行 {report['elapsed_seconds']:.1f}s。",
        '64 留句和 24 未见说话人样本。四个情感探针未用于训练；准确率不是 macro-F1。',
        '各模型使用相同跨情感参考；不可与历史 neutral enrollment 全集表直接比较。', '',
        '|模型|集合|均值口型 MAE|采样口型 MAE|4 个均值情感探针|风格码距离比|采样加速度/GT|',
        '|---|---|---:|---:|---|---:|---:|']
    for r in table:
        ps='/'.join(f"{r[f'probe{i}']:.3f}" for i in range(4))
        lines.append(f"|{r['model']}|{r['role']}|{r['mouth_mae']:.4f}|{r['sample_mouth_mae']:.4f}|{ps}|{r['style_code_ratio']:.3f}|{r['acceleration_ratio']:.2f}|")
    lines += ['', '风格码距离比不是独立风格质量证据；采样差异不是动态自然度证据。',
              '教师是 privileged oracle，不是部署结果。任意风格交换没有逐帧 GT。',
              '预算不同，不能据此把 pilot_a/b 差异全部归因于响应场。未达到论文模型验收。']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    for emotion in range(8):
        src={arm:dict(np.load(root/arm/f'held_sentence_emotion_{emotion}.npz',allow_pickle=False))
             for arm in ('retained_v5_cross_ref','repair')}
        base,new=src['retained_v5_cross_ref'],src['repair']
        if int(base['row'])!=int(new['row']): raise ValueError('different query')
        np.testing.assert_array_equal(base['motions'][0],new['motions'][0])
        valid=base['valid'];end=np.flatnonzero(valid)[-1]+1
        record=next(r for r in records if r['role']=='held_sentence' and r['row']==int(base['row']))
        arrays=np.stack([base['motions'][0],base['motions'][3],new['motions'][2],new['motions'][3],new['motions'][4]])[:,:end]
        names=['GT','retained_v5_sample','new_mean','new_flow_sample','visual_teacher']
        # Use the query's recorded supervision, never an assumed rig mask.
        if 'channel_mask' in base:
            mask=base['channel_mask']
        else:
            if not a.data:
                raise ValueError('legacy examples require --data to read actual channel_mask')
            with np.load(Path(a.data)/'held_sentence.npz',allow_pickle=False) as data:
                mask=data['channel_mask'][int(base['row'])]
        np.savez_compressed(root/f'render_{EMOTIONS[emotion]}.npz',channels=np.array(ARKIT_NAMES),
            motions=arrays,mode_names=np.array(names),times=base['times'][:end],valid=valid[:end],
            channel_mask=mask,clip_id=np.array(record['clip_id']))
        fig,axs=plt.subplots(4,1,figsize=(10,8),sharex=True)
        for ax,ch in zip(axs,['jawOpen','mouthSmileLeft','browDownLeft','browInnerUp']):
            for i,name in enumerate(names):
                y=arrays[i,:,ARKIT_NAMES.index(ch)].copy();y[~valid[:end]]=np.nan
                ax.plot(base['times'][:end],y,label=name,linewidth=1.4 if i==0 else .9)
            ax.set_ylabel(ch);ax.grid(alpha=.2)
        axs[0].legend(ncol=3,fontsize=8);axs[-1].set_xlabel('Native seconds')
        fig.suptitle(EMOTIONS[emotion]+': identical query, unmodified coefficients')
        fig.tight_layout();fig.savefig(root/f'curves_{EMOTIONS[emotion]}.png',dpi=150);plt.close(fig)
    print(json.dumps({'rows':len(table),'output':str(root.resolve())}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True)
    p.add_argument('--data',help='prepared arrays for legacy example channel masks');main(p.parse_args())
