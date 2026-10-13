"""Collect scored factor-pilot tables and matched native render inputs."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES
from kinetalk_b0.protocol import EMOTIONS


def main(a):
    root=Path(a.results);report=json.loads((root/'report.json').read_text(encoding='utf8'))
    table=[]
    for arm,roles in report['results'].items():
        for role,r in roles.items():
            p=r['independent_probes']
            table.append(dict(arm=arm,role=role,mean_mouth=r['mean_mouth_mae'],sample_mouth=r['sample_mouth_mae'],
                mean_full=r['mean_mae'],**{f'mean_probe_{i}':v['mean'] for i,v in p.items()},
                **{f'sample_probe_{i}':v['sample'] for i,v in p.items()},
                **{f'style_preserved_probe_{i}':v['style_class_preserved'] for i,v in p.items()},
                style_code_ratio=r['style_distance_ratio'],same_person_delta=r['reference_motion_delta'],
                cross_person_delta=r['cross_style_motion_delta'],seed_diversity=r['seed_diversity'],
                sample_accel_ratio=r['sample_acceleration']/r['gt_dynamics']['acceleration']))
    with (root/'comparison.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    lines=['# v5 因子修复短训对照','',f"总运行 {report['elapsed_seconds']:.1f}s。",
        '同查询、同跨情绪参考；四个评分探针未参与训练。准确率不是 macro-F1。',
        '参考交换没有逐帧动作GT；保持预测类别不等于保持真实情绪或自然个人风格。','',
        '|版本|集合|均值嘴部 MAE|采样嘴部 MAE|4个均值情感探针|4个风格交换类别保持率|',
        '|---|---|---:|---:|---|---|']
    for row in table:
        probe='/'.join(f"{row['mean_probe_'+str(i)]:.3f}" for i in range(4))
        style='/'.join(f"{row['style_preserved_probe_'+str(i)]:.3f}" for i in range(4))
        lines.append(f"|{row['arm']}|{row['role']}|{row['mean_mouth']:.4f}|{row['sample_mouth']:.4f}|{probe}|{style}|")
    lines+=['','TRAIN 256 MEAD +32 HDTF，留句64条、未见人24条/3人。',
        '代码所称趋势是条件均值状态，不是音频可唯一决定的表演或逐帧心理情感真值。',
        '这是诊断结果；不能据此声称三重解耦/三创新已成立。']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    if not a.render_modes:
        print(json.dumps({'output':str(root.resolve()),'rows':len(table)}));return
    modes=[item.split(':') for item in a.render_modes.split(',')]
    for emotion in range(8):
        first=modes[0][0]
        source=dict(np.load(root/first/f'held_sentence_emotion_{emotion}.npz',allow_pickle=False))
        arrays=[source['motions'][0]];names=['GT']
        if a.baseline:
            baseline=dict(np.load(Path(a.baseline)/a.baseline_arm/f'held_sentence_emotion_{emotion}.npz',allow_pickle=False))
            assert int(source['row'])==int(baseline['row'])
            np.testing.assert_array_equal(source['motions'][0],baseline['motions'][0])
            np.testing.assert_array_equal(source['times'],baseline['times'])
            np.testing.assert_array_equal(source['valid'],baseline['valid'])
            slot=list(baseline['mode_names']).index(a.baseline_mode)
            arrays.append(baseline['motions'][slot]);names.append(a.baseline_label)
        for arm,path_name in modes:
            example=dict(np.load(root/arm/f'held_sentence_emotion_{emotion}.npz',allow_pickle=False))
            assert int(example['row'])==int(source['row'])
            np.testing.assert_array_equal(example['motions'][0],source['motions'][0])
            np.testing.assert_array_equal(example['times'],source['times'])
            np.testing.assert_array_equal(example['valid'],source['valid'])
            slot=list(example['mode_names']).index(path_name)
            label={'audio_flow_sample':'sample','audio_modal':'mean'}.get(path_name,path_name)
            arrays.append(example['motions'][slot]);names.append(arm+'_'+label)
        end=np.flatnonzero(source['valid'])[-1]+1;arrays=np.stack(arrays)[:,:end]
        np.savez_compressed(root/f'render_{EMOTIONS[emotion]}.npz',motions=arrays,mode_names=np.array(names),
            channels=np.array(ARKIT_NAMES),valid=source['valid'][:end],times=source['times'][:end],
            channel_mask=source['channel_mask'],row=source['row'],emotion=source['emotion'])
        fig,axes=plt.subplots(4,1,figsize=(10,8),sharex=True)
        for ax,channel in zip(axes,['jawOpen','mouthSmileLeft','browDownLeft','browInnerUp']):
            for k,name in enumerate(names):
                curve=arrays[k,:,ARKIT_NAMES.index(channel)].copy();curve[~source['valid'][:end]]=np.nan
                ax.plot(source['times'][:end],curve,label=name,linewidth=1.4 if k==0 else 1.)
            ax.set_ylabel(channel);ax.grid(alpha=.2)
        axes[0].legend(fontsize=6,ncol=2);axes[-1].set_xlabel('Native seconds')
        fig.suptitle(EMOTIONS[emotion]+': identical query, unmodified coefficients')
        fig.tight_layout();fig.savefig(root/f'curves_{EMOTIONS[emotion]}.png',dpi=150);plt.close(fig)
    print(json.dumps({'output':str(root.resolve()),'rows':len(table)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--baseline')
    p.add_argument('--baseline-arm',default='retained_v5_cross_ref')
    p.add_argument('--baseline-mode',default='audio_flow_sample')
    p.add_argument('--baseline-label',default='retained_v5_sample')
    p.add_argument('--render-modes',help='comma-separated arm:mode_names entries')
    main(p.parse_args())
