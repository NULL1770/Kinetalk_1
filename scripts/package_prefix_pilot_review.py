"""Package every preselected pilot clip; upper curves only, never a full-face demo."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_temporal_repair import read, sha

CHANNELS = [(41,'browDownLeft'),(42,'browDownRight'),(43,'browInnerUp'),
            (44,'browOuterUpLeft'),(45,'browOuterUpRight'),(5,'eyeSquintLeft'),
            (6,'eyeWideLeft'),(12,'eyeSquintRight'),(13,'eyeWideRight')]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root;curves={};reports={};sources={}
    for arm in ('no_prefix','teacher_prefix'):
        path=root/'pilot30'/arm/'curves.pt'
        digest=sha(path)
        if digest!=read(path.with_name('complete.json'))['curves_sha256']:
            raise ValueError('Pilot curve hash differs')
        curves[arm]=torch.load(path,map_location='cpu',weights_only=False)
        reports[arm]=read(path.with_name('evaluation.json'))
        sources[arm]={'path':str(path.resolve()),'sha256':digest}
    ref=curves['no_prefix'];prefix=curves['teacher_prefix']
    if ref['clip_id']!=prefix['clip_id'] or not torch.equal(ref['target'],prefix['target']):
        raise ValueError('Pilot membership/targets differ')
    if len(ref['clip_id'])!=8:raise ValueError('Expected the complete fixed eight-clip pilot')
    pictures=root/'curves';pictures.mkdir(exist_ok=True)
    cards=[]
    for ix,cid in enumerate(ref['clip_id']):
        valid=ref['valid'][ix].numpy();clock=np.arange(len(valid))/25
        fig,axes=plt.subplots(3,3,figsize=(13.5,8),sharex=True)
        lines=[('Tracked reference',ref['target'][ix], '#171717','-'),
               ('No prefix',ref['predictions']['42/full'][ix],'#999999','-'),
               ('Generated prefix',prefix['predictions']['42/full'][ix],'#087f8c','-')]
        for ax,(channel,name) in zip(axes.flat,CHANNELS):
            for label,value,color,style in lines:
                y=value[:,channel].numpy().copy();y[~valid]=np.nan
                ax.plot(clock,y,label=label,color=color,linestyle=style,linewidth=1.2)
            for boundary in range(16,len(valid),16):ax.axvline(boundary/25,color='#df942d',alpha=.35,linewidth=.8)
            ax.axhline(0,color='#aaaaaa',linewidth=.5);ax.set_title(name,fontsize=10);ax.grid(alpha=.13)
        for ax in axes[-1]:ax.set_xlabel('Seconds (25 Hz)')
        fig.suptitle(cid+' | fixed seed 42 | TRAINING reconstruction',fontsize=10)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3)
        fig.tight_layout(rect=(0,.04,1,.96));filename=f'clip_{ix:02d}.png'
        fig.savefig(pictures/filename,dpi=130);plt.close(fig)
        cards.append(f'<details><summary>{html.escape(cid)}</summary><img loading="lazy" src="curves/{filename}" alt="Nine raw upper-face coefficient curves"></details>')
    rows=[];summary={}
    for group,label in [('brows','眉部'),('eyes_expression','表情眼部')]:
        for arm,arm_label in [('no_prefix','无前缀'),('teacher_prefix','生成前缀')]:
            modes=[reports[arm]['modes'][f'{seed}/full'] for seed in (42,123,2026)]
            values={metric:float(np.mean([m['metrics'][group][metric] for m in modes]))
                    for metric in ('raw_mse','centered_mse','centered_correlation','outside_fraction')}
            values['boundary_rms']=float(np.mean([m['boundaries'][group]['chunk_boundary']['pred_rms'] for m in modes]))
            summary[f'{group}/{arm}']=values
            rows.append(f'<tr><td>{label}</td><td>{arm_label}</td><td>{values["raw_mse"]:.6f}</td>'
                f'<td>{values["centered_mse"]:.6f}</td><td>{values["centered_correlation"]:.4f}</td>'
                f'<td>{values["boundary_rms"]:.5f}</td><td>{values["outside_fraction"]:.2%}</td></tr>')
    (root/'pilot_summary.json').write_text(json.dumps({'scope':'8 fixed exposed fit clips, cumulative30epochs180updates per arm',
        'sources':sources,'three_seed_means':summary,'test_loaded':False,'plots_seed':42,'clipping':False},indent=2),encoding='utf8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>显式动作前缀 · 小集诊断</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1160px;margin:35px auto;padding:0 22px;color:#20303b;background:#f5f6f8}h1{font-size:29px}section,details{background:white;border-radius:10px;padding:18px 23px;margin:18px 0}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #dde2e7;padding:8px}summary{cursor:pointer}a{color:#087f8c}.note{border-left:5px solid #dd9933}</style>
<h1>显式动作前缀：小集诊断</h1><p>2026-09-17 · 固定的 8 条 fit 片段 · 两组累计各 30 小集轮 / 180 次更新</p>
<section class="note"><b>连续性有改善，尚未证明推广有效。</b><p>两组在同一已暴露小集上训练。生成前缀的接缝位移下降，但眼部原始误差和眉部越界有代价。真实前缀接收门槛经过指标对象更正、一次固定延长后通过；这不是原始 15 轮试验通过，也不是音频能预测目标动作的证据。</p><p>本页仅展示九个上脸系数。小集文件其余 43 通道是未评分的零占位，不能据此评价口型、身份或整体表情。</p></section>
<section><h2>完整三 seed 均值</h2><p>部署方式：第一块无前缀，之后仅用自己生成的过去。GT oracle 不进入此表。数值未经裁剪或增益调整。</p><table><thead><tr><th>区域</th><th>方法</th><th>原始 MSE</th><th>中心 MSE</th><th>中心相关</th><th>接缝 RMS</th><th>越界比例</th></tr></thead><tbody>ROWS</tbody></table></section>
<section><h2>所有固定片段的原始曲线</h2><p>固定 seed 42；橙线为 16 帧分块边界。展开查看全部九通道，未按生成效果挑样例。灰线为无前缀对照，蓝绿线为生成前缀。</p>CARDS</section>
<section><h2>正式实验</h2><p>正式两组各 12 轮，从同一个原始 history12/no_history 基座重新开始，不使用小集适配权重。身份、全局情感和声学条件冻结，其余 43 通道复制已有完整模型的输出。训练后按固定 405 条内部开发片段、三 seed 比较；结果待正式训练完成。</p><p><a href="RESULTS.md">详细小集结论</a> · <a href="../../docs/PREFIX_FORMAL_PROTOCOL_20260917.md">正式实验协议</a> · <a href="pilot30/receiver_gate.json">修正后的接收诊断</a> · <a href="pilot_summary.json">数据来源与均值</a></p></section></html>'''
    (root/'index.html').write_text(page.replace('ROWS',''.join(rows)).replace('CARDS',''.join(cards)),encoding='utf8')
    print(root/'index.html')


if __name__=='__main__':main()
