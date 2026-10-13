"""Descriptive personal-transfer table; matched performances are not pure style GT."""
import argparse
import csv
import json
from pathlib import Path


def main(path):
    path=Path(path);report=json.loads(path.read_text(encoding='utf8'));rows=[]
    for arm,r in report['results'].items():
        rows.append(dict(arm=arm,**r['summary']))
    with path.with_name('comparison.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    lines=['# 匹配内容/情绪/强度的个人迁移方向审计','',
        f"{report['pair_count']} 个有向配对，{report['source_clip_count']} 条源片段，{report['person_count']} 个人。",
        '配对均为不同人、同句、同情绪、同已知强度；HDTF未知语义片段不进入此审计。',
        '输出统计采用均值/q10/q90、TRAIN通道尺度。余弦为预测交换变化与真实人间统计差异的方向相似度。',
        '这些未对齐的不同表演只提供描述性证据，不能作为纯个人风格真值；配对复用片段，不能当66个独立样本。','',
        '|版本|不迁移目标误差|迁移后误差↓|方向余弦↑|变化幅度|4个情感类别保持率|',
        '|---|---:|---:|---:|---:|---|']
    for r in rows:
        preserved='/'.join(f"{r['probe'+str(i)+'_class_preserved']:.3f}" for i in range(4))
        lines.append(f"|{r['arm']}|{r['unchanged_style_target_error']:.4f}|{r['transferred_style_target_error']:.4f}|{r['cosine']:.4f}|{r['achieved_style_change']:.4f}|{preserved}|")
    lines+=['','类别保持是相对于交换前预测，不能等同于正确情绪。正确率另见原报告。',
        '风格修复增强时，情感保持改善，但个人迁移方向/幅度略减；不据此宣称三重解耦成立。']
    path.with_name('comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);main(p.parse_args().report)
