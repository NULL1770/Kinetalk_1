"""Assemble verified development results, never mix them with final benchmarks."""
from pathlib import Path
import csv
import hashlib
import html
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.evaluate_paper_coefficients import cluster_interval, write_csv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/paper_metrics_20260919'
def load(p):return json.loads((OUT/p).read_text(encoding='utf8'))
def read_csv(p):return list(csv.DictReader((OUT/p).open(encoding='utf-8-sig')))
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'---|'*len(headers)+'\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)
def tex(path,headers,rows):
    esc=lambda x:str(x).replace('_',r'\_').replace('%',r'\%')
    text=['\\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(map(esc,headers))+r' \\',r'\midrule']
    text += [' & '.join(map(esc,r))+r' \\' for r in rows]
    text += [r'\bottomrule',r'\end{tabular}']
    path.write_text('\n'.join(text)+'\n',encoding='utf8')

def main():
    (OUT/'figures').mkdir(exist_ok=True);(OUT/'latex').mkdir(exist_ok=True)
    c=load('current_coefficients/coefficient_metrics.json');p=load('paper_motion_probes_v3.json')
    assert c['schema'].endswith('v2_raw_reference') and p['schema']=='paper_motion_probe_v3'
    d={r['arm']:r for r in c['distribution_metrics']};s=c['summaries'];arms=list(s)
    main_rows=[{'arm':a,**{k:s[a][k]['mean'] for k in ('lip_mae','mouth_mae','upper_mae','upper_std_absolute_gap','upper_intensity_mae','upper_velocity_mae_per_second','upper_pairwise_rms_diversity')},'centered_fair_es':d[a]['centered_fair_es'],'raw_fair_es':d[a]['raw_fair_es']} for a in arms]
    write_csv(OUT/'main_results.csv',main_rows)
    dist=read_csv('current_coefficients/distribution_per_clip.csv');by={a:{r['clip_id']:r for r in dist if r['arm']==a} for a in arms};paired=[]
    for a,b in [('prior','base'),('audio','prior'),('audio','matched_static'),('audio','static'),('audio','reverse'),('audio','mismatch')]:
        for metric in ('centered_fair_es','raw_fair_es'):
            ids=sorted(by[a]);delta=[float(by[a][i][metric])-float(by[b][i][metric]) for i in ids];z=cluster_interval(delta,[by[a][i]['sentence'] for i in ids])
            paired.append(dict(candidate=a,control=b,metric=metric,mean_delta=z['mean'],ci95_low=z['ci95'][0],ci95_high=z['ci95'][1]))
    write_csv(OUT/'distribution_paired_ablations.csv',paired)
    conditions=[]
    for name in ('affect_global','affect_global_intensity','identity_code','b9'):
        r=load(f'condition_probe_v2_{name}.json')
        for task,v in r['inner_validation'].items():
            conditions.append(dict(feature=name,task=task,n=206,accuracy=v['accuracy'],uar=v['balanced_accuracy'],macro_f1=v['macro_f1'],scope='input feature separability; inherited upstream exposure'))
    write_csv(OUT/'condition_separability.csv',conditions)
    probes=[]
    for name,v in [('real_inner206',p['real_validation']),*p['outer'].items()]:
        for task in ('emotion','motion_speaker_signature'):
            z=v[task];probes.append(dict(arm=name,task=task,n=v['n'],accuracy=z['accuracy'],uar=z['balanced_accuracy'],macro_f1=z['macro_f1'],scope='diagnostic only; generated features have severe domain shift'))
    write_csv(OUT/'motion_probe_diagnostic.csv',probes)
    q=load('motion_condition_probe_v1/report.json');coarse=[]
    for frames,r in q['results'].items():
        for arm,v in r['inner_validation'].items():
            z=v['sentence_equal'];coarse.append(dict(window_seconds=int(frames)/25,arm=arm,normalized_mse=z['mean_normalized_mse'],activity_mse=z['activity_normalized_mse'],direction_mse=z['direction_normalized_mse']))
    write_csv(OUT/'coarse_condition_results.csv',coarse)
    smallheaders=['Arm','Lip MAE','Upper MAE','Std gap','Intensity MAE','Centered ES','Diversity']
    smallrows=[[r['arm']]+[f'{r[k]:.6f}' for k in ('lip_mae','upper_mae','upper_std_absolute_gap','upper_intensity_mae','centered_fair_es','upper_pairwise_rms_diversity')] for r in main_rows]
    tex(OUT/'latex/table_current_coefficients.tex',smallheaders,smallrows)
    tex(OUT/'latex/table_distribution_ablation.tex',['Candidate','Control','Delta centered ES','CI low','CI high'],[[r['candidate'],r['control'],*[f'{r[k]:.6f}' for k in ('mean_delta','ci95_low','ci95_high')]] for r in paired if r['metric']=='centered_fair_es'])
    tex(OUT/'latex/table_condition_separability.tex',['Feature','Task','Acc','UAR','F1'],[[r['feature'],r['task'],*[f'{r[k]:.4f}' for k in ('accuracy','uar','macro_f1')]] for r in conditions])
    tex(OUT/'latex/table_motion_probe.tex',['Arm','Task','N','Acc','UAR','F1'],[[r['arm'],r['task'],r['n'],*[f'{r[k]:.4f}' for k in ('accuracy','uar','macro_f1')]] for r in probes if r['arm'] in ('real_inner206','reference','base','audio/seed42')])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    colors=['#94a3b8','#2563eb','#f97316','#0891b2','#14b8a6','#a78bfa','#64748b']
    fig,axs=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    for ax,key,title in zip(axs,['upper_mae','centered_fair_es'],['Upper-face coefficient error','Centered trajectory distribution score']):
        values=[r[key] for r in main_rows]; ax.bar(range(7),values,color=colors);ax.set_xticks(range(7),['Base','Prior','Audio','Trained\nstatic','Static','Reverse','Mismatch'],rotation=20);ax.set_ylabel(key.replace('_',' ')+' (lower is better)');ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Development diagnostics: 64 clips / 7 sentences / all 4 fixed draws')
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'figures/main_comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,4),layout='constrained')
    for arm,color in [('matched_static','#2563eb'),('audio','#f97316'),('audio_reverse','#64748b')]:
        rr=[r for r in coarse if r['arm']==arm];ax.plot([r['window_seconds'] for r in rr],[r['normalized_mse'] for r in rr],'o-',label=arm.replace('_',' '),color=color)
    ax.set(xlabel='Window length (seconds)',ylabel='Sentence-equal normalized MSE',title='New coarse-condition probe: 206 validation clips / 5 sentences');ax.legend();ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'figures/coarse_probe.{ext}',dpi=180)
    plt.close(fig)
    summary=f'''# KineTalk 论文实验数据包 · 2026-09-19

今天可以开始写方法、协议和开发实验。当前证据支持“运动先验改善眉眼运动分布且保持口部预测”，尚不支持“局部音频时序已学会”或“达到论文投稿效果”。本包没有外部方法复现分数，不构造SOTA排名。

## 1. 固定评估口径

本轮只对已完成 bounded_audio_formal24000 做同批评估：613条拟合、206条内层验证；生成对比为64条、7句、15个已见说话人、4类情感的历史开发集。该64条已在多轮调试使用，不是最终封存测试。随机方法固定4个种子，逐draw打分后按clip等权平均；不挑种子、不平均轨迹、不拟合时差、不裁剪原始预测。句子簇bootstrap 4096次仅作描述区间。

参考来自原始target52/valid/channel_mask；显示填充值不作GT。Lip为18:41，JawMouth为14:41，Upper9为[41,42,43,44,45,5,6,12,13]，不包含眨眼、眼球注视或头姿。速度单位为系数/秒；逐帧强度为Upper9绝对值均值，是系数代理，不是心理情绪测量。

## 2. 当前主表

{table(smallheaders,smallrows)}

误差、Std gap、ES越低越好；Diversity不是越大越好，须结合ES。基座确定性，diversity=0。Centered ES去除原生连续段均值，衡量中心化轨迹分布；Raw ES保留偏置。两者都不能单独证明音频事件对齐。

原先验相对基座：Upper MAE降低{(1-s['prior']['upper_mae']['mean']/s['base']['upper_mae']['mean'])*100:.2f}%，Centered ES降低{(1-d['prior']['centered_fair_es']/d['base']['centered_fair_es'])*100:.2f}%，速度MAE降低{(1-s['prior']['upper_velocity_mae_per_second']['mean']/s['base']['upper_velocity_mae_per_second']['mean'])*100:.2f}%。这些属于同数据同表示内部对比。

Audio相对Prior虽降低Upper MAE，但Centered ES变差{(d['audio']['centered_fair_es']/d['prior']['centered_fair_es']-1)*100:.2f}%；原始越界率为{s['audio']['raw_upper_oob_fraction']['mean']*100:.2f}%。独立静态训练的Upper MAE几乎相同，因此MAE提升不能归因于逐帧音频时序。保留原先验为对照，未替换默认模型。

口型：所有方案Lip MAE={s['base']['lip_mae']['mean']:.6f}、JawMouth MAE={s['base']['mouth_mae']['mean']:.6f}，因为本轮保护其余43个通道；这证明本轮没有改变口部数值，不能证明原基座的感知口型同步已达标。

## 3. 消融（Centered ES差值，候选减对照）

{table(['候选','对照','差值','句子簇95%区间'],[[r['candidate'],r['control'],f"{r['mean_delta']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]"] for r in paired if r['metric']=='centered_fair_es'])}

Trained static为同预算独立训练静态适配器；Static为audio模型输入时序置静态；Reverse和Mismatch为输入干预，不能替代独立训练消融。Audio与Trained static比较不通过，因此不把“优于反转/错配”写成有效时序学习的充分证据。

## 4. 情感与身份，哪些已经有证据

输入特征的固定线性探针只在613条真实拟合、206条跨句验证：

{table(['输入','任务','Acc','UAR','Macro F1'],[[r['feature'],r['task'],*[f"{r[k]*100:.2f}%" for k in ('accuracy','uar','macro_f1')]] for r in conditions])}

全局情感特征含有明显的情感标签信息；identity code能区分该内部闭集说话人。但affect_global仍可识别说话人（Acc 61.65%，15类），不能写成完全身份解耦。上述分数不评价生成脸，也继承上游特征历史曝光。

独立真实动作统计探针：内层真实动作情感UAR 86.30%，说话人动作特征UAR 95.90%；历史64条真实动作分别84.38%和96.67%。生成基座及各Upper9方案情感UAR31.25%、说话人动作特征UAR6.67%。**这些生成分数只放诊断附录，不作主表质量结论**：原基座统计出现严重域偏移，部分真实几乎恒定通道的fit std约1e-7至1e-5，标准化后绝对值99分位达22823.83（真实3.34），足以支配线性分类。需校准独立评估器并验证人类判断。不能据此定位到眉眼，更不能等价外观身份。

更正初版评估：v1/v2存在嵌套划分、stage合并/重复、说话人字符串映射问题，已标记废弃。v3使用规范元数据、原始掩码、逐arm逐seed，并保存权重和哈希；64条说话人均为已见，早前“未知身份无法评估”解释不成立。

## 5. 并行动作优化结果

新增粗粒度活动/方向条件预测已跑完：固定0.4/1/2秒窗口，4维log速度活动+4维有符号位移，固定岭回归，局部audio与独立trained-static对比。未用64条开发目标拟合或选模型。

{table(['窗口(s)','Trained static MSE','Audio MSE','Reverse MSE'],[[w]+[f"{next(r['normalized_mse'] for r in coarse if r['window_seconds']==w and r['arm']==a):.6f}" for a in ('matched_static','audio','audio_reverse')] for w in (.4,1.,2.)])}

等句子×尺度×8目标的Audio−Trained static为+0.012007，探索区间[-0.000056,+0.024069]；仅1/5句改善。方向条件差值+0.019189，区间[+0.003130,+0.035248]。该线性条件方案未通过，故未接生成器，不再花长训预算验证已失败的前提。结果仅否定这一紧凑线性方案，不证明音频信息原则上不可预测。

下一步优先修复可检验链条：先在拟合数据审计通道量纲与近恒定通道；校准能泛化到生成域的情感/身份评估；随后在固定613/206划分比较非线性音频事件概率/持续时间预测与同容量静态预测。只有跨句、跨种子稳定超过独立静态，并改善动态分布而不损口部，才接入冻结先验。保留随机过程生成多解，不要求唯一GT相位重建。当前无新生成器训练在运行。

## 6. 参考文章指标如何使用

MEDTalk（arXiv:2507.06071v2）在174维MetaHuman rig上报告MLE/MEE/EIE及std差FRD；MEE不是情感分类准确率，文中Emotion Acc是主观评分。FRD原文为有符号std差，不能无条件称越低越好。SubtleTalk（arXiv:2608.06408v1）使用FLAME顶点LVE/FDD/HDD；std型FDD不证明局部音频时间对齐。DEITalk原文访问受限、公开仓库未给完整指标协议，本包不编造其公式。

我们的系数代理在物理单位、拓扑和通道集合上不同，不能把本表数字直接放到这几篇原文表中排名。已缓存原文和来源哈希，详见PAPER_METRIC_SOURCE_MAP_20260919.md。

## 7. 今天可以写与还需补齐

可以写：方法动机、冻结口部/全局基座+随机运动先验、协议、内部主表、动态分布收益、静态/反转/错配消融、局部音频局限。先使用下附英文实验草稿。

投稿前待补：封存且无历史曝光的最终测试；同数据/表示/预处理下外部基线重跑；独立情感/身份与口型感知验证；自然性、音画同步与身份用户研究；正式方法部件消融（去全局情感、去身份、去蒸馏等需重新训练，当前尚无同协议数值）。不要用不同时期数据池/检查点拼成消融表。

## 8. 文件

- paper_results.xlsx：工作簿（主表、消融、条件探针、粗条件预测、情感身份诊断、每片/每组明细）。
- main_results.csv、distribution_paired_ablations.csv：直接取数。
- current_coefficients/：所有原始系数统计、区间、情感/身份分组。
- latex/：可复制LaTeX表；figures/：PNG/PDF/SVG图。
- paper_experiments_draft.md：英文实验段落初稿。
- paper_motion_probes_v3.*：经纠正的诊断探针与参数；probe_domain_diagnostic.json说明其局限。
- motion_condition_probe_v1/：新的优化验证，含预测/权重/清单与完整结果。
'''
    (OUT/'REPORT.md').write_text(summary,encoding='utf8')
    (ROOT/'docs/PAPER_RESULTS_20260919.md').write_text(summary,encoding='utf8')
    draft='''# Experimental section draft (development results only)

## Evaluation protocol
We evaluate ARKit52 coefficient predictions using a fixed development cohort of 64 clips from seven sentence groups, 15 previously seen speakers, and four emotion classes. This cohort was used in earlier development and is not a sealed test set. The prior and adapter use 613 fitting clips and 206 inner-validation clips. For stochastic configurations we report all four pre-specified draws, score each draw before averaging, and assign equal weight to each clip. Descriptive 95% intervals use 4,096 sentence-cluster bootstrap resamples. All errors use raw coefficients, native timestamps, and original per-channel observation masks. We do not optimize temporal shifts or select favorable seeds.

## Metrics and comparisons
We measure mouth coefficient error, upper-face coefficient error, temporal standard-deviation discrepancy, velocity error, coefficient-intensity error, sample diversity, and raw/centered fair energy scores. Our nine-channel upper-face region covers brow and eye-opening/squint controls; it excludes blinking, gaze, and head pose. These coefficient-space measures are inspired by prior work but are not numerically interchangeable with MetaHuman rig or FLAME vertex metrics. Centered energy score removes each contiguous run's mean and measures trajectory-distribution agreement; it alone does not establish audio-event alignment.

## Development results
The frozen motion prior decreases upper-face MAE from 0.060354 to 0.051790 and centered energy score from 0.378113 to 0.238428. Mouth predictions remain numerically unchanged by construction (lip MAE 0.051844). A bounded audio adapter further decreases upper-face MAE to 0.046665, but its centered energy score worsens to 0.248502. A separately trained static adapter achieves nearly the same upper-face MAE (0.046827) and a better centered energy score (0.240019). Consequently, these experiments support the distributional benefit of the motion prior, but do not establish an incremental time-local audio-conditioning benefit.

## Diagnostics and limitations
Source-condition linear probes show high emotion and closed-set speaker separability on the internal split, which does not measure generated-face quality or guarantee disentanglement. A real-motion probe has severe distribution shift on generated coefficients, especially in channels with near-zero training variance; generated classification scores are therefore reported only as diagnostics. Formal identity, emotion and perceptual lip-sync claims require independently validated evaluators and/or blinded human assessment. External methods must be rerun under a common representation and split before making comparative performance claims. Further tests should use a newly sealed final cohort.

## Tables and figures
Use latex/table_current_coefficients.tex and latex/table_distribution_ablation.tex with booktabs. Figures in figures/ are provided as PNG, PDF and SVG. Do not copy historical run12/v5 numbers into the current ablation table.
'''
    (OUT/'paper_experiments_draft.md').write_text(draft,encoding='utf8')
    digest={'schema':'paper_results_digest_v2_audited','scope':c['scope'],'coefficient_rows':main_rows,'distribution_ablations':paired,'condition_probes':conditions,'motion_probe_diagnostic':probes,'coarse_probe':coarse,'generated_probe_quality_claim_allowed':False,'generator_modified':False,'default_replaced':False}
    (OUT/'paper_results_digest.json').write_text(json.dumps(digest,ensure_ascii=False,indent=2),encoding='utf8')
    def ht(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
    page='''<!doctype html><meta charset="utf-8"><title>KineTalk · 论文实验数据</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f2f5fa;color:#18243b;margin:0}main{max-width:1180px;margin:32px auto;padding:36px;background:white;border-radius:14px}h1{font-size:30px}h2{margin-top:36px}p{max-width:1000px}a{color:#245ed1}table{border-collapse:collapse;width:100%;font-size:14px}th{background:#213452;color:white}td,th{padding:11px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}.notice{background:#fff4df;border-left:4px solid #edaa36;padding:16px}img{width:100%}.links{display:flex;gap:22px;flex-wrap:wrap}</style><main><p>KINETALK / 2026-09-19</p><h1>论文实验数据与消融</h1><p class="notice">已完成的开发集评估：64条、7句、4个固定生成种子。运动先验改善动态分布；逐帧音频适配尚未通过独立静态对照。不能当作最终测试或外部方法排名。</p><div class="links"><a href="paper_results.xlsx">Excel 工作簿</a><a href="REPORT.md">完整中文报告</a><a href="paper_experiments_draft.md">英文实验草稿</a><a href="main_results.csv">主表 CSV</a><a href="latex/table_current_coefficients.tex">LaTeX 主表</a><a href="../bounded_audio_20260919/review/index.html">已有视频对比</a></div><h2>当前同批指标</h2>'''+ht(smallheaders,smallrows)+'''<p>MAE、Std gap、ES越低越好；多样性须与分布保真一起看。口部数值未改，不等价于口型质量认证。</p><img src="figures/main_comparison.png"><h2>配对消融</h2>'''+ht(['候选','对照','Δ Centered ES','95%描述区间'],[[r['candidate'],r['control'],f"{r['mean_delta']:+.6f}",f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]"] for r in paired if r['metric']=='centered_fair_es'])+'''<h2>情感、身份的证据边界</h2><p>全局情感输入特征：内部验证情感UAR 99.38%；identity code：闭集说话人识别100%。这不是生成质量。真实动作统计探针在生成域有严重数值偏移，生成识别低分只放诊断附录；暂不作情感或外观身份的成功/失败结论。</p><h2>新优化验证</h2><p>粗粒度活动/方向预测已完成。总体Audio−Trained static MSE为+0.012007，未通过，因此没有将该条件接入生成器长训。</p><img src="figures/coarse_probe.png"><h2>论文使用范围</h2><p>可以写方法、协议、内部主表和消融。还需封存最终测试、统一条件外部基线、独立感知验证。MEDTalk的MetaHuman指标、SubtleTalk的FLAME指标不能直接与本表ARKit系数数字排名。</p><p><a href="current_coefficients/coefficient_summary.csv">全部均值与区间</a> · <a href="current_coefficients/subgroup_metrics.csv">按情感/身份分组</a> · <a href="motion_probe_diagnostic.csv">动作探针诊断</a> · <a href="coarse_condition_results.csv">新条件预测结果</a></p></main>'''
    (OUT/'index.html').write_text(page,encoding='utf8')
    print(json.dumps({'report':str(OUT/'REPORT.md'),'rows':len(main_rows),'ablations':len(paired)}))

if __name__=='__main__':main()
