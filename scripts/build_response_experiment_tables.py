"""Hash-bound development tables; never consumes sealed scores or ranks oracles."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def verified_report(root):
    evaluation=root/'evaluation'
    if not evaluation.exists():evaluation=root/'recovery_v1/evaluation'
    complete=read(root/'seed47/complete.json');report=read(evaluation/'report.json')
    assert read(evaluation/'state.json')['report_sha256']==sha(evaluation/'report.json')
    assert report['checkpoint_sha256']==complete['final_sha256']==sha(root/'seed47/final.pt')
    assert report['clips']==1367 and report['test_loaded'] is False
    return evaluation,report,read(root/'binding.json'),read(root/'seed47/protocol.json')

def canonical(metrics):
    d=dict(metrics)
    for dest,source in [('mouth_displacement_mse','mouth_jaw/displacement_mse'),
        ('jaw_centered_correlation','jawOpen/centered_correlation'),('jaw_q90_q10','jawOpen/pred_q90_q10')]:
        if source in d:d[dest]=d[source]
    return d

def emit(out,name,columns,rows):
    with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    def fmt(x):
        if x is None:return '—'
        if isinstance(x,(float,np.floating)):return f'{x:.6g}' if math.isfinite(x) else '—'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+
                     ['| '+' | '.join(fmt(r.get(k)) for k in columns)+' |' for r in rows])

def build(baseline_root,responses,out,inventory=None):
    out.mkdir(parents=True,exist_ok=True)
    base=read(baseline_root/'report.json');guard=read(baseline_root/'complete.json')
    assert guard['report_sha256']==sha(baseline_root/'report.json') and not base['test_loaded']
    assert guard['per_clip_sha256']==sha(baseline_root/'per_clip_metrics.npz')
    with np.load(baseline_root/'per_clip_metrics.npz',allow_pickle=False) as z:
        ids=z['clip_id'].tolist();labels=z['labels'].copy()
    assert len(ids)==len(set(ids))==1367
    sources=[{'path':str(baseline_root/'report.json'),'sha256':sha(baseline_root/'report.json')}]
    metadata=[];policy_rows={k:[] for k in ('raw','clip_all')};ablation=[];interventions=[];classes=[];groups=[]
    inv=read(inventory)['baselines'] if inventory else None
    for name,data in base['methods'].items():
        if data['provenance']['method']=='kinetalk':continue
        pr=data['provenance'];directory=Path(pr['checkpoint']).parent.name
        if inv is not None:
            assert inv[directory][Path(pr['checkpoint']).name]['sha256']==pr['checkpoint_sha256']
            assert inv[directory][Path(pr['predictions']).name]['sha256']==pr['prediction_sha256']
        metadata.append({'method':name,'fit_clips':12536,'epochs':pr['epochs'],'train_seed':42,
                         'scope':'shared-audio ARKit adaptation','draws':len(data['draw_seeds']),
                         'checkpoint_sha256':pr['checkpoint_sha256']})
        for policy in policy_rows:
            d=data['results'][policy];row={'method':name,**canonical(d['metrics'])}
            row.update({f'F1_{i+1}':v for i,v in enumerate(d['F1'])});policy_rows[policy].append(row)
        perdraw=data['results']['clip_all']['per_draw_probes']
        for emotion in base['classes']:
            classes.append({'method':name,'emotion':emotion,**{f'F1_{i+1}':float(np.mean([x[i]['per_class'][emotion]['f1'] for x in perdraw])) for i in range(4)}})
    per_response={}
    for name,root in responses:
        evaluation,r,binding,protocol=verified_report(root)
        assert r['data_manifest_sha256']==base['data_manifest_sha256'] and r['rig_sha256']==base['rig_sha256']
        assert [p['sha256'] for p in binding['probes']]==[p['sha256'] for p in base['probes']]
        clips=read(evaluation/'per_clip.json');assert clips['clip_ids']==ids
        sources.append({'path':str(evaluation/'report.json'),'sha256':sha(evaluation/'report.json')})
        metadata.append({'method':name,'fit_clips':protocol['data']['fit_clips'],
            'epochs':protocol['args']['epochs'],'updates':read(root/'seed47/complete.json')['updates'],
            'train_seed':protocol['args']['seed'],'scope':'neutral B0 + audio expression + independent reference','draws':'prior mean',
            'checkpoint_sha256':r['checkpoint_sha256']})
        for policy in policy_rows:
            d=r['results']['prior_mean/'+policy];row={'method':name,**canonical(d['metrics'])}
            row.update({f'F1_{i+1}':p['macro_f1'] for i,p in enumerate(d['probes'])});policy_rows[policy].append(row)
        native=policy_rows['clip_all'][-1]
        ablation.append({'method':name,'prior_variance':protocol['config'].get('prior_variance','learned'),
            'center_local':protocol['config'].get('center_local',False),
            'reference_training':protocol['args'].get('reference_training','single'),
            'fit_clips':protocol['data']['fit_clips'],'updates':metadata[-1]['updates'],**native})
        for mode,d in r['interventions'].items():
            interventions.append({'method':name,'intervention':mode,'n':d['n'],**canonical(d['metrics'])})
        for emotion in base['classes']:
            classes.append({'method':name,'emotion':emotion,**{f'F1_{i+1}':p['per_class'][emotion]['f1'] for i,p in enumerate(r['results']['prior_mean/clip_all']['probes'])}})
        values=clips['results']['prior_mean/clip_all'];per_response[name]=values
        for kind,keys in [('emotion',base['classes']),('speaker',sorted(set(cid.split('_')[1] for cid in ids)))]:
            for j,key in enumerate(keys):
                indexes=[i for i,cid in enumerate(ids) if (labels[i]==j if kind=='emotion' else cid.split('_')[1]==key)]
                assert indexes
                groups.append({'method':name,'group_type':kind,'group':key,'n':len(indexes),
                    **{k:float(np.mean([values[i][k] for i in indexes])) for k in ['arkit_mbe','arkit_lbe','lve_mean_mm_mean','eve_mean_mm_mean','jawOpen/centered_correlation','jawOpen/pred_q90_q10']}})
    sections=[]
    tables=[('table1_geometry',['method','lve_mean_mm_mean','eve_mean_mm_mean','vertex_lve_sqrt_mean','eye_forehead_eve_sqrt_mean','vertex_fdd_absolute_mm2_mean']),
            ('table2_coefficients',['method','arkit_mbe','arkit_lbe','arkit_fdd_absolute','arkit_fdd_signed']),
            ('table3_dynamics_semantics',['method','mouth_displacement_mse','jaw_centered_correlation','jaw_q90_q10','F1_1','F1_2','F1_3','F1_4'])]
    for policy,rows in policy_rows.items():
        for name,columns in tables:
            table=emit(out,name+'_'+policy,columns,rows)
            if policy=='clip_all':sections+=['## '+name,'',table,'']
    for name,columns,rows in [
        ('table4_trained_ablation',['method','prior_variance','center_local','reference_training','fit_clips','updates','arkit_mbe','arkit_lbe','lve_mean_mm_mean','jaw_centered_correlation','jaw_q90_q10','F1_1','F1_2','F1_3','F1_4'],ablation),
        ('table5_inference_interventions',['method','intervention','n','arkit_mbe','arkit_lbe','jaw_centered_correlation','jaw_q90_q10','brows/centered_correlation'],interventions),
        ('table6_emotion_breakdown',['method','emotion','F1_1','F1_2','F1_3','F1_4'],classes),
        ('table7_geometry_groups',['method','group_type','group','n','arkit_mbe','arkit_lbe','lve_mean_mm_mean','eve_mean_mm_mean','jawOpen/centered_correlation','jawOpen/pred_q90_q10'],groups),
        ('table8_training_protocol',['method','scope','fit_clips','epochs','updates','train_seed','draws'],metadata)]:
        sections+=['## '+name,'',emit(out,name,columns,rows),'']
    intro=['# 开发集实验表：同协议基线与训练消融','',
        'Material Passport: MODE=validate; STATUS=ANALYZED; sources为已完成、SHA核验的原报告；本次汇总未重新训练/重新推理。',
        '全部1,367条开发集、相同原生有效帧、51观察通道、固定rig和四冻结probe。主表统一clip[0,1]；raw完整表同时导出。未读取sealed。',
        '基线是ARKit共享音频改编，非原论文官方分数。训练数据/预算/seed不同，见table8；不是严格等预算SOTA结论。FaceDiffuser原预测无checkpoint哈希绑定，原生时钟/GT一致，历史报告有此限制；其余三基线原首batch权重重放通过。本次远端checkpoint/预测SHA重新核验。',
        'Lip/Expression mean为顶点欧氏距离平均(mm)；LVE-max/EVE-max另列，不能混用。这里mesh FDD为上区平方位移能量时间std差的绝对值(mm²)，是沿用实现，不能将所有论文同名FDD当同公式。MBE/LBE为现有系数区域L2误差定义，不能拿其他rig的裸数值比较。',
        '四F1依次为原128/原64/辅助128/辅助64整段统计probe；不是逐帧情感GT。jaw范围GT平均0.175279，范围越大不必越好；相关与位移误差联合判断，不能以平滑低误差替代正确动态。',
        'table4只列实际重训练消融；table5是冻结推理干预，不混为训练模块消融。oracle不进入方法排名。仅单seed结果，不报伪造多seed误差条。',
        '论文实践：EmoTalk Table5分别检查emotion disentangling encoder、emotion-guided attention、Lvel/Lcls、HDTF数据及encoder替换；MEDTalk §4.5/Table3检查overlap exchange、cycle exchange、disentangle、intensity和text。本项目应围绕自己的g/u职责、style/reference与teacher/student提出并重训练消融；不能直接照搬其模块名。',
        '统计核验11/11已检查：分组结果另列以检查聚合反转；不由3人推断总体个体；MEAD演员/伪GT选择偏差保留；无协变量调整/collider推断；报告8类而非只happy；不选极端片段宣称回归改善；所有1367无删坏样本；所有probe/raw保留；明确多轮开发探索；冻结交换非因果独立证明；教师看到GT非部署预测/因果反向结论。没有开展显著性检验，三开发身份及单训练seed不足以证明统计稳定。','']
    (out/'README.md').write_text('\n'.join(intro+sections),encoding='utf8')
    report={'sources':sources,'test_loaded':False,'rows':policy_rows,'trained_ablations':ablation,'protocols':metadata,
            'response_groups':groups,'metrics_unchanged':True,'validation_only':True}
    (out/'tables.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--response',action='append',required=True,help='Display label=local experiment root')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inventory',type=Path)
    a=p.parse_args();r=build(a.baseline,[(s.split('=',1)[0],Path(s.split('=',1)[1])) for s in a.response],a.output,a.inventory)
    print(json.dumps({'methods':len(r['rows']['clip_all']),'responses':len(r['trained_ablations']),'output':str(a.output),'test_loaded':False}))
