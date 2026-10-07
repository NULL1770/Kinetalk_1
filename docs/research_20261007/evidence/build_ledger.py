from pathlib import Path
import json,re,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1]
CLAIMS=[
 ('emote_export','Since we treat emotion as a sequence phenomenon', '情感约束是序列层面，不要求表达特征之间逐帧对齐。'),
 ('emote_export','Lip-reading loss:', '保持lip-reading特征而非固定嘴幅度。'),
 ('deeptalk','Generalized Pooling', 'DEE概率情感向量经过池化，不能等同逐帧轨迹。'),
 ('deeptalk','Unlike Lip Vertex', '作者区分多样性生成与单GT几何误差的评价目标。'),
 ('deeptalk','Emotion Consistency Loss.', '独立emotion空间用于生成语义一致性，不是F1主表。'),
 ('mimic_export','Style Encoder The', 'style从时序关系提取后池化。'),
 ('mimic_export','Auxiliary Inverse Classifier', '内容空间有GRL身份抑制。'),
 ('mimic_export','Paired Latent Cycle Losses', 'style/content循环不依赖交换动作GT，但不是因果解耦证明。'),
 ('mimic_export','Style cosine similarity', '风格评价使用独立训练motion分类器特征。'),
 ('diffposetalk_export','3.3.1 Speaking Style Encoder', '形状与短期说话style分别建模。'),
 ('diffposetalk_export','two proximate times', 'style正例建立在临近时间相似假设上。'),
 ('media2face_export','Upper Face Dynamics', 'FDD主要比较上脸时间std。'),
 ('unitalker_export','Pivot Identity Embedding', 'pivot身份用于减轻跨annotation偏差。'),
 ('unitalker_export','reposition the', '音频频率适配位置影响预训练迁移。'),
 ('said_export','5.3. Evaluation Metrics', '主要为同步、多样性、分布指标。'),
 ('exptalk_primary','Emotion encodings of corresponding frame', 'emotion2vec引导逐帧表达latent，但纯情感性仍需验证。'),
 ('exptalk_primary','4.2 Quantitative Evaluation', 'MVE/LVE/EVE/FDD，非本项目motion F1。'),
 ('pestalk','Emotional Styel Library', 'voiceprint与emotion库检索不是独立motion参考泛化。'),
 ('pestalk','(11)', 'Eq11目标方向需公式和代码确认，不照抄。'),
 ('wav2sem_export','3.4. Optimizing Function', '文本语义蒸馏属于内容特征增强。'),
 ('editemotalk','do not report emotion', '作者未报分类accuracy；其替代指标也不能证明时序正确。'),
 ('echo','deterministic anchor', '确定anchor与随机残差已有先例。'),
 ('wildwest_repo','3.4.3. FDD', 'FDD时间std不能区分时间顺序。'),
 ('wildwest_repo','3.4.6. CE', 'Coverage Error属于固定K的best-of-K。'),
 ('wildwest_repo','absence of a direct relationship', '客观指标排序不等于人评排序。'),
 ('perceptual2025','5. Evaluation Metrics', 'MTM/PLRS/SLCC区分时间、可读性和强度。'),
 ('perceptual2025','Speech and Lip Intensity Correlation', 'SLCC为clip层面强度相关。'),
 ('fmreward','65,574 annotated', '人类偏好数据与GT几何误差补充关系。'),
 ('shape_speech','uniform motion gain', '协同发音轨迹结构不能靠uniform gain修复。'),
 ('uncertainty_distill','it excludes the emotional feature', '3DGS内嘴分支与当前rig情感嘴设计不是相同任务。'),
]
rows=[]
for paper,anchor,claim in CLAIMS:
 p=ROOT/'fulltext'/f'{paper}.txt';s=p.read_text(encoding='utf-8')
 hit=re.search(re.escape(anchor),s,re.I)
 if hit is None:raise RuntimeError(f'Missing anchor: {paper}: {anchor}')
 before=s[:hit.start()];pages=list(re.finditer(r'\[PDF PAGE (\d+)\]',before))
 rows.append({'paper':paper,'file':str(p.relative_to(ROOT)), 'page':int(pages[-1].group(1)) if pages else None,'offset':hit.start(),'anchor':anchor,'claim_zh':claim,'excerpt':s[max(0,hit.start()-160):hit.end()+650]})
(ROOT/'evidence'/'claim_ledger.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
sources=[]
for p in sorted((ROOT/'papers').glob('*_retrieval.json')):
 d=json.loads(p.read_text(encoding='utf-8'))
 if not d.get('passed'):continue
 name=d['name'] if 'name' in d else p.stem.removesuffix('_retrieval')
 candidates=[ROOT/'papers'/f'{name}.pdf',ROOT/'papers'/f'{name}_curl.pdf']
 found=next((x for x in candidates if x.exists() and hashlib.sha256(x.read_bytes()).hexdigest()==d['sha256']),None)
 if found is None:raise RuntimeError('Hash unresolved: '+p.name)
 sources.append({'id':name,'url':d['url'],'pdf':str(found.relative_to(ROOT)),'sha256':d['sha256'],'pages':d['pages'],'metadata':str(p.relative_to(ROOT))})
manifest={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified_new_pdfs':len(sources),'sources':sources,'reused_sources':'../../final_experiment/evaluation/diagnostics/paper_protocol_audit_20261007/retrieval.json','notes':['21 new PDF acquisitions; reading scope differs by paper, see synthesis.','ProsodyTalker publisher fetch helper stopped after prolonged non-completion; all completed per-paper manifests preserved.','Wrong PESTalk DOI probe is excluded. Correct PDF DOI is 10.1145/3746027.3755190; independent lookup failed SSL.','ECHO DOI query404; use as arXiv preprint, not independently confirmed acceptance.']}
(ROOT/'evidence'/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'claims':len(rows),'verified_new_pdfs':len(sources)},ensure_ascii=False))
