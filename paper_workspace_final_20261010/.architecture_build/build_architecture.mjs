import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const root='D:/科研/小论文', build=root+'/.architecture_build', out=root+'/架构图';
const skill='C:/Users/zhh/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const python='C:/Users/zhh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const C={ink:'#233245',gray:'#697687',line:'#485B71',light:'#F7F9FC',blue:'#377EA6',bl:'#E6F1F8',green:'#398477',gl:'#E6F3EE',purple:'#7864A7',pl:'#F0ECF8',orange:'#B57B38',ol:'#FBF1E1',red:'#AF6471',rl:'#F7EAF0',border:'#C9D2DD'};
const deck=Presentation.create({slideSize:{width:1600,height:900}});
let count=0;
function shape(s,x,y,w,h,fill='none',stroke='none',r=0,name='shape'){return s.shapes.add({geometry:r?'roundRect':'rect',name:name+'_'+(++count),position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:stroke==='none'?0:1.5},...(r?{borderRadius:r}:{})});}
function txt(s,x,y,w,h,text,size=22,color=C.ink,bold=false,align='center',font='Arial'){
 const q=shape(s,x,y,w,h);q.text=text;q.text.style={typeface:font,fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:1,right:1,top:0,bottom:0}};return q;
}
function box(s,x,y,w,h,text,fill=C.light,stroke=C.border,size=22,font='Arial'){
 const q=shape(s,x,y,w,h,fill,stroke,9,'module');q.text=text;q.text.style={typeface:font,fontSize:size,color:C.ink,alignment:'center',verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:8,right:8,top:5,bottom:5}};return q;
}
function point(s,x,y){return shape(s,x-.2,y-.2,.4,.4);}
function route(s,pts,color=C.line,dash=false,arrow=true,width=2){
 for(let i=1;i<pts.length;i++){
 const a=point(s,...pts[i-1]),b=point(s,...pts[i]);
 const q=s.shapes.connect(a,b,{kind:'straight',line:{fill:color,width,style:dash?'dashed':'solid'},...(arrow&&i===pts.length-1?{tail:{type:'triangle',width:'sm',length:'sm'}}:{})});q.bringToFront();
 }
}
function circ(s,x,y,d,label,color=C.ink){const a=s.shapes.add({geometry:'ellipse',name:'operator_'+(++count),position:{left:x,top:y,width:d,height:d},fill:'#FFFFFF',line:{fill:color,width:1.8}});a.text=label;a.text.style={typeface:'Arial',fontSize:d*.57,color,alignment:'center',verticalAlignment:'middle',insets:{left:0,right:0,top:0,bottom:0}};return a;}
function panel(s,x,y,w,h,label,color,fill,font='Arial'){shape(s,x,y,w,h,fill,color,12,'region');txt(s,x+18,y+8,w-36,30,label,23,color,true,'left',font);}
function placeholder(s,x,y,w,h,label,font='Arial'){
 const q=shape(s,x,y,w,h,'#FCFCFD','#ABB6C3',4,'replace_with_real_frame');q.line={fill:'#AAB7C4',width:1.3,style:'dashed'};txt(s,x+5,y+h/2-24,w-10,48,label,18,C.gray,false,'center',font);return q;
}
function top(s,title,sub,cn=false){s.background.fill='#FFFFFF';txt(s,42,20,1185,48,title,34,C.ink,true,'left',cn?'Microsoft YaHei':'Arial');txt(s,1050,30,508,30,sub,17,C.gray,false,'right',cn?'Microsoft YaHei':'Arial');}
function notes(s,t){s.speakerNotes.textFrame.setText(t);}
const sources='Source snapshot: 2026-09-22. D:/实验室项目/新实验/kinetalk_b0_residual_train/kinetalk_b0/models/neutral_affect.py:53,337,429; scripts/train_full_staged.py:394,404,496,552; docs/CURRENT_SYSTEM_20260921.md. Figure depicts the mouth-protected v10 / Stage4 route. It is an implementation diagram, not a claim that identity, emotion, lip sync and dynamics have all passed a unified evaluation. B0 is trained in articulation stage and frozen thereafter. Identity describes motion execution, not geometric face identity. Only global representation distillation is drawn; no claim of full low-rate control distillation. Motion targets and labels are training-only.';
function overview(cn=false){
 const s=deck.slides.add();const f=cn?'Microsoft YaHei':'Arial';const t=(a,b)=>cn?b:a;
 top(s,t('KineTalk: reference-conditioned facial animation','KineTalk：参考条件下的语音驱动面部动画'),t('Protected Stage4 architecture','口部保护版 Stage4 架构'),cn);
 panel(s,40,88,1520,224,t('(a) Neutral reference enrollment','(a) 多句中性参考登记'),C.green,'#F6FAF8',f);
 box(s,62,139,170,53,t('Reference audio','参考语音'),C.gl,C.green,22,f);
 placeholder(s,62,209,170,65,t('Neutral motion\nreference frames','中性参考动作\n真实帧待补'),f);
 box(s,270,139,175,53,t('Shared B₀','共享发音基座 B₀'),C.bl,C.blue,21,f);
 circ(s,475,160,42,'−');
 route(s,[[232,165],[270,165]],C.green);route(s,[[445,165],[461,165],[461,181],[475,181]],C.blue);
 route(s,[[232,241],[496,241],[496,202]],C.green);
 txt(s,273,208,171,28,t('paired utterance','语音与动作配对'),17,C.gray,false,'center',f);
 box(s,555,147,218,83,t('Reference residual\nRʳ = Mʳ − B₀(Cʳ)','参考动作残差\nRʳ = Mʳ − B₀(Cʳ)'),C.gl,C.green,21,f);
 route(s,[[517,181],[555,181]],C.green);
 box(s,812,147,208,83,t('Statistics + encoder\nMulti-reference pooling','残差统计与共享编码\n多参考聚合'),C.gl,C.green,21,f);
 route(s,[[773,188],[812,188]],C.green);
 box(s,1060,147,172,83,t('Execution code\nS_id','个体执行表征\nS_id'),C.gl,C.green,23,f);
 route(s,[[1020,188],[1060,188]],C.green);
 box(s,1280,147,251,83,t('Static offset P_id\nMouth calibration','静态个体偏移 P_id\n参考口部校准'),C.gl,C.green,22,f);
 route(s,[[1232,188],[1280,188]],C.green);
 route(s,[[916,230],[916,267],[1405,267],[1405,230]],C.green);
 txt(s,996,245,364,23,t('reference mean for calibration','由参考均值构造静态口部偏移'),16,C.green,false,'center',f);
 txt(s,557,271,460,25,t('Enrollment is reusable across target utterances','登记结果可跨目标语句复用'),18,C.gray,false,'left',f);
 // Inference area
 txt(s,57,329,760,35,t('(b) Speech-driven synthesis','(b) 语音驱动的面部动画合成'),24,C.ink,true,'left',f);
 box(s,60,471,173,91,t('Driving speech\nA','目标语音\nA'),C.light,C.line,25,f);
 box(s,285,395,190,86,t('Acoustic features\nContent + affect','声学特征\n内容与情感特征'),C.bl,C.blue,22,f);
 box(s,530,395,190,86,t('Audio affect\nstudent','音频情感\n学生网络'),C.ol,C.orange,24,f);
 box(s,766,395,185,86,t('E_g   I_g   L_t\nAffect conditions','E_g   I_g   L_t\n全局与局部条件'),C.ol,C.orange,23,f);
 route(s,[[233,516],[259,516],[259,438],[285,438]],C.blue);route(s,[[475,438],[530,438]],C.blue);route(s,[[720,438],[766,438]],C.orange);
 box(s,285,572,190,75,t('Content features\nC','语音内容特征\nC'),C.bl,C.blue,23,f);
 box(s,530,572,190,75,t('Articulation base\nB₀  ·  frozen','发音基座 B₀\n表达阶段冻结'),C.bl,C.blue,22,f);
 route(s,[[259,516],[259,609],[285,609]],C.blue);route(s,[[475,609],[530,609]],C.blue);
 box(s,1000,417,205,159,t('Conditional\nexpression generator\nResidual DiT','条件表达生成器\nResidual DiT\n系数残差生成'),C.pl,C.purple,24,f);
 route(s,[[951,438],[975,438],[975,477],[1000,477]],C.orange);
 // enrollment conditioning. Distinct edge ports avoid hidden crossings
 route(s,[[1146,230],[1146,348],[1102,348],[1102,417]],C.green);
 txt(s,1124,328,76,25,'S_id',18,C.green,false,'center',f);
 route(s,[[720,609],[822,609],[822,548],[1000,548]],C.blue);
 txt(s,832,548,150,31,t('Hidden states H₀','内容隐状态 H₀'),18,C.blue,false,'center',f);
 box(s,967,333,92,45,'z ∼ N',C.light,C.border,20,f);route(s,[[1013,378],[1013,399],[1030,399],[1030,417]],C.gray);
 box(s,1244,454,106,87,t('Fixed\nmask\nS_exp','非口部\n硬约束\nS_exp'),C.pl,C.purple,21,f);
 route(s,[[1205,497],[1244,497]],C.purple);txt(s,1204,459,40,25,'R',18,C.purple);
 circ(s,1370,479,46,'+');route(s,[[1350,502],[1370,502]],C.purple);
 // P_id enters above, avoids output image and expression path
 route(s,[[1472,230],[1472,373],[1393,373],[1393,479]],C.green);
 txt(s,1396,387,143,30,'P_id',20,C.green,false,'left',f);
 // B0 skip along clear bottom lane
 route(s,[[625,647],[625,685],[1393,685],[1393,525]],C.blue);
 txt(s,938,652,265,27,t('Articulation sequence B₀(C)','发音序列 B₀(C)'),19,C.blue,false,'center',f);
 placeholder(s,1440,456,112,144,t('Avatar\nframes','生成头像\n真实帧待补'),f);
 route(s,[[1416,502],[1440,502]],C.line);
 txt(s,1419,418,147,28,'ARKit52',21,C.ink,true,'center',f);
 txt(s,1000,589,372,33,t('R_mouth/jaw = 0','R_mouth/jaw = 0'),23,C.purple,false,'center',f);
 txt(s,45,674,460,32,t('M̂ = B₀(C) + P_id + S_exp ⊙ R','M̂ = B₀(C) + P_id + S_exp ⊙ R'),24,C.ink,false,'left',f);
 // self-contained teacher training inset
 panel(s,40,731,1520,132,t('(c) Motion-guided affect learning · training only','(c) 动作引导的情感学习（仅训练期）'),C.orange,'#FFFAF2',f);
 box(s,61,784,292,57,'M − B₀(C) − P_id',C.ol,C.orange,24,f);
 box(s,391,784,246,57,t('Motion affect teacher','动作情感教师'),C.ol,C.orange,23,f);
 box(s,677,784,137,57,'E_gᴹ',C.ol,C.orange,25,f);
 box(s,861,784,267,57,t('Global feature alignment','全局情感表征对齐'),C.ol,C.orange,22,f);
 box(s,1172,784,137,57,'E_gᴬ',C.ol,C.orange,25,f);
 box(s,1350,784,188,57,t('Audio student','音频学生'),C.ol,C.orange,23,f);
 route(s,[[353,812],[391,812]],C.orange,true);route(s,[[637,812],[677,812]],C.orange,true);route(s,[[814,812],[861,812]],C.orange,true);route(s,[[1172,812],[1128,812]],C.orange,true);route(s,[[1350,812],[1309,812]],C.orange,true);
 txt(s,44,869,1400,24,t('Solid: synthesis / enrollment     Dashed: training supervision     Screenshot boxes: replace with real frames','实线：登记与合成路径    虚线：训练监督    截图框：后续替换为真实画面'),16,C.gray,false,'left',f);
 notes(s,sources+'\nCaption: KineTalk overview. Independent neutral audio-motion pairs provide an execution representation and a time-constant offset. The audio student and frozen articulation base condition an expression generator. A fixed non-mouth support mask protects the articulation sequence. Motion-derived affect supervises the global audio representation during training. E_g: global affect; I_g: clip intensity; L_t: local condition. The native-region gain experiment is intentionally separate. Screenshot placeholders are author-requested, not missing generated content.');
 return s;
}
overview(false);overview(true);
// Detail figure
{
 const s=deck.slides.add();top(s,'KineTalk: reference conditioning and output protection','Methods detail');
 panel(s,40,92,726,676,'(a) Reference residual conditioning',C.green,'#F8FBF9');
 panel(s,791,92,769,676,'(b) Protected coefficient composition',C.purple,'#FAF9FC');
 placeholder(s,66,159,147,112,'Neutral motion\nMʳ');box(s,65,314,148,65,'Paired audio\nAʳ',C.gl,C.green,22);
 box(s,269,314,180,65,'Shared B₀',C.bl,C.blue,23);route(s,[[213,346],[269,346]],C.blue);
 circ(s,489,238,50,'−');route(s,[[213,215],[514,215],[514,238]],C.green);route(s,[[449,346],[514,346],[514,288]],C.blue);
 box(s,575,233,161,63,'Residual Rʳ',C.gl,C.green,22);route(s,[[539,263],[575,263]],C.green);
 box(s,563,391,184,89,'Masked statistics\nmean / std',C.gl,C.green,23);route(s,[[655,296],[655,391]],C.green);
 box(s,316,391,195,89,'Shared encoder\nper reference',C.gl,C.green,23);route(s,[[563,436],[511,436]],C.green);
 box(s,66,391,197,89,'Average across\nindependent clips',C.gl,C.green,22);route(s,[[316,436],[263,436]],C.green);
 box(s,76,541,177,71,'Execution code\nS_id',C.gl,C.green,24);route(s,[[164,480],[164,541]],C.green);
 box(s,316,541,195,71,'Bounded bias\nhead',C.gl,C.green,23);route(s,[[253,577],[316,577]],C.green);
 box(s,563,541,184,71,'Static offset\nP_id',C.gl,C.green,24);route(s,[[511,577],[563,577]],C.green);
 box(s,310,653,250,64,'Fixed mouth calibration',C.gl,C.green,22);route(s,[[655,480],[755,480],[755,735],[435,735],[435,717]],C.green);route(s,[[560,685],[655,685],[655,612]],C.green);
 txt(s,66,642,203,80,'Cross-reference\nlearning',22,C.green);
 txt(s,819,151,712,46,'M̂ = B₀(C) + P_id + S_exp ⊙ R',31,C.ink,true);
 box(s,824,227,254,85,'Articulation output\nB₀(C)',C.bl,C.blue,25);
 box(s,824,348,254,85,'Time-constant offset\nP_id',C.gl,C.green,24);
 box(s,824,482,254,85,'Generated expression\nR',C.pl,C.purple,24);
 box(s,1143,482,158,85,'Non-mouth\nsupport mask',C.pl,C.purple,23);
 route(s,[[1078,525],[1143,525]],C.purple);
 circ(s,1388,367,56,'+');route(s,[[1078,268],[1416,268],[1416,367]],C.blue);route(s,[[1078,390],[1388,390]],C.green);route(s,[[1301,525],[1416,525],[1416,423]],C.purple);
 box(s,1344,608,165,66,'ARKit52 output',C.light,C.line,23);route(s,[[1416,525],[1416,608]],C.line);
 txt(s,817,607,500,57,'M̂_mouth(t) = B₀,mouth(t) + P_id,mouth',23,C.blue,false,'left');
 txt(s,817,681,694,53,'Δₜ M̂_mouth = Δₜ B₀,mouth',27,C.blue,true);
 txt(s,44,793,1510,62,'The reference branch captures motion execution. Static calibration changes the enrolled offset.\nThe fixed expression mask preserves the temporal changes of the articulation output.',24,C.ink,false,'left');
 notes(s,sources+'\nDetail figure: Identity encoder computes masked mean/std per reference and averages valid reference codes. Mouth calibration is fixed affine calibration of enrollment neutral residual mean. It does not scale B0 frame trajectories. Δt identity holds on valid supported coefficients before display-time clipping. Cross-reference identity learning uses independent neutral sets.\nSource neutral_affect.py:53–106,337–366,429–443; scripts/train_full_staged.py:520–540.');
}
// Current candidate separated from paper architecture
{
 const s=deck.slides.add();top(s,'Regional activity modulation','Current experimental branch');
 txt(s,43,75,1490,43,'Native regional activity v2 · amplitude control over a frozen motion carrier',25,C.gray,false,'left');
 panel(s,40,143,1520,226,'(a) Frozen motion carrier',C.blue,'#F6FAFD');
 box(s,70,214,241,100,'Frozen Stage4\nARKit52 prior',C.bl,C.blue,27);
 box(s,380,214,251,100,'Select upper9\nbrow + squint / wide',C.bl,C.blue,24);
 box(s,702,214,274,100,'5-frame smoothing\nwithin valid segments',C.bl,C.blue,25);
 box(s,1048,214,266,100,'Remove segment mean\nCentered carrier Q(t)',C.bl,C.blue,24);
 route(s,[[311,264],[380,264]],C.blue);route(s,[[631,264],[702,264]],C.blue);route(s,[[976,264],[1048,264]],C.blue);
 box(s,1375,228,156,72,'Mean μ',C.gl,C.green,24);route(s,[[1181,214],[1181,190],[1453,190],[1453,228]],C.green);
 panel(s,40,405,1520,297,'(b) Audio activity and bounded gain',C.orange,'#FFFBF5');
 box(s,70,479,241,103,'Audio features\n1540D / 25 fps',C.ol,C.orange,26);
 box(s,380,479,251,103,'Regional activity\npredictor',C.ol,C.orange,25);
 box(s,702,479,274,103,'Brow / eye envelope\nâ(t) ∈ R²₊',C.ol,C.orange,25);
 box(s,1048,479,266,103,'Bounded gain\ng = clip(â / a_Q)',C.ol,C.orange,24);
 route(s,[[311,530],[380,530]],C.orange);route(s,[[631,530],[702,530]],C.orange);route(s,[[976,530],[1048,530]],C.orange);
 route(s,[[1181,314],[1181,479]],C.blue);txt(s,1194,367,168,29,'Carrier activity a_Q',18,C.blue,false,'left');
 box(s,1375,479,156,103,'Scale Q(t)\nRecenter\nAdd μ',C.pl,C.purple,23);route(s,[[1314,530],[1375,530]],C.orange);route(s,[[1453,300],[1453,479]],C.green);
 txt(s,70,617,964,48,'Ŷ_upper = μ + center[g(t) ⊙ Q(t)]',30,C.ink,true,'left');
 route(s,[[1531,530],[1543,530],[1543,748],[1260,748]],C.purple);
 box(s,1021,719,239,68,'Copy other 43 channels',C.gl,C.green,21);
 txt(s,44,747,900,45,'Gain range [0.25, 2.5]     Blink and gaze remain in the copied channels',21,C.gray,false,'left');
 box(s,1310,800,224,61,'ARKit52 candidate',C.light,C.line,23);route(s,[[1140,787],[1140,831],[1310,831]],C.line);
 txt(s,44,817,1197,43,'Audio timing validation remains open. This branch is separate from the main architecture.',23,C.red,false,'left');
 notes(s,'This slide records a separate experimental branch, not a validated paper contribution. Source: D:/实验室项目/新实验/kinetalk_b0_residual_train/docs/NATIVE_REGIONAL_DYNAMICS_20260922.md; kinetalk_b0/models/audio_regional_envelope.py and temporal_motion_carrier.py. Targets are five-frame-smoothed RMS activity of clip-centered upper9. The carrier is the triangle-smoothed Stage4 prior with segment means restored. Bounded gain changes existing shape amplitude and cannot create absent events. Non-upper43 are copied bitwise. Raw coefficients may fall outside [0,1]; display clipping does not preserve their mean. Training targets and masks are not inference conditions. No extra stochastic micro-motion or new zero-DC flow is included. This candidate has not passed the complete audio timing gate.');
}
await fs.mkdir(out,{recursive:true});
const candidate=build+'/KineTalk_architecture_candidate.pptx';
await(await PresentationFile.exportPptx(deck)).save(candidate);
for(let i=0;i<deck.slides.items.length;i++){
 const s=deck.slides.items[i];const b=await deck.export({slide:s,format:'png',scale:1.2});
 await fs.writeFile(build+`/slide-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
 const l=await s.export({format:'layout'});await fs.writeFile(build+`/slide-${i+1}.layout.json`,await l.text());
}
const dest=out+'/KineTalk_方法架构图_可编辑.pptx';
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:dest,pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','15240000,8572500'],fontPolicy:{basis:'design',families:['Arial','Microsoft YaHei']},verifyArtifactToolImport:true,receiptPath:build+'/validation.json'});
console.log(JSON.stringify(result));
