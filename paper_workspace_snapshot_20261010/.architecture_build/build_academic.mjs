import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const root='D:/科研/小论文', build=root+'/.architecture_build', out=root+'/架构图';
const skill='C:/Users/zhh/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const python='C:/Users/zhh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/zhh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const deck=Presentation.create({slideSize:{width:1600,height:900}});
const C={ink:'#142632',edge:'#29323A',gray:'#647079',blue:'#328FC1',bl:'#D5ECFA',green:'#579849',gl:'#DDEECF',purple:'#8F7DAB',pl:'#E6DEEE',orange:'#D39243',ol:'#FFE6B7',aqua:'#4C9998',al:'#D0EFED',pink:'#BC8197',pk:'#F4D9E3'};let n=0;
function sh(s,x,y,w,h,fill='none',stroke='none',radius=0){return s.shapes.add({geometry:radius?'roundRect':'rect',name:'element_'+(++n),position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:stroke==='none'?0:1.1},...(radius?{borderRadius:radius}:{})});}
function tx(s,x,y,w,h,str,size=21,color=C.ink,bold=false,font='Arial',align='center') {let a=sh(s,x,y,w,h);a.text=str;a.text.style={typeface:font,fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};return a;}
function bx(s,x,y,w,h,str,fill,stroke,size=21,font='Arial'){let a=sh(s,x,y,w,h,fill,stroke,6);a.text=str;a.text.style={typeface:font,fontSize:size,color:C.ink,alignment:'center',verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:3,right:3,top:2,bottom:2}};return a;}
function pt(s,x,y){return sh(s,x-.1,y-.1,.2,.2);}
function ar(s,p,c=C.edge,d=false,end=true){for(let i=1;i<p.length;i++){let a=pt(s,...p[i-1]),b=pt(s,...p[i]);let q=s.shapes.connect(a,b,{kind:'straight',line:{fill:c,width:1.7,style:d?'dashed':'solid'},...(end&&i===p.length-1?{tail:{type:'triangle',width:'sm',length:'sm'}}:{})});q.bringToFront();}}
function op(s,x,y,label,d=31){let a=s.shapes.add({geometry:'ellipse',name:'operator_'+(++n),position:{left:x,top:y,width:d,height:d},fill:'#FFFFFF',line:{fill:C.edge,width:1.4}});a.text=label;a.text.style={typeface:'Arial',fontSize:23,alignment:'center',verticalAlignment:'middle',insets:{left:0,right:0,top:0,bottom:0}};return a;}
function poly(s,x,y,w,h,points,fill,stroke){return s.shapes.add({geometry:'custom',name:'encoder_'+(++n),position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:1.2},customPaths:[{width:w,height:h,commands:[{moveTo:{x:points[0][0],y:points[0][1]}},...points.slice(1).map(p=>({lineTo:{x:p[0],y:p[1]}})),{close:{}}]}]});}
function enc(s,x,y,w,h,label,fill,stroke,f='Arial'){poly(s,x,y,w,h,[[0,0],[w,h*.16],[w,h*.84],[0,h]],fill,stroke);tx(s,x+7,y+10,w-15,h-20,label,22,C.ink,false,f);}
function cube(s,x,y,color,light,rows=5,cols=3,cell=10){const w=cols*cell,h=rows*cell,d=9;poly(s,x,y-d,w+d,d,[[0,d],[d,0],[w+d,0],[w,d]],light,color);poly(s,x+w,y-d,d,h+d,[[0,d],[d,0],[d,h],[0,h+d]],light,color);sh(s,x,y,w,h,light,color);for(let i=1;i<cols;i++)ar(s,[[x+i*cell,y],[x+i*cell,y+h]],color,false,false);for(let i=1;i<rows;i++)ar(s,[[x,y+i*cell],[x+w,y+i*cell]],color,false,false);return {x,y,w:w+d,h};}
function cards(s,x,y,c,l,n=3,w=20,h=60){for(let i=0;i<n;i++)sh(s,x+i*10,y+i*8,w,h,l,c);}
function ph(s,x,y,w,h,label,f='Arial'){let a=sh(s,x,y,w,h,'#FCFCFC','#A7ADB3',3);a.line={fill:'#A7ADB3',width:1.1,style:'dashed'};tx(s,x+3,y+3,w-6,h-6,label,17,C.gray,false,f);}
function band(s,x,y,w,h,fill,stroke){sh(s,x,y,w,h,fill,stroke,20);}
function panelLabel(s,x,y,char,label,f='Arial'){bx(s,x,y,37,37,char,C.ol,C.edge,28);tx(s,x+51,y,w=650,37,label,24,C.ink,true,f,'left');}
const source='Sources: kinetalk_b0/models/neutral_affect.py:53,337,429; models/dit.py:25–184; models/slow_state_affect.py:239; scripts/train_full_staged.py:394,404,496,552. Root D:/实验室项目/新实验/kinetalk_b0_residual_train. Depicted system is protected Stage4, not a combination of historical branches. B0 is frozen after articulation training. S_id is motion execution, not geometric identity. Identity and motion target conditions obey valid masks. Figure is an implemented architecture, not a claim that all evaluations passed. Target motion/labels are training-only. Regional gain v2 is not shown because it is a separate unvalidated experiment. Screenshots are intentionally blank for author insertion.';
function main(cn){const s=deck.slides.add();s.background.fill='#FFFFFF';const f=cn?'Microsoft YaHei':'Arial',t=(a,b)=>cn?b:a;
 tx(s,30,15,950,35,t('(a) KineTalk overview','(a) KineTalk 总体架构'),28,C.ink,true,f,'left');
 ar(s,[[1225,35],[1225,810]],C.gray,true,false);
 tx(s,1250,15,320,35,t('(b) Expression generator','(b) 条件表达生成器'),26,C.ink,true,f);
 // Content lane
 band(s,214,69,577,122,'#F1F8FC','#A8CADD');
 ph(s,30,97,141,78,t('Driving speech\nA','目标语音\nA'),f);
 enc(s,236,90,112,64,t('Speech\nencoder','语音\n编码器'),C.bl,C.blue,f);
 bx(s,389,92,165,61,t('Content mapping\nB₀  (frozen)','发音映射 B₀\n表达阶段冻结'),C.bl,C.blue,20,f);
 ar(s,[[171,135],[236,135]]);ar(s,[[348,121],[389,121]]);
 cube(s,609,94,C.blue,C.bl,5,3,10);ar(s,[[554,121],[609,121]]);
 tx(s,656,108,92,30,'H₀',24,C.ink,false,'Times New Roman');
 tx(s,224,161,546,26,t('Audio content and articulation','语音内容与发音运动'),21,C.ink,true,f);
 // B0 coefficient bypass path over generator, output sum at 1090,558
 ar(s,[[470,92],[470,62],[1130,62],[1130,556],[1107,556]],C.blue);tx(s,810,40,190,31,'B₀(C)',23,C.blue,false,'Times New Roman');
 // Affect lane
 band(s,214,224,577,183,'#FFF8DE','#D8C996');
 enc(s,236,265,112,90,t('Affect\nencoder','情感\n编码器'),C.ol,C.orange,f);
 bx(s,389,258,150,55,t('Global pooling','全局聚合'),C.ol,C.orange,21,f);
 bx(s,389,337,150,48,t('Local projection','局部投影'),C.al,C.aqua,20,f);
 ar(s,[[190,135],[190,310],[236,310]]);ar(s,[[348,310],[367,310],[367,285],[389,285]]);ar(s,[[367,310],[367,361],[389,361]]);
 cube(s,589,263,C.orange,C.ol,3,3,11);ar(s,[[539,285],[589,285]]);tx(s,638,270,129,38,'Eᴀg , Iᴀg',25,C.ink,false,'Times New Roman');
 cards(s,591,338,C.aqua,C.al,3,16,39);ar(s,[[539,361],[591,361]]);tx(s,643,345,115,31,'L₁:ₜ',25,C.ink,false,'Times New Roman');
 tx(s,249,231,498,24,t('Audio affect student','音频情感学生'),21,C.ink,true,f);
 // reference lane
 band(s,214,443,577,184,'#EDF5E9','#B7C9AA');
 ph(s,30,451,142,108,t('Neutral reference\naudio + motion\n{Aʳ, Mʳ}','多句中性参考\n语音与动作\n{Aʳ, Mʳ}'),f);
 bx(s,236,478,133,89,'Mʳ − B₀(Cʳ)',C.gl,C.green,22,'Times New Roman');
 enc(s,401,478,115,89,t('Statistics\nencoder','残差统计\n编码器'),C.gl,C.green,f);
 bx(s,551,492,83,61,t('Pool','聚合'),C.gl,C.green,21,f);
 cube(s,678,492,C.green,C.gl,5,3,10);tx(s,650,553,107,35,'Sᵢd',26,C.ink,false,'Times New Roman');
 ar(s,[[172,510],[236,510]]);ar(s,[[369,522],[401,522]]);ar(s,[[516,522],[551,522]]);ar(s,[[634,522],[678,522]]);
 tx(s,231,595,544,23,t('Reference residual conditioning','参考残差个体条件'),21,C.ink,true,f);
 // generator and inputs, distinct routes
 enc(s,922,225,154,181,t('Conditional\nflow decoder\nDθ','条件流\n解码器\nDθ'),C.pl,C.purple,f);
 ar(s,[[648,118],[837,118],[837,244],[922,244]],C.blue);
 ar(s,[[628,279],[861,279],[861,293],[922,293]],C.orange);
 ar(s,[[636,360],[889,360],[889,340],[922,340]],C.aqua);
 ar(s,[[717,515],[837,515],[837,388],[922,388]],C.green);
 tx(s,889,160,221,31,'z ∼ N(0, I)',24,C.ink,false,'Times New Roman');ar(s,[[999,195],[999,225]]);
 // composition and output
 cube(s,980,451,C.purple,C.pl,5,3,10);ar(s,[[999,406],[999,442]],C.purple);tx(s,1030,454,70,33,'R₁:ₜ',23,C.ink,false,'Times New Roman');
 bx(s,922,529,113,55,t('Non-mouth\nmask','非口部\n约束'),C.pl,C.purple,20,f);ar(s,[[999,501],[999,529]],C.purple);
 op(s,1076,541,'+',31);ar(s,[[1035,556],[1076,556]]);
 // Pid bias from reference info, side path
 bx(s,833,587,201,53,t('Static offset Pᵢd','静态个体偏移 Pᵢd'),C.gl,C.green,21,f);
 ar(s,[[717,531],[809,531],[809,613],[833,613]],C.green);ar(s,[[1034,613],[1091,613],[1091,572]],C.green);
 tx(s,832,644,203,40,t('Reference mouth\ncalibration','参考口部静态校准'),17,C.green,false,f);
 ar(s,[[1091,572],[1195,572],[1195,673],[1075,673],[1075,696]],C.edge);
 ph(s,909,701,87,98,t('Frame 1','真实帧 1'),f);ph(s,1009,701,87,98,t('Frame 2','真实帧 2'),f);ph(s,1109,701,87,98,t('Frame 3','真实帧 3'),f);
 tx(s,906,802,288,28,t('ARKit52 facial animation','ARKit52 面部动画'),21,C.ink,true,f);
 // training inset
 const train=sh(s,30,670,760,150,'#FFFDF8','#D9BA84',12);train.line={fill:'#D9BA84',width:1.1,style:'dashed'};
 tx(s,46,674,722,28,t('Motion-guided global affect learning (training only)','动作引导的全局情感学习（仅训练期）'),21,C.ink,true,f,'left');
 bx(s,49,723,190,56,'M − B₀(C) − Pᵢd',C.ol,C.orange,21,'Times New Roman');
 enc(s,270,714,133,76,t('Motion\nteacher','动作情感\n教师'),C.ol,C.orange,f);
 cube(s,440,728,C.orange,C.ol,3,3,11);tx(s,429,780,70,29,'Eᴍg',23,C.ink,false,'Times New Roman');
 bx(s,526,729,122,47,'L_align',C.ol,C.orange,21,'Times New Roman');cube(s,703,728,C.orange,C.ol,3,3,11);tx(s,693,780,70,29,'Eᴀg',23,C.ink,false,'Times New Roman');
 ar(s,[[239,751],[270,751]],C.orange,true);ar(s,[[403,751],[440,751]],C.orange,true);ar(s,[[479,751],[526,751]],C.orange,true);ar(s,[[703,751],[648,751]],C.orange,true);
 // network detail
 tx(s,1257,66,313,59,'Hctx = Pₕ(H₀) + Pₗ(L₁:ₜ)\ng = T(τ) + Pₑ[Eᴀg,Iᴀg] + Pₛ(Sᵢd)',19,C.ink,false,'Times New Roman');
 bx(s,1278,147,271,44,'Pₓ(xτ) + Hctx',C.pl,C.purple,23,'Times New Roman');
 const repeat=sh(s,1251,214,319,332,'#FFFEFA',C.orange,18);repeat.line={fill:C.orange,width:1.4};
 const ys=[240,312,384,456];const labels=cn?['AdaLN + 自注意力','个体执行特征调制','AdaLN + 交叉注意力','AdaLN + 前馈网络']:['AdaLN + Self-Attention','Style Modulation','AdaLN + Cross-Attention','AdaLN + Feed Forward'];
 const colors=[[C.al,C.aqua],[C.gl,C.green],[C.ol,C.orange],[C.al,C.aqua]];
 labels.forEach((v,i)=>{bx(s,1280,ys[i],262,46,v,colors[i][0],colors[i][1],19,f);if(i)ar(s,[[1411,ys[i-1]+46],[1411,ys[i]]]);});ar(s,[[1411,191],[1411,240]]);
 tx(s,1258,514,303,24,t('× N blocks · gated residuals','× N 层（含门控残差连接）'),17,C.gray,false,f);
 tx(s,1275,363,260,19,'K,V = Hctx',16,C.orange,false,'Times New Roman','right');
 bx(s,1280,573,262,44,t('Output AdaLN + Linear','输出 AdaLN + Linear'),C.al,C.aqua,20,f);ar(s,[[1411,502],[1411,573]]);
 tx(s,1324,622,174,27,'vθ(xτ, τ)',23,C.ink,false,'Times New Roman');
 bx(s,1280,662,262,49,t('Flow integration','流积分采样'),C.pl,C.purple,21,f);ar(s,[[1411,648],[1411,662]]);
 tx(s,1263,720,295,57,'R = s · Integrate(vθ, z)\nM̂ = B₀(C) + Pᵢd + Sexp ⊙ R',21,C.ink,false,'Times New Roman');
 tx(s,1263,789,295,39,t('Sexp,mouth/jaw = 0','Sexp,mouth/jaw = 0'),22,C.purple,false,'Times New Roman');
 // figure caption
 ar(s,[[30,841],[1570,841]],'#A8B1B9',false,false);
 tx(s,30,853,1540,33,t('Figure 1. KineTalk combines reference execution features and audio affect conditions with a protected articulation pathway.','图 1. KineTalk 总体架构。中性参考提供个体执行条件，音频提供情感条件，表达生成通过固定输出约束保护发音时序。'),22,C.ink,false,f,'left');
 s.speakerNotes.textFrame.setText(source+'\nDetailed DiT: Hctx is content MLP + local linear. Global condition includes flow time, global emotion/intensity and style. Blocks apply global-modulated self-attention, gated style modulation, global-modulated cross-attention, and global-modulated FFN. Output AdaLN and linear return velocity, then Euler flow integration returns scaled residual. Sampling timestep τ differs from frame time. Main static offset box includes learned non-mouth bias and fixed mouth calibration from reference mean. The inset Sexp describes the fixed support that is also imposed internally on sampled state/velocity. Reference mean calibration is expanded on slide 3. Arrows are data/conditioning, dashed training inset arrows are supervision. Feature grids are schematic tensors, not empirical activations.');
}
main(false);main(true);
// Compact module-detail plate
{
 const s=deck.slides.add();s.background.fill='#FFFFFF';tx(s,30,18,1510,40,'KineTalk: reference encoding and motion-guided affect transfer',30,C.ink,true,'Arial','left');
 ar(s,[[800,77],[800,793]],C.gray,true,false);
 tx(s,40,81,719,40,'(a) Reference execution representation',26,C.ink,true);
 tx(s,831,81,718,40,'(b) Global affect transfer',26,C.ink,true);
 ph(s,45,169,143,109,'Neutral motion\nMʳ');ph(s,45,335,143,75,'Paired speech\nAʳ');
 bx(s,236,334,172,76,'Shared articulation\nbase B₀',C.bl,C.blue,23);ar(s,[[188,372],[236,372]]);
 op(s,447,230,'−',38);ar(s,[[188,223],[466,223],[466,230]]);ar(s,[[408,372],[466,372],[466,268]],C.blue);
 cube(s,541,215,C.green,C.gl,6,4,11);ar(s,[[485,249],[541,249]],C.green);tx(s,603,234,143,34,'Rʳ = Mʳ − B₀(Cʳ)',21,C.ink,false,'Times New Roman');
 bx(s,534,341,214,71,'Masked mean / std',C.gl,C.green,23);ar(s,[[563,281],[563,341]],C.green);
 enc(s,535,451,214,95,'Shared reference\nencoder',C.gl,C.green);ar(s,[[640,412],[640,451]],C.green);
 cards(s,379,457,C.green,C.gl,4,26,70);ar(s,[[535,493],[435,493]],C.green);
 bx(s,111,451,196,92,'Multi-reference\naveraging',C.gl,C.green,24);ar(s,[[379,492],[307,492]],C.green);cube(s,194,596,C.green,C.gl,5,4,11);ar(s,[[209,543],[209,587]],C.green);tx(s,170,658,105,35,'Sᵢd',28,C.ink,false,'Times New Roman');
 bx(s,341,596,182,66,'Bounded bias',C.gl,C.green,23);ar(s,[[247,623],[341,623]],C.green);
 bx(s,576,596,172,66,'Static offset Pᵢd',C.gl,C.green,22);ar(s,[[523,629],[576,629]],C.green);
 bx(s,422,711,244,55,'Fixed mouth calibration',C.gl,C.green,21);ar(s,[[748,376],[773,376],[773,786],[543,786],[543,766]],C.green);ar(s,[[666,738],[707,738],[707,662]],C.green);
 tx(s,830,152,702,44,'Training motion M and enrolled Pᵢd',23,C.ink,false);
 bx(s,854,221,310,68,'M − B₀(C) − Pᵢd',C.ol,C.orange,27,'Times New Roman');
 enc(s,854,350,168,91,'Motion affect\nteacher',C.ol,C.orange);ar(s,[[1009,289],[1009,328],[938,328],[938,350]],C.orange);
 cube(s,1100,368,C.orange,C.ol,4,4,11);ar(s,[[1022,395],[1100,395]],C.orange);tx(s,1100,421,65,32,'Eᴍg',25,C.ink,false,'Times New Roman');
 bx(s,1232,364,277,71,'Emotion / intensity\nsupervision',C.ol,C.orange,23);ar(s,[[1148,390],[1232,390]],C.orange,true);
 bx(s,1061,480,144,65,'L_align',C.ol,C.orange,28,'Times New Roman');ar(s,[[1122,414],[1122,480]],C.orange,true);
 enc(s,854,594,168,91,'Audio affect\nstudent',C.ol,C.orange);cube(s,1100,616,C.orange,C.ol,4,4,11);ar(s,[[1022,639],[1100,639]],C.orange);ar(s,[[1122,616],[1122,545]],C.orange,true);tx(s,1095,674,75,32,'Eᴀg',25,C.ink,false,'Times New Roman');
 bx(s,1250,604,259,83,'Conditional expression\ngenerator',C.pl,C.purple,23);ar(s,[[1148,639],[1250,639]],C.orange);
 tx(s,852,730,680,57,'Teacher features supervise global audio affect.\nTarget motion and emotion labels are training-only.',23,C.ink,false);
 ar(s,[[30,828],[1570,828]],'#A8B1B9',false,false);
 tx(s,30,843,1530,42,'Figure 2. Independent reference residuals construct the execution condition. Motion supervision transfers global affect to speech.',23,C.ink,false,'Arial','left');
 s.speakerNotes.textFrame.setText(source+'\nReference detail follows encode_identity including masked mean/std and fixed affine mouth calibration. Right is a training subgraph for global teacher-student alignment, not a full list of losses. Full teacher stage additionally trains generation. The audio student receives acoustic features; this input is abbreviated in the module detail. Caption and graphics are editable.');
}
await fs.mkdir(out,{recursive:true});const draft=build+'/academic_candidate.pptx';await(await PresentationFile.exportPptx(deck)).save(draft);
for(let i=0;i<deck.slides.items.length;i++){const s=deck.slides.items[i],b=await deck.export({slide:s,format:'png',scale:1.2});await fs.writeFile(build+`/academic-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));}
const result=await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:out+'/KineTalk_学术架构图_可编辑.pptx',pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','15240000,8572500'],fontPolicy:{basis:'design',families:['Arial','Times New Roman','Microsoft YaHei']},verifyArtifactToolImport:true,receiptPath:build+'/academic_validation.json'});console.log(JSON.stringify(result));
