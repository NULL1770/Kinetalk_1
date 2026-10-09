import fs from 'node:fs/promises';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
const p=Presentation.create({slideSize:{width:1600,height:900}});
const s=p.slides.add();
const a=s.shapes.add({geometry:'roundRect',position:{left:100,top:100,width:200,height:100},fill:'#E5F2F8',line:{fill:'#3B7AA0',width:1.5},borderRadius:10});
a.text='Audio encoder\n语音编码器'; a.text.style={typeface:'Microsoft YaHei',fontSize:25,alignment:'center',verticalAlignment:'middle',autoFit:'none',insets:{left:5,right:5,top:5,bottom:5}};
const b=s.shapes.add({geometry:'rect',position:{left:420,top:100,width:200,height:100},fill:'#EDEAF6'});b.text='Conditional generator';b.text.style={typeface:'Arial',fontSize:24};
s.shapes.connect(a,b,{kind:'straight',head:{type:'triangle'},line:{fill:'#39495C',width:2}});
await (await PresentationFile.exportPptx(p)).save('D:/科研/小论文/.architecture_build/smoke.pptx');
try { const z=await p.export({slide:s,format:'png',scale:1}); await fs.writeFile('D:/科研/小论文/.architecture_build/smoke.png',new Uint8Array(await z.arrayBuffer()));console.log('export ok'); } catch(e) { console.log(String(e));}
