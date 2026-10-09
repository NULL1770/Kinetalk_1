import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const src='D:/科研/小论文/架构图/ppt_refactor_build/KineTalk_学术架构图_学术命名版_draft.pptx';
const out='D:/科研/小论文/架构图/ppt_refactor_build/KineTalk_学术架构图_学术命名版_candidate.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(src));
const replacements = [
  ['(b) Conditional Residual Motion Decoder', '(b) Residual Motion Decoder'],
  ['Reference\nPooling', 'Motion\nPooling'],
  ['gₐ [B,64]', 'gₐ'],
  ['uₐ [B,T,64]', 'uₐ'],
  ['ε ∼ N(0,I)', 'ε'],
  ['Training Motion', 'Motion Sequence'],
  ['Condition inputs', 'Context inputs'],
  ['Noisy residual +\ncontext', 'Noisy residual'],
  ['Emotion and time\nmodulation', 'Emotion modulation'],
  ['Temporal context', 'Temporal attention'],
];
const snap = await p.inspect({kind:'textbox,notes', maxChars:50000});
const records=snap.ndjson.split(/\n/).filter(Boolean).map(x=>JSON.parse(x));
let changed=[];
for(const rec of records){
 if(rec.kind!=='textbox'||!rec.id) continue;
 const hit=replacements.find(([a])=>rec.text===a);
 if(hit){p.resolve(rec.id).text.replace(hit[0],hit[1]); changed.push(hit);}
}
await fs.mkdir('D:/科研/小论文/架构图/ppt_refactor_build',{recursive:true});
await (await PresentationFile.exportPptx(p)).save(out);
console.log(JSON.stringify({out,changed},null,2));
