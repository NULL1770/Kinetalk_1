import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const src='D:/科研/小论文/架构图/ppt_refactor_build/KineTalk_学术架构图_学术命名版_draft.pptx';
const out='D:/科研/小论文/架构图/ppt_refactor_build/refactor_preview.png';
const p=await PresentationFile.importPptx(await FileBlob.load(src));
const b=await p.export({slide:p.slides.getItem(0),format:'png',scale:1});
await fs.writeFile(out,new Uint8Array(await b.arrayBuffer()));
console.log(out);
