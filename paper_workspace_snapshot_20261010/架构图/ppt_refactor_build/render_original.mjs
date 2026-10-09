import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const src='D:/科研/小论文/架构图/KineTalk_学术架构图_新版.pptx';
const out='D:/科研/小论文/架构图/ppt_refactor_build/original.png';
const p=await PresentationFile.importPptx(await FileBlob.load(src));
const s=p.slides.getItem(0);
const b=await p.export({slide:s,format:'png',scale:1});
await fs.writeFile(out,new Uint8Array(await b.arrayBuffer()));
console.log(out);
