import path from 'node:path';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const src='D:/科研/小论文/架构图/KineTalk_学术架构图_新版.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(src));
const s=await p.inspect({kind:'slide,textbox,shape,image,notes,layout',maxChars:50000});
console.log(s.ndjson);
