import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const src='D:/科研/小论文/架构图/ppt_refactor_build/KineTalk_学术架构图_学术命名版_draft.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(src));
const s=await p.inspect({kind:'textbox',maxChars:30000});
console.log(s.ndjson);
