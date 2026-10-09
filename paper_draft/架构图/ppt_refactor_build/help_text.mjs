import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('D:/科研/小论文/架构图/KineTalk_学术架构图_新版.pptx'));
console.log(p.help('*',{search:'text setText replace textbox',include:['index','examples'],maxChars:10000}));
