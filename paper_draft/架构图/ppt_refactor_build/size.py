from zipfile import ZipFile
import re
p=r'D:\科研\小论文\架构图\KineTalk_学术架构图_新版.pptx'
with ZipFile(p) as z:
 s=z.read('ppt/presentation.xml').decode('utf-8')
 print(re.search(r'<p:sldSz[^>]+>',s).group(0))
