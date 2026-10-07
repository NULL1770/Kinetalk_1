from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,json,hashlib,io,datetime
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'emote':'https://arxiv.org/pdf/2306.08990','deeptalk':'https://arxiv.org/pdf/2408.06010',
 'diffposetalk':'https://arxiv.org/pdf/2310.00434','media2face':'https://arxiv.org/pdf/2401.15687',
 'unitalker':'https://arxiv.org/pdf/2408.00762','emoface':'https://arxiv.org/pdf/2408.11518',
 'said':'https://arxiv.org/pdf/2401.08655','facediffuser':'https://arxiv.org/pdf/2309.11306',
 'personalized_style':'https://arxiv.org/pdf/2310.17011','mimic':'https://arxiv.org/pdf/2312.10877',
 'wav2sem':'https://arxiv.org/pdf/2505.23290',
 'wildwest':'https://dspace.library.uu.nl/bitstreams/2cd8c7bf-5ccd-462e-a46d-91a3b28768ef/download',
 'exptalk':'https://www.ijcai.org/proceedings/2025/0202.pdf',
 'prosodytalker':'https://ojs.aaai.org/index.php/AAAI/article/download/32542/34697',
 'ethead':'https://arxiv.org/pdf/2608.01605','tokens_faces':'https://arxiv.org/pdf/2606.13630',
}
def f(kv):
    name,url=kv; dest=ROOT/'papers'/f'{name}_curl.pdf'
    result={'name':name,'url':url,'transport':'curl.exe Windows Schannel','retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        if name=='deeptalk' and dest.is_file():code=0
        else:
            p=subprocess.run(['curl.exe','--silent','--show-error','--location','--max-time','45','--fail','--output',str(dest),url],capture_output=True,text=True,timeout=50)
            code=p.returncode;result['stderr']=p.stderr[:1200]
        if code!=0:raise RuntimeError('curl exit '+str(code))
        data=dest.read_bytes()
        if not data.startswith(b'%PDF'):raise ValueError('not PDF')
        pages=PdfReader(io.BytesIO(data)).pages
        txt='\n\n'.join(f'[PDF PAGE {i+1}]\n'+(p.extract_text() or '') for i,p in enumerate(pages))
        (ROOT/'fulltext'/f'{name}.txt').write_text(txt,encoding='utf-8')
        result.update(passed=True,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),pages=len(pages),title_excerpt=txt[:550])
    except Exception as e:result.update(passed=False,error=repr(e))
    (ROOT/'papers'/f'{name}_curl_retrieval.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(name,result.get('passed'),result.get('pages'),result.get('error',''),flush=True)
    return name,result
with ThreadPoolExecutor(max_workers=4) as ex:
    result=dict(ex.map(f,SOURCES.items()))
(ROOT/'papers'/'curl_retrieval_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
