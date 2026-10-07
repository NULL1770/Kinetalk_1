from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, json, hashlib, datetime, io, re
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
ARXIV={
 'emote':'2306.08990', 'deeptalk':'2408.06010', 'diffposetalk':'2310.00434',
 'media2face':'2401.15687','unitalker':'2408.00762','emoface':'2408.11518',
 'said':'2401.08655','facediffuser':'2309.11306',
 'personalized_style':'2310.17011','content_style':'2408.07005',
 'editemotalk':'2601.10000','pestalk':'2512.05121','echo':'2609.05506',
 'fmreward':'2608.15296','shape_speech':'2610.03436','toktalk':'2605.31294',
}
DIRECT={
 'mimic':'https://ojs.aaai.org/index.php/AAAI/article/download/27945/27910',
 'exptalk':'https://www.ijcai.org/proceedings/2025/0202.pdf',
 'prosodytalker':'https://ojs.aaai.org/index.php/AAAI/article/download/32542/34697',
 'wav2sem':'https://openaccess.thecvf.com/content/CVPR2025/papers/Fan_Wav2Sem_Plug-and-Play_Audio_Semantic_Decoupling_for_3D_Speech-Driven_Facial_Animation_CVPR_2025_paper.pdf',
}

def fetch_pdf(name,url,arxiv_id=None):
    result={'name':name,'url':url,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arxiv_id':arxiv_id}
    try:
        r=requests.get(url,timeout=35,headers={'User-Agent':'Academic source verification/1.0'})
        result.update(status=r.status_code,resolved_url=r.url)
        r.raise_for_status();data=r.content
        if not data.startswith(b'%PDF'):raise ValueError('Response is not a PDF: '+r.headers.get('Content-Type',''))
        (ROOT/'papers'/f'{name}.pdf').write_bytes(data)
        pages=PdfReader(io.BytesIO(data)).pages
        txt='\n\n'.join(f'[PDF PAGE {i+1}]\n'+(p.extract_text() or '') for i,p in enumerate(pages))
        (ROOT/'fulltext'/f'{name}.txt').write_text(txt,encoding='utf-8')
        result.update(passed=True,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),pages=len(pages),title_excerpt=txt[:650])
        patterns=r'(?i)(evaluation metrics|quantitative evaluation|lip vertex error|facial dynamics|temporal|disentangl|style encoder|cross.modal|probabilis|contrastive|F1.score|t.SNE|metric|velocity|acceleration|reconstruct|teacher|student)'
        spans=[]
        for m in re.finditer(patterns,txt):
            a=max(0,m.start()-160);b=min(len(txt),m.end()+700)
            if spans and a<=spans[-1][1]:spans[-1]=(spans[-1][0],b)
            else:spans.append((a,b))
        excerpts='\n\n'.join(f'[TEXT OFFSET {a}]\n'+txt[a:b] for a,b in spans)
        (ROOT/'evidence'/f'{name}_excerpts.txt').write_text(excerpts,encoding='utf-8')
    except Exception as e:
        result.update(passed=False,error=repr(e))
    if arxiv_id:
        try:
            a=requests.get(f'https://arxiv.org/abs/{arxiv_id}',timeout=25)
            result['abs_status']=a.status_code
            (ROOT/'papers'/f'{name}_abs.html').write_text(a.text,encoding='utf-8')
            soup=BeautifulSoup(a.text,'html.parser')
            result['arxiv_meta']={m.get('name'):m.get('content') for m in soup.select('meta[name^="citation_"]')}
            h=soup.select_one('.submission-history');result['submission_history']=h.get_text(' ',strip=True) if h else None
            block=soup.select_one('blockquote.abstract');result['abstract']=block.get_text(' ',strip=True) if block else None
        except Exception as e:result['abs_error']=repr(e)
    (ROOT/'papers'/f'{name}_retrieval.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return name,result

if __name__=='__main__':
    result={}
    with ThreadPoolExecutor(max_workers=5) as ex:
        fs=[ex.submit(fetch_pdf,n,f'https://arxiv.org/pdf/{i}',i) for n,i in ARXIV.items()]+[ex.submit(fetch_pdf,n,u) for n,u in DIRECT.items()]
        for f in as_completed(fs):
            n,v=f.result();result[n]=v
            print(n,v.get('passed'),v.get('pages'),v.get('error',''),flush=True)
    (ROOT/'papers'/'retrieval_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
