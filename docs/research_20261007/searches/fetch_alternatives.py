from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests,json,hashlib,io,datetime
from bs4 import BeautifulSoup
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
IDS={'emote':'2306.08990','deeptalk':'2408.06010','diffposetalk':'2310.00434','media2face':'2401.15687','unitalker':'2408.00762','emoface':'2408.11518','said':'2401.08655','facediffuser':'2309.11306','personalized_style':'2310.17011','mimic':'2312.10877','wav2sem':'2505.23290','ethead':'2608.01605','tokens_faces':'2606.13630'}

def f(name,aid):
    attempts=[]
    for url in [f'https://arxiv.org/html/{aid}v1',f'https://arxiv.org/pdf/{aid}v1']:
        try:
            r=requests.get(url,timeout=25);r.raise_for_status();data=r.content
            if data.startswith(b'%PDF'):
                pages=PdfReader(io.BytesIO(data)).pages
                txt='\n\n'.join(f'[PDF PAGE {i+1}]\n'+(p.extract_text() or '') for i,p in enumerate(pages));ext='pdf'
            else:
                soup=BeautifulSoup(data,'html.parser')
                for x in soup(['script','style','nav','footer']):x.decompose()
                article=soup.select_one('article')
                if not article:raise ValueError('No HTML article')
                txt=article.get_text('\n',strip=True);ext='html'
            (ROOT/'papers'/f'{name}.{ext}').write_bytes(data)
            (ROOT/'fulltext'/f'{name}.txt').write_text(txt,encoding='utf-8')
            attempts.append({'url':url,'passed':True,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'text_chars':len(txt),'first_excerpt':txt[:400]})
            break
        except Exception as e:attempts.append({'url':url,'passed':False,'error':repr(e)})
    (ROOT/'papers'/f'{name}_alternative_retrieval.json').write_text(json.dumps({'name':name,'arxiv_id':aid,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempts':attempts},ensure_ascii=False,indent=2),encoding='utf-8')
    print(name,attempts[-1].get('passed'),attempts[-1].get('text_chars'),flush=True)

with ThreadPoolExecutor(max_workers=5) as ex:
    list(ex.map(lambda kv:f(*kv),IDS.items()))
