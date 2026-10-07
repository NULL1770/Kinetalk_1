from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, json, hashlib, datetime, re
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
QUERIES = [
    'speech driven 3D facial animation emotion disentanglement',
    'audio driven 3D facial animation emotional style',
    '3D talking face motion identity disentanglement',
    '3D facial animation emotion dynamics',
    'MEDTalk', 'EMOTE emotional speech driven animation',
    'DEEPTalk', 'DiffPoseTalk', 'Media2Face', 'UniTalker',
    'SAiD speech animation diffusion', 'EmoFace',
    'FaceDiffuser', 'DESTalker',
    'speech driven facial animation 2026',
    'speech driven 3D facial animation 2025 emotion style',
]

def get_openalex(q):
    url = 'https://api.openalex.org/works'
    params = {'search': q, 'per-page': 15, 'filter': 'from_publication_date:2023-01-01,to_publication_date:2026-10-07'}
    try:
        r = requests.get(url, params=params, timeout=35)
        data = r.json() if r.ok else None
        stem = re.sub('[^a-z0-9]+', '_', q.lower()).strip('_')
        (ROOT / 'searches' / f'{stem}.json').write_text(json.dumps({'query':q,'url':r.url,'status':r.status_code,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'data':data,'error_body':None if r.ok else r.text[:1000]},ensure_ascii=False,indent=2),encoding='utf-8')
        out=[]
        for x in (data or {}).get('results',[]):
            inv=x.get('abstract_inverted_index') or {}
            words={i:w for w,idx in inv.items() for i in idx}
            abstract=' '.join(words[i] for i in sorted(words))
            out.append({'title':x['title'],'date':x['publication_date'],'doi':x.get('doi'),'openalex':x['id'],'venue':(x.get('primary_location') or {}).get('source'),'locations':x.get('locations'),'abstract':abstract})
        return q,out
    except Exception as e:
        return q, {'error':str(e)}

def get_cvf(event):
    url=f'https://openaccess.thecvf.com/{event}?day=all'
    try:
        r=requests.get(url,timeout=40)
        p=ROOT/'searches'/f'{event}.html'; p.write_text(r.text,encoding='utf-8')
        soup=BeautifulSoup(r.text,'html.parser')
        rows=[]
        for dt in soup.select('dt.ptitle'):
            a=dt.find('a'); title=dt.get_text(' ',strip=True)
            if re.search('talk|facial|speech.*3d|3d.*speech',title,re.I):
                rows.append({'title':title,'url':'https://openaccess.thecvf.com'+a['href'] if a and a['href'].startswith('/') else (a or {}).get('href')})
        return event,{'status':r.status_code,'url':url,'sha256':hashlib.sha256(r.content).hexdigest(),'matches':rows}
    except Exception as e:
        return event,{'error':str(e)}

if __name__=='__main__':
    results={}
    with ThreadPoolExecutor(max_workers=5) as ex:
        fs=[ex.submit(get_openalex,q) for q in QUERIES]+[ex.submit(get_cvf,e) for e in ['CVPR2025','ICCV2025','CVPR2026']]
        for f in as_completed(fs):
            k,v=f.result();results[k]=v
            print(k, 'items='+str(len(v)) if isinstance(v,list) else v.get('status',v.get('error')),flush=True)
    (ROOT/'searches'/'harvest_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    for q,v in results.items():
        print('\nQUERY',q)
        if isinstance(v,list):
            for x in v[:10]:print(x['date'],x['title'],x['doi'])
        else:
            for x in v.get('matches',[]):print(x['title'],x['url'])
