from fetch_papers import fetch_pdf,ROOT
from concurrent.futures import ThreadPoolExecutor
import json
SOURCES={
 'emote_export':'https://export.arxiv.org/pdf/2306.08990',
 'diffposetalk_export':'https://export.arxiv.org/pdf/2310.00434',
 'media2face_export':'https://export.arxiv.org/pdf/2401.15687',
 'unitalker_export':'https://export.arxiv.org/pdf/2408.00762',
 'emoface_export':'https://export.arxiv.org/pdf/2408.11518',
 'said_export':'https://export.arxiv.org/pdf/2401.08655',
 'mimic_export':'https://export.arxiv.org/pdf/2312.10877',
 'wav2sem_export':'https://export.arxiv.org/pdf/2505.23290',
 'wildwest_repo':'https://dspace.library.uu.nl/bitstreams/2cd8c7bf-5ccd-462e-a46d-91a3b28768ef/download',
 'exptalk_primary':'https://www.ijcai.org/proceedings/2025/0202.pdf',
 'prosodytalker_primary':'https://ojs.aaai.org/index.php/AAAI/article/download/32542/34697',
 'personalized_style_export':'https://export.arxiv.org/pdf/2310.17011',
}
def f(kv):
 n,r=fetch_pdf(*kv);print(n,r.get('passed'),r.get('pages'),r.get('error',''),flush=True);return n,r
with ThreadPoolExecutor(max_workers=3) as ex:
 result=dict(ex.map(f,SOURCES.items()))
(ROOT/'papers'/'export_retrieval_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
