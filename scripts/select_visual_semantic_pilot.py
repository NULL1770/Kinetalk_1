"""Metadata-only, sentence-disjoint inner pilot selection from prior1053 fit."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def select(protocol, video_root, output):
    if output.exists(): raise FileExistsError(output)
    rows = json.loads(protocol.read_text(encoding='utf8'))['split']['fit']
    if len(rows) != 1053: raise ValueError('Expected frozen1053 fit pool')
    key = lambda s: hashlib.sha256(('visual_semantic_pilot_v1:'+s).encode()).hexdigest()
    sentences = sorted({r['sentence'] for r in rows}, key=key)
    held = set(sentences[:8])
    selected = []
    for role, total in [('train',192), ('holdout',64)]:
        groups = defaultdict(list)
        for row in rows:
            if (row['sentence'] in held) != (role == 'holdout'): continue
            groups[(row['speaker'],row['emotion'])].append(row)
        groups = {k:sorted(v,key=lambda r:key(r['clip_id'])) for k,v in groups.items()}
        picks=[]
        while len(picks)<total:
            count=0
            for cell in sorted(groups):
                if groups[cell] and len(picks)<total:
                    picks.append(groups[cell].pop(0));count+=1
            if not count: raise ValueError('Insufficient metadata pool')
        for row in picks:
            parts=row['clip_id'].split('_')
            video=video_root/parts[1]/'video'/'front'/parts[2]/('level_'+parts[3][1:])/(parts[4]+'.mp4')
            selected.append({**row,'split':role,'speaker_name':'mead_'+parts[1],
                             'video':str(video.resolve()),'video_sha256':sha(video),'needs_resample':True})
    assert not ({r['sentence'] for r in selected if r['split']=='train'} & {r['sentence'] for r in selected if r['split']=='holdout'})
    output.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema':'visual_semantic_pilot_selection_v1','clips':selected,
             'source_protocol_sha256':sha(protocol),'source_pool':'frozen1053 fit only',
             'rule':'SHA256 sentence split8 held sentences, round robin speaker/emotion cells; no motion or semantic outcomes',
             'code_sha256':sha(__file__),'counts':{'train':192,'holdout':64},
             'holdout_sentences':sorted(held),'historical_upstream_exposure':True,
             'sealed_test_loaded':False,'dev405_loaded':False,
             'semantic_stride_frames':5,'epochs':30,'train_seed':20260918,
             'sample_seeds':[42,123,2026,77]}
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'counts':payload['counts'],'path':str(output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--video-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();select(a.protocol,a.video_root,a.output)
