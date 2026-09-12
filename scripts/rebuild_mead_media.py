"""Rebuild auditable MEAD video/WAV/BS/text manifests without modifying inputs."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
import unicodedata
import wave
import numpy as np

VERSION = 'video-audio-text-v1'
ALIASES = {'disgusted': 'disgust', 'surprised': 'surprise'}
SOURCE_EMOTION = {'disgust': 'disgusted', 'surprise': 'surprised'}
ANNOTATION = re.compile(r'^([MW]\d+)_level_(\d+)_([a-z]+)_(\d+)\s+(.+)$')
CLIP = re.compile(r'^mead_([MW]\d+)_([a-z]+)_L(\d+)_(\d+)$')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_text(text):
    # Conservative: case and whitespace only; no fuzzy merging of words,
    # contractions, numbers or near-matching sentences.
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def content_id(text):
    return 'text_' + hashlib.sha256(canonical_text(text).encode('utf8')).hexdigest()[:24]


def parse_annotations(path):
    result = {}
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip(): continue
        match = ANNOTATION.fullmatch(line.strip())
        if match is None: raise ValueError(f'invalid annotation at line {number}')
        speaker, intensity, emotion, clip, text = match.groups()
        key = (speaker, ALIASES.get(emotion, emotion), int(intensity), int(clip))
        if key in result and canonical_text(result[key]['text']) != canonical_text(text):
            raise ValueError(f'conflicting annotation for {key}')
        result[key] = {'text': text, 'annotation_line': number, 'annotation_key': line.split()[0]}
    return result


def clip_key(cid):
    match = CLIP.fullmatch(cid)
    if match is None: raise ValueError(f'invalid MEAD clip id: {cid}')
    speaker, emotion, intensity, clip = match.groups()
    return speaker, ALIASES.get(emotion, emotion), int(intensity), int(clip)


def probe(path, executable):
    p = subprocess.run([executable, '-v','error','-show_streams','-show_format','-of','json',str(path)],
                       capture_output=True, check=True, timeout=90)
    return json.loads(p.stdout)


def audio_info(path):
    with wave.open(str(path), 'rb') as f:
        sr, n = f.getframerate(), f.getnframes()
        result = {'sample_rate':sr,'samples':n,'channels':f.getnchannels(),
                  'sample_width':f.getsampwidth(),'duration':n/sr}
        if (sr, f.getnchannels(), f.getsampwidth()) != (16000,1,2) or n < 1:
            raise ValueError('invalid WAV format')
        pcm = np.frombuffer(f.readframes(n),dtype='<i2').astype(np.float64)/32768
    result['rms'] = float(np.sqrt(np.mean(pcm**2)))
    result['peak'] = float(np.max(np.abs(pcm)))
    if not np.isfinite(pcm).all() or result['rms'] <= 1e-6:
        raise ValueError('empty/silent/nonfinite PCM')
    return result


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    tmp.replace(path)


def portable_relative(path, root):
    """Return a forward-slash relative path when path is under root."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()
    except ValueError:
        return None


def process(item, args):
    cid, video, bs = item['clip_id'], Path(item['video']), Path(item['bs'])
    result = dict(item)
    try:
        for p in [video, bs]:
            if not p.is_file(): raise FileNotFoundError(str(p))
        stat = video.stat()
        recipe = {'version':VERSION,'video':str(video),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                  'annotation_sha256':args.annotation_sha256,'bs_sha256':sha256(bs)}
        signature = hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        wav = args.out_root/'wav'/f'{cid}.wav'
        sidecar = args.out_root/'provenance'/f'{cid}.json'
        cached = json.loads(sidecar.read_text(encoding='utf8')) if sidecar.exists() else None
        with np.load(bs,allow_pickle=False) as z:
            motion = z['coeffs']
            n = len(motion)
            if motion.ndim != 2 or motion.shape[1] != 52: raise ValueError('BS dimensions are not T x 52')
            if item['in_final'] and not np.isfinite(motion).all(): raise ValueError('nonfinite final BS')
        if cached and cached['signature']==signature and wav.exists() and sha256(wav)==cached['wav_sha256']:
            info, extracted, metadata = cached['media_probe'], cached['wav_info'], cached
        else:
            info = probe(video, args.ffprobe)
            vs = [s for s in info['streams'] if s['codec_type']=='video']
            aus = [s for s in info['streams'] if s['codec_type']=='audio']
            if len(vs)!=1 or len(aus)!=1: raise ValueError('require exactly one video and one audio stream')
            v, a = vs[0], aus[0]
            start = float(a.get('start_time',0))-float(v.get('start_time',0))
            if abs(start) > 1/16000:
                # Do not discard a nonzero stream offset by resetting timestamps.
                raise ValueError(f'nonzero audio/video start offset: {start:.6f}s')
            tmp = wav.with_suffix('.part.wav')
            subprocess.run([args.ffmpeg,'-nostdin','-v','error','-y','-threads','1','-i',str(video),
                            '-map',f"0:{a['index']}",'-vn','-sn','-dn','-ac','1','-ar','16000',
                            '-c:a','pcm_s16le',str(tmp)],check=True,capture_output=True,timeout=90)
            extracted = audio_info(tmp)
            tmp.replace(wav)
            metadata = dict(recipe,signature=signature,media_probe=info,wav_info=extracted,
                            wav_sha256=sha256(wav),extraction='original_video_audio_stream; no time stretch; no GT lag shift')
            write_json(sidecar,metadata)
        v = next(s for s in info['streams'] if s['codec_type']=='video')
        a = next(s for s in info['streams'] if s['codec_type']=='audio')
        duration = float(v.get('duration',info['format']['duration']))
        reasons = []
        if abs(n/25-duration) > .06: reasons.append('bs_video_duration_mismatch')
        if abs(extracted['duration']-duration) > .08: reasons.append('wav_video_duration_mismatch')
        if n != item.get('n_frames',n): reasons.append('manifest_bs_frame_count_mismatch')
        if not item.get('text'): reasons.append('missing_annotation')
        valid_frames = min(n,int(np.ceil(extracted['duration']*25)))
        result.update(status='review' if reasons else 'ok',reasons=reasons,audio=str(wav.resolve()),
            audio_rel=portable_relative(wav, args.out_root),
            provenance_rel=portable_relative(sidecar, args.out_root),
            video_rel=portable_relative(video, args.mead_root),
            bs_rel=portable_relative(bs, args.data_root),
            bs_sha256=recipe['bs_sha256'],audio_sha256=metadata['wav_sha256'],n_frames=n,
            motion_fps=25,source_video_fps=float(Fraction(v['avg_frame_rate'])),
            video_duration=duration,audio_duration=extracted['duration'],
            audio_start_s=float(a.get('start_time',0)),video_start_s=float(v.get('start_time',0)),
            audio_source='original_video_embedded_track',video_audio_clock_verified=True,
            audio_valid_frames=valid_frames,audio_valid_end_frame_exclusive=valid_frames,
            provenance=str(sidecar),source_video_bytes=stat.st_size,
            training_eligible=not reasons and item['in_final'])
    except Exception as exc:
        result.update(status='error',training_eligible=False,reason=f'{type(exc).__name__}: {exc}')
    return result


def make_pairs(manifest):
    groups = defaultdict(list)
    for row in manifest: groups[(row['speaker'],row['split'],row['content_id'])].append(row)
    pairs, no_neutral = [], []
    for (_, _, _), group in sorted(groups.items()):
        neutral = [r for r in group if r['emotion']=='neutral']
        if not neutral:
            no_neutral.extend(r['clip_id'] for r in group)
            continue
        for row in group:
            ref = row if row['emotion']=='neutral' else min(neutral,key=lambda r:(-r.get('detection_rate',0),r['intensity'],r['clip_id']))
            pairs.append({'source_clip_id':row['clip_id'],'reference_clip_id':ref['clip_id'],
                          'content_id':row['content_id'],'text':row['text'],
                          'content_verified':True,'content_verification':'exact normalized supplied annotation',
                          'annotation_sha256':row['annotation_sha256'],
                          'sync_verified':True,'sync_verification':'same original video as BS; zero stream offset; duration checks',
                          'source_audio_offset_s':0.,'reference_audio_offset_s':0.,
                          'dtw_quality_verified':False})
    return pairs, sorted(no_neutral)


def write_rows(path, data):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=Path('D:/kinetalk_data'))
    ap.add_argument('--mead-root',type=Path,default=Path('E:/mead'))
    ap.add_argument('--annotations',type=Path,default=Path('E:/mead/list_full_mead_annotated.txt'))
    ap.add_argument('--out-root',type=Path,required=True)
    ap.add_argument('--ffmpeg',default='ffmpeg')
    ap.add_argument('--ffprobe',default='ffprobe')
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--limit',type=int,default=0)
    args = ap.parse_args()
    ann = parse_annotations(args.annotations)
    args.annotation_sha256 = sha256(args.annotations)
    final = {r['clip_id']:r for r in rows(args.data_root/'manifests/05_final.jsonl') if r['dataset']=='mead'}
    media = {r['clip_id']:r for r in rows(args.data_root/'manifests/02_media.jsonl') if r['dataset']=='mead'}
    raw = {p.stem:p for p in (args.data_root/'processed/coeffs_raw/mead').glob('*.npz')}
    items = []
    for cid in sorted(set(final)|set(raw)):
        sp, emo, level, number = key = clip_key(cid)
        r = final.get(cid,{})
        video = args.mead_root/sp/'video/front'/SOURCE_EMOTION.get(emo,emo)/f'level_{level}'/f'{number:03d}.mp4'
        recorded = media.get(cid,{}).get('src')
        if recorded and Path(recorded).resolve() != video.resolve():
            raise ValueError(f'BS source-video provenance conflict: {cid}')
        text = ann.get(key,{})
        bs_path = Path(r.get('npz_final',str(raw.get(cid,''))))
        items.append(dict(r,clip_id=cid,clip_number=f'{number:03d}',legacy_sentence_id=r.get('sentence_id'),
                          video=str(video.resolve()),video_rel=portable_relative(video,args.mead_root),
                          bs=str(bs_path.resolve()) if bs_path else '',bs_rel=portable_relative(bs_path,args.data_root),
                          in_final=cid in final,raw_bs=str(raw.get(cid,'')),
                          content_id=content_id(text['text']) if text else None,
                          text_normalized=canonical_text(text['text']) if text else None,
                          annotation_source=str(args.annotations),annotation_sha256=args.annotation_sha256,
                          source_video_manifest_recorded=bool(recorded),**text))
    if args.limit: items=items[:args.limit]
    for name in ['wav','provenance']: (args.out_root/name).mkdir(parents=True,exist_ok=True)
    results = []
    with (args.out_root/'coverage.jsonl').open('w',encoding='utf8') as journal:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process,item,args) for item in items]
            for f in as_completed(futures):
                r=f.result();results.append(r);journal.write(json.dumps(r,ensure_ascii=False)+'\n');journal.flush()
                if len(results)%250==0: print(json.dumps({'processed':len(results),'total':len(items),'status':dict(Counter(r['status'] for r in results))}),flush=True)
    results.sort(key=lambda r:r['clip_id'])
    manifest = []
    for result in results:
        if not result['training_eligible']: continue
        row=dict(result)
        row['audio_legacy']=final[row['clip_id']].get('audio')
        row['sentence_id']=row['content_id']
        row['npz_final']=row['bs']
        manifest.append(row)
    pairs,no_neutral=make_pairs(manifest)
    numbers=defaultdict(set)
    for row in manifest: numbers[(row['speaker'],row['clip_number'])].add(row['content_id'])
    by_id={r['clip_id']:r for r in manifest}
    old_pairs=old_wrong=0
    old_groups=defaultdict(list)
    for row in manifest:old_groups[(row['speaker'],row['clip_number'])].append(row)
    for group in old_groups.values():
        neutrals=[r for r in group if r['emotion']=='neutral']
        if not neutrals:continue
        reference=min(neutrals,key=lambda r:(r['intensity'],r['clip_id']))
        for row in group:
            if row['emotion']=='neutral':continue
            old_pairs+=1;old_wrong+=row['content_id']!=reference['content_id']
    no_neutral_records=[by_id[cid] for cid in no_neutral]
    summary={'version':VERSION,'input_final_bs':len(final),'input_raw_bs':len(raw),'scanned_bs':len(items),
             'annotations':len(ann),'annotation_sha256':args.annotation_sha256,
             'status':dict(Counter(r['status'] for r in results)),
             'final_manifest_rows':len(manifest),'bs_without_usable_wav':sum(r['status']!='ok' for r in results),
             'final_bs_excluded':len(final)-len(manifest),
             'missing_annotation':sum(not r.get('text') for r in results),
             'number_groups':len(numbers),'number_groups_multiple_texts':sum(len(v)>1 for v in numbers.values()),
             'old_non_neutral_pairs':old_pairs,'old_non_neutral_pairs_different_text':old_wrong,
             'content_ids':len(set(r['content_id'] for r in manifest)),
             'neutral_pairs_including_identity':len(pairs),
             'cross_emotion_neutral_pairs':sum(p['source_clip_id']!=p['reference_clip_id'] for p in pairs),
             'no_same_speaker_neutral':len(no_neutral),
             'no_same_speaker_neutral_by_emotion':dict(Counter(r['emotion'] for r in no_neutral_records)),
             'no_same_speaker_neutral_by_split':dict(Counter(r['split'] for r in no_neutral_records)),
             'split_counts':dict(Counter(r['split'] for r in manifest)),
             'wav_bytes':sum(Path(r['audio']).stat().st_size for r in results if r.get('audio_source')),
             'all_final_bs_have_matching_video_wav':len(final)==len(manifest) and not args.limit,
             'dtw_ready_for_alignment_only':True,'training_features_rebuilt':False,
             'limitations':['Annotation is supplied third-party text; no fuzzy text merge.',
                            'Container clocks verify provenance, not perceptual lip-sync ground truth.',
                            'No-neutral records remain valid for native self reconstruction, not neutral cross teachers.',
                            'Old audio features and DTW are invalidated by new audio/text provenance.']}
    write_rows(args.out_root/'coverage.jsonl',results)
    write_rows(args.out_root/'manifest_mead.jsonl',manifest)
    write_rows(args.out_root/'pairs_neutral.jsonl',pairs)
    write_rows(args.out_root/'without_neutral.jsonl',no_neutral_records)
    write_rows(args.out_root/'excluded.jsonl',[r for r in results if not r['training_eligible']])
    write_json(args.out_root/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
