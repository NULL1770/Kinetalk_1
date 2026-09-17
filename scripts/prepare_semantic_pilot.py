"""Select a reproducible MEAD pilot from audited native-clock artifacts.

Run locally to prepare the video-label extraction list and remote manifests.
No video, coefficient, intensity or VA values are synthesized by this script.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


EMOTIONS = ['neutral', 'angry', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-source', type=Path, required=True)
    p.add_argument('--val-source', type=Path, required=True)
    p.add_argument('--video-root', type=Path, required=True)
    p.add_argument('--native-root', required=True)
    p.add_argument('--remote-output', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-speakers', nargs='+', default=['mead_M003','mead_M005','mead_M007','mead_M009','mead_M011','mead_M012'])
    p.add_argument('--val-speakers', nargs='+', default=['mead_M025','mead_M037'])
    p.add_argument('--per-cell', type=int, default=2)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    videos, counts = [], {}
    for split, path, speakers in [('train',args.train_source,args.train_speakers),('val',args.val_source,args.val_speakers)]:
        groups = defaultdict(list)
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            r = json.loads(line)
            if r['speaker'] not in speakers or r['dataset'] != 'mead':
                continue
            emotion = EMOTIONS[int(r['emotion'])]
            level = int(r['intensity'])
            if emotion != 'neutral' and level not in (1,3):
                continue
            parts = r['clip_id'].split('_')
            video_emotion = {'disgust': 'disgusted', 'surprise': 'surprised'}.get(emotion, emotion)
            video = args.video_root / parts[1] / 'video' / 'front' / video_emotion / f'level_{int(parts[-2][1:])}' / (parts[-1]+'.mp4')
            if not video.is_file():
                continue
            groups[(r['speaker'],emotion,level)].append((r,video))
        selected = []
        for key, candidates in sorted(groups.items()):
            seen = set()
            for r,video in sorted(candidates,key=lambda item:item[0]['clip_id']):
                if r['sentence'] in seen:
                    continue
                seen.add(r['sentence'])
                native = str(Path(args.native_root) / r['artifact']).replace('\\','/')
                selected.append(dict(
                    clip_id=r['clip_id'],dataset='mead',speaker=r['speaker'],sentence_id=r['sentence'],
                    emotion=key[1],intensity_id=key[2],intensity_valid=True,split=split,
                    motion_path=native,content_path=native,audio_path=native,
                    va_path=args.remote_output.rstrip('/')+'/va/'+r['clip_id']+'.npz',
                    va_source='visual_teacher',source_audio_offset_s=0.0,
                    source_artifact_sha256=r.get('artifact_sha256'),
                    native_schema=r.get('schema'),
                ))
                videos.append(dict(clip_id=r['clip_id'],video=str(video),split=split,
                                   speaker=r['speaker'],emotion=key[1],intensity_id=key[2],
                                   output=str(args.output/'va'/(r['clip_id']+'.npz')),
                                   motion_fps=25.0))
                if len(seen) >= args.per_cell:
                    break
        if set(speakers) != {r['speaker'] for r in selected}:
            raise ValueError(f'Missing requested {split} speakers')
        (args.output/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
        counts[split] = {'clips':len(selected),'speaker_counts':dict(Counter(r['speaker'] for r in selected)),
                         'emotion_counts':dict(Counter(r['emotion'] for r in selected)),
                         'intensity_counts':dict(Counter(r['intensity_id'] for r in selected))}
    if set(args.train_speakers)&set(args.val_speakers):
        raise ValueError('Speaker overlap')
    (args.output/'videos.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in videos),encoding='utf-8')
    counts['scope']='MEAD pilot only; no test split accessed; CREMA-D not validated in this run'
    (args.output/'selection.json').write_text(json.dumps(counts,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(counts,ensure_ascii=False))


if __name__ == '__main__':
    main()
