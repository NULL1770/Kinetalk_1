"""Independent ASR check of the claimed same-sentence pairing."""
import argparse
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def normalized(text):
    return re.findall(r"[a-z]+", text.lower())


def main():
    from faster_whisper import WhisperModel
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', default='small.en')
    ap.add_argument('--groups', type=int, default=3)
    ap.add_argument('--all-intensities', action='store_true')
    args = ap.parse_args()
    groups = defaultdict(list)
    for line in args.manifest.read_text().splitlines():
        row = json.loads(line)
        groups[(row['speaker'], row['sentence_id'])].append(row)
    model = WhisperModel(args.model, device='cpu', compute_type='int8', cpu_threads=6)
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf8') as out:
        for group in list(groups.values())[:args.groups]:
            # All emotions, lowest available intensity per class.
            by_emotion = defaultdict(list)
            for row in group:
                by_emotion[row['emotion']].append(row)
            selected = group if args.all_intensities else [min(v, key=lambda r: r['intensity']) for v in by_emotion.values()]
            selected.sort(key=lambda r: (r['emotion'] != 'neutral', r['emotion']))
            reference = []
            for row in selected:
                segments, info = model.transcribe(row['audio'], language='en', beam_size=5,
                                                   word_timestamps=True, condition_on_previous_text=False)
                segments = list(segments)
                text = ' '.join(s.text.strip() for s in segments)
                tokens = normalized(text)
                if row['emotion'] == 'neutral':
                    reference = tokens
                score = SequenceMatcher(None, reference, tokens, autojunk=False).ratio()
                result = {k: row[k] for k in ['clip_id','speaker','sentence_id','emotion','intensity']}
                result.update(text=text, word_sequence_similarity=score,
                              model=args.model, avg_logprob=[s.avg_logprob for s in segments],
                              words=[{'word': w.word, 'start': w.start, 'end': w.end,
                                      'probability': w.probability} for s in segments for w in (s.words or [])])
                out.write(json.dumps(result) + '\n'); out.flush()
                results.append(result)
                print(json.dumps({k: result[k] for k in ['clip_id','text','word_sequence_similarity']}), flush=True)
    print(json.dumps({'clips': len(results), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
