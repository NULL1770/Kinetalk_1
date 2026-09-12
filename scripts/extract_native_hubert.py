"""Extract native-clock HuBERT features for DTW auditing.

The existing hubert_content_v1 files were resized to motion length.  This
script deliberately keeps HuBERT's native output rate (about 50 Hz) and saves
the feature-frame times alongside the embeddings.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor


def audio_to_float(path: Path, sr: int = 16000) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return np.frombuffer(p.stdout, dtype="<f4").copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--media-root", type=Path, default=None,
                    help="Root containing audio_rel paths from a portable manifest.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--groups", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("dataset") == "mead" and r.get("qc_pass", True)]
    if args.limit:
        rows = rows[: args.limit]
    if args.groups:
        groups = defaultdict(list)
        for r in rows:
            groups[(r['speaker'], str(r['sentence_id']))].append(r)
        by_speaker = defaultdict(list)
        for key, group in sorted(groups.items()):
            if any(r['emotion'] == 'neutral' for r in group):
                by_speaker[key[0]].append(group)
        rng = random.Random(args.seed)
        for groups in by_speaker.values():
            rng.shuffle(groups)
        selected = []
        while len(selected) < args.groups and any(by_speaker.values()):
            for speaker in sorted(by_speaker):
                if by_speaker[speaker] and len(selected) < args.groups:
                    selected.append(by_speaker[speaker].pop())
        rows = [r for group in selected for r in group]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.model), local_files_only=True)
    # This pinned environment has the official PyTorch checkpoint but no
    # safetensors file.  Keep loading local-only and never fall back to Hub.
    model = HubertModel.from_pretrained(str(args.model), local_files_only=True, use_safetensors=True).to(device).eval()
    torch.set_num_threads(4)
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / 'selected_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
    recipe = {'schema': 2, 'layer': args.layer, 'sampling_rate': 16000,
              'padding': 'none', 'config': model.config.to_dict()}
    signature = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    (args.out_root / 'recipe.json').write_text(json.dumps(recipe, indent=2), encoding='utf8')
    hop, receptive = 1, 1
    for kernel, stride in zip(model.config.conv_kernel, model.config.conv_stride):
        receptive += (kernel - 1) * hop
        hop *= stride
    done = 0
    for row in rows:
        path = args.out_root / 'mead' / f"{row['clip_id']}.npz"
        if path.exists() and not args.overwrite:
            with np.load(path, allow_pickle=False) as cached:
                if 'signature' in cached and str(cached['signature']) == signature:
                    done += 1
                    continue
            raise ValueError(f'Incompatible cached extraction: {path}')
        audio_path = None
        if args.media_root is not None and row.get('audio_rel'):
            audio_path = args.media_root / row['audio_rel']
        if audio_path is None or not audio_path.exists():
            candidate = Path(row.get('audio', ''))
            if candidate.exists():
                audio_path = candidate
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError(
                f"audio not found for {row['clip_id']}; checked audio_rel={row.get('audio_rel')} "
                f"under media-root={args.media_root} and audio={row.get('audio')}"
            )
        wave = audio_to_float(audio_path)
        inputs = extractor(wave, sampling_rate=16000, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(device)
        attention = getattr(inputs, "attention_mask", None)
        if attention is not None:
            attention = attention.to(device)
        with torch.inference_mode():
            out = model(input_values, attention_mask=attention, output_hidden_states=True)
        feat = out.hidden_states[args.layer][0].float().cpu().numpy()
        validation = out.hidden_states[9][0].float().cpu().numpy()
        n = int(model._get_feat_extract_output_lengths(len(wave)))
        if n != len(feat):
            raise ValueError('HuBERT convolution length mismatch')
        times = (np.arange(n, dtype=np.float64) * hop + (receptive - 1) / 2) / 16000
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, content=feat.astype(np.float16), validation=validation.astype(np.float16),
                            times=times, signature=signature, audio_samples=len(wave),
                            sample_rate=16000, hop_samples=hop, receptive_samples=receptive)
        done += 1
        if done % 25 == 0:
            print(f"processed {done}/{len(rows)}", flush=True)
    print(json.dumps({"processed": done, "total": len(rows), "out_root": str(args.out_root), "device": device}))


if __name__ == "__main__":
    main()

