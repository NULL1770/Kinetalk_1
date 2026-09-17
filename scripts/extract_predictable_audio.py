"""Audio-only frozen intermediate states and native-clock prosody sidecars.

Uses audited emotion2vec weights already installed, not a WavLM replication.
No text, labels, target motion, fitted dataset statistics or network downloads.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_emotion2vec_pilot import load_extractor, sha, expected_length, align_features


def prosody(wave, targets, valid, sample_rate=16000):
    """40-ms centered autocorrelation pitch, log RMS, periodicity and voicing.

    Raw-wave energy is measured before whole-utterance normalization. Unvoiced
    pitch is zero with an explicit voicing flag, not fabricated interpolation.
    """
    wave = np.asarray(wave, dtype=np.float64)
    positions = np.rint(np.asarray(targets) * sample_rate).astype(int)
    length = 640
    lo, hi = int(sample_rate / 500), int(sample_rate / 60)
    result = np.zeros((len(targets), 4), dtype=np.float32)
    for i, center in enumerate(positions):
        if not valid[i]:
            continue
        indices = np.arange(center - length // 2, center + length // 2)
        segment = wave[np.clip(indices, 0, len(wave) - 1)]
        rms = np.sqrt(np.mean(segment ** 2))
        signal = (segment - segment.mean()) * np.hanning(length)
        acf = np.correlate(signal, signal, mode='full')[length-1:]
        if acf[0] > 1e-12:
            # Hann-window correction limits systematic suppression at long lags.
            window = np.hanning(length)
            correction = np.correlate(window, window, mode='full')[length-1:]
            acf = acf / np.maximum(correction, 1e-10)
            acf = acf / max(acf[0], 1e-12)
            peaks = [j for j in range(lo, hi + 1) if acf[j] > acf[j-1] and acf[j] >= acf[j+1]]
            lag = max(peaks, key=lambda j: acf[j]) if peaks else lo
            best = float(np.clip(acf[lag], 0, 1)) if peaks else 0.
            # Among similarly periodic candidates prefer the first peak to
            # reduce subharmonic selection for nearly periodic speech.
            plausible = [j for j in peaks if acf[j] >= max(.65, best * .95)]
            if plausible:
                lag = plausible[0]
            confidence = float(np.clip(acf[lag], 0, 1)) if peaks else 0.
        else:
            lag, confidence = lo, 0.
        voiced = rms > 1e-4 and confidence >= .65
        result[i] = [np.log(sample_rate / lag) if voiced else 0., np.log(max(rms, 1e-7)), confidence, float(voiced)]
    return result


@torch.inference_mode()
def extract(clip, model, geometry, layers, device):
    import soundfile as sf
    provenance = clip['metadata']['provenance']
    path = Path(provenance['audio_path'])
    if sha(path) != provenance['audio_sha256']:
        raise ValueError(f"Wave hash changed: {path}")
    wave, sr = sf.read(path, dtype='float32')
    if sr != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError('Expected finite audited 16k mono source')
    captures = {}
    hooks = []
    for number in layers:
        def capture(module, inputs, output, key=number):
            captures[key] = output[0].detach() if isinstance(output, tuple) else output.detach()
        hooks.append(model.blocks[number - 1].register_forward_hook(capture))
    try:
        sequence = torch.from_numpy(wave).to(device)
        out = model.extract_features(F.layer_norm(sequence, sequence.shape)[None], mask=False, remove_extra_tokens=True)
    finally:
        for hook in hooks:
            hook.remove()
    n = expected_length(len(wave), geometry)
    extra = int(model.modality_encoders['AUDIO'].modality_cfg.num_extra_tokens)
    normalized = []
    layer_std = {}
    for number in layers:
        x = captures[number]
        if x.shape != (1, n + extra, 768):
            raise ValueError(f'Unexpected intermediate hidden layout {number}: {x.shape}')
        x = x[:, extra:]
        layer_std[str(number)] = float(x.std(1).mean())
        normalized.append(F.layer_norm(x, (768,)))
    # Prespecified average with frame-wise LN; no learned weights or data-fit.
    middle = torch.stack(normalized).mean(0)[0].float().cpu().numpy()
    final = out['x'][0].float().cpu().numpy()
    times = (geometry['center_offset_samples'] + np.arange(n) * geometry['hop_samples']) / 16000
    targets = clip['times'].numpy() + float(provenance['audio_offset_s'])
    valid = clip['valid'].numpy()
    middle, covered, extended = align_features(middle, times, targets, valid, len(wave), geometry)
    final, _, _ = align_features(final, times, targets, valid, len(wave), geometry)
    p = prosody(wave, targets, valid)
    return {'clip_id': clip['clip_id'], 'sentence_id': clip['sentence_id'],
        'middle': torch.from_numpy(middle).half(), 'final': torch.from_numpy(final).half(),
        'prosody': torch.from_numpy(p), 'valid': clip['valid'], 'times': clip['times'],
        'record': {'wave_sha256': provenance['audio_sha256'], 'audio_offset_s': provenance['audio_offset_s'],
            'layer_temporal_std': layer_std, 'boundary_extended_frames': int(extended.sum()),
            'voiced_fraction': float(p[valid, 3].mean())}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--layers', type=int, nargs='+', default=[2, 4, 6])
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    model, geometry, info = load_extractor(args.model_dir, args.device)
    if any(l < 1 or l >= len(model.blocks) for l in args.layers):
        raise ValueError(f'Intermediate layers must be below last block {len(model.blocks)}')
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {'schema':'predictable_audio_v1', 'model':info, 'geometry':geometry, 'layers':args.layers,
        'layer_mean':'equal mean of per-frame LayerNorm intermediate block output hidden states',
        'prosody':['log_f0_unvoiced_zero','log_rms','periodicity','voiced'],
        'prosody_method':'40ms centered window, corrected autocorrelation 60-500Hz; confidence .65',
        'offline':True, 'statistics_fitted':False, 'script_sha256':sha(__file__), 'source_cache_sha256':{}}
    for split in ('train','heldout'):
        cache_path = args.data / (split + '.pt')
        dataset = torch.load(cache_path, map_location='cpu', weights_only=False)
        records = []
        for index, clip in enumerate(dataset['queries']):
            records.append(extract(clip, model, geometry, args.layers, args.device))
            if index % 25 == 0 or index + 1 == len(dataset['queries']):
                print(json.dumps({'split':split,'done':index+1,'total':len(dataset['queries'])}),flush=True)
        meta['source_cache_sha256'][split] = sha(cache_path)
        torch.save({'clips':records,'provenance':dict(meta)}, args.output / (split + '.pt'))
    (args.output / 'provenance.json').write_text(json.dumps(meta, indent=2),encoding='utf8')


if __name__ == '__main__':
    main()
