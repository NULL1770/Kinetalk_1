"""Attribute upper-face regressions without changing weights or target inputs."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_isolated_audio_state import load_context, center
from scripts.train_full_staged import subset, batch_identity, obs
from scripts.compact_native_curves import expand_curves
from scripts.train_formal_predictable_projection import save_json
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    for key in ('data', 'source', 'curves', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    data, system, _, identities, _ = load_context(a.data, a.source, 'cuda')
    q = data['splits']['validation']
    c = expand_curves(torch.load(a.curves, map_location='cpu', weights_only=False))
    assert c['clip_id'] == q['clip_id']
    valid = q['valid']; cc = list(UPPER_INDICES)
    base, previous = c['predictions']['42/base'], c['predictions']['42/full']
    bc, pc = center(base[..., cc], valid), center(previous[..., cc], valid)
    bm = (base[..., cc] - bc) * valid[..., None]
    pm = (previous[..., cc] - pc) * valid[..., None]
    variants = {'base': base, 'previous': previous,
                'base_mean_new_timing': compose_upper_face(base, bm + pc, valid),
                'new_mean_base_timing': compose_upper_face(base, pm + bc, valid),
                'base_mean_only': compose_upper_face(base, bm, valid)}
    result = {'scope': 'full development diagnostic, not model selection', 'clips': len(valid),
              'test_loaded': False, 'variants': {}}
    for name, pred in variants.items():
        correct = 0
        for ix in torch.arange(len(valid)).split(32):
            b = subset(q, ix, 'cuda'); ident = batch_identity(identities, b)
            out = system.encode_motion(torch.where(obs(b), pred[ix].cuda() - b['b0']
                                       - ident['baseline'][:, None], 0.), b['valid'])
            correct += int((out['emotion_logits'].argmax(-1) == b['emotion_id']).sum())
        result['variants'][name] = {'internal_emotion_accuracy': correct / len(valid),
            'upper_raw_mse': float((pred[..., cc] - q['motion'][..., cc])[valid].square().mean()),
            'upper_centered_mse': float((center(pred[..., cc], valid) - center(q['motion'][..., cc], valid))[valid].square().mean())}
    save_json(a.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
