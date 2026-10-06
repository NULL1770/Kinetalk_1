"""Frozen temporal interventions that never change other affect conditions."""
import hashlib
import torch

MODES = ('audio', 'zero', 'static', 'reverse', 'shuffle', 'content_static')


def temporal_variant(local, valid, mode, clip_ids, shuffle_seed=20261006):
    if mode not in MODES[:5]:
        raise ValueError(mode)
    if (local.ndim != 3 or valid.shape != local.shape[:2] or valid.dtype != torch.bool
            or len(clip_ids) != len(local) or not valid.any(1).all()):
        raise ValueError('Expected nonempty native [B,T,D] condition and Boolean mask')
    if not torch.isfinite(local[valid]).all():
        raise ValueError('Observed u_a contains nonfinite values')
    if mode == 'audio':
        return local
    out = torch.zeros_like(local)
    for j, cid in enumerate(clip_ids):
        ix = valid[j].nonzero().flatten()
        if mode == 'static':
            out[j, ix] = local[j, ix].mean(0)
        elif mode == 'reverse':
            out[j, ix] = local[j, ix.flip(0)]
        elif mode == 'shuffle':
            seed = int.from_bytes(hashlib.sha256(f'{shuffle_seed}:{cid}'.encode()).digest()[:8], 'little') % (2**63)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            order = torch.randperm(len(ix), generator=generator).to(ix.device)
            out[j, ix] = local[j, ix[order]]
    return out


def content_static_features(features, valid):
    """Remove HuBERT frame order only for alternate u_a extraction."""
    if (features.ndim != 3 or features.shape[-1] != 1540
            or valid.shape != features.shape[:2] or valid.dtype != torch.bool
            or not valid.any(1).all() or not torch.isfinite(features[valid]).all()):
        raise ValueError('Expected native 1540D audio features and Boolean support')
    updated = features.clone()
    for i in range(len(features)):
        updated[i, valid[i], :768] = features[i, valid[i], :768].mean(0)
    return updated


def replace_temporal(affect, temporal):
    if temporal.shape != affect['u_a'].shape:
        raise ValueError('u_a shape changed')
    updated = dict(affect)
    for name in ('u_a', 'temporal', 'local'):
        if name in updated:
            updated[name] = temporal
    assert all(updated[k] is v for k, v in affect.items() if k not in ('u_a', 'temporal', 'local'))
    return updated


def canonical_flags(system, audio):
    # Preserve the native final evaluator's dispatch flags without gradients.
    if torch.is_grad_enabled():
        raise RuntimeError('Frozen diagnosis requires no_grad')
    system.requires_grad_(False)
    audio.requires_grad_(True)
    system.renderer.requires_grad_(True)
    assert all(p.grad is None for m in (system, audio) for p in m.parameters())
