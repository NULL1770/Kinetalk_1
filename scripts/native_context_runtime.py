"""Native-length context operations without altering historical experiment code.

The caller owns membership, complete audio/native provenance, padding masks,
optimizer policy and matched update budgets. Encoding consumes audio/content
and independent anchors only. The loss uses tracked motion as supervision and
strictly past training context; it is not an inference function.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from scripts import train_prefix_upper as p
from scripts.train_full_staged import base_forward
from kinetalk_b0.models.slow_state_affect import lift_slow_state


def _valid_mask(valid):
    if (not torch.is_tensor(valid) or valid.ndim != 2 or valid.dtype != torch.bool
            or min(valid.shape) < 1 or not valid.any(1).all()):
        raise ValueError('Every native row needs a nonempty Boolean observation mask')
    return valid


def _sequence(value, valid, width, name):
    if (not torch.is_tensor(value) or value.shape != (*valid.shape, width)
            or not value.is_floating_point() or value.device != valid.device
            or not torch.isfinite(value[valid]).all()):
        raise ValueError(f'{name} must be finite observed [B,T,{width}] on the mask device')
    return value


@torch.no_grad()
def encode_full_conditions(system, audio, b, target_scales):
    """Return a new batch with freshly encoded frozen whole-input conditions.

    ``base_forward`` uses the actual last valid frame of each row, matching
    ``cache_current_base`` and preserving its true-length B0 positional path.
    Frozen audio reads the supplied complete/native window, never motion.
    ``static_upper`` uses its valid-frame mean state and independent anchors.
    Local features are deliberately not returned: a trainable local encoder
    must be called outside this no-grad function.
    """
    if any(key not in b for key in ('valid', 'content', 'audio_features', 'anchors', 'anchor_valid')):
        raise ValueError('Native content/audio, independent anchors and masks required')
    if getattr(system, 'training', False) or getattr(audio, 'training', False):
        raise ValueError('Frozen system and audio must already be in eval mode')
    for name, module in (('system', system), ('audio', audio)):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError(name+' must be frozen before native condition encoding')
    valid = _valid_mask(b['valid'])
    _sequence(b['content'], valid, 768, 'content')
    _sequence(b['audio_features'], valid, 1540, 'audio_features')
    anchor, anchor_valid = b['anchors'], b['anchor_valid']
    if (anchor.shape != (len(valid), 52) or not anchor.is_floating_point()
            or anchor.device != valid.device or anchor_valid.shape != anchor.shape
            or anchor_valid.dtype != torch.bool or anchor_valid.device != valid.device
            or not anchor_valid[:, p.CC].all() or not torch.isfinite(anchor[:, p.CC]).all()):
        raise ValueError('Nine independently observed finite upper anchors required')
    if (not torch.is_tensor(target_scales) or target_scales.shape != (52,)
            or not target_scales.is_floating_point() or not torch.isfinite(target_scales).all()
            or not (target_scales > 0).all()):
        raise ValueError('Frozen target channel scales must contain52 positive finite values')
    base = base_forward(system, b['content'], valid)
    affect = audio(b['audio_features'], valid)
    if any(key not in affect for key in ('global', 'intensity_value', 'state')):
        raise ValueError('Frozen audio must supply global, intensity and four-state predictions')
    state = _sequence(affect['state'], valid, 4, 'audio state')
    mean = torch.where(valid[..., None], state, 0.).sum(1, keepdim=True)/valid.sum(1)[:, None, None]
    static = anchor[:, None]+lift_slow_state(mean, target_scales.to(device=mean.device, dtype=mean.dtype))
    if not torch.isfinite(static[:, 0, p.CC]).all():
        raise ValueError('Audio static upper origin is nonfinite')
    for key in ('global', 'intensity_value'):
        if affect[key].ndim != 2 or len(affect[key]) != len(valid) or not torch.isfinite(affect[key]).all():
            raise ValueError('Frozen audio clip conditions must be finite [B,D]')
    if affect['intensity_value'].shape[-1] != 1:
        raise ValueError('Frozen audio intensity must be [B,1]')
    # No motion/label/reference dictionary is passed to either model. Existing
    # batch fields are retained for the caller's later loss, without mutation.
    # Old local caches refer to a different window or to a frozen earlier
    # parameter state. Remove them instead of allowing an accidental detached
    # path around the trainable native local call in the caller.
    updated = {key: value for key, value in b.items() if key not in ('prefix_local', 'audio_local')}
    return {**updated, 'b0': base['b0'], 'h0': base['h0'],
            'audio_global': affect['global'], 'audio_intensity': affect['intensity_value'],
            'static_upper': static[:, 0, p.CC]}


def variable_context_loss(upper, b, ident, native, scales, noise, time):
    """Standard unknown-only FM for arbitrary padded native length.

    Caller supplies one complete noise tensor and one flow-time scalar per
    clip. Every chunk sees at most the previous eight native frames of clean
    GT motion as training context; missing frames never get time-compressed.
    Chunk losses are weighted by their observed unknown frame counts, yielding
    one scalar for a single optimizer update. Tail noise is padded to24 slots.
    """
    if any(key not in b for key in ('valid', 'motion', 'static_upper', 'h0', 'audio_global', 'audio_intensity')):
        raise ValueError('Native supervised batch and audio conditions required')
    valid = _valid_mask(b['valid'])
    frames = valid.shape[1]
    motion = b['motion']
    if (not torch.is_tensor(motion) or motion.shape != (*valid.shape, 52)
            or not motion.is_floating_point() or motion.device != valid.device):
        raise ValueError('Motion must share the native padded52-channel clock')
    # Only nine channels are supervised. Other channels may be unavailable
    # placeholders and must not become an implicit data-quality prerequisite.
    _sequence(motion[..., p.CC], valid, 9, 'upper motion')
    if native.ndim != 3 or native.shape[:2] != valid.shape:
        raise ValueError('Native local conditions must share the padded clock')
    _sequence(native, valid, native.shape[-1], 'native local')
    if ('channel_mask' in b and (b['channel_mask'].shape != (len(valid), 52)
            or b['channel_mask'].dtype != torch.bool or not b['channel_mask'][:, p.CC].all())):
        raise ValueError('All nine supervised upper channels must be observed')
    if (b['static_upper'].shape != (len(valid), 9) or not torch.isfinite(b['static_upper']).all()
            or b['static_upper'].device != valid.device):
        raise ValueError('Finite audio-predicted static upper origin required')
    if (not torch.is_tensor(scales) or scales.shape != (9,) or not scales.is_floating_point()
            or not torch.isfinite(scales).all() or not (scales > 0).all()):
        raise ValueError('Nine finite positive frozen residual scales required')
    if (not torch.is_tensor(noise) or noise.shape != (*valid.shape, 9) or not noise.is_floating_point()
            or not torch.isfinite(noise[valid.to(noise.device)]).all()):
        raise ValueError('Caller noise must match the full native clock and be finite where observed')
    if (not torch.is_tensor(time) or time.shape != (len(valid),) or not time.is_floating_point()
            or not torch.isfinite(time).all() or ((time < 0.) | (time > 1.)).any()):
        raise ValueError('Caller flow time must be one finite [0,1] scalar per clip')
    target = p.h.normalized_target(b, scales.to(b['motion']))
    loss = target.new_zeros(())
    for start in range(0, frames, p.CHUNK):
        stop = min(start+p.CHUNK, frames)
        ids = valid[:, start:stop].any(1).nonzero(as_tuple=True)[0]
        if not len(ids):
            continue
        # Fixed history/current slot layout even for a short final chunk.
        current_noise = target.new_zeros(len(ids), p.HISTORY+p.CHUNK, 9)
        current_noise[:, p.HISTORY:p.HISTORY+stop-start] = noise[ids.to(noise.device), start:stop].to(target)
        part = p.patch_loss(upper, b, ident, native, target, target, ids,
                           torch.full_like(ids, start), current_noise, time[ids.to(time.device)].to(target), empty=False)
        loss = loss+part*valid[ids, start:stop].sum()
    return loss/valid.sum()


def _integer_metadata(value, name):
    if torch.is_tensor(value):
        if value.ndim != 1 or value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(name+' must be one-dimensional integer metadata')
        return value.detach().to(device='cpu', dtype=torch.long)
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or not value or any(type(item) is not int for item in value)):
        raise ValueError(name+' must be a nonempty sequence of integer metadata')
    return torch.tensor(value, dtype=torch.long)


def paired_noise(native_lengths, center_starts, generator, mode='full'):
    """Draw paired native/central noise without reading target or valid values.

    Returns ``(selected_noise, time, draws)`` on CPU. ``draws['full_noise']``
    and ``draws['time']`` identify the identical random draw in either arm.
    The selected full length is a multiple of16 covering both native content
    and every96-frame center slice, including its short-clip padding. Callers
    must pad full batches to ``draws['draw_frames']`` and use false masks beyond
    actual lengths. Separate matched generators must begin at the same state.
    """
    lengths = _integer_metadata(native_lengths, 'native_lengths')
    starts = _integer_metadata(center_starts, 'center_starts')
    if (lengths.shape != starts.shape or not len(lengths) or (lengths < 1).any()
            or (starts < 0).any() or (starts > (lengths-96).clamp_min(0)).any()):
        raise ValueError('Center crops must stay on the declared native clock; short clips start at zero')
    if mode not in ('full', 'center'):
        raise ValueError('Paired noise mode must be full or center')
    if not isinstance(generator, torch.Generator) or generator.device.type != 'cpu':
        raise ValueError('A dedicated CPU generator is required for paired training noise')
    needed = int(torch.maximum(lengths, starts+96).max())
    draw_frames = ((needed+p.CHUNK-1)//p.CHUNK)*p.CHUNK
    full_noise = torch.randn(len(lengths), draw_frames, 9, generator=generator)
    time = torch.rand(len(lengths), generator=generator)
    if mode == 'full':
        selected = full_noise
    else:
        selected = torch.stack([full_noise[row, int(start):int(start)+96] for row, start in enumerate(starts)])
    return selected, time, {'full_noise': full_noise, 'time': time, 'draw_frames': draw_frames,
                            'native_lengths': lengths.clone(), 'center_starts': starts.clone()}


__all__ = ['encode_full_conditions', 'variable_context_loss', 'paired_noise']
