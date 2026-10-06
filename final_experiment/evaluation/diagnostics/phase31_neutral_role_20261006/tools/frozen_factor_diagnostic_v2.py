"""Small, replay-gated identity/u_a diagnosis of immutable historical models.

Uses each model's own frozen source, original evaluation batches and noise.
No optimizer or training. Cross-person outputs have response statistics only,
because the original speaker's GT is not their reconstruction target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

TEMPORAL_MODES = ('zero', 'static', 'reverse', 'shuffle')
IDENTITY_MODES = ('own_A', 'own_B', 'other_1', 'other_2', 'zero_code',
                  'zero_bias', 'code_other_1', 'bias_other_1')
REGIONS = {'mouth': list(range(14, 41)), 'jaw': [17],
           'smile': [23, 24], 'brows': list(range(41, 46)), 'eyes': list(range(14))}
REGION_COLUMNS = ('mse', 'mean_bias_mse', 'centered_mse', 'pred_centered_rms',
                  'gt_centered_rms', 'centered_correlation', 'displacement_mse',
                  'pred_q90_q10', 'gt_q90_q10')
RESPONSE_COLUMNS = ('total_rms', 'mean_rms', 'centered_rms', 'displacement_rms')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def model_digest(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        v = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def stratified_indices(clip_ids, speakers, emotions, levels):
    """One lexicographically first clip per observed person/emotion/level cell."""
    if not (len(clip_ids) == len(speakers) == len(emotions) == len(levels)):
        raise ValueError('Metadata lengths differ')
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError('Duplicate clip ID')
    chosen = {}
    for i, cid in enumerate(clip_ids):
        key = (int(speakers[i]), int(emotions[i]), int(levels[i]))
        if key not in chosen or cid < clip_ids[chosen[key]]:
            chosen[key] = i
    return sorted(chosen.values())


def original_batches(selected, count, batch_size):
    if batch_size < 1 or any(i < 0 or i >= count for i in selected):
        raise ValueError('Invalid native indices/batch size')
    return [list(range(start, min(start + batch_size, count)))
            for start in sorted({i // batch_size * batch_size for i in selected})]


def identity_variant(own, own_a, own_b, other_1, other_2, mode):
    if mode == 'own_A':
        return own_a
    if mode == 'own_B':
        return own_b
    if mode == 'other_1':
        return other_1
    if mode == 'other_2':
        return other_2
    if mode == 'zero_code':
        return {**own, 'code': torch.zeros_like(own['code'])}
    if mode == 'zero_bias':
        return {**own, 'baseline': torch.zeros_like(own['baseline'])}
    if mode == 'code_other_1':
        return {**own, 'code': other_1['code']}
    if mode == 'bias_other_1':
        return {**own, 'baseline': other_1['baseline']}
    raise ValueError(mode)


def response_values(value, reference, valid, channels, indices):
    indices = [i for i in indices if channels[i]]
    if not indices or not valid.any() or not (valid[1:] & valid[:-1]).any():
        raise ValueError('Response needs observed channels and adjacent frames')
    delta = (value - reference)[:, indices].double()
    d = delta[valid]; mean = d.mean(0)
    velocity = delta.diff(dim=0)[valid[1:] & valid[:-1]]
    return [float(d.square().mean().sqrt()), float(mean.square().mean().sqrt()),
            float((d - mean).square().mean().sqrt()), float(velocity.square().mean().sqrt())]


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@torch.no_grad()
def run(a):
    binding = json.loads(a.binding.read_text())
    spec = binding['models'][a.model]
    out = a.output
    if out.exists():
        raise FileExistsError('Fresh output required')
    out.mkdir(parents=True)
    state_path = out / 'state.json'

    def update(status, **kwargs):
        state_path.write_text(json.dumps(dict(status=status, updated_at=time.time(),
            model=a.model, training_performed=False, test_loaded=False, **kwargs), indent=2))
        print(json.dumps(dict(status=status, model=a.model, **kwargs)), flush=True)

    update('binding')
    try:
        if sha(__file__) != binding['diagnostic_sha256']:
            raise ValueError('Diagnostic source differs')
        if sha(a.protocol) != binding['protocol_sha256']:
            raise ValueError('Intervention protocol differs')
        import importlib.util
        pspec = importlib.util.spec_from_file_location('frozen_temporal_protocol', a.protocol)
        protocol = importlib.util.module_from_spec(pspec); pspec.loader.exec_module(protocol)
        code = Path(spec['code_root'])
        for name, digest in spec['source_sha256'].items():
            if sha(code / name) != digest:
                raise ValueError('Frozen model source differs: ' + name)
        sys.path.insert(0, str(code))
        from scripts.packed_trainval_cache import load_packed
        from scripts.train_full_staged import subset, cache_current_base, encode_ref, batch_identity, audio_affect
        from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
        from kinetalk_b0.models.slow_state_affect import SlowStateAffect

        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = spec['stable_runtime']
        torch.backends.cudnn.benchmark = False
        if spec['stable_runtime']:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
                torch.backends.cuda.enable_cudnn_sdp(False)
        cp = Path(spec['checkpoint_root'])
        for name in ('audio/final.pt', 'audio/curves.pt'):
            if sha(cp / name) != spec['files'][name]:
                raise ValueError('Checkpoint/curves hash differs')
        ck = torch.load(cp / 'audio/final.pt', map_location='cpu', weights_only=False)
        saved = torch.load(cp / 'audio/curves.pt', map_location='cpu', weights_only=False)
        data_root = Path(binding['data'])
        marker = json.loads((data_root/'packed_cache.json').read_text())
        if sha(data_root/'packed_cache.json') != binding['data_marker_sha256']:
            raise ValueError('Packed marker differs')
        if sha(data_root/marker['metadata']) != binding['packed_metadata_sha256']:
            raise ValueError('Packed metadata differs')
        for name, digest in binding['validation_array_sha256'].items():
            if sha(data_root/name) != digest:
                raise ValueError('Validation array differs: ' + name)
        data = load_packed(binding['data'], materialize=False, with_refs=True)
        if set(data['splits']) != {'train', 'validation'} or data['provenance']['test_loaded']:
            raise ValueError('Packed boundary differs')
        if data['provenance']['manifest_sha256'] != ck['data_manifest_sha256']:
            raise ValueError('Dataset differs')
        q = data['splits']['validation']; data['splits'] = {'validation': q}
        if q['clip_id'] != saved['clip_id'] or len(q['clip_id']) != 1367:
            raise ValueError('Validation order differs')
        selected = stratified_indices(q['clip_id'], q['speaker_id'], q['emotion_id'], q['intensity_id'])
        if selected != binding['selected_indices']:
            raise ValueError('Predeclared sample differs')
        groups = original_batches(selected, len(q['clip_id']), spec['batch_size'])
        selected_set = set(selected)
        device = torch.device(a.device)
        system = NeutralAffectSystem(ck['config']).to(device).eval()
        system.load_state_dict(ck['system'], strict=True)
        audio = SlowStateAffect(ck['feature_stats']['mean'], ck['feature_stats']['std'],
                                stride=spec['stride']).to(device).eval()
        audio.load_state_dict(ck['audio'], strict=True)
        initial_system, initial_audio = model_digest(system), model_digest(audio)
        system.requires_grad_(False)
        audio.requires_grad_(True); system.renderer.requires_grad_(True)

        # Reproduce original Stage1 caching batches (32), separately from the
        # renderer evaluation batches (2 or 16). TF32 kernels can vary with
        # Stage1 true-length group batch shape even though clips are independent.
        cache_current_base(system, data, device, batch_size=32)
        # Enrollment residuals are also cached in each model's own coordinates.
        refs = {}
        for sid, r in data['refs'].items():
            refs[sid] = {}
            for name, ids in [('full', range(len(r['valid']))), ('A', [0]), ('B', [1])]:
                encoded = encode_ref(system, data, sid, ids, device)
                refs[sid][name] = {k: encoded[k] for k in ('code', 'baseline')}
        identities = {sid: r['full'] for sid, r in refs.items()}
        donor_sids = sorted(data['dev_sids'])
        if len(donor_sids) != 3:
            raise ValueError('Expected three independent development identities')
        reference_rows = []
        for sid, r in refs.items():
            other = [s for s in sorted(refs) if s != sid]
            reference_rows.append(dict(sid=sid, role='validation' if sid in donor_sids else 'train',
                clips=data['refs'][sid]['clip_id'], same_code_rms=float((r['A']['code']-r['B']['code']).square().mean().sqrt()),
                same_bias_rms=float((r['A']['baseline']-r['B']['baseline']).square().mean().sqrt()),
                other_code_rms_mean=float(torch.stack([(r['A']['code']-refs[s]['B']['code']).square().mean().sqrt() for s in other]).mean()),
                other_bias_rms_mean=float(torch.stack([(r['A']['baseline']-refs[s]['B']['baseline']).square().mean().sqrt() for s in other]).mean())))

        # Retain original full-validation padded RNG stream before batch slicing.
        noises = {draw: torch.randn(len(q['valid']), q['valid'].shape[1], 52,
                   generator=torch.Generator().manual_seed(draw)) for draw in (42, 123, 2026)}
        cache = []; replay = []
        update('native_replay', cells=len(selected), original_batches=len(groups))
        for gi, indices in enumerate(groups):
            ix = torch.tensor(indices)
            b = subset(q, ix, device); base = {'b0': b['b0'], 'h0': b['h0']}
            width = b['valid'].shape[1]
            for name, src in [('motion', 'target'), ('valid', 'valid'), ('times', 'times'), ('channel_mask', 'channel_mask')]:
                want = saved[src][ix] if name == 'channel_mask' else saved[src][ix, :width]
                torch.testing.assert_close(b[name].cpu(), want, rtol=0, atol=0)
            torch.testing.assert_close(base['b0'].cpu(), saved['b0'][ix, :width], rtol=0, atol=0)
            ident = batch_identity(identities, b); affect = audio_affect(audio, b['audio_features'], b['valid'])
            original = None
            for draw, noise in noises.items():
                pred = system.generate(b['content'], b['valid'], ident, affect,
                    initial_noise=noise[ix, :width].to(device), steps=12, base=base)['motion']
                want = saved['predictions'][f'{draw}/full'][ix, :width].to(device)
                error = float((pred-want).abs().max())
                torch.testing.assert_close(pred, want, rtol=0, atol=0)
                replay.append(dict(batch=gi, draw=draw, max_abs=error, bit_exact=True))
                if draw == 42: original = pred
            cache.append((ix, b, base, ident, affect, original))
            if gi % 10 == 0: update('native_replay', batches_done=gi+1, total=len(groups))
        (out/'replay.json').write_text(json.dumps(replay, indent=2))
        update('interventions_after_exact_replay', replay_entries=len(replay), cells=len(selected))

        # Scores are descriptive native-frame region statistics; no small-set F1.
        def region_values(value, target, valid, channels, indices):
            indices = [i for i in indices if channels[i]]
            x, y = value[valid][:,indices].double(), target[valid][:,indices].double()
            xc, yc = x-x.mean(0), y-y.mean(0)
            adj = valid[1:] & valid[:-1]
            dx, dy = value.diff(dim=0)[adj][:,indices].double(), target.diff(dim=0)[adj][:,indices].double()
            denom = xc.square().sum().sqrt()*yc.square().sum().sqrt()
            corr = float((xc*yc).sum()/denom) if denom > 1e-12 else float('nan')
            return [float((x-y).square().mean()), float((x.mean(0)-y.mean(0)).square().mean()),
                float((xc-yc).square().mean()), float(xc.square().mean().sqrt()), float(yc.square().mean().sqrt()),
                corr, float((dx-dy).square().mean()),
                float((torch.quantile(x,.9,dim=0)-torch.quantile(x,.1,dim=0)).mean()),
                float((torch.quantile(y,.9,dim=0)-torch.quantile(y,.1,dim=0)).mean())]

        rows = []; exports = {}
        for gi, (ix, b, base, ident, affect, original) in enumerate(cache):
            width = b['valid'].shape[1]; noise = noises[42][ix, :width].to(device)
            fixed = [b['content'], b['valid'], b['times'], base['b0'], base['h0'],
                     affect['global'], affect['intensity_value'], affect['u_a'],
                     ident['code'], ident['baseline'], noise]
            fixed_copies = [value.clone() for value in fixed]
            ref_variants = {}
            for mode in IDENTITY_MODES:
                pieces = []
                for sid_tensor in b['speaker_id']:
                    sid = int(sid_tensor); position = donor_sids.index(sid)
                    other_1 = identities[donor_sids[(position+1)%3]]
                    other_2 = identities[donor_sids[(position+2)%3]]
                    pieces.append(identity_variant(identities[sid], refs[sid]['A'], refs[sid]['B'], other_1, other_2, mode))
                ref_variants['id_'+mode] = {k: torch.cat([p[k] for p in pieces]) for k in ('code', 'baseline')}
            variants = [('original', ident, affect, original)]
            for mode in TEMPORAL_MODES:
                temporal = protocol.temporal_variant(affect['u_a'], b['valid'], mode, b['clip_id'])
                variants.append(('ua_'+mode, ident, protocol.replace_temporal(affect, temporal), None))
            variants.extend((mode, changed, affect, None) for mode, changed in ref_variants.items())
            for mode, changed_id, changed_affect, pred in variants:
                if pred is None:
                    pred = system.generate(b['content'], b['valid'], changed_id, changed_affect,
                        initial_noise=noise, steps=12, base=base)['motion']
                if mode == 'id_zero_bias':
                    torch.testing.assert_close(pred, torch.where(b['valid'][...,None],
                        original-ident['baseline'][:,None], 0.), rtol=0, atol=2e-6)
                if mode == 'id_bias_other_1':
                    torch.testing.assert_close(pred, torch.where(b['valid'][...,None],
                        original-ident['baseline'][:,None]+changed_id['baseline'][:,None], 0.), rtol=0, atol=2e-6)
                for j, i in enumerate(ix.tolist()):
                    if i not in selected_set: continue
                    valid, channel = b['valid'][j].cpu(), b['channel_mask'][j].cpu()
                    value, reference, target = pred[j].cpu(), original[j].cpu(), b['motion'][j].cpu()
                    for policy in ('raw', 'clip_all'):
                        x = value if policy == 'raw' else value.clamp(0,1)
                        y = reference if policy == 'raw' else reference.clamp(0,1)
                        # Other person's output has no paired target in this audit.
                        score_allowed = not any(s in mode for s in ('other', 'zero_code', 'zero_bias'))
                        row = dict(index=i, clip_id=b['clip_id'][j], speaker=int(b['speaker_id'][j]),
                            emotion=int(b['emotion_id'][j]), intensity=int(b['intensity_id'][j]), mode=mode, policy=policy,
                            response={k:response_values(x,y,valid,channel,indices) for k,indices in REGIONS.items()},
                            target_score=None if not score_allowed else {k:region_values(x,target,valid,channel,indices) for k,indices in REGIONS.items()})
                        if mode == 'original':
                            base_value = base['b0'][j].cpu()
                            if policy == 'clip_all': base_value = base_value.clamp(0,1)
                            row['b0_score'] = {k:region_values(base_value,target,valid,channel,indices) for k,indices in REGIONS.items()}
                        rows.append(row)
                    # Fixed first validation speaker, each emotion/level, draw42.
                    if int(b['speaker_id'][j]) == donor_sids[0]:
                        n = int(valid.nonzero()[-1])+1; cid=b['clip_id'][j]
                        exports[cid+'/'+mode] = value[:n].numpy()
                        if mode == 'original':
                            for name, val in [('target', target),('b0',base['b0'][j].cpu())]: exports[cid+'/'+name] = val[:n].numpy()
                            exports[cid+'/valid'] = valid[:n].numpy()
                            exports[cid+'/times'] = b['times'][j,:n].cpu().numpy()
                if not all(torch.equal(value, reference) for value, reference in zip(fixed, fixed_copies)):
                    raise ValueError('Intervention mutated a fixed condition or noise')
            if gi % 10 == 0: update('interventions_after_exact_replay', batches_done=gi+1,total=len(cache))
        if initial_system != model_digest(system) or initial_audio != model_digest(audio):
            raise ValueError('Model state changed')
        if any(p.grad is not None for m in (system,audio) for p in m.parameters()):
            raise ValueError('Unexpected gradients')
        for name in ('audio/final.pt', 'audio/curves.pt'):
            if sha(cp/name) != spec['files'][name]: raise ValueError('Input file changed')
        report = dict(schema='frozen_factor_small_diagnostic_v1', model=a.model,
            binding_sha256=sha(a.binding), training_performed=False, optimizer_created=False,
            test_loaded=False, model_state_unchanged=True, replay_bit_exact=True,
            selected_indices=selected, selected_clip_ids=[q['clip_id'][i] for i in selected],
            intervention_draw=42, replay_draws=[42,123,2026], region_columns=REGION_COLUMNS,
            response_columns=RESPONSE_COLUMNS, regions=REGIONS, reference_rows=reference_rows, rows=rows,
            scope='One deterministic validation clip per observed speaker/emotion/level cell. Historical mixed-input diagnosis, not 772D or full performance. Other identity has no paired GT score. u_a OOD interventions establish sensitivity only.')
        (out/'report.json').write_text(json.dumps(finite_json(report),indent=2,allow_nan=False))
        np.savez_compressed(out/'fixed_exports.npz',**exports)
        update('complete', report_sha256=sha(out/'report.json'),exports_sha256=sha(out/'fixed_exports.npz'),
               replay_sha256=sha(out/'replay.json'),cells=len(selected))
    except Exception as exc:
        update('failed', error_type=type(exc).__name__, error=str(exc)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--model',choices=('neutral','native'),required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    run(p.parse_args())
