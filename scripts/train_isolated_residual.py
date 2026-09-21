"""Train a zero-DC upper-face residual around a frozen audio state predictor.

All full-face baselines retain the frozen protected Stage4 output outside the
nine upper channels.  No deterministic/state/global/identity optimizer is
constructed.  The separate static arm has identical budgets and initialization.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_isolated_audio_state import (load_context, make_model, forward,
    old_prediction, trim, center)
from scripts.train_full_staged import subset, obs, batch_identity, region_report, NOT_UPPER
from scripts.train_formal_predictable_projection import (save_json, save_checkpoint,
    capture_rng, restore_rng, canonical_hash)
from scripts.train_predictable_renderer import state_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.compact_native_curves import compact_curves
from scripts.mouth_protection import protection_report
from scripts.paper_generation_report import report_generation
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face
from kinetalk_b0.models.audio_residual_flow import fair_trajectory_es
from kinetalk_b0.models.dc_protected_temporal_flow import DCProtectedTemporalFlow, project_temporal_dc


SEEDS = (42, 123, 2026)
MODES = ('full', 'base', 'deterministic', 'static', 'reverse')


def validate_state_checkpoint(saved, data, source_sha256):
    protocol = saved.get('protocol', {})
    if (saved.get('protocol_sha256') != canonical_hash(protocol)
            or protocol.get('schema') != 'isolated_mean_state_v1'
            or protocol.get('source_sha256') != source_sha256
            or protocol.get('data', {}).get('manifest_sha256') != data['provenance']['manifest_sha256']
            or protocol.get('data', {}).get('index_sha256') != data['provenance']['index_sha256']
            or protocol.get('test_loaded') is not False
            or protocol.get('mode') != 'audio'):
        raise ValueError('Frozen audio state checkpoint provenance/source binding mismatch')
    if not {'model', 'config', 'dynamic_scales'} <= saved.keys():
        raise ValueError('Incomplete frozen state checkpoint')


def condition_inputs(b, state_model, local_adapter, mode):
    """GT-free state inference; adapter gradients stop at the frozen predictor."""
    if mode not in ('audio', 'static', 'reverse'):
        raise ValueError('Unknown audio condition intervention')
    with torch.no_grad():
        out = {k: v.detach() for k, v in forward(state_model, b, mode).items()}
    valid = b['valid']; mask = valid[..., None]
    local = torch.where(mask, local_adapter(out['local']), 0.)
    h0 = torch.where(mask, b['h0'].detach(), 0.)
    if mode == 'static':
        h0 = torch.where(mask, h0.sum(1, keepdim=True) / valid.sum(1)[:, None, None], 0.)
    elif mode == 'reverse':
        h0 = h0.clone()
        for index in range(len(h0)):
            native = torch.nonzero(valid[index], as_tuple=False).flatten()
            h0[index, native] = h0[index, native.flip(0)].clone()
    q = {'valid': valid, 'h0': h0}
    if 'channel_mask' in b: q['channel_mask'] = b['channel_mask']
    identity = {'code': b['identity_code'].detach()}
    affect = {'global': b['global_code'].detach(), 'intensity_value': b['intensity_value'].detach()}
    return q, identity, affect, local, out


def fixed_residual_target(b, upper, scales):
    """Training-only fixed target. Every observed upper channel is required."""
    cc = list(UPPER_INDICES)
    if (b['channel_mask'].dtype != torch.bool or b['channel_mask'].shape != (len(upper), 52)
            or not b['channel_mask'][:, cc].all()):
        raise ValueError('Residual fitting requires all nine observed upper channels')
    return project_temporal_dc((b['motion'][..., cc].detach() - upper.detach()) / scales[cc], b['valid']).detach()


def compose_upper_residual(upper, residual, scales, valid):
    return torch.where(valid[..., None], upper.detach() + scales[list(UPPER_INDICES)] * residual, 0.)


def objective(flow, adapter, state_model, b, scales, mode, noise, flow_time,
              second_noise=None, decode_steps=12):
    q, identity, affect, local, out = condition_inputs(b, state_model, adapter, mode)
    target = fixed_residual_target(b, out['upper'], scales)
    loss = flow.flow_loss(target, q, identity, affect, local, out['state'], noise, flow_time)
    values = {'flow': loss.detach()}
    if second_noise is not None:
        residuals = [flow.decode(q, identity, affect, local, out['state'], z, decode_steps)
                     for z in (noise, second_noise)]
        raw = torch.stack([compose_upper_residual(out['upper'], r, scales, b['valid']) for r in residuals])
        normalized = raw / scales[list(UPPER_INDICES)]
        target_raw = torch.where(b['valid'][..., None], b['motion'][..., list(UPPER_INDICES)].detach(), 0.)
        target_normalized = target_raw / scales[list(UPPER_INDICES)]
        raw_es = fair_trajectory_es(normalized, target_normalized, b['valid'])
        centered_es = fair_trajectory_es(normalized, target_normalized, b['valid'], centered=True)
        observed = raw[:, b['valid']]
        domain = (F.relu(-observed) + F.relu(observed - 1.)).mean()
        loss = loss + .1 * raw_es + .1 * centered_es + .1 * domain
        values.update(raw_fair_es=raw_es.detach(), centered_fair_es=centered_es.detach(), domain=domain.detach())
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite residual objective')
    return loss, values


@torch.no_grad()
def evaluate(flow, adapter, state_model, data, system, audio, identities, device,
             mode='audio', decode_steps=12):
    flow.eval(); adapter.eval(); state_model.eval()
    q = data['splits']['validation']; valid = q['valid']; predictions = {}
    n = len(valid); max_dc = {}; nonupper = {}; teacher_accuracy = {}; mouth = {}
    for seed in SEEDS:
        # Common fullface noise is fixed across conditions and independent arms;
        # upper9 uses those same frame/channel draws, not another RNG stream.
        noise = torch.randn(q['motion'].shape, generator=torch.Generator().manual_seed(seed))
        chunks = {key: [] for key in MODES}
        for ids in torch.arange(n).split(16):
            b = subset(q, ids, device); b['frozen_local'] = audio(b['audio_features'], b['valid'])['local']
            base = old_prediction(system, b, identities, noise[ids].to(device))
            chunks['base'].append(base.cpu())
            for key, intervention in (('full', mode), ('static', 'static'), ('reverse', 'reverse')):
                cond, identity, affect, local, out = condition_inputs(b, state_model, adapter, intervention)
                residual = flow.decode(cond, identity, affect, local, out['state'],
                    noise[ids][..., list(UPPER_INDICES)].to(device), decode_steps)
                upper = compose_upper_residual(out['upper'], residual, data['target_scales'].to(device), b['valid'])
                pred = compose_upper_face(base, upper, b['valid'])
                chunks[key].append(pred.cpu())
                mask = b['valid'][..., None]
                dc = torch.where(mask, upper - out['upper'], 0.).double().sum(1) / mask.sum(1)
                max_dc[key] = max(max_dc.get(key, 0.), float(dc.abs().max()))
                if key == 'full': chunks['deterministic'].append(compose_upper_face(base, out['upper'], b['valid']).cpu())
        for key, values in chunks.items(): predictions[f'{seed}/{key}'] = torch.cat(values)
    for key, pred in predictions.items():
        ref = predictions[key.split('/')[0] + '/base']
        nonupper[key] = bool(torch.equal(pred[..., NOT_UPPER], ref[..., NOT_UPPER]))
        mouth[key] = protection_report(pred, ref, q['motion'], valid, q['channel_mask'], q['emotion_id'])
        correct = 0
        for ids in torch.arange(n).split(16):
            b = subset(q, ids, device); identity = batch_identity(identities, b)
            enc = system.encode_motion(torch.where(obs(b), pred[ids].to(device) - b['b0'] - identity['baseline'][:, None], 0.), b['valid'])
            correct += int((enc['emotion_logits'].argmax(-1) == b['emotion_id']).sum())
        teacher_accuracy[key] = correct / n
    report = {'schema': 'isolated_dc_residual_evaluation_v1', 'clips': n, 'mode': mode,
        'test_loaded': False, 'scope': 'complete development partition',
        'regions': {k: region_report(v, q) for k, v in predictions.items()},
        'mouth_protection': mouth, 'nonupper43_exact': nonupper,
        'max_absolute_upper_mean_drift': max_dc,
        'generated_teacher_accuracy_nonindependent': teacher_accuracy,
        'input_audio_accuracy': float((q['emotion_logits'].argmax(-1) == q['emotion_id']).float().mean()),
        'independent_emotion_AV_visual_pending': True}
    curves = {'clip_id': q['clip_id'], 'target': q['motion'], 'valid': valid, 'times': q['times'],
        'channel_mask': q['channel_mask'], 'b0': q['b0'], 'predictions': predictions, 'noise_seeds': list(SEEDS)}
    return report, curves


def acceptance_checks(report, benchmark):
    keys = {f'{seed}/{mode}' for seed in SEEDS for mode in MODES}
    for field in ('mouth_protection', 'nonupper43_exact', 'generated_teacher_accuracy_nonindependent'):
        if set(report.get(field, {})) != keys: raise ValueError('Incomplete evaluation coverage: ' + field)
    for mode in MODES:
        if benchmark.get(mode, {}).get('prediction_keys') != [f'{seed}/{mode}' for seed in SEEDS]:
            raise ValueError('Benchmark seed coverage mismatch')
    emotions = {mode: float(np.mean([report['generated_teacher_accuracy_nonindependent'][f'{seed}/{mode}'] for seed in SEEDS]))
                for mode in MODES}
    mbe = {mode: benchmark[mode]['coefficient']['arkit_mbe']['value'] for mode in MODES}
    if not all(np.isfinite(v) for v in (*emotions.values(), *mbe.values())):
        raise ValueError('Nonfinite acceptance metric')
    drift = report['max_absolute_upper_mean_drift']
    if set(drift) != {'full', 'static', 'reverse'} or not all(np.isfinite(v) for v in drift.values()):
        raise ValueError('Incomplete/nonfinite DC verification')
    return {'mouth_preserved': all(v.get('passed') is True for v in report['mouth_protection'].values()),
        'nonupper43_exact': all(v is True for v in report['nonupper43_exact'].values()),
        'upper_mean_preserved': max(drift.values()) <= 1e-5,
        'mbe_no_worse_than_stage4': mbe['full'] <= mbe['base'],
        'teacher_drop_at_most_3pp': emotions['full'] >= max(emotions['base'], emotions['deterministic']) - .03}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'source', 'state-checkpoint', 'output'): p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--epochs', type=int, default=40); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--mode', choices=('audio', 'static'), default='audio')
    p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, default=47)
    p.add_argument('--decode-steps', type=int, default=12)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.decode_steps) < 1: raise ValueError('Positive budgets required')
    if a.output.exists() and not a.resume: raise FileExistsError('Fresh residual output required')
    a.output.mkdir(parents=True, exist_ok=True); started = time.monotonic(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    data, system, audio, identities, ds = load_context(a.data, a.source, a.device, a.seed)
    state_saved = torch.load(a.state_checkpoint, map_location='cpu', weights_only=False)
    source_sha = sha(a.source); validate_state_checkpoint(state_saved, data, source_sha)
    if not a.smoke and state_saved['protocol'].get('smoke') is not False:
        raise ValueError('Full training cannot consume a smoke state predictor')
    if not torch.equal(state_saved['dynamic_scales'].cpu(), ds.cpu()): raise ValueError('TRAIN dynamic scales changed')
    state_model = make_model(data, ds).to(a.device)
    if state_model.export_config() != state_saved['config']: raise ValueError('Frozen state model config differs')
    state_model.load_state_dict(state_saved['model'], strict=True); state_model.requires_grad_(False).eval()
    if a.smoke:
        for role, split in data['splits'].items(): data['splits'][role] = subset(split, torch.arange(min(16, len(split['valid']))), 'cpu')
    torch.manual_seed(a.seed)
    flow = DCProtectedTemporalFlow(data['config']).to(a.device)
    adapter = nn.Linear(state_model.hidden, flow.emotion_dim).to(a.device)
    parameters = list(flow.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-5)
    frozen = {k: state_hash(m.state_dict()) for k, m in [('system', system), ('audio', audio), ('state', state_model)]}
    root = Path(__file__).resolve().parents[1]
    sources = ['scripts/train_isolated_residual.py', 'scripts/train_isolated_audio_state.py',
        'kinetalk_b0/models/isolated_audio_state.py', 'kinetalk_b0/models/dc_protected_temporal_flow.py',
        'kinetalk_b0/models/temporal_audio_residual_flow.py', 'kinetalk_b0/models/audio_residual_flow.py',
        'kinetalk_b0/models/slow_state_affect.py', 'scripts/train_full_staged.py',
        'scripts/prepare_paper_full_data.py', 'scripts/mouth_protection.py',
        'scripts/paper_generation_report.py', 'scripts/compact_native_curves.py', 'scripts/joint_motion_metrics.py']
    protocol = {'schema': 'isolated_dc_residual_training_v1', 'source_sha256': source_sha,
        'state_checkpoint_sha256': sha(a.state_checkpoint), 'state_protocol_sha256': state_saved['protocol_sha256'],
        'data': data['provenance'], 'epochs': 1 if a.smoke else a.epochs, 'mode': a.mode,
        'seed': a.seed, 'batch_size': a.batch_size, 'decode_steps': a.decode_steps, 'smoke': a.smoke,
        'flow_config': flow.architecture_config, 'flow_model_config': data['config']['model'],
        'adapter_config': {'in_features': state_model.hidden, 'out_features': flow.emotion_dim},
        'objective': 'fixed DC residual FM; every fourth batch .1 raw fairES + .1 centered fairES + .1 two-draw domain',
        'freeze_hashes': frozen, 'sources': {s: sha(root / s) for s in sources},
        'test_loaded': False, 'default_replaced': False, 'selection': 'fixed final epoch',
        'train_clips': len(data['splits']['train']['valid']),
        'validation_clip_ids': data['splits']['validation']['clip_id'],
        'noise_seeds': list(SEEDS), 'evaluation_modes': list(MODES)}
    if a.resume:
        if json.loads((a.output / 'protocol.json').read_text()) != protocol: raise ValueError('Resume protocol mismatch')
    else:
        save_json(a.output / 'protocol.json', protocol)
        for source in sources:
            dest = a.output / 'source' / source; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((root / source).read_bytes())
    gen = torch.Generator().manual_seed(a.seed); first = 0; elapsed_before = 0.
    if a.resume:
        checkpoint = torch.load(a.output / 'last.pt', map_location='cpu', weights_only=False)
        if checkpoint['protocol_sha256'] != canonical_hash(protocol): raise ValueError('Resume checkpoint protocol differs')
        flow.load_state_dict(checkpoint['flow']); adapter.load_state_dict(checkpoint['adapter'])
        optimizer.load_state_dict(checkpoint['optimizer']); first = checkpoint['epoch']
        elapsed_before = checkpoint['elapsed_seconds']; restore_rng(checkpoint['rng'], gen)
    scales = data['target_scales'].to(a.device); train = data['splits']['train']
    for epoch in range(first, protocol['epochs']):
        tick = time.monotonic(); sums = {}; flow.train(); adapter.train()
        for batch_index, ids in enumerate(torch.randperm(len(train['valid']), generator=gen).split(a.batch_size)):
            b = trim(train, ids, a.device)
            shape = (*b['valid'].shape, 9)
            noise = torch.randn(shape, generator=gen).to(a.device)
            flow_time = torch.rand(len(ids), generator=gen).to(a.device)
            second = torch.randn(shape, generator=gen).to(a.device) if batch_index % 4 == 0 else None
            loss, values = objective(flow, adapter, state_model, b, scales, a.mode, noise, flow_time, second, a.decode_steps)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True); optimizer.step()
            for key, value in {'total': loss.detach(), **values}.items(): sums.setdefault(key, []).append(float(value))
            if batch_index % 25 == 0:
                print(json.dumps({'event': 'batch', 'epoch': epoch + 1, 'batch': batch_index + 1,
                    'loss': float(loss.detach()), 'grad_norm': float(norm)}), flush=True)
        record = {'status': 'training', 'epoch': epoch + 1, 'epochs': protocol['epochs'], 'mode': a.mode,
            'clips': len(train['valid']), 'seconds': time.monotonic() - tick,
            'elapsed_seconds': elapsed_before + time.monotonic() - started,
            'losses': {k: float(np.mean(v)) for k, v in sums.items()}, 'test_loaded': False}
        save_json(a.output / f'epoch{epoch+1:03d}.json', record); save_json(a.output / 'status.json', record)
        print(json.dumps(record), flush=True)
        save_checkpoint(a.output / 'last.pt', {'flow': flow.state_dict(), 'adapter': adapter.state_dict(),
            'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'rng': capture_rng(gen),
            'elapsed_seconds': record['elapsed_seconds'], 'protocol_sha256': canonical_hash(protocol)})
    report, curves = evaluate(flow, adapter, state_model, data, system, audio, identities, a.device, a.mode, a.decode_steps)
    benchmark = report_generation(curves, data, a.output, 'residual',
        SimpleNamespace(condition_mode=a.mode, artifact_dir=a.output / 'scores'))
    checks = acceptance_checks(report, benchmark)
    report['mbe_vs_deterministic_diagnostic'] = {
        'full': benchmark['full']['coefficient']['arkit_mbe']['value'],
        'deterministic': benchmark['deterministic']['coefficient']['arkit_mbe']['value'],
        'reason': 'Point error and calibrated stochastic diversity can trade off; Stage4 coefficient protection remains mandatory'}
    for name, model in [('system', system), ('audio', audio), ('state', state_model)]:
        if state_hash(model.state_dict()) != frozen[name]: raise RuntimeError('Frozen model changed: ' + name)
        if any(p.grad is not None for p in model.parameters()): raise RuntimeError('Frozen model received gradients: ' + name)
    checks['frozen_models_unchanged'] = True
    report.update(checks=checks, protection_passed=all(checks.values()) and not a.smoke,
        quantitative_timing_passed=None, paired_independent_static_pending=True,
        passed=False, caution='Protection checks do not establish audio timing; matched static arm acceptance is external')
    save_json(a.output / 'evaluation.json', report)
    save_json(a.output / 'mouth_protection.json', report['mouth_protection'])
    manifest = json.loads((a.data / 'manifest.json').read_text())
    lengths = {r['clip_id']: r['frames'] for r in manifest['roles']['val']['query']}
    save_checkpoint(a.output / 'native_curves.pt', compact_curves(curves, lengths))
    save_checkpoint(a.output / 'final.pt', {'flow': flow.state_dict(), 'adapter': adapter.state_dict(),
        'protocol': protocol, 'protocol_sha256': canonical_hash(protocol), 'test_loaded': False})
    save_json(a.output / 'status.json', {'status': 'complete', 'protection_passed': report['protection_passed'],
        'quantitative_timing_passed': None, 'elapsed_seconds': elapsed_before + time.monotonic() - started,
        'test_loaded': False, 'final_sha256': sha(a.output / 'final.pt'), 'native_curves_sha256': sha(a.output / 'native_curves.pt')})


if __name__ == '__main__': main()
