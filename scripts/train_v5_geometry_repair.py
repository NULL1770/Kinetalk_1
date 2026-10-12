"""Matched TRAIN fine-tuning for the demonstrated v4 observer-support bug.

Both arms start from the same immutable v4 checkpoint, data order and budget.
The repaired arm cleans asset-level numerical noise and refits TRAIN scales.
Architecture and losses otherwise remain unchanged, separating this engineering
fix from subsequent research changes. No held or validation targets are fitted.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v4.unified_model_v4 import KineTalkV4Config, KineTalkV4Model
from kinetalk_v4.geometry_support import clean_geometry_jacobian, RIG_NUMERICAL_TOLERANCE_MM
from kinetalk_v4.rollout_statistics import expression_mean_loss, pooled_state_loss
from kinetalk_v2.unified_model import anchor_matrix
from scripts.train_expression_response import configure, load_runtime, development_fold, state_digest, write, save, sha
from scripts.train_unified_v4 import (select_balanced, reference_ids, build_base_cache,
    attach_base, make_refs, fit_geometry_scale, motion_targets, objective, evaluate, pair_batches)


def run(a):
    configure(a.seed); device = torch.device(a.device)
    root = Path(a.output)
    if root.exists(): raise FileExistsError('Fresh output required')
    root.mkdir(parents=True)
    binding = json.loads(Path(a.binding).read_text())
    data, base = load_runtime(binding, device); frozen = state_digest(base.stage1)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False); s = ck['model']
    cfg = KineTalkV4Config(**ck['config'])
    model = KineTalkV4Model(cfg, s['feature_mean'], s['feature_std'], s['scales'],
        s['decoder.geometry_jacobian'], s['decoder.geometry_offset'], s['decoder.geometry_scale']).to(device)
    model.load_state_dict(s)
    q = data['splits']['train']; fold = development_fold(q, data['fit_sids'])
    fit = select_balanced(q, fold['train'], a.limit, a.seed)
    held = select_balanced(q, fold['speaker_dev'] + fold['sentence_dev'], a.eval_limit, a.seed)
    pool = {}
    for i in fold['train']: pool.setdefault(int(q['speaker_id'][i]), []).append(i)
    rng = random.Random(a.seed)
    refs = {i: reference_ids(q, [i], pool, rng)[0] for i in fit}
    print(json.dumps({'event': 'cache_start', 'fit_clips': len(fit)}), flush=True)
    cache = build_base_cache(q, base, sorted(set(fit + held + list(refs.values()))), device)
    old_scale = model.decoder.geometry_scale.detach().cpu().tolist()
    if a.repair:
        cleaned = clean_geometry_jacobian(model.decoder.geometry_jacobian)
        scale = fit_geometry_scale(q, fit, cache, cleaned, model.decoder.geometry_offset, device)
        with torch.no_grad():
            model.decoder.geometry_jacobian.copy_(cleaned)
            model.decoder.geometry_scale.copy_(scale)
        # Preserve the pretrained physical geometry target on changing units.
        with torch.no_grad():
            ratio = torch.tensor(old_scale, device=device) / scale.to(device)
            model.decoder.geometry_delta.weight.mul_(ratio[:, None])
            model.decoder.geometry_delta.bias.mul_(ratio)
    channels = ck['protocol']['geometry_channels']
    protocol = dict(schema='v5_matched_geometry_repair_v1', seed=a.seed, epochs=a.epochs,
        fit_clips=len(fit), internal_dev_clips=len(held), source_checkpoint_sha256=sha(a.checkpoint),
        binding_sha256=sha(a.binding), source_sha256=sha(__file__), repair=a.repair,
        rig_numerical_tolerance_mm=RIG_NUMERICAL_TOLERANCE_MM if a.repair else None,
        old_geometry_scale=old_scale, geometry_scale_train_fitted=model.decoder.geometry_scale.cpu().tolist(),
        geometry_channels=channels, text_enabled=False, hdtf_enabled=False, test_loaded=False,
        flags=vars(a), rollout_mean_weight=a.rollout_mean_weight,
        unchanged='event likelihood/execution; no pointwise sampled-GT timing fit')
    write(root / 'protocol.json', protocol)
    write(root / 'selection.json', {'fit_ids': fit, 'held_ids': held, 'references': refs})
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    step = 0; history = []; begin = time.monotonic()
    order_rng = random.Random(a.seed)
    for epoch in range(a.epochs):
        model.train(); sums = {}; batches = pair_batches(q, fit, a.batch_size, order_rng)
        for ids in batches:
            b = attach_base(q.batch(torch.tensor(ids), device), ids, cache)
            r = make_refs(q, [refs[i] for i in ids], device)
            target = motion_targets(b)
            optimizer.zero_grad(set_to_none=True)
            loss, outputs, logs = objective(model, b, r, target, warmup=False, channels=channels)
            if a.rollout_mean_weight:
                # Schedule sampling remains detached from the new mean objective.
                # Shape/state/style gradients follow the deployment decoder path.
                rollout = model.decode(b['h0'], b['b0'], b['valid'], outputs['conditioned'],
                    outputs['style'], outputs['sampled_schedule']['events'].detach())['motion']
                pose = expression_mean_loss(rollout, b['motion'], b['valid'], b['channel_mask'], model.scales)
                measured = (rollout - b['b0']) @ anchor_matrix().to(rollout)
                sustained = pooled_state_loss(measured, target['state'], b['valid'], target['quality'])
                loss = loss + a.rollout_mean_weight * (pose + sustained)
                logs.update(rollout_mean=float(pose), rollout_state_mean=float(sustained))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite objective')
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); step += 1
            for k, v in logs.items(): sums[k] = sums.get(k, 0.) + v
            if step % 20 == 1:
                state = dict(status='training', epoch=epoch + 1, step=step,
                    total_updates=len(batches) * a.epochs, loss=float(loss), grad_norm=float(grad))
                write(root / 'state.json', state); print(json.dumps(state), flush=True)
        dev = evaluate(model, q, held, cache, data, device, a.seed, channels)
        if state_digest(base.stage1) != frozen: raise RuntimeError('Frozen B0 changed')
        history.append(dict(epoch=epoch + 1, step=step, train={k: v / len(batches) for k, v in sums.items()}, internal_dev=dev))
        payload = dict(config=asdict(cfg), model=model.state_dict(), protocol=protocol, history=history,
                       epoch=epoch + 1, step=step, optimizer=optimizer.state_dict(),
                       python_rng=order_rng.getstate(), torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        save(root / 'resume.pt', payload); write(root / 'history.json', history)
    save(root / 'final.pt', {k: v for k, v in payload.items() if k != 'optimizer'})
    write(root / 'complete.json', dict(status='complete', updates=step,
        elapsed_seconds=time.monotonic() - begin, neutral_frozen_sha256=frozen,
        final_internal_dev=history[-1]['internal_dev'], paper_result=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--binding', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--repair', action='store_true')
    p.add_argument('--seed', type=int, default=47); p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--limit', type=int, default=1024); p.add_argument('--eval-limit', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--device', default='cuda')
    p.add_argument('--rollout-mean-weight', type=float, default=0.)
    run(p.parse_args())
