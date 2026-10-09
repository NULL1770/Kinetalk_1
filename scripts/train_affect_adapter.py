"""Train a zero-start affect-only additive adapter on a frozen parent.

The parent prior, posterior, style encoder, B0 and legacy decoder remain
frozen.  The adapter sees only global/local affect latents and reference
response code; local content directions are reduced to one RMS envelope.  The
objective is the existing position plus adjacent-displacement reconstruction
objective, so this experiment isolates decoder capacity rather than adding a
new loss or a new target.
"""
from __future__ import annotations

import argparse, hashlib, json, shutil, sys, time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import (
    ExpressionResponse, ResponseConfig, motion_objective,
)
from scripts.train_expression_response import (
    configure, load_runtime, cache_base, development_fold, audit_data,
    reference_batch, quick_evaluate, state_digest, write, save, sha,
)


def restore_parent(binding, device):
    spec = binding['parent_checkpoint']
    if sha(spec['path']) != spec['sha256']:
        raise ValueError('Parent checkpoint hash mismatch')
    parent = torch.load(spec['path'], map_location=device, weights_only=False)
    old_cfg = ResponseConfig(**parent['config'])
    if old_cfg.response_head == 'affect_adapter':
        raise ValueError('Parent must be the retained pre-adapter model')
    cfg = replace(old_cfg, response_head='affect_adapter')
    model = ExpressionResponse(cfg, parent['model']['feature_mean'],
                               parent['model']['feature_std'],
                               parent['model']['scales']).to(device)
    missing, unexpected = model.load_state_dict(parent['model'], strict=False)
    if unexpected or any(not name.startswith('decoder.affect_adapter.') for name in missing):
        raise ValueError(f'Unexpected parent load differences: missing={missing}, unexpected={unexpected}')
    return model, parent


def freeze_parent(model):
    model.eval().requires_grad_(False)
    model.decoder.affect_adapter.requires_grad_(True)
    return list(model.decoder.affect_adapter.parameters())


def frozen_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith('decoder.affect_adapter.'):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def objective(model, batch, refs, generator=None):
    with torch.no_grad():
        style = model.encode_style(refs)['code']
        prior = model.audio_prior(batch['audio_features'], batch['valid'])
        posterior = model.motion_posterior(
            batch['motion'], batch['b0'], style, batch['valid'],
            batch['channel_mask'], batch['times'])

    def reconstruction(distribution, sample):
        prediction = model.decode(batch['b0'], distribution, style,
                                  batch['valid'], sample=sample,
                                  generator=generator)
        loss, parts = motion_objective(
            prediction, batch['motion'].detach(), batch['valid'],
            batch['channel_mask'], batch['times'], model.scales)
        return loss, parts

    q_loss, q_parts = reconstruction(posterior, True)
    p_loss, p_parts = reconstruction(prior, False)
    loss = q_loss + .5 * p_loss
    return loss, {
        'loss': loss,
        'q_reconstruction': q_loss,
        'q_position': q_parts['position'],
        'q_velocity': q_parts['velocity'],
        'p_reconstruction': p_loss,
        'p_position': p_parts['position'],
        'p_velocity': p_parts['velocity'],
    }


def train(args):
    configure(args.seed)
    device = torch.device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise FileExistsError('Completed run must not be restarted')
    binding = json.loads(Path(args.binding).read_text(encoding='utf8'))
    data, base = load_runtime(binding, device)
    fold = development_fold(data['splits']['train'], data['fit_sids'])
    audit = audit_data(data, fold)
    ids = fold['train']
    if args.smoke:
        ids = [next(i for i in ids if int(data['splits']['train']['emotion_id'][i]) == e)
               for e in range(8)]
        cache_base(data, base, device, {'train': ids, 'validation': []})
    else:
        cache_base(data, base, device)

    model, parent = restore_parent(binding, device)
    params = freeze_parent(model)
    frozen = frozen_digest(model)
    neutral = state_digest(base.stage1)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    order_rng = torch.Generator().manual_seed(args.seed + 30000)
    sample_rng = torch.Generator(device=device).manual_seed(args.seed + 40000)
    protocol = dict(
        schema='phase65_affect_adapter_v1',
        config=model.checkpoint_config(), args={k: v for k, v in vars(args).items()},
        data=audit, binding_sha256=sha(args.binding),
        parent_checkpoint_sha256=binding['parent_checkpoint']['sha256'],
        initial_model_digest=state_digest(model), frozen_digest=frozen,
        neutral_digest=neutral, trainable_parameters=sum(p.numel() for p in params),
        trainable_module='decoder.affect_adapter',
        inputs='global/local affect latents + reference response + RMS local envelope; no B0/content',
        loss='existing q position + .5 adjacent displacement + .5 prior reconstruction',
        test_loaded=False, default_replaced=False,
    )
    write(out / 'protocol.json', protocol)
    write(out / 'data_audit.json', audit)
    write(out / 'fold.json', fold)

    step, history, durations, losses = 0, [], [], []
    started = time.time()
    for epoch in range(1 if args.smoke else args.epochs):
        if shutil.disk_usage(out).free < 250 * 2**20:
            raise RuntimeError('Checkpoint safety disk floor reached')
        model.eval(); model.decoder.affect_adapter.train()
        tick, rows = time.time(), []
        order = torch.tensor(ids)[torch.randperm(len(ids), generator=order_rng)]
        batches = [order] * args.smoke_steps if args.smoke else list(order.split(args.batch_size))
        for index, sub in enumerate(batches):
            batch = data['splits']['train'].batch(sub, device)
            refs = reference_batch(data, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, logs = objective(model, batch, refs, sample_rng)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite affect-adapter objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            if any(p.grad is not None for n, p in model.named_parameters()
                   if not n.startswith('decoder.affect_adapter.')):
                raise RuntimeError('Gradient escaped affect adapter')
            optimizer.step(); step += 1
            row = {k: float(v.detach()) for k, v in logs.items()}
            row['grad_norm'] = float(norm); rows.append(row); losses.append(row['q_position'])
            if index % 100 == 0 or (args.smoke and index % 20 == 0):
                print(json.dumps(dict(event='update', epoch=epoch + 1, step=step,
                                      batch=index + 1,
                                      seconds_per_update=(time.time() - tick) / (index + 1),
                                      **row)), flush=True)
        duration = time.time() - tick; durations.append(duration)
        entry = dict(epoch=epoch + 1, updates=step, seconds=duration,
                     train={k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
        history.append(entry)
        assert frozen_digest(model) == frozen and state_digest(base.stage1) == neutral
        assert all(p.grad is None for n, p in model.named_parameters()
                   if not n.startswith('decoder.affect_adapter.'))
        ck = dict(model=model.state_dict(), config=model.checkpoint_config(),
                  epoch=epoch + 1, step=step, history=history,
                  optimizer=optimizer.state_dict(), order_rng=order_rng.get_state(),
                  sample_rng=sample_rng.get_state(), torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                  neutral_digest=neutral, frozen_digest=frozen,
                  binding_sha256=sha(args.binding),
                  parent_checkpoint_sha256=binding['parent_checkpoint']['sha256'],
                  test_loaded=False, default_replaced=False)
        save(out / 'last.pt', ck); write(out / 'history.json', history)
        print(json.dumps(dict(event='epoch_complete', **entry)), flush=True)

    model.eval()
    final = {k: v for k, v in ck.items()
             if k not in ('optimizer', 'order_rng', 'sample_rng', 'torch_rng', 'cuda_rng')}
    save(out / 'final.pt', final)
    if args.smoke:
        first, last = float(np.mean(losses[:10])), float(np.mean(losses[-10:]))
        passed = len(losses) == args.smoke_steps and last < .9 * first
        write(out / 'smoke.json', dict(passed=passed, first10=first, last10=last,
                                       ratio=last / first, updates=step,
                                       frozen_conditions_exact=True,
                                       test_loaded=False, seconds=time.time() - started))
        if not passed:
            raise RuntimeError('Affect-adapter learnability gate failed')
    write(out / 'complete.json', dict(status='complete', updates=step,
        epochs=ck['epoch'], final_sha256=sha(out / 'final.pt'),
        test_loaded=False, default_replaced=False))
    return model, data, base


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--binding', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, default=47)
    p.add_argument('--epochs', type=int, default=8); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=120)
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    train(a)
