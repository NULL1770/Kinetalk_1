"""Adapt only the response decoder to frozen deployment expression conditions.

Both arms aggregate the same two independent neutral references. No gradient
from motion reconstruction reaches the prior, posterior, style, or B0.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import motion_objective
from scripts.refine_expression_prior import restore_model
from scripts.train_expression_response import (configure, load_runtime, cache_base,
    development_fold, audit_data, reference_batch, quick_evaluate, state_digest, write, save, sha)


def freeze_conditions(model):
    model.eval().requires_grad_(False)
    model.zero_grad(set_to_none=True)
    model.decoder.requires_grad_(True)
    return list(model.decoder.parameters())


def frozen_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith('decoder.'):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def objective(model, batch, refs, mode, generator=None):
    if mode not in ('mixed', 'deploy'):
        raise ValueError('Unknown receiver conditioning mode')
    if any(value.shape[1] != 2 for value in refs.values()):
        raise ValueError('Exactly two independent deployment supports required')
    with torch.no_grad():
        style = model.encode_style(refs)['code']
        prior = model.audio_prior(batch['audio_features'], batch['valid'])
        if mode == 'mixed':
            posterior = model.motion_posterior(batch['motion'], batch['b0'], style,
                batch['valid'], batch['channel_mask'], batch['times'])
    def reconstruction(distribution, sample):
        pred = model.decode(batch['b0'], distribution, style, batch['valid'],
                            sample=sample, generator=generator)
        return motion_objective(pred, batch['motion'].detach(), batch['valid'],
                                batch['channel_mask'], batch['times'], model.scales)
    p_loss, p_parts = reconstruction(prior, False)
    logs = {'p_reconstruction': p_loss, 'p_position': p_parts['position'],
            'p_velocity': p_parts['velocity']}
    if mode == 'mixed':
        q_loss, q_parts = reconstruction(posterior, True)
        loss = q_loss + .5 * p_loss
        logs.update(q_reconstruction=q_loss, q_position=q_parts['position'], q_velocity=q_parts['velocity'])
    else:
        loss = 1.5 * p_loss
    return loss, dict(logs, loss=loss)


def inherited_budget(parent):
    analytic = parent.get('analytic_fit', {})
    return {'parent_epochs': parent.get('parent_epochs', analytic.get('parent_epochs', 0)) + parent['epoch'],
            'parent_updates': parent.get('parent_updates', analytic.get('parent_updates', 0)) + parent['step'],
            'inherited_analytic_fit_passes': parent.get('inherited_analytic_fit_passes', analytic.get('analytic_fit_passes', 0)),
            'inherited_analytic_fit_clips': parent.get('inherited_analytic_fit_clips', analytic.get('analytic_fit_clips', 0))}


def scientific_args(args):
    return dict({k: v for k, v in vars(args).items() if k != 'resume'},
                reference_training='two_aggregate', matching='receiver_' + args.mode)


def train(args):
    configure(args.seed)
    device = torch.device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        raise FileExistsError('Completed run must not be restarted')
    if not args.resume and (out/'protocol.json').exists():
        raise FileExistsError('Existing run requires explicit --resume')
    binding = json.loads(Path(args.binding).read_text(encoding='utf8'))
    if binding['mode'] != args.mode:
        raise ValueError('Mode disagrees with source binding')
    for name, digest in binding['source_files'].items():
        if sha(Path(__file__).resolve().parents[1]/name) != digest:
            raise ValueError('Changed source: ' + name)
    data, base = load_runtime(binding, device)
    fold = development_fold(data['splits']['train'], data['fit_sids'])
    audit = audit_data(data, fold)
    ids = fold['train']
    if args.smoke:
        ids = [next(i for i in ids if int(data['splits']['train']['emotion_id'][i]) == e) for e in range(8)]
        cache_base(data, base, device, {'train': ids, 'validation': []})
    else:
        cache_base(data, base, device)
    model, parent = restore_model(binding, device)
    params = freeze_conditions(model)
    frozen, neutral = frozen_digest(model), state_digest(base.stage1)
    if neutral != parent['neutral_digest']:
        raise ValueError('Parent neutral B0 mismatch')
    budget = inherited_budget(parent)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    order_rng = torch.Generator().manual_seed(args.seed + 30000)
    sample_rng = torch.Generator(device=device).manual_seed(args.seed + 40000)
    protocol = dict(schema='phase46_frozen_conditions_receiver_v1', config=model.checkpoint_config(),
        args=scientific_args(args), data=audit, binding_sha256=sha(args.binding),
        parent_checkpoint_sha256=binding['parent_checkpoint']['sha256'],
        initial_model_digest=state_digest(model), frozen_digest=frozen, neutral_digest=neutral,
        trainable_parameters=sum(p.numel() for p in params), trainable_module='decoder',
        reconstruction_passes_per_update=2 if args.mode == 'mixed' else 1,
        loss='original position + .5 adjacent displacement; mixed=q_sample+.5p_mean; deploy=1.5p_mean',
        test_loaded=False, default_replaced=False, **budget)
    epoch0, step, history = 0, 0, []
    if args.resume:
        if json.loads((out/'protocol.json').read_text(encoding='utf8')) != protocol:
            raise ValueError('Scientific protocol changed on resume')
        ck = torch.load(out/'last.pt', map_location=device, weights_only=False)
        if ck['binding_sha256'] != sha(args.binding):
            raise ValueError('Resume binding changed')
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        order_rng.set_state(ck['order_rng'].cpu())
        sample_rng.set_state(ck['sample_rng'].cpu())
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
        epoch0, step, history = ck['epoch'], ck['step'], ck['history']
        if frozen_digest(model) != frozen:
            raise ValueError('Frozen conditions changed on resume')
    else:
        write(out/'protocol.json', protocol)
        write(out/'data_audit.json', audit)
        write(out/'fold.json', fold)
    durations, motion_losses = [], []
    began = time.time()
    for epoch in range(epoch0, 1 if args.smoke else args.epochs):
        if shutil.disk_usage(out).free < 250 * 2**20:
            raise RuntimeError('Checkpoint safety disk floor reached')
        model.eval()
        model.decoder.train()
        tick, rows = time.time(), []
        order = torch.tensor(ids)[torch.randperm(len(ids), generator=order_rng)]
        batches = [order] * args.smoke_steps if args.smoke else list(order.split(args.batch_size))
        for index, sub in enumerate(batches):
            batch = data['splits']['train'].batch(sub, device)
            refs = reference_batch(data, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, logs = objective(model, batch, refs, args.mode, sample_rng)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite receiver objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            if any(p.grad is not None for n, p in model.named_parameters() if not n.startswith('decoder.')):
                raise RuntimeError('Reconstruction gradient escaped decoder')
            optimizer.step()
            step += 1
            row = {k: float(v.detach()) for k, v in logs.items()}
            row['grad_norm'] = float(norm)
            rows.append(row)
            motion_losses.append(row['p_reconstruction'])
            if index % 100 == 0 or (args.smoke and index % 20 == 0):
                entry = dict(event='update', epoch=epoch+1, step=step, batch=index+1,
                             seconds_per_update=(time.time()-tick)/(index+1), **row)
                print(json.dumps(entry), flush=True)
                write(out/'state.json', dict(status='training', **entry))
        duration = time.time() - tick
        durations.append(duration)
        entry = dict(epoch=epoch+1, updates=step, seconds=duration,
                     train={k: float(np.mean([row[k] for row in rows])) for k in rows[0]})
        if not args.smoke:
            for role in ('speaker_dev', 'sentence_dev'):
                chosen = fold[role]
                positions = np.linspace(0, len(chosen)-1, min(256, len(chosen)), dtype=int)
                entry[role] = quick_evaluate(model, data, [chosen[i] for i in positions], device)
        history.append(entry)
        assert frozen_digest(model) == frozen and state_digest(base.stage1) == neutral
        assert all(p.grad is None for p in base.stage1.parameters())
        ck = dict(model=model.state_dict(), config=model.checkpoint_config(), reference_training='two_aggregate',
            epoch=epoch+1, step=step, history=history, optimizer=optimizer.state_dict(),
            order_rng=order_rng.get_state(), sample_rng=sample_rng.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            neutral_digest=neutral, binding_sha256=sha(args.binding), frozen_digest=frozen,
            parent_checkpoint_sha256=binding['parent_checkpoint']['sha256'], test_loaded=False, **budget)
        save(out/'last.pt', ck)
        write(out/'history.json', history)
        eta = max(0, args.epochs-epoch-1) * float(np.mean(durations[-3:]))
        write(out/'state.json', dict(status='training', epoch=epoch+1, updates=step, epoch_seconds=duration,
            remaining_training_seconds=eta, last_checkpoint_sha256=sha(out/'last.pt'), test_loaded=False))
        print(json.dumps(dict(event='epoch_complete', **entry, remaining_training_seconds=eta)), flush=True)
    model.eval()
    final = {k: v for k, v in ck.items() if k not in ('optimizer','order_rng','sample_rng','torch_rng','cuda_rng')}
    save(out/'final.pt', final)
    if args.smoke:
        first, last = float(np.mean(motion_losses[:10])), float(np.mean(motion_losses[-10:]))
        passed = len(motion_losses) == args.smoke_steps and last < .9 * first
        write(out/'smoke.json', dict(passed=passed, first10=first, last10=last, ratio=last/first,
            updates=step, frozen_conditions_exact=True, seconds=time.time()-began, test_loaded=False))
        if not passed:
            raise RuntimeError('Small-data receiver learnability gate failed')
    write(out/'complete.json', dict(status='complete', updates=step, epochs=ck['epoch'],
        final_sha256=sha(out/'final.pt'), test_loaded=False, **budget))
    write(out/'state.json', dict(status='training_complete', updates=step, epochs=ck['epoch'], test_loaded=False))
    return model, data, base


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--binding', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--mode', choices=('mixed','deploy'), required=True)
    p.add_argument('--seed', type=int, default=47); p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=16); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--smoke-steps', type=int, default=120)
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    try:
        train(a)
    except Exception as exc:
        write(Path(a.output)/'failure.json', {'type':type(exc).__name__,'message':str(exc),'time':time.time()})
        raise
