"""Matched flow-only continuation; immutable teacher, semantics and decoder.

Both arms start from the SAME short-pilot checkpoint, use the same TRAIN
examples, noise seeds, optimizer and number of updates. Only base covariance
differs. The unhelpful personal response is explicitly ablated in BOTH arms;
the original pilot checkpoint remains intact for the response-on comparison.
"""
from dataclasses import replace
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_v5.factor_rebuild import V5FactorRebuild
from scripts.pilot_v5_factor_rebuild import raw_audio, evaluate
from scripts.train_v7_pilot import load, batch, sha
from scripts.train_expression_response import configure, load_runtime, write, save


def frozen_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('flow.'):
            continue
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(a):
    configure(a.seed); device = torch.device(a.device); started = time.time()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('fresh output required')
    out.mkdir(parents=True)
    if sha(a.checkpoint) != a.sha256:
        raise ValueError('common starting checkpoint hash mismatch')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    original = V5FactorRebuild.from_checkpoint(ck).to(device).eval().requires_grad_(False)
    # Measured response field had no paired reconstruction benefit. Ablate it
    # consistently, without inventing that this solves reference-code leakage.
    with torch.no_grad():
        original.style_response.basis[-1].weight.zero_()
        original.style_response.basis[-1].bias.zero_()
    binding = json.loads(Path(a.binding).read_text())
    runtime, base = load_runtime(binding, torch.device('cpu'))
    mean, std = runtime['feature_stats']['mean'].cpu(), runtime['feature_stats']['std'].cpu()
    del runtime, base
    fit = raw_audio(load(Path(a.data) / 'fit.npz'), mean, std)
    rng = np.random.default_rng(a.seed)
    order = [rng.choice(len(fit['motion']), min(a.batch, len(fit['motion'])), replace=False).tolist()
             for _ in range(a.steps)]
    # Cache only true TRAIN teacher/prior trajectories to keep this ablation
    # fast and identical between arms. No held-out trajectory is fitted.
    cached = []
    with torch.no_grad():
        for ids in order:
            b, _ = batch(fit, ids, device)
            p = original.affect_audio(b['audio_features'], b['valid'])
            t = original.visual_teacher(b)
            c = original.core.condition(p, b['valid'])
            cached.append(tuple(x.detach() for x in (t['state'], p['state'], p['hidden'],
                               c['emotion'], c['intensity'], b['valid'], b['times'])))
    expected = frozen_digest(original)
    save(out / 'frozen_start.pt', original.checkpoint())
    models = {'frozen_start': original}
    history = []
    arms = [('white', 1, 1.), ('correlated', a.kernel, 1.)]
    if a.include_mixed:
        arms.append(('mixed', a.kernel, .5))
    for name, kernel, correlation in arms:
        configure(a.seed)
        model = copy.deepcopy(original)
        model.rebuild_config = replace(model.rebuild_config, flow_noise_kernel=kernel,
                                       flow_noise_correlation=correlation)
        model.flow.noise_kernel = kernel
        model.flow.noise_correlation = correlation
        model.flow.requires_grad_(True).train()
        params = list(model.flow.parameters())
        opt = torch.optim.AdamW(params, lr=a.lr)
        for step, args in enumerate(cached):
            opt.zero_grad(set_to_none=True)
            loss = model.flow.loss(*args[:6], times=args[6])
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite flow objective')
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.); opt.step()
            row = dict(arm=name, step=step+1, loss=float(loss), elapsed_seconds=time.time()-started)
            history.append(row)
            if (step+1) % 32 == 0:
                write(out / 'state.json', row); print(json.dumps(row), flush=True)
        model.eval().requires_grad_(False)
        if frozen_digest(model) != expected:
            raise RuntimeError('non-flow state changed during single-variable ablation')
        save(out / (name + '.pt'), model.checkpoint()); models[name] = model
    del cached
    # Held diagnostic datasets/probes are loaded only after all training.
    results = {}
    for name, model in models.items():
        dest = out / name; dest.mkdir()
        results[name] = {}
        for role in ('held_sentence', 'unseen_speaker'):
            data = raw_audio(load(Path(a.data) / (role + '.npz')), mean, std)
            results[name][role] = evaluate(model, data, device, binding, a.seed, dest, role)
    report = dict(schema='v5_flow_covariance_ablation_v1', elapsed_seconds=time.time()-started,
        base_checkpoint=a.checkpoint, base_sha256=a.sha256, seed=a.seed, steps_per_arm=a.steps,
        lr=a.lr, batch=a.batch, kernels={'white':1, 'correlated':a.kernel},
        arm_config={name:dict(kernel=k, correlation=rho) for name,k,rho in arms},
        training_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        data_sha256={role:sha(Path(a.data)/(role+'.npz')) for role in ('fit','held_sentence','unseen_speaker')},
        common_frozen_digest=expected, frozen_routes_verified=True, personal_response_enabled=False,
        results=results, paper_ready=False, independent_probes_used_in_training=False,
        sealed_test_loaded=False, limits=['short-crop pilot, finite seeds',
            'Gaussian covariance is a known design choice, not a standalone novelty',
            'style-code invariance and stochastic naturalness remain unproven',
            'three unseen persons are insufficient for population conclusions'])
    write(out / 'history.json', history); write(out / 'report.json', report)
    print(json.dumps({'completed':str(out), 'elapsed_seconds':report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('checkpoint','sha256','binding','data','output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--seed', type=int, default=47)
    p.add_argument('--steps', type=int, default=256); p.add_argument('--batch', type=int, default=8)
    p.add_argument('--kernel', type=int, default=9); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--include-mixed', action='store_true')
    train(p.parse_args())
