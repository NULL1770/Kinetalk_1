"""One TRAIN-only analytic fit of existing mean heads, with a frozen receiver.

Consumes the immutable diagnostic fit. No new parameters or reconstruction loss;
the original covariance and semantic heads are preserved, so sampling improvement
is not claimed. g/u/gu are finite matched analytic-fit ablations, not SGD epochs.
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.expression_response import ExpressionResponse, ResponseConfig, masked_pool
from scripts.diagnose_expression_targets import frame_features, ClipRidge
from scripts.train_expression_response import (configure, load_runtime, cache_base,
    development_fold, reference_batch, state_digest, sha, write, save)


@torch.no_grad()
def replace_means(model, fitted, mode):
    if mode not in ('g', 'u', 'gu') or not model.cfg.center_local:
        raise ValueError('Expected centered model and g/u/gu candidate')
    cfg = model.cfg
    shapes = {'global': (cfg.hidden+1, cfg.global_dim), 'hidden': (cfg.hidden, cfg.local_dim)}
    weights = {}
    for key, shape in shapes.items():
        value = torch.as_tensor(fitted[key], device=model.scales.device, dtype=model.scales.dtype)
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError('Invalid fit matrix: '+key)
        weights[key] = value
    old = {k:v.clone() for k,v in model.state_dict().items()}
    if 'g' in mode:
        model.prior.global_head.weight[:cfg.global_dim].copy_(weights['global'][:-1].T)
        model.prior.global_head.bias[:cfg.global_dim].copy_(weights['global'][-1])
    if 'u' in mode:
        model.prior.local_head.weight[:cfg.local_dim].copy_(weights['hidden'].T)
        # The bias cancels after native interpolation + valid-time centering.
    allowed = {}
    if 'g' in mode:
        allowed.update({'prior.global_head.weight': cfg.global_dim,
                        'prior.global_head.bias': cfg.global_dim})
    if 'u' in mode:
        allowed['prior.local_head.weight'] = cfg.local_dim
    for key, value in model.state_dict().items():
        if key in allowed:
            assert torch.equal(value[allowed[key]:], old[key][allowed[key]:]), key
        else:
            assert torch.equal(value, old[key]), key
    return {'passed':True, 'changed_mean_slices':allowed,
            'all_other_parameters_and_variance_rows_exact':True,
            'local_bias_preserved':True,
            'fitted_coefficients':sum(old[k][:n].numel() for k,n in allowed.items())}


@torch.no_grad()
def calibrate(a):
    configure(47)
    out = Path(a.output)
    if (out/'final.pt').exists():
        raise FileExistsError('Refuse to overwrite fitted candidate')
    out.mkdir(parents=True, exist_ok=True)
    binding = json.loads(Path(a.binding).read_text())
    root = Path(__file__).resolve().parents[1]
    for name, digest in binding['source_files'].items():
        assert sha(root/name) == digest, name
    audit = Path(a.audit)
    report = json.loads((audit/'report.json').read_text())
    state = json.loads((audit/'state.json').read_text())
    parent = binding['parent_checkpoint']
    assert sha(parent['path']) == parent['sha256'] == report['checkpoint_sha256']
    assert sha(audit/'report.json') == state['report_sha256'] == binding['audit']['report_sha256']
    assert sha(audit/'mean_head_fit.npz') == state['fit_sha256'] == binding['audit']['fit_sha256']
    assert sha(audit/'fold.json') == binding['audit']['fold_sha256']
    assert report['data_manifest_sha256'] == binding['data_manifest_sha256']
    assert state['status'] == 'complete' and not report['smoke'] and report['fit_clips'] == 10903
    assert report['frozen_state_exact'] and report['existing_mean_head_mapping_verified']
    assert not report['test_loaded'] and not report['external_validation_used']
    for role in ('speaker_dev','sentence_dev'):
        m=report['results'][role]['normalized_latent_mse']
        for key in a.mode:
            assert m['hidden_'+key] < m['student_'+key], (role,key)
    with np.load(audit/'mean_head_fit.npz', allow_pickle=False) as z:
        fitted={k:z[k].copy() for k in ('global','hidden')}
    device=torch.device(a.device)
    ck=torch.load(parent['path'],map_location=device,weights_only=False)
    cfg=ResponseConfig(**ck['config'])
    model=ExpressionResponse(cfg,ck['model']['feature_mean'],ck['model']['feature_std'],ck['model']['scales']).to(device)
    model.load_state_dict(ck['model'],strict=True)
    model.eval().requires_grad_(False)
    original=copy.deepcopy(model)
    checks=replace_means(model,fitted,a.mode)
    data,base=load_runtime(binding,device)
    fold=development_fold(data['splits']['train'],data['fit_sids'])
    assert fold == json.loads((audit/'fold.json').read_text())
    # Include longest real TRAIN clips and ordinary shorter clips, with their
    # original valid gaps. No external validation is needed for this gate.
    q=data['splits']['train']
    ids=list(dict.fromkeys(sorted(fold['train'], key=lambda i:int(q['_lengths'][i]))[-2:]+fold['train'][:2]))
    cache_base(data,base,device,{'train':ids,'validation':[]})
    base_digest=state_digest(base.stage1)
    b=q.batch(torch.tensor(ids),device)
    refs=reference_batch(data,b,device)
    prior=model.audio_prior(b['audio_features'],b['valid'])
    old=original.audio_prior(b['audio_features'],b['valid'])
    for key in ('g_logvar','u_logvar','u_mask'):
        assert torch.equal(prior[key],old[key]),key
    h=model.prior.encoder((b['audio_features'][...,768:]-model.feature_mean)/model.feature_std,b['valid'])
    g,u=model.conditions(prior,b['valid'])
    if 'g' in a.mode:
        expected=ClipRidge.predict(masked_pool(h,b['valid']),torch.as_tensor(fitted['global'],device=device),True)
        torch.testing.assert_close(g,expected,rtol=1e-4,atol=1e-5)
    else:
        assert torch.equal(prior['g_mean'],old['g_mean'])
    if 'u' in a.mode:
        expected=frame_features(model,b,h)['hidden'] @ torch.as_tensor(fitted['hidden'],device=device)
        torch.testing.assert_close(u,expected,rtol=1e-4,atol=1e-5)
    else:
        assert torch.equal(prior['u_mean'],old['u_mean'])
    prediction=model.predict(b['audio_features'],b['b0'],b['valid'],refs)
    poisoned=b['audio_features'].clone();poisoned[...,:768]=float('nan')
    assert torch.equal(prediction,model.predict(poisoned,b['b0'],b['valid'],refs))
    assert torch.isfinite(prediction).all()
    assert state_digest(base.stage1)==base_digest==ck['neutral_digest']
    checks.update(real_gpu=a.device.startswith('cuda'),native_mean_mapping=True,
        prior_variances_exact=True,content_nan_isolation=True,neutral_digest_exact=True,
        checked_train_indices=ids,test_loaded=False)
    parent_protocol=json.loads((Path(parent['path']).parent/'protocol.json').read_text())
    protocol={'schema':'phase45_analytic_mean_fit_v1','config':ck['config'],
        'args':{'seed':47,'epochs':0,'matching':'analytic_ridge_'+a.mode,
                'reference_training':ck.get('reference_training','single')},
        'data':parent_protocol['data'],'parent_epochs':ck['epoch'],'parent_updates':ck['step'],
        'analytic_fit_passes':1,'analytic_fit_clips':len(fold['train']),
        'ridge':report['ridge'],'fitted_coefficients':checks['fitted_coefficients'],
        'scope':'Existing mean slices fitted from TRAIN; one shared closed-form solve; not SGD-budget-matched',
        'source_checkpoint_sha256':parent['sha256'],'fit_sha256':state['fit_sha256'],
        'report_sha256':state['report_sha256'],'fold_sha256':sha(audit/'fold.json'),
        'binding_sha256':sha(a.binding),'test_loaded':False,'default_replaced':False}
    write(out/'protocol.json',protocol)
    final={'model':model.cpu().state_dict(),'config':ck['config'],'epoch':0,'step':0,'history':[],
           'neutral_digest':ck['neutral_digest'],'binding_sha256':sha(a.binding),
           'reference_training':ck.get('reference_training','single'),
           'analytic_fit':protocol,'test_loaded':False}
    save(out/'final.pt',final)
    restored=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v,restored['model'][k]) for k,v in model.state_dict().items())
    checks['saved_state_exact']=True
    write(out/'runtime_checks.json',checks)
    write(out/'complete.json',{'status':'complete','updates':0,'epochs':0,'analytic_fit_passes':1,
          'analytic_fit_clips':len(fold['train']),'final_sha256':sha(out/'final.pt'),'test_loaded':False})
    print(json.dumps({'mode':a.mode,'checks':checks,'output':str(out)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True)
    p.add_argument('--audit',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=('g','u','gu'),required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args()
    try:calibrate(a)
    except Exception as exc:
        write(Path(a.output)/'failure.json',{'type':type(exc).__name__,'message':str(exc)});raise
