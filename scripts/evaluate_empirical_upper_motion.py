"""Fixed-budget empirical upper-motion distribution feasibility experiment.

TRAIN donors preserve real joint motion shapes at 25 fps. This is a retrieval
baseline, not a new learned audio-timing method. All draws, including failures
to find a long enough donor (deterministic fallback), enter evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kinetalk_b0.models.empirical_upper_motion import fit_empirical_bank, sample_empirical_motion
from kinetalk_b0.models.reference_intensity_decoder import ReferenceIntensityDecoder, ReferenceIntensityStudent
from kinetalk_b0.models.slow_state_affect import compose_upper_face
from scripts.reference_decoder_data import load_reference_context
from scripts.train_reference_intensity_decoder import prepare_cache, predict, SOURCE_SHA, UPPER
from scripts.train_audio_regional_envelope import _paired_cluster_ci
from scripts.joint_motion_metrics import score_clip, summarize
from scripts.train_formal_predictable_projection import save_json, canonical_hash
from scripts.extract_emotion2vec_pilot import sha
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES

SEEDS = (42, 123, 2026, 47, 53, 79, 101, 211)
TEMPERATURES = (0., .5, 1.)


def selection_prediction(q, checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (ck.get('schema') != 'reference_intensity_decoder_v1'
            or canonical_hash(ck['protocol']) != ck['protocol_sha256']
            or ck['protocol']['source_sha256'] != SOURCE_SHA):
        raise ValueError('Reference selection checkpoint provenance differs')
    decoder = ReferenceIntensityDecoder(**ck['decoder_config']).to(device).eval()
    student = ReferenceIntensityStudent(**ck['student_config']).to(device).eval()
    decoder.load_state_dict(ck['decoder']); student.load_state_dict(ck['student'])
    cache = prepare_cache(q, ck['stats'])
    ids = torch.tensor(ck['protocol']['calibration_ids'], dtype=torch.long)
    with torch.no_grad():
        center, _ = predict(decoder, student, cache, ids, device)
    return ck, ids, center


def make_bank(q, ids):
    return fit_empirical_bank(q['motion'][ids][..., UPPER], q['valid'][ids],
        q['global_code'][ids], q['identity_code'][ids],
        [q['speaker'][i] for i in ids.tolist()],
        [q['sentence_id'][i] for i in ids.tolist()],
        [q['clip_id'][i] for i in ids.tolist()])


def fair_variogram(samples, target, valid, scales):
    """Fair finite-ensemble VS for lags 1/4/16/32, no pairs across gaps."""
    x, y = samples.double() / scales, target.double() / scales
    errors, empirical = [], []
    for lag in (1, 4, 16, 32):
        if len(valid) <= lag:
            continue
        pair = valid.unfold(0, lag + 1, 1).all(-1)
        if not pair.any():
            continue
        xd = (x[:, lag:] - x[:, :-lag]).abs().sqrt()[:, pair]
        yd = (y[lag:] - y[:-lag]).abs().sqrt()[pair]
        error = (xd.mean(0) - yd).square()
        empirical.append(error.flatten())
        errors.append((error - xd.var(0, unbiased=True) / len(samples)).flatten())
    return (float(torch.cat(errors).mean()), float(torch.cat(empirical).mean())) if errors else (None, None)


def evaluate(q, ids, center, bank, scales, temperature, mode='conditional', keep=()):
    rows, per, metadata, stored = [], [], {}, {}
    for row, i in enumerate(ids.tolist()):
        n = int(q['valid'][i].nonzero(as_tuple=True)[0][-1]) + 1
        valid = q['valid'][i, :n]; mu = center[row, :n]
        try:
            generated, donors = sample_empirical_motion(bank, mu[None], valid[None],
                q['global_code'][i:i+1], q['identity_code'][i:i+1],
                [q['speaker'][i]], [q['sentence_id'][i]], list(SEEDS),
                temperature, top_k=32, mode=mode)
            samples = generated[:, 0]
            fallback = None
        except ValueError as error:
            # Only a documented lack of a long-enough, disjoint donor is an
            # allowed fallback. API/shape/finite errors must stop this job.
            if not any(word in str(error).lower() for word in ('eligible', 'candidate', 'donor')):
                raise
            samples = mu[None].expand(len(SEEDS), -1, -1).clone()
            donors, fallback = [], str(error)
        target = q['motion'][i, :n, UPPER]
        score = score_clip(samples.numpy(), target.numpy(), valid.numpy(), scales.numpy())
        rows.append(score)
        fair, empirical = fair_variogram(samples, target, valid, scales)
        observed = samples[:, valid]
        delta = observed - target[valid]
        average = observed.mean(1)
        d = samples[:, 1:] - samples[:, :-1]
        pair = valid[1:] & valid[:-1]
        a = d[:, 1:] - d[:, :-1]; triple = valid[2:] & valid[1:-1] & valid[:-2]
        record = {'clip_id':q['clip_id'][i], 'speaker':q['speaker'][i], 'sentence':q['sentence_id'][i],
            'raw_es':score['joint_fair_es']['raw'], 'centered_es':score['joint_fair_es']['centered'],
            'fair_variogram':fair, 'empirical_variogram':empirical,
            'raw_mse':float(delta.square().mean()),
            'mean_shift_from_center':float((average - mu[valid].mean(0)).abs().mean()),
            'brow_std':float(observed[..., :5].std(1, unbiased=False).mean()),
            'eye_std':float(observed[..., 5:].std(1, unbiased=False).mean()),
            'acceleration_energy':float(a[:, triple].square().mean()) if triple.any() else None,
            'velocity_energy':float(d[:, pair].square().mean()) if pair.any() else None,
            'out_of_bounds':float(((observed < 0) | (observed > 1)).float().mean()),
            'fallback':fallback}
        per.append(record)
        metadata[q['clip_id'][i]] = {'donors':donors, 'fallback':fallback}
        if q['clip_id'][i] in keep:
            stored[q['clip_id'][i]] = samples
    aggregate = summarize(rows)
    for key in ('raw_es','centered_es','fair_variogram','empirical_variogram','raw_mse',
                'mean_shift_from_center','brow_std','eye_std','acceleration_energy','velocity_energy','out_of_bounds'):
        values = [r[key] for r in per if r[key] is not None]
        aggregate[key] = float(np.mean(values)) if values else None
    aggregate.update(fallback_clips=sum(r['fallback'] is not None for r in per), per_clip=per)
    return aggregate, stored, metadata


def comparisons(reports, main):
    ref = reports[main]['per_clip']
    speakers, sentences = [r['speaker'] for r in ref], [r['sentence'] for r in ref]
    return {name:{metric:_paired_cluster_ci([r[metric] for r in ref],
            [r[metric] for r in value['per_clip']], speakers, sentences)
        for metric in ('raw_es', 'centered_es', 'fair_variogram')}
        for name, value in reports.items() if name != main}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data','source','reference-run','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--selection-only', action='store_true')
    args = p.parse_args(); torch.set_num_threads(4); started = time.monotonic()
    if args.output.exists(): raise FileExistsError('Fresh output required')
    args.output.mkdir(parents=True)
    save_json(args.output/'status.json', {'status':'loading_train','test_loaded':False})
    data, system, audio, identities = load_reference_context(args.data, args.source, args.device,
        role='train', expected_source_sha256=SOURCE_SHA)
    q = data['splits']['train']
    ck, ids, center = selection_prediction(q, args.reference_run/'selection.pt', args.device)
    if ck['protocol']['data']['manifest_sha256'] != data['provenance']['manifest_sha256']:
        raise ValueError('Reference selection data manifest differs')
    fit = torch.tensor(ck['protocol']['fit_ids'], dtype=torch.long)
    bank = make_bank(q, fit); reports = {}
    torch.save({'bank':bank,'center':center,'ids':ids,'valid':q['valid'][ids]},args.output/'selection_replay.pt')
    protocol = {'schema':'empirical_upper_motion_feasibility_v1','source_sha256':SOURCE_SHA,
        'reference_selection_sha256':sha(args.reference_run/'selection.pt'),
        'reference_final_sha256':sha(args.reference_run/'final.pt'), 'data':data['provenance'],
        'temperatures':list(TEMPERATURES), 'seeds':list(SEEDS),'top_k':32,'window':5,
        'selection_rule':'lowest centered fair ES among temperatures with raw ES <= zero*1.005 and fair VS < zero; otherwise zero',
        'donor_exclusion':'same speaker OR same sentence excluded; native full-run crop only; missing donor => deterministic fallback',
        'condition':'global frozen audio64 + .25 standardized neutral identity128 squared distance; no local timing or query target',
        'selection_reuse':'TRAIN cal255 previously used to select deterministic model; exploratory, not independent confirmation',
        'mean_scope':'bounded raw residual can shift mean; report explicitly; no per-query GT scale or offset fitting',
        'residual_scope':'heuristic addition of observed centered motion to a dynamic deterministic center; not statistically orthogonal prediction residuals; double-counting remains possible',
        'bank_smoothing':bank['smoothing'], 'selection_replay_sha256':sha(args.output/'selection_replay.pt'),
        'innovation_claim':False,'audio_timing_success_claim':False,'test_loaded':False,'default_replaced':False,
        'source_files':{s:sha(ROOT/s) for s in ('scripts/evaluate_empirical_upper_motion.py','kinetalk_b0/models/empirical_upper_motion.py')}}
    save_json(args.output/'protocol.json', protocol)
    for temperature in TEMPERATURES:
        name = str(temperature)
        reports[name], _, donors = evaluate(q, ids, center, bank, ck['stats']['scales9'], temperature)
        save_json(args.output/('selection_donors_'+name+'.json'),donors)
        save_json(args.output/'selection_evaluation.json',reports)
        print(json.dumps({'stage':'temperature','temperature':temperature,
            'raw_es':reports[name]['raw_es'],'centered_es':reports[name]['centered_es'],
            'fair_variogram':reports[name]['fair_variogram'],'fallback_clips':reports[name]['fallback_clips']}),flush=True)
    base = reports['0.0']; choices = [0.]
    for t in TEMPERATURES[1:]:
        r = reports[str(t)]
        if r['raw_es'] <= base['raw_es']*1.005 and r['fair_variogram'] < base['fair_variogram']:
            choices.append(t)
    selected = min(choices,key=lambda t:reports[str(t)]['centered_es'])
    reports['unconditional'], _, donors = evaluate(q, ids, center, bank, ck['stats']['scales9'], selected, 'unconditional')
    save_json(args.output/'selection_donors_unconditional.json',donors)
    save_json(args.output/'selection_evaluation.json',reports)
    save_json(args.output/'selection.json',{'temperature':selected,'selection_only':args.selection_only,
        'paired_comparisons':comparisons(reports,str(selected)), 'confirmatory':False})
    if args.selection_only or selected == 0.:
        save_json(args.output/'status.json', {'status':'complete','selected_temperature':selected,
            'development_loaded':False,'test_loaded':False,'elapsed_seconds':time.monotonic()-started})
        return
    # No error residual estimated from in-sample predictions: bank holds TRAIN
    # observed, run-centered motion shapes, not regression fitting residuals.
    bank = make_bank(q, torch.arange(len(q['valid'])))
    torch.save({'schema':'empirical_upper_motion_v1','bank':bank,'temperature':selected,
        'protocol':protocol,'protocol_sha256':canonical_hash(protocol)}, args.output/'bank.pt')
    del q,data,system,audio,identities,center,ck
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    data, system, audio, identities = load_reference_context(args.data,args.source,args.device,
        role='validation',expected_source_sha256=SOURCE_SHA)
    q=data['splits']['validation']; ids=torch.arange(len(q['valid']))
    curves=torch.load(args.reference_run/'curves.pt',map_location='cpu',weights_only=True)
    ck=torch.load(args.reference_run/'final.pt',map_location='cpu',weights_only=True)
    if (ck.get('schema') != 'reference_intensity_decoder_v1'
            or canonical_hash(ck['protocol']) != ck['protocol_sha256']
            or ck['protocol']['source_sha256'] != SOURCE_SHA
            or ck['protocol']['data']['manifest_sha256'] != data['provenance']['manifest_sha256']
            or curves['selected_checkpoint_sha256'] != sha(args.reference_run/'final.pt')
            or curves['source_sha256'] != SOURCE_SHA
            or curves['protocol_sha256'] != ck['protocol_sha256']):
        raise ValueError('Frozen development curve provenance differs')
    if curves['clip_id'] != q['clip_id'] or not torch.equal(curves['valid'],q['valid']):
        raise ValueError('Development order/mask differs from frozen reference curves')
    center=curves['predictions']['full'][...,UPPER]
    previews=json.loads((args.reference_run/'preview_manifest.json').read_text())
    keep=[r['clip_id'] for r in previews]
    reports={}; stored={}; metadata={}
    for name,temperature,mode in (('deterministic',0.,'conditional'),
                                ('empirical',selected,'conditional'),('unconditional',selected,'unconditional')):
        reports[name],stored[name],metadata[name]=evaluate(q,ids,center,bank,ck['stats']['scales9'],temperature,mode,keep)
    final={'modes':reports,'paired_comparisons':comparisons(reports,'empirical'),
        'test_loaded':False,'medtalk_parity_established':False,'audio_timing_established':False,
        'sample_policy':'all eight draws scored, first predeclared seed42 rendered, no best-of-K',
        'scope':'one additional development evaluation after TRAIN-only temperature selection; development historically reused'}
    protected={}
    for clip in keep:
        i=q['clip_id'].index(clip); n=len(stored['empirical'][clip][0]); valid=q['valid'][i:i+1,:n]
        prior=curves['predictions']['original_prior'][i:i+1,:n]
        full=curves['predictions']['full'][i,:n]
        generated={name:compose_upper_face(prior,stored[name][clip][0:1],valid)[0]
                   for name in ('empirical','unconditional')}
        not_upper=[j for j in range(52) if j not in UPPER]
        protected[clip]={name:torch.equal(v[:,not_upper].contiguous().view(torch.int32),
                                      prior[0,:,not_upper].contiguous().view(torch.int32)) for name,v in generated.items()}
        np.savez_compressed(args.output/(clip+'_comparison.npz'), channels=np.array(ARKIT_NAMES),
            mode_names=np.array(['gt','smooth_prior','deterministic','empirical','unconditional']),
            motions=torch.stack([q['motion'][i,:n],curves['predictions']['smooth_prior'][i,:n],full,
                                 generated['empirical'],generated['unconditional']]).numpy(),
            times=q['times'][i,:n].numpy(),valid=valid[0].numpy(),channel_mask=q['channel_mask'][i].numpy(),
            clip_id=clip,noise_seed=SEEDS[0])
    final['preview_nonupper43_bit_exact']=protected
    final.update(bank_sha256=sha(args.output/'bank.pt'),reference_curves_sha256=sha(args.reference_run/'curves.pt'))
    save_json(args.output/'evaluation.json',final);save_json(args.output/'all_development_donors.json',metadata)
    save_json(args.output/'status.json',{'status':'complete','selected_temperature':selected,
        'test_loaded':False,'development_loaded':True,'default_replaced':False,
        'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'status':'complete','temperature':selected}),flush=True)


if __name__ == '__main__': main()
