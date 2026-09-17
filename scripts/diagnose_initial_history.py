"""Read-only one-time GT initialization diagnosis for the context12 receiver.

This is an upper-nine mechanism diagnostic, never a deployable GT-conditioned
result. All arms generate the original first16 slots, then oracle arms replace
only the known8 slots of the start16 call. Every subsequent prefix is generated.
No full target or target-derived mean is accepted by the decoder interface.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_context_mechanism as context
from scripts import train_prefix_upper as p
from scripts.audit_temporal_repair import read, sha, load_pt, paired_metrics

SCHEMA = 'one_time_initial_history_diagnostic_v1'
SEEDS = (42, 123, 2026)
INITIALIZATIONS = ('normal', 'gt_prefix16', 'gt_terminal_static')
CONDITIONS = ('full', 'local_static')
CONDITION_KEYS = ('valid', 'h0', 'audio_global', 'audio_intensity')
GROUPS = {'brows': list(range(5)), 'eyes_expression': list(range(5, 9))}
WINDOWS = {'suffix16': 16, 'late48': 48}


def historical_fit_indices(train, selection):
    """Verify the historical128 selection against current2315 metadata only."""
    ids = train['clip_id']
    rows = selection['clips']
    if (len(ids) != 2315 or len(set(ids)) != len(ids) or len(rows) != 128
            or selection['count'] != 128 or selection['selection_uses_motion'] is not False
            or len({row['clip_id'] for row in rows}) != 128):
        raise ValueError('Expected unique historical128 metadata inside current2315fit')
    lookup = {clip_id: i for i, clip_id in enumerate(ids)}
    selected = []
    for row in rows:
        index = lookup.get(row['clip_id'])
        if (index is None or index != row['original_fit_index']
                or int(train['speaker_id'][index]) != row['speaker_id']
                or int(train['emotion_id'][index]) != row['emotion_id']):
            raise ValueError('Historical fit clip/member metadata differs: ' + row['clip_id'])
        selected.append(index)
    return torch.tensor(selected, dtype=torch.long)


def eligibility(valid, channel_mask):
    """Same metadata/mask-only cohort for all arms; native gaps stay invalid."""
    if valid.dtype != torch.bool or valid.ndim != 2 or valid.shape[1] != 96:
        raise ValueError('Native96 Boolean validity required')
    if channel_mask.dtype != torch.bool or channel_mask.shape != (len(valid), 9):
        raise ValueError('Upper-nine Boolean channel mask required')
    reasons = {
        'missing_upper_channel': ~channel_mask.all(1),
        'native_index15_invalid': ~valid[:, 15],
        'no_observed_first_generated_chunk': ~valid[:, 16:32].any(1),
        'fewer_than_two_late_observations': valid[:, 48:].sum(1) < 2,
        'no_adjacent_late_pair': ~(valid[:, 49:] & valid[:, 48:-1]).any(1),
    }
    eligible = ~torch.stack(list(reasons.values())).any(0)
    return eligible, reasons


def local_condition(native, valid, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown local condition')
    clean = torch.where(valid[..., None], native, 0.)
    if condition == 'full':
        return clean
    mean = clean.sum(1, keepdim=True)/valid.sum(1)[:, None, None].clamp_min(1)
    return torch.where(valid[..., None], mean, 0.)


@torch.no_grad()
def decode_initial_history(upper, conditions, identity, native, noise, *,
                           initialization='normal', condition='full', gt_prefix=None, steps=12):
    """Decode on the exact context12 clock with at most one GT prefix injection.

    gt_prefix is normalized [B,16,9], not a full target. Only indices8:16 can
    reach a receiver call. Static initialization repeats index15 only where
    original past slots are valid; no gap is filled and no index is shifted.
    Output0:16 stays generated in all arms and is excluded from main scores.
    """
    if set(conditions) != set(CONDITION_KEYS) or set(identity) != {'code'}:
        raise ValueError('Only declared acoustic conditions and independent identity code accepted')
    if initialization not in INITIALIZATIONS or condition not in CONDITIONS:
        raise ValueError('Unknown initialization or local condition')
    valid = conditions['valid']
    if (valid.dtype != torch.bool or valid.ndim != 2 or valid.shape[1] != 96
            or noise.shape != (*valid.shape, 9) or native.shape[:2] != valid.shape
            or conditions['h0'].shape[:2] != valid.shape or steps != 12):
        raise ValueError('Fixed native96 shapes and twelve solver steps required')
    if not torch.isfinite(noise[valid]).all() or not torch.isfinite(native[valid]).all():
        raise ValueError('Finite observed noise and local features required')
    if (not valid.any(1).all() or not torch.isfinite(conditions['h0'][valid]).all()
            or not torch.isfinite(conditions['audio_global']).all()
            or not torch.isfinite(conditions['audio_intensity']).all()
            or not torch.isfinite(identity['code']).all()):
        raise ValueError('Finite declared acoustic/identity inputs and observed clips required')
    if initialization == 'normal':
        if gt_prefix is not None:
            raise ValueError('Normal deployment rejects GT prefix')
    else:
        if (gt_prefix is None or gt_prefix.shape != (len(valid), 16, 9)
                or not valid[:, 15].all() or not torch.isfinite(gt_prefix[:, 8:16][valid[:, 8:16]]).all()):
            raise ValueError('Oracle needs explicit16-frame prefix with valid fixed index15')
    local = local_condition(native, valid, condition)
    output = torch.zeros_like(noise)
    for start in range(0, 96, 16):
        stop = start + 16
        ids = valid[:, start:stop].any(1).nonzero(as_tuple=True)[0]
        if not len(ids):
            continue
        # Window construction always reads generated source. Oracle injection
        # changes only the already-masked known slots of one receiver call.
        args, known, mask, _ = p.window_batch(conditions, identity, local, output,
                                             ids, torch.full_like(ids, start))
        if start == 16 and initialization != 'normal':
            values = gt_prefix[ids, 8:16]
            if initialization == 'gt_terminal_static':
                values = gt_prefix[ids, 15:16].expand(-1, 8, -1)
            known[:, :8] = torch.where(mask[:, :8, None], values, 0.)
        initial = known.new_zeros(known.shape)
        # Preserve the source receiver's complete native noise draw, including
        # masked slots; decode_prefix owns validity masking of its ODE state.
        initial[:, 8:] = noise[ids, start:stop]
        result = upper.decode_prefix(*args, initial, known=known, known_mask=mask, steps=steps)
        if not torch.equal(result[mask], known[mask]):
            raise RuntimeError('Receiver changed the supplied known prefix')
        current = result[:, 8:]
        if not torch.isfinite(current[valid[ids, start:stop]]).all():
            raise RuntimeError('Nonfinite observed generated upper motion')
        output[ids, start:stop] = torch.where(valid[ids, start:stop, None], current, 0.)
    return output


def _displacement(pred, target, pairs):
    count = int(pairs.sum())
    if count == 0:
        return {'observed_channel_pairs': 0, 'prediction_rms': None, 'reference_rms': None,
                'rms_ratio': None, 'displacement_mse': None}
    x, y = pred[pairs].double(), target[pairs].double()
    px, py = x.square().mean().sqrt(), y.square().mean().sqrt()
    return {'observed_channel_pairs': count, 'prediction_rms': float(px), 'reference_rms': float(py),
            'rms_ratio': float(px/py) if py > 0 else None, 'displacement_mse': float((x-y).square().mean())}


def score_window(pred, target, valid, channel_mask, start):
    """Center and score only the specified suffix; never include supplied GT."""
    if start not in WINDOWS.values() or pred.shape != target.shape or pred.shape != (*valid.shape, 9):
        raise ValueError('Matching native96 upper arrays and declared suffix required')
    groups = {}
    for name, cc in GROUPS.items():
        x, y = pred[:, start:, cc], target[:, start:, cc]
        observed = valid[:, start:, None] & channel_mask[:, None, cc]
        if not observed.any():
            raise ValueError('Scored suffix has no observed values')
        paired = paired_metrics(x, y, observed)
        pair_mask = observed[:, 1:] & observed[:, :-1]
        dx, dy = x[:, 1:]-x[:, :-1], y[:, 1:]-y[:, :-1]
        # start itself is excluded: every seam has BOTH frames in the suffix.
        grid = (torch.arange(start+1, 96, device=valid.device) % 16 == 0)[None, :, None]
        speed = _displacement(dx, dy, pair_mask)
        seams = _displacement(dx, dy, pair_mask & grid)
        groups[name] = {'paired': paired, 'speed': speed, 'seams': seams,
                        'observed_values': int(observed.sum()),
                        'mean_error_mse': paired['raw_mse']-paired['centered_mse']}
    return {'native_start_index': start, 'native_stop_exclusive': 96, 'groups': groups,
            'centering_scope': 'Each clip/channel uses only this suffix valid observed frames.',
            'seam_rule': 'Adjacent valid native pairs inside suffix with right index divisible by16; no15->16 startup pair.'}


def initial_connection(pred, target, valid, channel_mask, supplied_endpoint):
    """Keep the actual15->16 handoff separate from later generated seams."""
    result = {}
    for name, cc in GROUPS.items():
        mask = (valid[:, 15] & valid[:, 16])[:, None] & channel_mask[:, cc]
        result[name] = _displacement(pred[:, 16, cc]-supplied_endpoint[:, cc],
                                     target[:, 16, cc]-target[:, 15, cc], mask)
    return result


@torch.no_grad()
def evaluate_initial_history(upper, q, identities, scales, *, batch_size=16, device='cpu',
                             max_clips=0, save_curves=True):
    """Evaluate a preselected fit128 or dev405 cohort, filtering only masks."""
    if (getattr(upper, 'training', False) or type(batch_size) is not int or batch_size < 1
            or type(max_clips) is not int or max_clips < 0):
        raise ValueError('Evaluation model, positive batch size, nonnegative smoke limit required')
    n = len(q['clip_id'])
    if (len(set(q['clip_id'])) != n or q['motion'].shape != (n, 96, 52)
            or q['times'].shape != (n, 96) or not torch.isfinite(q['times']).all()
            or not torch.allclose(q['times'][:, 1:]-q['times'][:, :-1],
                                  torch.full_like(q['times'][:, 1:], .04), atol=1e-7, rtol=1e-5)
            or scales.shape != (9,) or not torch.isfinite(scales).all() or not (scales > 0).all()):
        raise ValueError('Unique native96 cohort,25Hz times and finite source scales required')
    valid = q['valid'].cpu()
    cmask = q['channel_mask'][:, p.CC].cpu()
    eligible, reasons = eligibility(valid, cmask)
    ids = eligible.nonzero(as_tuple=True)[0]
    eligible_count = len(ids)
    if max_clips:
        ids = ids[:max_clips]
    if not len(ids):
        raise ValueError('No eligible fixed-index initial-history clips')
    if q['static_upper'].shape != (n,9) or not torch.isfinite(q['static_upper'][ids]).all():
        raise ValueError('Finite audio-derived static upper origin required')
    # Reference and inference dictionaries are separated BEFORE any decode.
    truth = q['motion'][ids][..., p.CC].detach().cpu().float()
    observed = valid[ids, :, None] & cmask[ids, None]
    if not torch.isfinite(truth[observed]).all():
        raise ValueError('Finite observed reference required')
    truth = torch.where(observed, truth, 0.)
    reference_valid, reference_mask = valid[ids], cmask[ids]
    inference = {key: q[key] for key in (*CONDITION_KEYS, 'prefix_local', 'static_upper', 'speaker_id')}
    selected_ids = [q['clip_id'][int(i)] for i in ids]
    records, curves = {}, {}
    started = time.monotonic()
    for seed in SEEDS:
        # Draw for original cohort order before eligibility/smoke filtering.
        # This preserves normal noise indices against historical128/dev405.
        noise = torch.randn(n, 96, 9, generator=torch.Generator().manual_seed(seed))
        for initialization in INITIALIZATIONS:
            for condition in CONDITIONS:
                values = []
                for selected in ids.split(batch_size):
                    b = p.r.subset(inference, selected, device)
                    ident = p.r.batch_identity(identities, b)
                    fixed = scales.to(b['prefix_local'])
                    gt_prefix = None
                    if initialization != 'normal':
                        # Only16 GT frames are copied to the generation device;
                        # normalization never computes any target time statistic.
                        prefix = q['motion'][selected, :16][..., p.CC].to(device)
                        gt_prefix = torch.where(b['valid'][:, :16, None],
                            (prefix-b['static_upper'][:, None])/fixed, 0.)
                    conditions = {key: b[key] for key in CONDITION_KEYS}
                    dynamic = decode_initial_history(upper, conditions, {'code': ident['code']},
                        b['prefix_local'], noise[selected].to(device), initialization=initialization,
                        condition=condition, gt_prefix=gt_prefix)
                    raw = b['static_upper'][:, None]+dynamic*fixed
                    values.append(torch.where(b['valid'][..., None], raw, 0.).cpu())
                prediction = torch.cat(values)
                key = f'{seed}/{initialization}/{condition}'
                endpoint = prediction[:, 15] if initialization == 'normal' else truth[:, 15]
                records[key] = {
                    'GT_was_input': initialization != 'normal',
                    'GT_input_native_indices': list(range(8, 16)) if initialization == 'gt_prefix16' else
                                               ([15] if initialization == 'gt_terminal_static' else []),
                    'initial_connection': initial_connection(prediction, truth, reference_valid, reference_mask, endpoint),
                    'windows': {name: score_window(prediction, truth, reference_valid, reference_mask, begin)
                                for name, begin in WINDOWS.items()},
                }
                if save_curves:
                    curves[key] = prediction
                # Keep one deployment curve per seed solely for invariance checks.
                if initialization == 'normal':
                    records[key]['first16_generation_sha256'] = p.state_hash({'first16': prediction[:, :16]})
                else:
                    baseline = records[f'{seed}/normal/{condition}']['first16_generation_sha256']
                    if p.state_hash({'first16': prediction[:, :16]}) != baseline:
                        raise RuntimeError('One-time history changed history-free first16 generation')
    mean_scores = {}
    for initialization in INITIALIZATIONS:
        for condition in CONDITIONS:
            modes = [records[f'{seed}/{initialization}/{condition}'] for seed in SEEDS]
            result = {}
            for window in WINDOWS:
                result[window] = {}
                for region in GROUPS:
                    result[window][region] = {}
                    for section in ('paired', 'speed', 'seams'):
                        items = [r['windows'][window]['groups'][region][section] for r in modes]
                        result[window][region][section] = {
                            name: sum(v[name] for v in items)/3 if all(v[name] is not None for v in items) else None
                            for name in items[0]}
            mean_scores[initialization+'/'+condition] = result
    report = {'schema': SCHEMA, 'source_cohort_count': n, 'eligible_count_before_smoke_limit': eligible_count,
              'evaluated_count': len(ids), 'clip_id': selected_ids, 'cohort_indices': ids.tolist(),
              'eligibility': {'rule': 'All9 channels observed; fixed native index15 valid; some valid16:32; at least2 late48: observations and1 adjacent late pair. Past gaps are retained.',
                              'excluded_by_reason_nonexclusive': {k: int(v.sum()) for k,v in reasons.items()},
                              'excluded_clip_ids': [q['clip_id'][i] for i in range(n) if not bool(eligible[i])]},
              'noise_seeds': list(SEEDS), 'noise_indexing': 'Full original cohort [N,96,9] draw before mask/smoke selection.',
              'max_clips_smoke_only': max_clips, 'decode_steps': 12,
              'modes': records, 'mean_over_three_seeds': mean_scores,
              'GT_prefix_oracles_are_not_deployment': True, 'normal_GT_was_input': False,
              'future_GT_or_mean_input': False, 'local_static_keeps_h0_global_static_identity': True,
              'nonupper_scored': False, 'fullface_deployment_evaluation': False,
              'weights_updated': False, 'test_loaded': False, 'seconds': time.monotonic()-started,
              'limitations': ['GT is injected only once into start16 known8 slots; all later history is generated.',
                             'Eight-frame receiver reads only8:16 of the nominal16-frame start;0:8 never enter history.',
                             'Oracle success cannot prove audio predicts initial state or identify a missing4D input.',
                             'Raw/centered scores use common suffix16:96 and late48:96;15->16 handoff is separate.',
                             'Acoustic/global features are offline; past-only motion does not imply causal audio.',
                             'Eligibility is mask-selected; repeated fit/dev evidence is not a sealed test.']}
    archive = {'schema': SCHEMA, 'clip_id': selected_ids, 'cohort_indices': ids,
               'target_upper9': truth, 'valid': reference_valid, 'channel_mask_upper': reference_mask,
               'times': q['times'][ids].cpu(), 'upper_predictions9': curves,
               'upper_channel_indices': list(p.CC), 'noise_seeds': list(SEEDS),
               'prefix0_16_excluded_from_all_main_scores': True, 'invalid_upper_is_zero_placeholder': True,
               'contains_explicit_GT_initialization_diagnostics': True, 'not_fullface_render_input': True}
    return report, archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','audio','targets','enrollment','native-root','trained-run',
                 'history-run','context-run','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--fit-selection', type=Path)
    parser.add_argument('--split', choices=('fit','development','both'), default='fit')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-clips', type=int, default=0, help='First eligible clips only; smoke scope, never formal results.')
    parser.add_argument('--no-curves', action='store_true', help='Only small JSON; default saves compact upper9 diagnostics.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    if args.batch_size < 1 or args.max_clips < 0:
        raise ValueError('Positive batch size and nonnegative smoke limit required')
    args.pilot = False
    args.pilot_selection = None
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    # Existing loader reads locked train/validation inputs, no test split. It
    # necessarily builds its acoustic caches; no full52 model outputs are loaded.
    data, system, audio, local, identities, old_source, source_recipe, _, steps, _, frozen = p.load_context(args)
    if set(data['splits']) != {'train','validation'} or steps != 12 or [
            len(data['splits'][role]['valid']) for role in ('train','validation')] != [2315,405]:
        raise ValueError('Only locked2315fit/405development source permitted')
    sourcepath = args.context_run/'chunk_teacher/final.pt'
    complete = read(sourcepath.with_name('complete.json'))
    if sha(sourcepath) != complete['final_sha256']:
        raise ValueError('Context source checkpoint hash differs')
    source = load_pt(sourcepath)
    previous = read(sourcepath.with_name('provenance.json'))
    if (source['schema'] != context.SCHEMA or source['arm'] != 'chunk_teacher'
            or source['completed_epochs'] != 12 or source['frozen'] != frozen
            or source['recipe_sha256'] != complete['recipe_sha256']
            or source['recipe_sha256'] != p.canonical_hash(previous['recipe'])
            or previous['recipe_sha256'] != source['recipe_sha256']
            or previous['recipe']['source_recipe_sha256'] != p.canonical_hash(source_recipe)
            or not torch.equal(source['scales'], old_source['scales'])):
        raise ValueError('Context checkpoint acoustic/coordinate lineage differs')
    selection_path = args.fit_selection or args.context_run/'fit_selection.json'
    if sha(selection_path) != previous['recipe']['fit_selection_sha256']:
        raise ValueError('Historical128 source selection hash differs')
    selection = read(selection_path)
    fit_ids = historical_fit_indices(data['splits']['train'], selection)
    keys = ('clip_id','valid','times','channel_mask','motion','h0','audio_global','audio_intensity',
            'static_upper','prefix_local','speaker_id')
    cohorts = {}
    if args.split in ('fit','both'):
        cohorts['fit128'] = p.r.subset({k:data['splits']['train'][k] for k in keys}, fit_ids, 'cpu')
    if args.split in ('development','both'):
        cohorts['development405'] = {k:data['splits']['validation'][k] for k in keys}
    upper = p.PrefixUpperFlow(data['config']).to(args.device).eval().requires_grad_(False)
    upper.load_state_dict(source['upper'], strict=True)
    scales = source['scales'].cpu()
    before = p.state_hash(upper.state_dict())
    provenance = {'schema': SCHEMA, 'context_source_sha256': sha(sourcepath),
                  'context_complete_sha256': sha(sourcepath.with_name('complete.json')),
                  'context_recipe_sha256': source['recipe_sha256'], 'frozen_source_hashes': frozen,
                  'fit_selection_sha256': sha(selection_path), 'historical_fit128_membership_verified': True,
                  'source_data_provenance': data['provenance'], 'split': args.split,
                  'weights_before': before, 'script_sha256': sha(Path(__file__)), 'test_loaded': False,
                  'max_clips_smoke_only': args.max_clips, 'no_optimization': True}
    del data, system, audio, local, source, old_source
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    args.output.mkdir(parents=True)
    p.save_json(args.output/'provenance.json', provenance)
    files = {'provenance.json': sha(args.output/'provenance.json')}
    for name, q in cohorts.items():
        report, curves = evaluate_initial_history(upper, q, identities, scales,
            batch_size=args.batch_size, device=args.device, max_clips=args.max_clips, save_curves=not args.no_curves)
        report['population'] = name
        report['context_source_sha256'] = provenance['context_source_sha256']
        path = args.output/(name+'.json')
        p.save_json(path, report)
        files[path.name] = sha(path)
        if not args.no_curves:
            path = args.output/(name+'_upper9.pt')
            p.save_checkpoint(path, curves)
            files[path.name] = sha(path)
        del curves
    after = p.state_hash(upper.state_dict())
    if before != after:
        raise RuntimeError('Read-only diagnostic modified receiver weights')
    p.save_json(args.output/'complete.json', {'status':'complete','schema':SCHEMA,'files':files,
        'weights_before':before,'weights_after':after,'weights_unchanged':True,'test_loaded':False,
        'smoke':bool(args.max_clips),'no_training':True})
    print('INITIAL_HISTORY_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
