"""Frozen full-face diagnostic for a completed reference-controlled prior.

Every query in the prior's prelocked metadata selection is exported, in its
original order. Seed42 is always displayed. Query motion supplies only binding
checks and the GT comparison; no identity, audio baseline or motion generator
receives it. Forty-three non-upper coefficients and invalid frames are copied
bit-for-bit from the frozen run12 baseline.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_clocked_fullface_examples as legacy


SCHEMA = 'controlled_prior_frozen_fullface_diagnostic_v1'
PRIOR_SCHEMA = 'independent_reference_continuous_prior_v1'
MODES = ('GT', 'run12 audio baseline', 'bounded medoid prior',
         'controlled stochastic prior', 'independent reference swap')
ARMS = ('medoid', 'process', 'reference_swap')
UPPER, OTHER = legacy.UPPER, legacy.OTHER


def load_prior(root):
    """Bind saved generation, metadata-only query selection and source run."""
    root = Path(root)
    protocol = legacy.read(root/'protocol.json')
    status = legacy.read(root/'status.json')
    selection = legacy.read(root/'query_selection.json')
    saved = legacy.load(root/'predictions.pt')
    if (protocol.get('schema') != PRIOR_SCHEMA or status.get('schema') != PRIOR_SCHEMA
            or saved.get('schema') != PRIOR_SCHEMA or status.get('status') != 'complete'
            or status.get('smoke') is not False or status.get('test_loaded') is not False
            or status.get('default_replaced') is not False or protocol.get('test_loaded') is not False
            or protocol.get('extra_expression_reference') is not True
            or protocol.get('audio_timing_claim') is not False):
        raise ValueError('Completed formal standalone reference-controlled prior required')
    picks = selection['queries']
    ids = [row['clip_id'] for row in picks]
    if (not ids or len(ids) != len(set(ids)) or ids != protocol['query_clip_ids']
            or set(ids) != set(saved['curves']) or status.get('queries') != len(ids)):
        raise ValueError('Prior selection, predictions and protocol membership differ')
    if protocol.get('seeds', [None])[0] != 42:
        raise ValueError('The first saved prior draw must be the predeclared seed42')
    for row in picks:
        curves = saved['curves'][row['clip_id']]
        if any(curves['metadata'][key] != row[key] for key in ('clip_id', 'sentence', 'speaker', 'emotion')):
            raise ValueError('Prediction metadata differ from prelocked selection')
        valid = np.asarray(curves['valid'])
        if valid.dtype != np.bool_ or valid.ndim != 1 or not valid.any():
            raise ValueError('Saved native validity mask required')
        for arm in ARMS:
            values = np.asarray(curves['samples'][arm])
            if (values.shape != (len(protocol['seeds']), len(valid), 9)
                    or not np.isfinite(values[:, valid]).all()):
                raise ValueError('Expected all saved seed draws of finite native upper9: '+arm)
    source = Path(protocol['source_run'])
    for name, digest in protocol['source_files_sha256'].items():
        if legacy.sha(source/name) != digest:
            raise ValueError('Frozen prior source artifact changed: '+name)
    source_protocol = legacy.read(source/'protocol.json')
    expected = {row['clip_id']: row for row in source_protocol['split']['confirmation']}
    for row in picks:
        if row['clip_id'] not in expected or any(row[key] != expected[row['clip_id']][key]
                                                for key in ('sentence', 'speaker', 'emotion')):
            raise ValueError('Selected query outside the source consumed-development membership')
    return protocol, status, selection, saved, picks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio', 'targets', 'native-root', 'native-manifest', 'delta-dir',
                 'audio-checkpoint', 'prior-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--enrollment', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh full-face diagnostic output required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    protocol, status, selection, saved, picks = load_prior(args.prior_root)
    checkpoint = legacy.load(args.audio_checkpoint)
    complete = legacy.read(args.audio_checkpoint.with_name('complete.json'))
    if (checkpoint.get('schema') != 'full_staged_spline_innovation_v1'
            or checkpoint.get('stage') not in ('audio', 'dynamics')
            or complete.get('final_sha256') != legacy.sha(args.audio_checkpoint)
            or checkpoint.get('inference_only') is not True):
        raise ValueError('Hash-bound completed run12 checkpoint required')
    run_root = args.audio_checkpoint.parent.parent
    recipe = legacy.read(run_root/'provenance.json')['recipe']
    canonical_audio = run_root/'audio/final.pt'
    audio = legacy.load_frozen_audio(canonical_audio, args.device)
    canonical = legacy.load(canonical_audio)
    for module in ('system', 'audio'):
        if (set(checkpoint[module]) != set(canonical[module]) or any(
                not torch.equal(value, canonical[module][key]) for key, value in checkpoint[module].items())):
            raise ValueError('Protected system/audio differ from the pinned completed source')
    system = legacy.NeutralAffectSystem(checkpoint['config'])
    system.load_state_dict(checkpoint['system'], strict=True)
    system.to(args.device).eval().requires_grad_(False)
    steps = int(recipe['args']['decode_steps'])
    enrollment = args.enrollment or Path(recipe['paths']['enrollment'])
    archive, targets = legacy.load(args.audio), legacy.load(args.targets)
    cache_path = args.cache or Path(archive['provenance']['renderer_cache'])
    cache_hash = legacy.sha(cache_path)
    if cache_hash != legacy._renderer_cache_binding(archive) or cache_hash != legacy._renderer_cache_binding(targets):
        raise ValueError('Renderer/audio/targets source binding differs')
    lineage = {'cache': cache_hash, 'audio': legacy.sha(args.audio), 'targets': legacy.sha(args.targets)}
    binding = legacy.assert_source_binding(audio, lineage)
    if binding != protocol['frozen_audio'] or any(protocol['source'][key] != value for key, value in lineage.items()):
        raise ValueError('Prior and frozen full-face sources differ')
    cache = legacy.load(cache_path)
    query = copy.copy(cache['splits']['train']['q'])
    raw_audio, raw_targets = archive['splits']['train'], targets['splits']['train']
    if query['clip_id'] != raw_audio['clip_id'] or query['clip_id'] != raw_targets['clip_id']:
        raise ValueError('TRAIN query/audio/anchor membership differs')
    query.update(content=raw_audio['features'][..., :768], audio_features=raw_audio['features'],
                 anchors=raw_targets['anchors'], anchor_valid=raw_targets['anchor_valid'])
    mapping = {cid: index for index, cid in enumerate(query['clip_id'])}
    if any(pick['clip_id'] not in mapping for pick in picks):
        raise ValueError('Selected query outside historical TRAIN allowlist')
    indices = [mapping[pick['clip_id']] for pick in picks]
    speakers = {query['speaker'][i]: int(query['speaker_id'][i]) for i in indices}
    identities, identity_refs = legacy.load_identities(system, targets, enrollment, args.native_root.resolve(),
                                                       query, speakers, args.device)
    selected_query = {key: value[indices] if torch.is_tensor(value) else [value[i] for i in indices]
        for key, value in query.items() if torch.is_tensor(value) or isinstance(value, (list, tuple))}
    store = legacy.NativeContextStore(selected_query, args.native_root, args.native_manifest, args.delta_dir, role='train')
    args.output.mkdir(parents=True)
    (args.output/'video_npz').mkdir(); (args.output/'audio').mkdir()
    jobs, records = [], []
    for index, pick in enumerate(picks):
        cid = pick['clip_id']
        if Path(cid).name != cid or cid in ('.', '..'):
            raise ValueError('Clip ID must be a safe output basename')
        native = store.clip(index); valid = native['valid']; curves = saved['curves'][cid]
        if int(native['speaker_id']) != pick['speaker'] or int(native['emotion_id']) != pick['emotion']:
            raise ValueError('Selected metadata differ from native query')
        # Query GT is used only here for the saved-target lineage check and
        # below for the GT panel; it is never an input to baseline inference.
        if (not np.array_equal(curves['valid'], valid.numpy()) or
                not np.array_equal(curves['target'], native['motion'][:, UPPER].numpy())):
            raise ValueError('Saved prior target/mask differ from selected native query')
        baseline, inference = legacy.infer_baseline(system, audio, native['content'], native['audio_features'],
            valid, identities[int(native['speaker_id'])], cid, steps, args.device)
        variants = []
        for arm in ARMS:
            upper = torch.from_numpy(np.asarray(curves['samples'][arm][0])).to(baseline)
            composed = legacy.compose_upper_face(baseline[None], upper[None], valid[None])[0]
            if not torch.equal(composed[:, OTHER], baseline[:, OTHER]) or not torch.equal(composed[~valid], baseline[~valid]):
                raise RuntimeError('Protected non-upper43 or invalid frame changed: '+arm)
            variants.append(composed)
        motions = torch.stack([native['motion'], baseline, *variants]).numpy()
        native_provenance = native['metadata']['provenance']
        waveform = Path(native_provenance['audio_path'])
        if legacy.sha(waveform) != native_provenance['audio_sha256']:
            raise ValueError('Bound native waveform changed')
        copied = Path('audio')/(cid+waveform.suffix)
        shutil.copy2(waveform, args.output/copied)
        relative = Path('video_npz')/(cid+'.npz')
        np.savez_compressed(args.output/relative, channels=np.asarray(legacy.ARKIT_NAMES), mode_names=np.asarray(MODES),
            clip_id=np.asarray(cid), noise_seed=np.asarray(42), times=native['times'].numpy(),
            valid=valid.numpy(), channel_mask=native['channel_mask'].numpy(), motions=motions,
            audio_relative_path=np.asarray(copied.as_posix()), audio_sha256=np.asarray(native_provenance['audio_sha256']),
            audio_offset_seconds=np.asarray(float(native_provenance['audio_offset_s'])))
        inspection = legacy.inspect_input(args.output/relative, 25)[-1]
        records.append({**pick, 'speaker_name': native['speaker'], 'emotion_name': legacy.EMOTIONS[pick['emotion']],
            'npz': relative.as_posix(), 'sha256': legacy.sha(args.output/relative), 'native_frames': len(valid),
            'native_metadata': native['metadata'], 'inference': inference, 'display_report': inspection})
        jobs.append({'input': relative.as_posix(), 'output': cid, 'fps': 25, 'columns': 3,
            'tile_size': 360, 'samples': 16, 'max_frames': 0, 'expected_video': cid+'/comparison.mp4',
            'audio': copied.as_posix(), 'audio_sha256': native_provenance['audio_sha256'],
            'audio_offset_seconds': float(native_provenance['audio_offset_s'])})
        print('CONTROLLED_FULLFACE_EXPORTED', cid, len(valid), 'frames; protected43 exact', flush=True)
    legacy.write(args.output/'render_jobs.json', {'schema': SCHEMA, 'jobs': jobs, 'rendered': False,
        'driver': 'scripts/render_dynamic_rig_comparison.py', 'paths_relative_to': 'directory containing this JSON'})
    source_files = ('protocol.json', 'status.json', 'predictions.pt', 'query_selection.json', 'references.json')
    legacy.write(args.output/'provenance.json', {'schema': SCHEMA, 'selection_rule': selection['rule'],
        'selection_uses_metadata_only': True, 'all_prelocked_queries_exported': True, 'outcome_based_selection': False,
        'clips': records, 'mode_names': list(MODES), 'neutral_reference_bindings': identity_refs,
        'expression_reference_bindings': legacy.read(args.prior_root/'references.json'),
        'checkpoint_sha256': legacy.sha(args.audio_checkpoint), 'prior_status': status,
        'source_prior_sha256': {name: legacy.sha(args.prior_root/name) for name in source_files},
        'decode_steps': steps, 'prior_seed': 42, 'prior_draw_selection': 'First of eight predeclared draws, never best-of-eight',
        'baseline_seed_scope': 'Same clip-keyed42 baseline as frozen clocked exporter; newly inferred',
        'query_teacher_inference': False, 'nonupper43_exact': True, 'invalid_exact': True,
        'full_native_clock': True, 'raw_clamping': False, 'query_target_amplitude_rescaling': False,
        'extra_expression_reference': True, 'development_only': True, 'test_loaded': False, 'dev405_indexed': False,
        'identity_mouth_emotion_certified': False, 'perceptual_naturalness_certified': False,
        'audio_timing_claim': False, 'default_replaced': False,
        'limits': 'Diagnostic composition of a reference-controlled stochastic process. Channel preservation is not a visual or publication-quality certificate.',
        'script_sha256': legacy.sha(__file__), 'helper_sha256': legacy.sha(Path(legacy.__file__))})
    print('CONTROLLED_FULLFACE_EXPORT_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
