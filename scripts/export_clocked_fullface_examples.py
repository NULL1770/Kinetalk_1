"""Frozen run12 full-face diagnostic with saved clocked upper-face predictions.

Select the first lexicographic clip ID per emotion from the consumed353
development membership. Only those native TRAIN queries and their independent
neutral enrollment recordings are opened. Inference never receives query
motion; GT is read only for binding and exported reference. No model is trained
or default replaced. The output is renderer-compatible raw52 NPZ plus audio.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.models.neutral_affect import NeutralAffectSystem
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face
from kinetalk_b0.neutral_data import native_clip, stack_clips, EMOTIONS
from scripts.full_native_context_data import NativeContextStore
from scripts.full_staged_data import sha, _renderer_cache_binding
from scripts.joint_prior_audio_context import load_frozen_audio, assert_source_binding
from scripts.train_full_staged import base_forward
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input


SCHEMA = 'clocked_frozen_fullface_diagnostic_v1'
MODES = ('GT', 'run12 audio baseline', 'bounded static prior', 'bounded temporal prior')
UPPER = list(UPPER_INDICES)
OTHER = [i for i in range(52) if i not in UPPER]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def select_metadata(protocol):
    rows = protocol['split']['confirmation']
    chosen = []
    for emotion in sorted({row['emotion'] for row in rows}):
        chosen.extend(sorted((row for row in rows if row['emotion'] == emotion),
                             key=lambda row: row['clip_id'])[:1])
    if not chosen or len({row['clip_id'] for row in chosen}) != len(chosen):
        raise ValueError('Unique metadata-only development examples required')
    return chosen


def contained(root, relative):
    target = (root/str(relative)).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Reference artifact leaves native root')
    return target


@torch.no_grad()
def load_identities(system, targets, enrollment_path, native_root, query, speakers, device):
    """Recompute current B0 residuals and identities from independent refs only."""
    if targets['provenance']['source_sha256']['enrollment'] != sha(enrollment_path):
        raise ValueError('Enrollment differs from independently bound anchors')
    rows = [json.loads(line) for line in Path(enrollment_path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    query_ids, query_sentences = set(query['clip_id']), set(query['sentence_id'])
    identities, bindings = {}, []
    for speaker, sid in sorted(speakers.items()):
        selected = [row for row in rows if row['speaker'] == speaker]
        expected = targets['by_speaker'][speaker]
        if (len(selected) < 2 or int(expected['speaker_id']) != sid
                or expected['reference_clip_ids'] != [row['clip_id'] for row in selected]):
            raise ValueError('Reference membership differs from anchor binding')
        reference_clips = []
        for row in selected:
            if (row['split'] != 'train' or int(row.get('emotion_id', row.get('emotion', -1))) != 0
                    or int(row['speaker_id']) != sid or row['clip_id'] in query_ids
                    or row['sentence'] in query_sentences):
                raise ValueError('Reference must be independent native-TRAIN neutral recording')
            path = contained(native_root, row['artifact'])
            if sha(path) != row['artifact_sha256']:
                raise ValueError('Reference artifact changed')
            item = native_clip({**row, 'emotion_id': 0,
                'intensity_id': int(row.get('intensity_id', row.get('intensity', 0)))},
                native_root, window=96, min_valid_frames=32)
            reference_clips.append(item)
            bindings.append({'clip_id': row['clip_id'], 'speaker_id': sid,
                             'artifact_sha256': row['artifact_sha256']})
        refs = stack_clips(reference_clips)
        content, valid = refs['content'].to(device), refs['valid'].to(device)
        base = base_forward(system, content, valid)
        observed = valid[..., None] & refs['channel_mask'][:, None].to(device)
        residual = torch.where(observed, refs['motion'].to(device)-base['b0'], 0.)
        identity = system.encode_identity(residual[None], valid[None])
        identities[sid] = {key: identity[key] for key in ('code', 'baseline')}
    return identities, bindings


@torch.no_grad()
def infer_baseline(system, audio, content, features, valid, identity, clip_id, steps, device):
    """Canonical model API: acoustic arrays, independent identity, fixed noise."""
    # Pad only for audio/native compatibility; true-length base_forward keeps
    # the original B0 positional behavior. No query motion argument exists.
    n = len(valid); length = ((n+15)//16)*16
    x = torch.zeros(1, length, 768, device=device)
    f = torch.zeros(1, length, 1540, device=device)
    mask = torch.zeros(1, length, dtype=torch.bool, device=device)
    x[0, :n], f[0, :n], mask[0, :n] = content.to(device), features.to(device), valid.to(device)
    base = base_forward(system, x, mask)
    affect = audio(f, mask)
    key = int.from_bytes(hashlib.sha256(('clocked_fullface:42:'+clip_id).encode()).digest()[:8], 'little') % (2**63-1)
    noise = torch.randn(1, length, 52, generator=torch.Generator().manual_seed(key)).to(device)
    output = system.generate(x, mask, identity, affect, initial_noise=noise, steps=steps, base=base)['motion']
    return output[0, :n].cpu(), {'seed': 42, 'clip_seed': key,
        'audio_emotion_prediction': int(affect['emotion_logits'].argmax(-1)[0]),
        'audio_intensity_prediction': float(affect['intensity_value'][0, 0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio', 'targets', 'native-root', 'native-manifest', 'delta-dir',
                 'audio-checkpoint', 'residual-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--enrollment', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    protocol = read(args.residual_root/'protocol.json')
    status = read(args.residual_root/'status.json')
    if status.get('status') != 'complete' or protocol.get('schema') != 'clocked_bounded_static_residual_v1':
        raise ValueError('Require completed bounded residual experiment')
    picks = select_metadata(protocol)
    # Both supported source stages carry the identical protected system/audio.
    checkpoint = load(args.audio_checkpoint)
    complete = read(args.audio_checkpoint.with_name('complete.json'))
    if (checkpoint.get('schema') != 'full_staged_spline_innovation_v1'
            or checkpoint.get('stage') not in ('audio', 'dynamics')
            or complete.get('final_sha256') != sha(args.audio_checkpoint)
            or checkpoint.get('inference_only') is not True):
        raise ValueError('Expected hash-bound completed run12 system checkpoint')
    run_root = args.audio_checkpoint.parent.parent
    recipe = read(run_root/'provenance.json')['recipe']
    canonical_audio = run_root/'audio/final.pt'
    audio = load_frozen_audio(canonical_audio, args.device)
    canonical = load(canonical_audio)
    for module in ('system', 'audio'):
        if (set(checkpoint[module]) != set(canonical[module]) or
                any(not torch.equal(value, canonical[module][name]) for name, value in checkpoint[module].items())):
            raise ValueError('Selected checkpoint differs from pinned audio-stage protected module')
    system = NeutralAffectSystem(checkpoint['config'])
    system.load_state_dict(checkpoint['system'], strict=True)
    system.to(args.device).eval().requires_grad_(False)
    steps = int(recipe['args']['decode_steps'])
    enrollment = args.enrollment or Path(recipe['paths']['enrollment'])
    archive, targets = load(args.audio), load(args.targets)
    cache_path = args.cache or Path(archive['provenance']['renderer_cache'])
    if sha(cache_path) != _renderer_cache_binding(archive) or sha(cache_path) != _renderer_cache_binding(targets):
        raise ValueError('Renderer/audio/targets binding differs')
    lineage = {'cache': sha(cache_path), 'audio': sha(args.audio), 'targets': sha(args.targets)}
    assert_source_binding(audio, lineage)
    if any(protocol['source'][key] != lineage[key] for key in lineage):
        raise ValueError('Prior and frozen face source inputs differ')
    cache = load(cache_path)
    query = copy.copy(cache['splits']['train']['q'])
    raw_audio, raw_targets = archive['splits']['train'], targets['splits']['train']
    if query['clip_id'] != raw_audio['clip_id'] or query['clip_id'] != raw_targets['clip_id']:
        raise ValueError('TRAIN query/audio/anchor membership differs')
    query.update(content=raw_audio['features'][..., :768], audio_features=raw_audio['features'],
                 anchors=raw_targets['anchors'], anchor_valid=raw_targets['anchor_valid'])
    mapping = {cid: index for index, cid in enumerate(query['clip_id'])}
    if any(pick['clip_id'] not in mapping for pick in picks):
        raise ValueError('Selected development clip outside historical TRAIN allowlist')
    indices = [mapping[pick['clip_id']] for pick in picks]
    speakers = {query['speaker'][i]: int(query['speaker_id'][i]) for i in indices}
    identities, refs = load_identities(system, targets, enrollment, args.native_root.resolve(),
                                       query, speakers, args.device)
    # Restrict NativeContextStore itself to these selected TRAIN queries.
    chosen_query = {key: value[indices] if torch.is_tensor(value) else [value[i] for i in indices]
                    for key, value in query.items() if torch.is_tensor(value) or isinstance(value, (list, tuple))}
    store = NativeContextStore(chosen_query, args.native_root, args.native_manifest, args.delta_dir, role='train')
    source_curves = {arm: load(args.residual_root/('confirmation_'+arm+'.pt'))
                     for arm in ('static_trained', 'temporal')}
    args.output.mkdir(parents=True)
    (args.output/'video_npz').mkdir(); (args.output/'audio').mkdir()
    jobs, records = [], []
    for index, pick in enumerate(picks):
        cid = pick['clip_id']
        if Path(cid).name != cid or cid in ('.', '..'):
            raise ValueError('Clip ID is not a safe output basename')
        raw = store.clip(index); valid = raw['valid']
        if int(raw['speaker_id']) != pick['speaker'] or int(raw['emotion_id']) != pick['emotion']:
            raise ValueError('Selected metadata differs from native query')
        for arm in source_curves:
            prior = source_curves[arm][cid]
            if (not np.array_equal(prior['valid'], valid.numpy()) or
                    not np.array_equal(prior['target'], raw['motion'][:, UPPER].numpy())):
                raise ValueError('Saved prior target/mask differs from selected native query')
        baseline, inference = infer_baseline(system, audio, raw['content'], raw['audio_features'], valid,
            identities[int(raw['speaker_id'])], cid, steps, args.device)
        variants = []
        for arm in ('static_trained', 'temporal'):
            upper = torch.from_numpy(source_curves[arm][cid]['samples'][0]).to(baseline)
            composed = compose_upper_face(baseline[None], upper[None], valid[None])[0]
            if not torch.equal(composed[:, OTHER], baseline[:, OTHER]) or not torch.equal(composed[~valid], baseline[~valid]):
                raise RuntimeError('Protected43 or invalid frame changed')
            variants.append(composed)
        motions = torch.stack([raw['motion'], baseline, *variants]).numpy()
        provenance = raw['metadata']['provenance']
        waveform = Path(provenance['audio_path'])
        if sha(waveform) != provenance['audio_sha256']:
            raise ValueError('Bound native waveform changed')
        copied = Path('audio')/(cid+waveform.suffix)
        shutil.copy2(waveform, args.output/copied)
        relative = Path('video_npz')/(cid+'.npz')
        np.savez_compressed(args.output/relative, channels=np.asarray(ARKIT_NAMES), mode_names=np.asarray(MODES),
            clip_id=np.asarray(cid), noise_seed=np.asarray(42), times=raw['times'].numpy(),
            valid=valid.numpy(), channel_mask=raw['channel_mask'].numpy(), motions=motions,
            audio_relative_path=np.asarray(copied.as_posix()), audio_sha256=np.asarray(provenance['audio_sha256']),
            audio_offset_seconds=np.asarray(float(provenance['audio_offset_s'])))
        display = inspect_input(args.output/relative, 25)[-1]
        record = {**pick, 'speaker_name': raw['speaker'], 'emotion_name': EMOTIONS[pick['emotion']],
            'npz': relative.as_posix(), 'sha256': sha(args.output/relative), 'native_frames': len(valid),
            'native_metadata': raw['metadata'], 'inference': inference, 'display_report': display}
        records.append(record)
        jobs.append({'input': relative.as_posix(), 'output': cid, 'fps': 25, 'columns': 2,
            'tile_size': 360, 'samples': 16, 'max_frames': 0, 'expected_video': cid+'/comparison.mp4',
            'audio': copied.as_posix(), 'audio_sha256': provenance['audio_sha256'],
            'audio_offset_seconds': float(provenance['audio_offset_s'])})
        print('EXPORTED', cid, len(valid), 'native frames, protected43 exact', flush=True)
    write(args.output/'render_jobs.json', {'schema': SCHEMA, 'jobs': jobs, 'rendered': False,
        'driver': 'scripts/render_dynamic_rig_comparison.py', 'paths_relative_to': 'directory containing this JSON'})
    write(args.output/'provenance.json', {'schema': SCHEMA, 'selection_rule': 'First lexicographic ID per emotion from consumed353',
        'selection_uses_metadata_only': True, 'outcome_based_selection': False, 'clips': records,
        'mode_names': MODES, 'neutral_reference_bindings': refs, 'checkpoint_sha256': sha(args.audio_checkpoint),
        'prior_protocol_sha256': sha(args.residual_root/'protocol.json'), 'prior_status': status,
        'source_curves_sha256': {arm: sha(args.residual_root/('confirmation_'+arm+'.pt')) for arm in source_curves},
        'decode_steps': steps, 'baseline_seed_scope': 'New clip-keyed42 noise, not historical saved curve reproduction',
        'prior_draw': 'first of eight fixed draws (seed42), not best-of-eight',
        'query_teacher_inference': False, 'nonupper43_exact': True, 'invalid_exact': True,
        'full_native_clock': True, 'raw_clamping': False, 'amplitude_rescaling': False,
        'development_only': True, 'test_loaded': False, 'dev405_indexed': False,
        'identity_mouth_emotion_certified': False, 'default_replaced': False,
        'limits': 'Diagnostic composition only. Prior timing/quality gates may fail; global expression/identity and lips need visual review.',
        'script_sha256': sha(__file__)})
    print('FULLFACE_EXPORT_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
