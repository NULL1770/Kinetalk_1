"""Build isolated reset candidates for the 32 locked clips after audit completion.

These are same-model re-extractions, not verified ground truth or training data.
No inference, video decoding, original coefficient overwrite, or manifest adoption.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import scipy
from scipy.signal import savgol_filter
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_tracking_reset import SCHEMA as AUDIT_SCHEMA, comparison, load_coefficients, sha, write_json

SCHEMA = 'locked_reset_supervision_sidecars_v1'
LOCKED_SELECTION_SHA256 = 'dccffb96b6a52b34ec2046566d257e1b225745668bda0408508cda2e2665e7e8'
ARMS = ('fresh_forward', 'reused_forward', 'fresh_reverse', 'reused_reverse')
ATOL, RTOL = 1e-6, 1e-5


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def interpolate_invalid(coeffs, valid):
    """Historical 05_postprocess_coeffs.py interpolation, including endpoints."""
    out = coeffs.copy()
    idx = np.arange(coeffs.shape[0])
    good = np.where(valid)[0]
    if good.size == 0:
        return np.zeros_like(out)
    if good.size == coeffs.shape[0]:
        return out
    for c in range(coeffs.shape[1]):
        out[:, c] = np.interp(idx, good, coeffs[good, c])
    return out


def smooth(coeffs, window=5, poly=2):
    """Historical short-clip handling and float32 Savitzky-Golay output."""
    frames = coeffs.shape[0]
    window = min(window, frames if frames % 2 == 1 else frames - 1)
    if window < poly + 2 or window < 3:
        return coeffs
    if window % 2 == 0:
        window -= 1
    return savgol_filter(coeffs, window_length=window, polyorder=poly,
                         axis=0, mode='interp').astype(np.float32)


def postprocess(coeffs, valid):
    return np.clip(smooth(interpolate_invalid(coeffs.astype(np.float32), valid)), 0., 1.)


def validate_audit(root):
    """Require completed immutable audit and exact metadata-locked membership."""
    status = read(root/'status.json')
    if status.get('schema') != AUDIT_SCHEMA or status.get('status') != 'complete':
        raise ValueError('Completed reset audit required before building sidecars')
    manifest = read(root/'manifest.json')
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Completed audit manifest required')
    for name, record in manifest.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or Path(name).is_absolute():
            raise ValueError('Unsafe audit manifest path: '+name)
        if (not path.is_file() or path.stat().st_size != record['bytes']
                or sha(path) != record['sha256']):
            raise ValueError('Audit manifest hash/size differs: '+name)
    required = {'status.json', 'summary.json', 'provenance.json', 'selection.json',
                'activity_thresholds.json', 'source_auditor.py'}
    if not required.issubset(manifest):
        raise ValueError('Completed audit records missing from manifest')
    if sha(root/'selection.json') != LOCKED_SELECTION_SHA256:
        raise ValueError('Only the exact locked 32-clip selection is supported')
    selection, summary, provenance = [read(root/name) for name in
                                     ('selection.json', 'summary.json', 'provenance.json')]
    ids = [row['clip_id'] for row in selection['clips']]
    if (len(ids) != 32 or len(set(ids)) != 32 or set(ids) != set(summary['clips'])
            or status.get('clips') != 32 or summary.get('clip_count') != 32
            or any(not isinstance(cid, str) or Path(cid).name != cid or '\\' in cid or ':' in cid
                   or cid in ('', '.', '..') for cid in ids)):
        raise ValueError('Exactly the locked 32 completed clip IDs are required')
    if (any(v.get('schema') != AUDIT_SCHEMA for v in (summary, provenance))
            or summary.get('independent_ground_truth') is not False
            or summary.get('training_started') is not False
            or summary.get('default_changed') is not False
            or provenance.get('source_files_modified') is not False
            or provenance.get('fresh_vs_reused_same_absolute_timestamps') is not True
            or provenance.get('same_model_landmark_proxy_is_independent_ground_truth') is not False
            or provenance.get('fps') != 25 or set(provenance.get('arms', [])) != set(ARMS)):
        raise ValueError('Source-preserving same-clock four-arm audit required')
    if (summary['provenance_sha256'] != sha(root/'provenance.json')
            or provenance['selection_sha256'] != sha(root/'selection.json')
            or provenance['activity_thresholds_sha256'] != sha(root/'activity_thresholds.json')
            or provenance['code_sha256'] != sha(root/'source_auditor.py')):
        raise ValueError('Audit provenance bindings differ')
    expected = {'arrays/'+cid+'_'+arm+'.npz' for cid in ids for arm in ARMS}
    actual = {p.relative_to(root).as_posix() for p in (root/'arrays').glob('*.npz')}
    manifested = {name for name in manifest if name.startswith('arrays/') and name.endswith('.npz')}
    if expected != actual or expected != manifested:
        raise ValueError('Four-arm array membership differs from locked selection')
    thresholds = np.asarray(read(root/'activity_thresholds.json')['thresholds'], dtype=float)
    if (thresholds.shape != (4,) or not np.isfinite(thresholds).all()
            or (thresholds < 1e-6).any()
            or not np.array_equal(thresholds, provenance['activity_thresholds'])):
        raise ValueError('Pinned activity thresholds differ')
    return selection, summary, provenance, thresholds


def load_arm(path):
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in
                  ('coeffs', 'valid', 'head_pose', 'times', 'timestamps_ms', 'rgb_sha256')}
    c, v = values['coeffs'], values['valid']
    if (c.dtype != np.float32 or c.ndim != 2 or c.shape[1] != 52
            or not 1 <= len(c) <= 400 or v.dtype != np.bool_ or v.shape != (len(c),)
            or not np.isfinite(c[v]).all() or values['head_pose'].shape != (len(c), 3)
            or values['rgb_sha256'].shape != (len(c),)
            or values['timestamps_ms'].shape != (len(c),)
            or not np.issubdtype(values['timestamps_ms'].dtype, np.integer)
            or not np.array_equal(values['times'], np.arange(len(c))/25)
            or not np.all(np.diff(values['timestamps_ms']) == 40)):
        raise ValueError('Invalid native 25Hz arm array: '+str(path))
    return values


def order_consistency(forward, reverse, thresholds):
    """Tolerance applies only to jointly observed coefficients; no GT claim."""
    common = forward['valid'] & reverse['valid']
    same_mask = bool(np.array_equal(forward['valid'], reverse['valid']))
    close = bool(np.allclose(forward['coeffs'][common], reverse['coeffs'][common],
                            atol=ATOL, rtol=RTOL)) if common.any() else None
    offset = reverse['timestamps_ms'] - forward['timestamps_ms']
    return {'atol': ATOL, 'rtol': RTOL, 'valid_masks_equal': same_mask,
            'coefficients_allclose_on_shared_observations': close,
            'consistent_within_tolerance': bool(same_mask and close) if close is not None else None,
            'absolute_timestamp_offset_ms': int(offset[0]),
            'absolute_timestamp_offset_constant': bool(np.all(offset == offset[0])),
            'comparison': comparison(forward['coeffs'], forward['valid'],
                                     reverse['coeffs'], reverse['valid'], thresholds),
            'meaning': 'Fresh clip order also changes absolute timestamp offsets; sensitivity, not correctness'}


def support(valid):
    runs = ''.join('1' if value else '0' for value in valid).split('1')
    return {'observed_frames': int(valid.sum()), 'missing_frames': int((~valid).sum()),
            'observation_rate': float(valid.mean()), 'longest_missing_run': max(map(len, runs)),
            'all_invalid': not bool(valid.any()), 'historical_qc_pass_inherited': False}


def build(audit_root, output, source_root):
    root, output, source_root = (Path(p).resolve() for p in (audit_root, output, source_root))
    if output.exists():
        raise FileExistsError('Fresh sidecar output directory required; no overwrite')
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Sidecars must be separate from the completed audit directory')
    selection, summary, provenance, thresholds = validate_audit(root)
    processing_path = source_root/'scripts/05_postprocess_coeffs.py'
    config_path = source_root/'configs/default.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf8'))
    pp = config['postprocess']
    if (config['media']['fps'] != 25 or pp['savgol_window'] != 5 or pp['savgol_polyorder'] != 2
            or pp['clip_min'] != 0. or pp['clip_max'] != 1.
            or sha(config_path) != provenance['historical_source_sha256']['configs/default.yaml']):
        raise ValueError('Historical SG5/2, [0,1], 25Hz configuration differs')
    processing_hash = sha(processing_path)
    sources = {}
    records, payloads = {}, {}
    for row in selection['clips']:
        cid = row['clip_id']
        for field, hash_key in (('raw_npz', 'raw_sha256'), ('final_npz', 'final_sha256')):
            path = Path(row[field]).resolve()
            if sha(path) != row[hash_key]:
                raise ValueError('Locked historical source hash differs: '+cid+'/'+field)
            if path.is_relative_to(output):
                raise ValueError('Output cannot contain original coefficient files')
            sources[str(path)] = row[hash_key]
        old_raw, old_valid = load_coefficients(row['raw_npz'])
        old_final, old_final_valid = load_coefficients(row['final_npz'], inherited_valid=old_valid)
        arms = {arm: load_arm(root/'arrays'/f'{cid}_{arm}.npz') for arm in ARMS}
        fresh = arms['fresh_forward']
        if (len(old_raw) != len(old_final) or len(old_raw) != len(fresh['coeffs'])
                or len(old_raw) != summary['clips'][cid]['frames']):
            raise ValueError('Historical/fresh native frame count differs: '+cid)
        for arm, data in arms.items():
            if (not np.array_equal(fresh['times'], data['times'])
                    or not np.array_equal(fresh['rgb_sha256'], data['rgb_sha256'])):
                raise ValueError('Four-arm native clock/pixel hashes differ: '+cid+'/'+arm)
        for order in ('forward', 'reverse'):
            if not np.array_equal(arms['fresh_'+order]['timestamps_ms'], arms['reused_'+order]['timestamps_ms']):
                raise ValueError('Same-order fresh/reused absolute times differ: '+cid)
        final = postprocess(fresh['coeffs'], fresh['valid'])
        meta = row['media_record']
        labels = {key: np.asarray(meta[key]) for key in ('emotion', 'intensity', 'speaker', 'dataset')}
        shared = {'valid': fresh['valid'].copy(), 'times': fresh['times'].copy(),
                  'timestamps_ms': fresh['timestamps_ms'].copy(), 'clip_id': np.asarray(cid)}
        payloads[cid] = {
            'raw': {'coeffs': fresh['coeffs'].copy(), 'head_pose': fresh['head_pose'].astype(np.float32),
                    'rgb_sha256': fresh['rgb_sha256'].copy(), **shared, **labels},
            'final': {'coeffs': final, 'interpolated': ~fresh['valid'], **shared, **labels}}
        records[cid] = {'frames': len(final), 'old_raw_sha256': row['raw_sha256'],
            'old_final_sha256': row['final_sha256'], 'source_video_sha256': row['video_sha256'],
            'source_arm_sha256': {arm: sha(root/'arrays'/f'{cid}_{arm}.npz') for arm in ARMS},
            'fresh_support': support(fresh['valid']), 'original_support': support(old_valid),
            'fresh_forward_reverse': order_consistency(fresh, arms['fresh_reverse'], thresholds),
            'historical_raw_vs_reset_raw': comparison(old_raw, old_valid, fresh['coeffs'], fresh['valid'], thresholds),
            'historical_final_vs_reset_final': comparison(old_final, old_final_valid, final, fresh['valid'], thresholds),
            'verified_ground_truth': False, 'incorporated_into_training': False}
    # Complete all source validation before creating any output.
    output.mkdir(parents=True)
    (output/'raw').mkdir(); (output/'final').mkdir()
    write_json(output/'status.json', {'schema': SCHEMA, 'status': 'building'})
    try:
        for cid, data in payloads.items():
            for stage in ('raw', 'final'):
                target = output/stage/(cid+'.npz')
                np.savez_compressed(target, **data[stage])
                records[cid]['new_'+stage+'_sha256'] = sha(target)
                records[cid][stage+'_sidecar'] = target.relative_to(output).as_posix()
        if any(sha(path) != digest for path, digest in sources.items()):
            raise RuntimeError('Historical source changed while building sidecars')
        validate_audit(root)
        if sha(processing_path) != processing_hash:
            raise RuntimeError('Historical processing source changed during build')
        shutil.copyfile(__file__, output/'source_builder.py')
        report = {'schema': SCHEMA, 'status': 'complete', 'clip_count': 32,
            'frames': sum(r['frames'] for r in records.values()), 'source_arm': 'fresh_forward',
            'audit_root': str(root), 'audit_manifest_sha256': sha(root/'manifest.json'),
            'audit_provenance_sha256': sha(root/'provenance.json'),
            'locked_selection_sha256': LOCKED_SELECTION_SHA256, 'builder_sha256': sha(__file__),
            'historical_processing_source': str(processing_path), 'historical_processing_sha256': processing_hash,
            'historical_config_sha256': sha(config_path), 'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
            'processing': {'input': 'float32 fresh_forward coefficients', 'interpolation': 'np.interp per channel, nearest endpoint; all-invalid zero',
                'savgol_window': 5, 'savgol_polyorder': 2, 'savgol_mode': 'interp', 'clip': [0., 1.],
                'speaker_neutral_subtraction': False, 'fps': 25},
            'validity': 'Both sidecars preserve fresh original detection mask. Interpolation/SG5 creates values, not observations. Final valid marks source detection support, not absence of neighboring interpolation influence.',
            'fresh_order_tolerance': {'atol': ATOL, 'rtol': RTOL, 'scope': 'Jointly valid coefficients; masks separately exact',
                'clips_outside_tolerance': [cid for cid,r in records.items() if r['fresh_forward_reverse']['consistent_within_tolerance'] is False],
                'clips_without_shared_support': [cid for cid,r in records.items() if r['fresh_forward_reverse']['consistent_within_tolerance'] is None]},
            'verified_ground_truth': False, 'incorporated_into_training': False, 'original_data_modified': False,
            'training_manifests_modified': False, 'training_started': False, 'default_changed': False,
            'limitations': ['Only these metadata-locked 32 fit clips; no dataset-wide adoption or generalization claim.',
                'Fresh extraction and geometry share MediaPipe; no independent/manual correctness verification.',
                'Fresh order comparison changes clip order and absolute timestamp offsets; tolerance failure is reported, never silently filtered.',
                'Historical raw/final comparison can also reflect unknown original worker state/version; reset alone is not isolated.',
                'Original QC pass is not inherited. New missing runs are reported; long gaps and all-invalid clips require separate eligibility review.',
                'SG5 intentionally matches historical processing; it is not proof of event fidelity or final label quality.',
                'Original video files are not decoded or rehashed here; video SHA is inherited from completed audit selection.',
                'No evidence from these sidecars alone that audio predicts timing or that natural generation succeeds.'],
            'clips': records}
        write_json(output/'migration_report.json', report)
        write_json(output/'status.json', {'schema': SCHEMA, 'status': 'complete', 'clips': 32,
                                        'incorporated_into_training': False})
        write_json(output/'manifest.json', {p.relative_to(output).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
                                           for p in output.rglob('*') if p.is_file() and p.name != 'manifest.json'})
        return report
    except Exception as exc:
        write_json(output/'status.json', {'schema': SCHEMA, 'status': 'failed', 'error': repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path('D:/实验室项目/新实验/数据集'))
    args = parser.parse_args()
    result = build(args.audit_root, args.output, args.source_root)
    print(json.dumps({'output': str(args.output.resolve()), 'clips': result['clip_count'],
                      'incorporated_into_training': False}), flush=True)


if __name__ == '__main__':
    main()
