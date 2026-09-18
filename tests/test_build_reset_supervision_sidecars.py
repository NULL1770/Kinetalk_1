"""Sidecar contracts use synthetic completed audits; no real dataset writes."""
import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import savgol_filter

from scripts import build_reset_supervision_sidecars as side


@pytest.fixture
def completed(tmp_path, monkeypatch):
    root = tmp_path/'audit'; root.mkdir(); (root/'arrays').mkdir()
    originals = tmp_path/'originals'; originals.mkdir()
    source = tmp_path/'upstream'; (source/'configs').mkdir(parents=True); (source/'scripts').mkdir()
    (source/'scripts/05_postprocess_coeffs.py').write_text('# synthetic source provenance\n')
    (source/'configs/default.yaml').write_text('media: {fps: 25}\npostprocess: {savgol_window: 5, savgol_polyorder: 2, clip_min: 0.0, clip_max: 1.0}\n')
    rows, records = [], {}
    valid = np.ones(40, dtype=bool); valid[[0, 10, 39]] = False
    coeffs = np.tile(np.linspace(0., 1., 40, dtype=np.float32)[:, None], (1, 52))
    coeffs[~valid] = 0.
    for index in range(32):
        cid = f'locked_{index:02d}'
        raw, final = originals/(cid+'_raw.npz'), originals/(cid+'_final.npz')
        np.savez_compressed(raw, coeffs=coeffs, valid=valid, head_pose=np.zeros((40, 3), np.float32))
        # Actual historical final schema has scalar labels and no validity field.
        np.savez_compressed(final, coeffs=side.postprocess(coeffs, valid),
                            emotion=np.asarray('neutral'), intensity=np.asarray(1),
                            speaker=np.asarray('speaker'), dataset=np.asarray('mead'))
        rows.append({'clip_id': cid, 'raw_npz': str(raw), 'final_npz': str(final),
                     'raw_sha256': side.sha(raw), 'final_sha256': side.sha(final), 'video_sha256': 'a'*64,
                     'media_record': {'emotion': 'neutral', 'intensity': 1, 'speaker': 'speaker', 'dataset': 'mead'}})
        records[cid] = {'frames': 40}
        for arm in side.ARMS:
            offset = 100000 if arm.endswith('reverse') else 0
            np.savez_compressed(root/'arrays'/f'{cid}_{arm}.npz', coeffs=coeffs, valid=valid,
                                head_pose=np.zeros((40, 3), np.float64), times=np.arange(40)/25,
                                timestamps_ms=np.arange(40, dtype=np.int64)*40+offset,
                                rgb_sha256=np.asarray([str(i).zfill(64) for i in range(40)]))
    side.write_json(root/'selection.json', {'clips': rows})
    monkeypatch.setattr(side, 'LOCKED_SELECTION_SHA256', side.sha(root/'selection.json'))
    side.write_json(root/'activity_thresholds.json', {'thresholds': [.1, .2, .3, .4]})
    (root/'source_auditor.py').write_text('# synthetic auditor\n')
    provenance = {'schema': side.AUDIT_SCHEMA, 'selection_sha256': side.sha(root/'selection.json'),
                  'activity_thresholds_sha256': side.sha(root/'activity_thresholds.json'),
                  'activity_thresholds': [.1, .2, .3, .4], 'code_sha256': side.sha(root/'source_auditor.py'),
                  'historical_source_sha256': {'configs/default.yaml': side.sha(source/'configs/default.yaml')},
                  'source_files_modified': False, 'fresh_vs_reused_same_absolute_timestamps': True,
                  'same_model_landmark_proxy_is_independent_ground_truth': False, 'fps': 25, 'arms': list(side.ARMS)}
    side.write_json(root/'provenance.json', provenance)
    side.write_json(root/'summary.json', {'schema': side.AUDIT_SCHEMA, 'clip_count': 32, 'clips': records,
               'independent_ground_truth': False, 'training_started': False, 'default_changed': False,
               'provenance_sha256': side.sha(root/'provenance.json')})
    side.write_json(root/'status.json', {'schema': side.AUDIT_SCHEMA, 'status': 'complete', 'clips': 32})
    rehash(root)
    return root, tmp_path/'sidecars', source


def rehash(root):
    side.write_json(root/'manifest.json', {p.relative_to(root).as_posix(): {'sha256': side.sha(p), 'bytes': p.stat().st_size}
                                          for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'})


def test_refuse_incomplete_before_creating_output(tmp_path):
    root = tmp_path/'audit'; root.mkdir()
    side.write_json(root/'status.json', {'schema': side.AUDIT_SCHEMA, 'status': 'running'})
    output = tmp_path/'sidecars'
    with pytest.raises(ValueError, match='Completed reset audit'):
        side.build(root, output, tmp_path)
    assert not output.exists()


def test_completed_migration_preserves_originals_masks_clock_and_schema(completed):
    root, output, source = completed
    old_manifest = side.sha(root/'manifest.json')
    report = side.build(root, output, source)
    assert report['clip_count'] == 32 and report['frames'] == 1280
    assert not report['verified_ground_truth'] and not report['incorporated_into_training']
    assert not report['fresh_order_tolerance']['clips_outside_tolerance']
    assert side.sha(root/'manifest.json') == old_manifest
    for cid, record in report['clips'].items():
        row = next(r for r in side.read(root/'selection.json')['clips'] if r['clip_id'] == cid)
        assert side.sha(row['raw_npz']) == record['old_raw_sha256']
        assert side.sha(row['final_npz']) == record['old_final_sha256']
        with np.load(root/'arrays'/f'{cid}_fresh_forward.npz') as fresh, \
                np.load(output/record['raw_sidecar']) as raw, np.load(output/record['final_sidecar']) as final:
            np.testing.assert_array_equal(raw['coeffs'], fresh['coeffs'])
            for z in (raw, final):
                np.testing.assert_array_equal(z['valid'], fresh['valid'])
                np.testing.assert_array_equal(z['times'], fresh['times'])
                assert z['coeffs'].dtype == np.float32 and z['emotion'].item() == 'neutral'
            np.testing.assert_array_equal(final['interpolated'], ~fresh['valid'])
            assert final['coeffs'][0, 0] > 0 and not final['valid'][0]
        assert record['fresh_forward_reverse']['absolute_timestamp_offset_ms'] == 100000
        assert not record['fresh_support']['historical_qc_pass_inherited']
    with pytest.raises(FileExistsError, match='Fresh sidecar'):
        side.build(root, output, source)


def test_hash_tampering_and_extra_membership_are_refused(completed):
    root, output, source = completed
    target = root/'arrays/locked_00_reused_reverse.npz'
    target.write_bytes(target.read_bytes()+b'tampered')
    with pytest.raises(ValueError, match='manifest hash/size'):
        side.build(root, output, source)
    assert not output.exists()
    rehash(root)
    (root/'arrays/extra.npz').write_bytes(b'extra')
    with pytest.raises(ValueError, match='array membership'):
        side.build(root, output, source)


def test_selection_cannot_expand_or_shrink_even_with_new_manifest(completed, monkeypatch):
    root, output, source = completed
    selection = side.read(root/'selection.json'); selection['clips'].pop()
    side.write_json(root/'selection.json', selection); rehash(root)
    with pytest.raises(ValueError, match='exact locked 32-clip'):
        side.build(root, output, source)
    monkeypatch.setattr(side, 'LOCKED_SELECTION_SHA256', side.sha(root/'selection.json'))
    with pytest.raises(ValueError, match='Exactly the locked 32'):
        side.build(root, output, source)


def test_pixel_clock_integrity_checked_beyond_file_hashes(completed):
    root, output, source = completed
    target = root/'arrays/locked_00_fresh_reverse.npz'
    with np.load(target) as z:
        payload = {name: z[name] for name in z.files}
    payload['rgb_sha256'][3] = 'f'*64
    np.savez_compressed(target, **payload); rehash(root)
    with pytest.raises(ValueError, match='clock/pixel hashes'):
        side.build(root, output, source)
    assert not output.exists()


def test_order_sensitivity_reported_without_silently_dropping_clip(completed):
    root, output, source = completed
    target = root/'arrays/locked_00_fresh_reverse.npz'
    with np.load(target) as z:
        payload = {name: z[name] for name in z.files}
    payload['coeffs'][2, 41] += .03
    np.savez_compressed(target, **payload); rehash(root)
    report = side.build(root, output, source)
    assert report['fresh_order_tolerance']['clips_outside_tolerance'] == ['locked_00']
    assert (output/'raw/locked_00.npz').is_file()
    record = report['clips']['locked_00']['fresh_forward_reverse']
    assert record['valid_masks_equal'] and not record['coefficients_allclose_on_shared_observations']
    assert record['comparison']['overall']['max_abs'] > .029


def test_interpolation_short_clips_and_all_invalid_preserve_semantics():
    coeffs = np.tile(np.array([np.nan, .2, np.nan, .6, np.nan], np.float32)[:, None], (1, 52))
    mask = np.array([False, True, False, True, False])
    np.testing.assert_allclose(side.interpolate_invalid(coeffs, mask)[:, 0], [.2, .2, .4, .6, .6])
    np.testing.assert_allclose(side.postprocess(coeffs[:3], mask[:3])[:, 0], [.2, .2, .2])
    np.testing.assert_array_equal(side.postprocess(coeffs, np.zeros(5, bool)), np.zeros((5, 52), np.float32))
    np.testing.assert_array_equal(mask, [False, True, False, True, False])
    assert side.support(np.zeros(5, bool))['longest_missing_run'] == 5


def test_parity_with_actual_historical_functions_when_available():
    path = Path('D:/实验室项目/新实验/数据集/scripts/05_postprocess_coeffs.py')
    if not path.is_file():
        pytest.skip('Historical external dataset source unavailable')
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('interpolate_invalid', 'smooth')]
    assert len(selected) == 2
    namespace = {'np': np, 'savgol_filter': savgol_filter}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    rng = np.random.default_rng(218)
    for frames in (1, 2, 3, 4, 5, 6, 39, 40):
        values = rng.normal(.5, .4, (frames, 52)).astype(np.float32)
        for mask in (np.ones(frames, bool), np.zeros(frames, bool), rng.random(frames) > .25):
            expected = np.clip(namespace['smooth'](namespace['interpolate_invalid'](values, mask), 5, 2), 0., 1.)
            np.testing.assert_array_equal(side.postprocess(values, mask), expected)
