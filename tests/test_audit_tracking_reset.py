"""Synthetic contracts only: no selected real videos or MediaPipe inference."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_tracking_reset as audit


def landmark_fixture():
    points = np.full((478, 3), .5)
    for corners, brow, top, bottom, center in [
        ([33,133], [63,105,66], [160,159,158], [144,145,153], .3),
        ([362,263], [296,334,293], [385,386,387], [380,374,373], .7)]:
        points[corners[0], :2] = [center-.1, .5]; points[corners[1], :2] = [center+.1, .5]
        points[brow, :2] = [center, .35]; points[top, :2] = [center, .47]; points[bottom, :2] = [center, .53]
    return points


def test_geometry_is_dimensionless_and_invariant_to_planar_similarity():
    points = landmark_fixture()
    expected = [.6, .6, .3, .3]
    np.testing.assert_allclose(audit.geometry_proxies(points, 100, 100), expected)
    angle = .4; rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    altered = points.copy(); altered[:, :2] = (points[:, :2]@rot.T)*2+[.1, -.2]
    np.testing.assert_allclose(audit.geometry_proxies(altered, 100, 100), expected)
    altered = points.copy(); altered[:, 0] /= 2
    np.testing.assert_allclose(audit.geometry_proxies(altered, 200, 100), expected)
    assert np.isnan(audit.geometry_proxies(np.zeros((478, 3)), 100, 100)).all()


def test_rotation_removes_scale_and_reports_expected_yaw():
    angle = np.deg2rad(20); matrix = np.eye(4)
    matrix[:3, :3] = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])*3
    np.testing.assert_allclose(audit.matrix_rotation(matrix), [20,0,0], atol=1e-10)


def test_name_mapping_is_order_invariant_and_missing_names_fail():
    names = [n for n in audit.ARKIT_NAMES if n != 'tongueOut']+['_neutral']
    categories = [SimpleNamespace(category_name=name, score=i/100) for i,name in enumerate(names)]
    first, valid = audit.named_coefficients(categories)
    reverse, _ = audit.named_coefficients(categories[::-1])
    assert valid and first[-1] == 0
    np.testing.assert_array_equal(first, reverse)
    categories[2].category_name = 'unknown'
    with pytest.raises(ValueError, match='names missing'):
        audit.named_coefficients(categories)
    assert audit.named_coefficients([])[1] is False


def test_activity_clock_and_masked_smoothing_never_cross_holes():
    time = np.arange(80)[:,None]
    values = time*np.ones((1,9))*.01
    valid = np.ones(80, bool); valid[40] = False; values[40] = np.nan
    starts, energy = audit.activity_windows(values, valid)
    np.testing.assert_array_equal(starts, [0,8,41,48])
    np.testing.assert_allclose(energy, .25)
    speed = audit.masked_speed(values, valid)
    assert np.isnan(speed[38:42]).all()
    assert np.isfinite(speed[1:38]).all()
    np.testing.assert_allclose(speed[2], .25)


def test_comparison_identity_and_first_frames_localization():
    original = np.zeros((40,52)); changed = original.copy(); changed[:3,41] = .2
    valid = np.ones(40, bool)
    result = audit.comparison(original,valid,changed,valid)
    assert result['first8']['rms'] > 0 and result['after8']['rms'] == 0
    exact = audit.comparison(original,valid,original,valid)
    assert exact['overall']['max_abs'] == 0 and exact['activity_windows'] == 2


def test_selection_requires_pinned_files_and_refuses_more_than32(tmp_path):
    files = {}
    for key in ['video','raw_npz','final_npz']:
        path=tmp_path/key; path.write_bytes(b'x'); files[key]=str(path)
        files[key.replace('_npz','')+'_sha256']=audit.sha(path)
    row={'clip_id':'clip1','needs_resample':True,**files}
    manifest=tmp_path/'selection.json'; manifest.write_text(json.dumps({'clips':[row]}),encoding='utf8')
    assert audit.read_selection(manifest)['clips'][0]['clip_id']=='clip1'
    row['video_sha256']='0'*64; manifest.write_text(json.dumps({'clips':[row]}),encoding='utf8')
    with pytest.raises(ValueError,match='hash differs'):
        audit.read_selection(manifest)
    manifest.write_text(json.dumps({'clips':[row]*33}),encoding='utf8')
    with pytest.raises(ValueError,match='one to32'):
        audit.read_selection(manifest)


def test_coupling_constant_signals_report_unavailable_not_nan():
    coefficients=np.zeros((40,52)); geometry=np.ones((40,4)); head=np.zeros((40,3))
    result=audit.coupling(coefficients,np.ones(40,bool),geometry,head)
    assert result['group_geometry_level_correlation']==[None]*4
    json.dumps(result,allow_nan=False)


def test_historical_final_npz_inherits_raw_detection_mask_explicitly(tmp_path):
    raw_valid=np.ones(40,bool); raw_valid[[0,8,39]]=False
    path=tmp_path/'final.npz'
    np.savez_compressed(path,coeffs=np.full((40,52),.2,dtype=np.float32),
                        emotion=np.asarray('neutral'),intensity=np.asarray(1),
                        speaker=np.asarray('mead_M003'),dataset=np.asarray('mead'))
    with pytest.raises(ValueError,match='explicit paired raw mask'):
        audit.load_coefficients(path)
    values,mask=audit.load_coefficients(path,inherited_valid=raw_valid)
    assert values.shape==(40,52)
    np.testing.assert_array_equal(mask,raw_valid)
    mask[0]=True
    assert not raw_valid[0]
