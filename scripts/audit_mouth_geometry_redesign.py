"""Read-only fixed-rig mouth geometry audit; no model fitting or test access."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def geometry_asset(path):
    with np.load(path, allow_pickle=False) as z:
        names = [str(x) for x in z['group_names']]
        weights = z['lip_group_weights'].astype(np.float64)
        rest = z['neutral_mm'].astype(np.float64)
        delta = z['blendshape_deltas_mm'].astype(np.float64)
        def centroid(name):
            w = weights[:, names.index(name)]
            if w.sum() <= 0:
                raise ValueError('Empty independently named lip group')
            w = w / w.sum()
            return w @ rest, np.einsum('v,cva->ca', w, delta)
        left, dl = centroid('lLipCorner')
        right, dr = centroid('rLipCorner')
        upper, du = centroid('LipUpperMiddle')
        lower, dd = centroid('LipLowerMiddle')
        sign = np.sign(left[0] - right[0])
        keys = ['corner_width_x_mm', 'corner_lift_z_mm', 'middle_aperture_z_mm', 'corner_depth_y_mm']
        offset = np.array([sign*(left[0]-right[0]), (left[2]+right[2])/2-(upper[2]+lower[2])/2,
                           upper[2]-lower[2], (left[1]+right[1])/2])
        jacobian = np.stack([sign*(dl[:,0]-dr[:,0]), (dl[:,2]+dr[:,2])/2-(du[:,2]+dd[:,2])/2,
                             du[:,2]-dd[:,2], (dl[:,1]+dr[:,1])/2])
        return keys, offset, jacobian, z['channels'].copy()


def audit(asset, files):
    keys, offset, jacobian, channels = geometry_asset(asset)
    records = []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(channels, z['channels']):
                raise ValueError('Rig/motion channel order differs')
            valid = z['valid'].astype(bool)
            observed = z['channel_mask'].astype(bool)
            # Same fixed display/evaluation policy for all columns; raw is also retained.
            # The render contract disables missing channels (including tongueOut).
            # This is a common display proxy, never an imputed GT observation.
            raw = np.where(observed[None,None], z['motions'].astype(np.float64)[:, valid], 0.)
            motion = np.clip(raw, 0, 1)
            values = motion @ jacobian.T + offset
            raw_values = raw @ jacobian.T + offset
            rows = {}
            for i, name in enumerate(z['mode_names']):
                rows[str(name)] = {key: {
                    'mean_mm': float(values[i,:,j].mean()),
                    'p95_p05_mm': float(np.ptp(np.quantile(values[i,:,j], [.05,.95]))),
                    'mae_to_gt_mm': float(np.abs(values[i,:,j]-values[0,:,j]).mean()),
                    'raw_mean_mm': float(raw_values[i,:,j].mean()),
                } for j, key in enumerate(keys)}
            records.append({'clip_id': str(z['clip_id'].item()), 'valid_frames': int(valid.sum()),
                            'disabled_unobserved_channels': channels[~observed].tolist(),
                            'motion_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': rows})
    return {'schema': 'fixed_rig_mouth_geometry_diagnostic_v1', 'test_loaded': False,
            'fit_performed': False, 'asset_sha256': hashlib.sha256(asset.read_bytes()).hexdigest(),
            'geometry_rule': 'Named skin-group weighted centroids selected independently of fitted/generated motion',
            'limitations': ['Shared linear rig proxy, not official dataset landmarks or physiological contact.',
                            'Two display clips do not establish general angry-expression or content performance.',
                            'Fitted GT can contain fitting bias; original video spot-check remains necessary.'],
            'records': records}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset', type=Path, default=Path('final_experiment/rig/lip_constraints_v2_20261010.npz'))
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.asset, a.inputs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    for record in result['records']:
        print(record['clip_id'])
        for name, row in record['rows'].items():
            print(name, {k: round(v['mean_mm'],3) for k,v in row.items()})
