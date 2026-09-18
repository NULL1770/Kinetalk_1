"""Bounded MediaPipe VIDEO reset/order diagnostic on explicitly locked clips.

Never overwrites existing coefficients or selects clips by motion outcomes.
For each of two fixed clip orders, fresh and reused VIDEO instances receive
the exact same RGB frames and absolute timestamps. Landmark measurements are
same-model geometry proxies, not an independent tracker or ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES

SCHEMA = 'tracking_reset_order_diagnostic_v1'
FPS, MAX_CLIPS, MAX_FRAMES = 25, 32, 400
UPPER = [41, 42, 43, 44, 45, 5, 6, 12, 13]
GROUPS = ((2, 3, 4), (0, 1), (5, 7), (6, 8))
GROUP_NAMES = ('brow_up', 'brow_down', 'eye_squint', 'eye_wide')
PROXY_NAMES = ('right_brow_to_upper_eye_over_width', 'left_brow_to_upper_eye_over_width',
               'right_eye_aperture_over_width', 'left_eye_aperture_over_width')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def read_selection(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    rows = payload['clips']
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CLIPS:
        raise ValueError('Explicit selection of one to32 clips required; no full-dataset mode')
    ids = [row['clip_id'] for row in rows]
    if len(ids) != len(set(ids)) or any(not isinstance(cid, str) or Path(cid).name != cid or cid in ('', '.', '..') for cid in ids):
        raise ValueError('Unique safe clip IDs required')
    for row in rows:
        if type(row['needs_resample']) is not bool:
            raise ValueError('Explicit Boolean needs_resample required')
        for key in ('video', 'raw_npz', 'final_npz'):
            target = Path(row[key])
            if not target.is_absolute() or not target.is_file() or target.stat().st_size == 0:
                raise ValueError('Nonempty absolute input path required: '+key)
            expected = row.get(key.replace('_npz', '')+'_sha256')
            if not isinstance(expected, str) or len(expected) != 64 or sha(target) != expected:
                raise ValueError('Locked source hash differs: '+key+'/'+row['clip_id'])
        if row.get('native_npz'):
            if sha(row['native_npz']) != row['native_sha256']:
                raise ValueError('Locked native source hash differs: '+row['clip_id'])
    return payload


def load_coefficients(path, *, inherited_valid=None):
    with np.load(path, allow_pickle=False) as z:
        key = 'coeffs' if 'coeffs' in z else 'motion'
        values = np.asarray(z[key], dtype=np.float64)
        if 'valid' in z or 'mask' in z:
            valid = np.asarray(z['valid'] if 'valid' in z else z['mask'])
        elif inherited_valid is not None:
            # Historical final NPZ stores coefficients plus clip labels only.
            # Its SG5 interpolation does not restore observation support;
            # inherit the paired original raw detection mask explicitly.
            valid = np.asarray(inherited_valid).copy()
        else:
            raise ValueError('Coefficient archive lacks validity; explicit paired raw mask required')
    if (values.ndim != 2 or values.shape[1] != 52 or not 1 <= len(values) <= MAX_FRAMES
            or valid.dtype != np.bool_ or valid.shape != values.shape[:1]
            or not np.isfinite(values[valid]).all()):
        raise ValueError('At most400 native frames of finite observed coefficients[T,52] and Boolean mask required')
    return values, valid


def video_frames(path, resample, ffmpeg):
    """Match historical ffmpeg fps=25 RGB decoding, without spatial resize."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError('Cannot open selected source video: '+str(path))
    if not resample:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        return
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if min(width, height) <= 0:
        raise ValueError('Source dimensions unavailable')
    command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-i', str(path),
               '-vf', 'fps=25', '-an', '-sn', '-dn', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    completed = False
    try:
        size = width*height*3
        while True:
            raw = proc.stdout.read(size)
            if not raw:
                completed = True
                break
            if len(raw) != size:
                raise RuntimeError('Truncated RGB video frame')
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        if not completed:
            proc.terminate()
        message = proc.stderr.read().decode('utf8', errors='replace')
        code = proc.wait()
        proc.stderr.close()
        if completed and code:
            raise RuntimeError('FFmpeg failed: '+message[-500:])


def matrix_rotation(matrix):
    """Nearest proper rotation and historical yaw/pitch/roll degrees."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        return np.full(3, np.nan)
    u, _, vt = np.linalg.svd(value[:3, :3])
    rotation = u@vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u@vt
    sin_pitch = float(np.clip(-rotation[1, 2], -1, 1))
    pitch = np.arcsin(sin_pitch)
    if abs(sin_pitch) < .99999:
        yaw, roll = np.arctan2(rotation[0, 2], rotation[2, 2]), np.arctan2(rotation[1, 0], rotation[1, 1])
    else:
        yaw, roll = np.arctan2(-rotation[2, 0], rotation[0, 0]), 0.
    return np.degrees([yaw, pitch, roll])


def geometry_proxies(landmarks, width, height):
    """Pixel-aspect-corrected2D distances / eye width, from same-model landmarks.

These cancel2D translation/scale/roll, not out-of-plane head pose. Right/left
refer to the subject. Fixed indices are stored in the run's provenance.
    """
    value = np.asarray(landmarks, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 388 or not np.isfinite(value).all() or min(width, height) <= 0:
        return np.full(4, np.nan)
    xy = value[:, :2]*np.array([width, height])
    def side(corners, brow, top, bottom):
        eye_width = np.linalg.norm(xy[corners[0]]-xy[corners[1]])
        if eye_width <= 1e-6:
            return np.nan, np.nan
        a, b, d = xy[brow].mean(0), xy[top].mean(0), xy[bottom].mean(0)
        return np.linalg.norm(a-b)/eye_width, np.linalg.norm(b-d)/eye_width
    right = side((33, 133), [63, 105, 66], [160, 159, 158], [144, 145, 153])
    left = side((362, 263), [296, 334, 293], [385, 386, 387], [380, 374, 373])
    return np.asarray([right[0], left[0], right[1], left[1]])


def named_coefficients(categories):
    """Explicit names only; absent tongue is zero, no positional fallback."""
    if categories is None or len(categories) != 52:
        return np.zeros(52, dtype=np.float32), False
    lookup = {str(getattr(c, 'category_name', '')).lower(): float(c.score) for c in categories}
    if len(lookup) != 52 or any(name.lower() not in lookup for name in ARKIT_NAMES if name != 'tongueOut'):
        raise ValueError('MediaPipe names missing/duplicate; positional fallback forbidden')
    values = np.asarray([lookup.get(name.lower(), 0.) for name in ARKIT_NAMES], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite MediaPipe coefficient output')
    return values, True


def make_landmarker(model):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    options = vision.FaceLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
        output_facial_transformation_matrixes=True, min_face_detection_confidence=.5,
        min_face_presence_confidence=.5, min_tracking_confidence=.5)
    return vision.FaceLandmarker.create_from_options(options)


def observation(result, width, height):
    coefficient, valid = named_coefficients(result.face_blendshapes[0] if result.face_blendshapes else None)
    landmarks = np.full((478, 3), np.nan, dtype=np.float32)
    if result.face_landmarks:
        values = np.asarray([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)
        if values.shape != landmarks.shape:
            raise ValueError('Expected478 face landmarks for pinned FaceLandmarker task')
        landmarks = values
    matrix = np.full((4, 4), np.nan)
    if result.facial_transformation_matrixes:
        matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
    return {'coeffs': coefficient, 'valid': valid, 'landmarks': landmarks,
            'matrix': matrix, 'head_pose': matrix_rotation(matrix),
            'geometry': geometry_proxies(landmarks, width, height)}


def stack_observations(rows, timestamps, frame_hashes):
    return {**{key: np.asarray([row[key] for row in rows]) for key in rows[0]},
            'timestamps_ms': np.asarray(timestamps, dtype=np.int64),
            'rgb_sha256': np.asarray(frame_hashes), 'times': np.arange(len(rows), dtype=np.float64)/FPS}


def continuous_runs(valid):
    edge = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def activity_windows(values, valid):
    """Historical probe-compatibleH32/hop8, centered3frame/8transition speed."""
    values, valid = np.asarray(values, dtype=np.float64), np.asarray(valid)
    if values.shape != (len(valid), 9) or valid.dtype != np.bool_ or not np.isfinite(values[valid]).all():
        raise ValueError('Finite observed upper9 and matching Boolean mask required')
    starts, energy = [], []
    for left, right in continuous_runs(valid):
        if right-left < 32:
            continue
        offsets = list(range(int(left), int(right)-31, 8))
        if offsets[-1] != right-32:
            offsets.append(int(right)-32)
        for start in offsets:
            window = values[start:start+32]
            smooth = (window[10:19]+window[11:20]+window[12:21])/3
            delta = np.diff(smooth, axis=0)*FPS
            starts.append(start)
            energy.append([float(np.sqrt(np.square(delta[:, group]).mean())) for group in GROUPS])
    return np.asarray(starts, dtype=np.int64), np.asarray(energy, dtype=np.float64).reshape(-1, 4)


def correlation(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a)&np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return None
    a, b = a-a.mean(), b-b.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a@b/denom, -1., 1.)) if denom > 1e-12 else None


def comparison(left, left_valid, right, right_valid, thresholds=None):
    """Raw upper9 difference on shared valid support, with no fitting or lag."""
    left, right = np.asarray(left), np.asarray(right)
    left_valid, right_valid = np.asarray(left_valid), np.asarray(right_valid)
    if left.shape != right.shape or left.shape[1] != 52 or left_valid.shape != right_valid.shape:
        raise ValueError('Comparison requires same native frame clock')
    valid = left_valid & right_valid
    error = left[:, UPPER]-right[:, UPPER]
    def stat(mask):
        if not mask.any():
            return None
        return {'frames': int(mask.sum()), 'rms': float(np.sqrt(np.square(error[mask]).mean())),
                'p95_abs': float(np.quantile(np.abs(error[mask]), .95)),
                'max_abs': float(np.abs(error[mask]).max()),
                'group_rms': [float(np.sqrt(np.square(error[mask][:, group]).mean())) for group in GROUPS]}
    s1, e1 = activity_windows(left[:, UPPER], valid)
    s2, e2 = activity_windows(right[:, UPPER], valid)
    if not np.array_equal(s1, s2):
        raise RuntimeError('Comparison window clocks differ')
    return {'valid_agreement': float(np.mean(left_valid == right_valid)), 'common_valid': int(valid.sum()),
            'overall': stat(valid), 'first8': stat(valid & (np.arange(len(valid)) < 8)),
            'after8': stat(valid & (np.arange(len(valid)) >= 8)),
            'activity_windows': len(s1),
            'activity_rms_difference': np.sqrt(np.square(e1-e2).mean(0)).tolist() if len(s1) else None,
            'activity_correlation': [correlation(e1[:, j], e2[:, j]) for j in range(4)] if len(s1) else None,
            'activity_label_disagreement': float(np.mean((e1 > thresholds) != (e2 > thresholds))) if len(s1) and thresholds is not None else None,
            'activity_label_disagreement_groups': np.mean((e1 > thresholds) != (e2 > thresholds), axis=0).tolist() if len(s1) and thresholds is not None else None,
            'activity_left_mean_energy': e1.mean(0).tolist() if len(s1) else None,
            'activity_right_mean_energy': e2.mean(0).tolist() if len(s1) else None}


def masked_speed(values, valid):
    """Centered3point smoother; a difference is valid only with all4 inputs."""
    value = np.asarray(values, dtype=np.float64)
    if value.ndim == 1:
        value = value[:, None]
    mask = np.asarray(valid) & np.isfinite(value).all(1)
    out = np.full((max(len(value)-1, 0), value.shape[1]), np.nan)
    if len(value) >= 4:
        smooth = (value[:-2]+value[1:-1]+value[2:])/3
        ok = mask[:-3]&mask[1:-2]&mask[2:-1]&mask[3:]
        delta = np.diff(smooth, axis=0)*FPS
        out[1:-1] = np.where(ok[:, None], delta, np.nan)
    return out


def coupling(coefficients, valid, geometry, head_pose):
    """Same-model diagnostic associations, not independent expression accuracy."""
    group = np.stack([np.asarray(coefficients)[:, UPPER][:, g].mean(1) for g in GROUPS], axis=1)
    geo = np.asarray(geometry)
    paired_geometry = np.column_stack([geo[:, :2].mean(1)]*2+[geo[:, 2:].mean(1)]*2)
    mask = np.asarray(valid)&np.isfinite(group).all(1)&np.isfinite(paired_geometry).all(1)
    group_speed = masked_speed(group, mask)
    geometry_speed = masked_speed(paired_geometry, mask)
    blink = np.asarray(coefficients)[:, [0, 7]].mean(1)
    blink_speed = masked_speed(blink, mask)[:, 0]
    head_speed = masked_speed(head_pose, np.asarray(valid))
    head_norm = np.sqrt(np.square(head_speed).mean(1))
    return {'scope': 'Same tracker geometry proxy; associations can include tracking/pose/blink coupling',
            'expected_geometry_sign': [1, -1, -1, 1], 'common_geometry_frames': int(mask.sum()),
            'group_geometry_level_correlation': [correlation(group[mask, j], paired_geometry[mask, j]) for j in range(4)],
            'group_geometry_signed_speed_correlation': [correlation(group_speed[:, j], geometry_speed[:, j]) for j in range(4)],
            'group_blink_absolute_speed_correlation': [correlation(np.abs(group_speed[:, j]), np.abs(blink_speed)) for j in range(4)],
            'group_head_speed_correlation': [correlation(np.abs(group_speed[:, j]), head_norm) for j in range(4)]}


def save_contact(path, frames, indices):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (4*256, 4*172), '#edf1f5')
    draw = ImageDraw.Draw(canvas)
    for tile, index in enumerate(indices):
        rgb = frames[int(index)]
        height, width = rgb.shape[:2]
        # Same fixed normalized rectangle for every source/identity/time.
        crop = Image.fromarray(rgb).crop((int(.30*width), int(.27*height), int(.70*width), int(.62*height)))
        crop.thumbnail((256, 145))
        x, y = tile%4*256, tile//4*172
        canvas.paste(crop, (x+(256-crop.width)//2, y+20))
        draw.text((x+5, y+2), f'frame {index:03d} / {index/FPS:.2f}s', fill='#213448')
    canvas.save(path, quality=90)


def save_plot(path, baseline, baseline_valid, fresh):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(len(baseline))/FPS
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout='constrained')
    mask = lambda x, valid: np.where(valid, x, np.nan)
    for j, name in enumerate(GROUP_NAMES):
        axes[0].plot(t, mask(baseline[:, UPPER][:, GROUPS[j]].mean(1), baseline_valid), label=name, lw=1)
        axes[1].plot(t, mask(fresh['coeffs'][:, UPPER][:, GROUPS[j]].mean(1), fresh['valid']), label=name, lw=1)
    for j, name in enumerate(PROXY_NAMES):
        axes[2].plot(t, fresh['geometry'][:, j], label=name, lw=1)
    for j, name in enumerate(('yaw', 'pitch', 'roll')):
        axes[3].plot(t, fresh['head_pose'][:, j], label=name, lw=1)
    for ax, title in zip(axes, ('Historical final coefficients', 'Fresh VIDEO raw coefficients',
                               'Same-model image geometry / eye width (not independent GT)', 'Same-model head rotation degrees')):
        ax.set_ylabel(title, fontsize=8); ax.grid(alpha=.2); ax.legend(fontsize=7, loc='upper right', ncol=2)
    axes[-1].set_xlabel('Native seconds, no lag/amplitude fitting; missing geometry stays blank')
    fig.savefig(path); plt.close(fig)


def run(args):
    started = time.monotonic()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Fresh output directory required; existing data are never overwritten')
    selection = read_selection(args.selection)
    thresholds_path = Path(args.activity_thresholds)
    threshold_record = json.loads(thresholds_path.read_text(encoding='utf8'))
    thresholds = np.asarray(threshold_record['thresholds'], dtype=np.float64)
    if thresholds.shape != (4,) or not np.isfinite(thresholds).all() or (thresholds < 1e-6).any():
        raise ValueError('Four saved positive fitted activity thresholds required')
    import yaml
    source_root = Path(args.source_root)
    config = yaml.safe_load((source_root/'configs/default.yaml').read_text(encoding='utf8'))
    confidence_keys = ('min_face_detection_confidence', 'min_face_presence_confidence', 'min_tracking_confidence')
    if any(config['landmarker'][key] != .5 for key in confidence_keys):
        raise ValueError('Upstream confidence options differ from pinned .5/.5/.5 audit protocol')
    if config['media']['fps'] != FPS:
        raise ValueError('Upstream native FPS differs from pinned25Hz')
    model, ffmpeg = Path(args.model), Path(args.ffmpeg)
    if not model.is_file() or not ffmpeg.is_file():
        raise FileNotFoundError('Explicit local model and ffmpeg binary required')
    historical = {}
    for row in selection['clips']:
        raw, rv = load_coefficients(row['raw_npz']); final, fv = load_coefficients(row['final_npz'], inherited_valid=rv)
        if len(raw) != len(final):
            raise ValueError('Historical raw/final frame count differs: '+row['clip_id'])
        historical[row['clip_id']] = (raw, rv, final, fv)
    import mediapipe as mp
    import cv2
    output.mkdir(parents=True)
    (output/'arrays').mkdir(); (output/'visual').mkdir()
    shutil.copyfile(args.selection, output/'selection.json')
    shutil.copyfile(thresholds_path, output/'activity_thresholds.json')
    shutil.copyfile(__file__, output/'source_auditor.py')
    provenance = {'schema': SCHEMA, 'selection_sha256': sha(args.selection), 'model_sha256': sha(model),
        'ffmpeg_sha256': sha(ffmpeg), 'mediapipe_version': mp.__version__, 'opencv_version': cv2.__version__,
        'activity_thresholds_sha256': sha(thresholds_path), 'activity_thresholds': thresholds.tolist(),
        'activity_thresholds_refitted': False,
        'fps': FPS, 'maximum_clips': MAX_CLIPS, 'maximum_frames_per_clip': MAX_FRAMES,
        'options': {'mode': 'VIDEO', 'num_faces': 1, 'detection': .5, 'presence': .5, 'tracking': .5},
        'arms': ['fresh_forward', 'reused_forward', 'fresh_reverse', 'reused_reverse'],
        'fresh_vs_reused_same_absolute_timestamps': True, 'source_files_modified': False,
        'same_model_landmark_proxy_is_independent_ground_truth': False, 'geometry_names': list(PROXY_NAMES),
        'geometry_indices': {'right': {'corners': [33,133], 'brow': [63,105,66], 'top': [160,159,158], 'bottom': [144,145,153]},
                             'left': {'corners': [362,263], 'brow': [296,334,293], 'top': [385,386,387], 'bottom': [380,374,373]}},
        'landmark_geometry_limit': 'Image-plane distances remove scale/roll but retain out-of-plane pose; same model as blendshapes',
        'history_comparison_limit': 'Historical raw comparison also changes unknown worker context/version; final additionally includes SG5/clamp',
        'historical_final_validity': 'Final NPZ has no mask; paired historical raw detection mask inherited, not interpolated support',
        'sampling': '16 uniform native-frame indices per locked clip, never selected by motion or predictions',
        'code_sha256': sha(__file__)}
    source_files = ['scripts/03_extract_blendshapes.py', 'configs/default.yaml', 'kinetalk_data/arkit.py', 'kinetalk_data/media.py']
    provenance['historical_source_sha256'] = {name: sha(source_root/name) for name in source_files}
    write_json(output/'provenance.json', provenance)
    records, canonical_hashes = {}, {}
    try:
        for order_name, clip_rows in (('forward', selection['clips']), ('reverse', selection['clips'][::-1])):
            base_timestamp = 0
            reused = make_landmarker(model)
            try:
                for row in clip_rows:
                    cid = row['clip_id']; n = len(historical[cid][0])
                    indices = np.rint(np.linspace(0, n-1, 16)).astype(int)
                    captured, fresh_rows, reused_rows, hashes, timestamps = {}, [], [], [], []
                    fresh = make_landmarker(model)
                    frames = video_frames(row['video'], row['needs_resample'], ffmpeg)
                    try:
                        for index, rgb in enumerate(frames):
                            if index >= MAX_FRAMES or index >= n:
                                raise ValueError('Decoded frames exceed pinned bounded native clock: '+cid)
                            ts = base_timestamp+index*40
                            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                            # Within each order both instances receive identical
                            # absolute times and identical decoded pixel arrays.
                            fresh_rows.append(observation(fresh.detect_for_video(image, ts), rgb.shape[1], rgb.shape[0]))
                            reused_rows.append(observation(reused.detect_for_video(image, ts), rgb.shape[1], rgb.shape[0]))
                            timestamps.append(ts); hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                            if order_name == 'forward' and index in indices:
                                captured[index] = rgb.copy()
                    finally:
                        frames.close(); fresh.close()
                    if len(fresh_rows) != n:
                        raise ValueError('Decoded count differs from historical native clock: '+cid)
                    if cid in canonical_hashes and canonical_hashes[cid] != hashes:
                        raise RuntimeError('Forward/reverse decoded RGB hashes differ: '+cid)
                    canonical_hashes[cid] = hashes
                    for arm, values in (('fresh', fresh_rows), ('reused', reused_rows)):
                        payload = stack_observations(values, timestamps, hashes)
                        np.savez_compressed(output/'arrays'/f'{cid}_{arm}_{order_name}.npz', **payload)
                    if order_name == 'forward':
                        save_contact(output/'visual'/f'{cid}_contact.jpg', captured, indices)
                        save_plot(output/'visual'/f'{cid}_curves.png', historical[cid][2], historical[cid][3],
                                  stack_observations(fresh_rows, timestamps, hashes))
                    base_timestamp += n*40+1  # Exactly historical worker clock rule.
                    write_json(output/'status.json', {'schema': SCHEMA, 'status': 'running', 'order': order_name, 'last_clip': cid})
                    print('TRACKED', order_name, cid, n, flush=True)
            finally:
                reused.close()
        for row in selection['clips']:
            cid = row['clip_id']; raw, rv, final, fv = historical[cid]
            arrays = {}
            for order in ('forward', 'reverse'):
                for arm in ('fresh', 'reused'):
                    key = arm+'_'+order
                    with np.load(output/'arrays'/f'{cid}_{key}.npz', allow_pickle=False) as z:
                        arrays[key] = {name: z[name] for name in z.files}
            report = {'metadata': {key: value for key, value in row.items() if key not in ('raw_npz', 'final_npz', 'video')},
                      'frames': len(raw), 'same_decoded_frames_all_arms': True, 'comparisons': {},
                      'fresh_detection_rate': float(arrays['fresh_forward']['valid'].mean())}
            for name, a, b in [('reset_forward', 'fresh_forward', 'reused_forward'), ('reset_reverse', 'fresh_reverse', 'reused_reverse'),
                               ('fresh_order_timestamp_control', 'fresh_forward', 'fresh_reverse'),
                               ('reused_order', 'reused_forward', 'reused_reverse')]:
                left, right = arrays[a], arrays[b]
                report['comparisons'][name] = comparison(left['coeffs'], left['valid'], right['coeffs'], right['valid'], thresholds)
            fresh = arrays['fresh_forward']
            report['comparisons']['historical_raw_vs_fresh'] = comparison(raw, rv, fresh['coeffs'], fresh['valid'], thresholds)
            report['comparisons']['historical_final_vs_fresh_raw_processing_differs'] = comparison(final, fv, fresh['coeffs'], fresh['valid'], thresholds)
            report['coupling_historical_final_to_fresh_geometry'] = coupling(final, fv & fresh['valid'], fresh['geometry'], fresh['head_pose'])
            report['coupling_fresh_coefficients_to_same_geometry'] = coupling(fresh['coeffs'], fresh['valid'], fresh['geometry'], fresh['head_pose'])
            reset = [report['comparisons'][name] for name in ('reset_forward', 'reset_reverse')]
            report['reset_warrants_further_investigation'] = any(
                (r['overall'] is not None and (r['overall']['p95_abs'] > .02 or r['overall']['max_abs'] > .1))
                or (r['activity_label_disagreement'] is not None and r['activity_label_disagreement'] > .05) for r in reset)
            report['no_shared_valid_support'] = any(r['overall'] is None for r in reset)
            records[cid] = report
        read_selection(args.selection)  # Re-hash all read-only sources after the run.
        if sha(model) != provenance['model_sha256'] or sha(ffmpeg) != provenance['ffmpeg_sha256']:
            raise RuntimeError('Model/decoder changed during diagnostic')
        summary = {'schema': SCHEMA, 'clips': records, 'clip_count': len(records), 'seconds': time.monotonic()-started,
                   'scope': 'Selected fit clips only; reset/order sensitivity and same-model geometry associations',
                   'independent_ground_truth': False, 'training_started': False, 'default_changed': False,
                   'provenance_sha256': sha(output/'provenance.json'),
                   'clips_warranting_reset_investigation': [cid for cid, r in records.items() if r['reset_warrants_further_investigation']],
                   'clips_without_shared_support': [cid for cid, r in records.items() if r['no_shared_valid_support']],
                   'decision_scope': 'Per-clip thresholds highlight reset/order sensitivity; no label correctness or causal training-failure conclusion'}
        write_json(output/'summary.json', summary)
        write_json(output/'status.json', {'schema': SCHEMA, 'status': 'complete', 'clips': len(records), 'seconds': summary['seconds']})
        write_json(output/'manifest.json', {p.relative_to(output).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in output.rglob('*') if p.is_file() and p.name != 'manifest.json'})
        return summary
    except Exception as exc:
        write_json(output/'status.json', {'schema': SCHEMA, 'status': 'failed', 'error': repr(exc), 'seconds': time.monotonic()-started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--activity-thresholds', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path('D:/实验室项目/新实验/数据集'))
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({'clips': result['clip_count'], 'seconds': result['seconds'], 'output': str(args.output.resolve())}), flush=True)


if __name__ == '__main__':
    main()
