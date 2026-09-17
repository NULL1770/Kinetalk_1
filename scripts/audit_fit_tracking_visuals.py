"""Read only preselected current-fit video/coefficients; audit upstream tracking.

No model training, alignment fitting, tracker rerun, or held-out motion reads.
Selection is fixed from allowlist metadata before any video/coefficients load.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter


REGIONS = {'brows': [41, 42, 43, 44, 45], 'eyes_expression': [5, 6, 12, 13], 'blink': [0, 7]}
NAMES = ['browDownL', 'browDownR', 'browInnerUp', 'browOuterUpL', 'browOuterUpR',
         'eyeSquintL', 'eyeWideL', 'eyeSquintR', 'eyeWideR']
EMOTIONS = (0, 1, 5, 6)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def choose(allowlist):
    rows = json.loads(allowlist.read_text(encoding='utf-8-sig'))['clips']
    if len(rows) != 2315 or len({row['clip_id'] for row in rows}) != len(rows):
        raise ValueError('Expected current locked 2315 unique fit clips')
    speakers = sorted({row['speaker'] for row in rows})[:2]
    chosen = [min((row for row in rows if row['speaker'] == speaker and row['emotion_id'] == emotion),
                  key=lambda row: row['clip_id']) for speaker in speakers for emotion in EMOTIONS]
    return {'selection_rule': 'First 2 lexicographic current-fit speakers; each neutral/angry/happy/sad smallest clip_id; fixed before target/video reading',
            'fit_allowlist_sha256': sha(allowlist), 'clips': chosen}


def metadata_subset(path, selected):
    wanted = {row['clip_id'] for row in selected}
    found = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        row = json.loads(line)
        if row['clip_id'] in wanted:
            found[row['clip_id']] = row
    if found.keys() != wanted:
        raise ValueError(f'Missing selected metadata: {wanted - found.keys()}')
    return found


def manifest_subset(path, wanted):
    found = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        row = json.loads(line)
        if row['clip_id'] in wanted:
            found[row['clip_id']] = row
    return found


def load_source_module(path):
    spec = importlib.util.spec_from_file_location('tracking_audit_arkit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def static_audit(source_root, ffmpeg):
    paths = [source_root / 'scripts' / name for name in ('02_preprocess_media.py', '03_extract_blendshapes.py', '05_postprocess_coeffs.py')]
    paths += [source_root / 'kinetalk_data' / name for name in ('arkit.py', 'media.py')]
    paths += [source_root / 'configs/default.yaml']
    arkit = load_source_module(source_root / 'kinetalk_data/arkit.py')
    perm = np.random.default_rng(814).permutation(52)
    names = [arkit.MEDIAPIPE_52[i] for i in perm]
    indices, missing = arkit.build_gather_index(names)
    result = arkit.gather_to_arkit(perm.astype(np.float32)[None], indices)
    expected = np.asarray([i if i is not None else 0 for i in arkit.ARKIT_FROM_MEDIAPIPE], dtype=np.float32)[None]
    if not np.array_equal(result, expected) or missing != ['TongueOut']:
        raise ValueError('Name-based upstream channel mapping fails permutation test')
    render_source = Path('scripts/render_dynamic_rig_comparison.py').read_text(encoding='utf-8')
    import ast
    assigned = next(node for node in ast.parse(render_source).body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'ARKIT_NAMES' for target in node.targets))
    render_names = ast.literal_eval(assigned.value)
    if [name.lower() for name in render_names] != [name.lower() for name in arkit.ARKIT_52]:
        raise ValueError('Upstream and current render channel orders differ')
    version = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    return {'source_hashes': {str(path): sha(path) for path in paths},
            'ffmpeg_version': version, 'ffmpeg_path': ffmpeg,
            'permuted_category_name_mapping_passed': True, 'current_render_order_matches_upstream': True,
            'upper_mapping': [{'arkit_index': i, 'arkit_name': arkit.ARKIT_52[i], 'mediapipe_index': arkit.ARKIT_FROM_MEDIAPIPE[i]}
                              for i in REGIONS['brows'] + REGIONS['eyes_expression']],
            'implemented_sampling': 'keep_normalized_video=false -> source decoded by ffmpeg fps=25 -> MediaPipe VIDEO timestamps idx*40ms',
            'implemented_postprocess': 'invalid-frame linear interpolation; Savitzky-Golay window=5,poly=2,axis=time,mode=interp; clip[0,1]; no per-speaker resting subtraction',
            'tracking_caveat': 'MediaPipe VIDEO mode worker reuses tracker across clips and advances timestamps, rather than reinitializing per clip; effect not established by this audit.'}


def frame_clock(video, ffmpeg, output):
    command = [ffmpeg, '-hide_banner', '-loglevel', 'info', '-i', str(video), '-an',
               '-vf', 'showinfo@source,fps=25,showinfo@sample', '-f', 'null', '-']
    run = subprocess.run(command, capture_output=True, text=True, check=True, timeout=90)
    (output / 'ffmpeg_frame_clock.log').write_text(run.stderr, encoding='utf-8')
    pattern = re.compile(r'showinfo@(source|sample).*?\bn:\s*(\d+).*?pts_time:([\d.eE+-]+).*?checksum:([A-F0-9]+)')
    rows = {'source': [], 'sample': []}
    for line in run.stderr.splitlines():
        match = pattern.search(line)
        if match:
            kind, index, pts, checksum = match.groups()
            rows[kind].append({'index': int(index), 'pts_s': float(pts), 'checksum': checksum})
    if not rows['source'] or not rows['sample']:
        raise ValueError('No decoded source/sample timestamps')
    checksums = {}
    for row in rows['source']:
        checksums.setdefault(row['checksum'], []).append(row)
    mapping = []
    for row in rows['sample']:
        candidates = checksums.get(row['checksum'], [])
        if not candidates:
            raise ValueError('fps filter produced frame not matching a decoded source checksum')
        source = min(candidates, key=lambda candidate: abs(candidate['pts_s'] - row['pts_s']))
        mapping.append({'sample_index': row['index'], 'sample_pts_s': row['pts_s'], 'source_index': source['index'],
                        'source_pts_s': source['pts_s'], 'source_minus_sample_s': source['pts_s'] - row['pts_s'],
                        'checksum_matches': len(candidates)})
    sample_pts = np.asarray([row['sample_pts_s'] for row in mapping])
    if not np.allclose(sample_pts, np.arange(len(mapping)) / 25, rtol=0, atol=1e-6):
        raise ValueError('fps output is not the declared 25Hz clock')
    delta = [row['source_minus_sample_s'] for row in mapping]
    result = {'source_frames': len(rows['source']), 'sample_frames': len(mapping),
              'source_median_interval_s': float(np.median(np.diff([row['pts_s'] for row in rows['source']]))),
              'source_minus_sample_range_s': [min(delta), max(delta)], 'mapping': mapping,
              'scope': 'Re-executed historical configured ffmpeg fps=25 and matched decoded checksums; original extraction did not save per-frame PTS.'}
    save(output / 'frame_clock.json', result)
    return result


def statistics(raw, final, valid, indices):
    x, y = raw[valid][:, indices].astype(float), final[valid][:, indices].astype(float)
    xc, yc = x - x.mean(0), y - y.mean(0)
    adjacent = valid[:-1] & valid[1:]
    dx, dy = np.diff(raw[:, indices], axis=0)[adjacent], np.diff(final[:, indices], axis=0)[adjacent]
    raw_energy, final_energy = float(np.square(xc).sum()), float(np.square(yc).sum())
    corr = float((xc * yc).sum() / max((raw_energy * final_energy) ** .5, 1e-12))
    return {'raw_temporal_rms': float(np.sqrt(np.mean(xc ** 2))), 'final_temporal_rms': float(np.sqrt(np.mean(yc ** 2))),
            'temporal_rms_retention': float(np.sqrt(final_energy / max(raw_energy, 1e-12))),
            'temporal_energy_retention': final_energy / max(raw_energy, 1e-12), 'raw_final_centered_correlation': corr,
            'frame_displacement_rms_retention': float(np.sqrt(np.mean(dy ** 2) / max(np.mean(dx ** 2), 1e-12))),
            'observed_raw_minmax': [float(x.min()), float(x.max())], 'observed_final_minmax': [float(y.min()), float(y.max())],
            'raw_centered_energy': raw_energy, 'final_centered_energy': final_energy,
            'raw_velocity_energy': float(np.square(dx).sum()), 'final_velocity_energy': float(np.square(dy).sum()),
            'observed_channel_samples': int(x.size), 'observed_adjacent_channel_samples': int(dx.size)}


def source_crops(video, indices):
    cap = cv2.VideoCapture(str(video)); frames = {}; wanted = set(indices)
    index = 0
    while index <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Failed source frame {index}')
        if index in wanted:
            height, width = frame.shape[:2]
            # One fixed normalized rectangle across both speakers/all clips.
            frames[index] = cv2.cvtColor(frame[int(height*.31):int(height*.64), int(width*.30):int(width*.67)], cv2.COLOR_BGR2RGB)
        index += 1
    cap.release()
    return [frames[index] for index in indices]


def make_plot(cid, raw, final, valid, frame_ids, crops, mapping, out, crop_start, crop_stop, train_valid):
    times = np.arange(len(raw)) / 25
    fig = plt.figure(figsize=(20, 10)); grid = fig.add_gridspec(5, 8, height_ratios=[1.4, 1, 1, 1, 1], hspace=.65, wspace=.04)
    for j, (frame, crop) in enumerate(zip(frame_ids, crops)):
        ax = fig.add_subplot(grid[0, j]); ax.imshow(crop); ax.axis('off')
        unobserved = '\nnot observed in training' if train_valid is not None and not train_valid[frame] else ''
        ax.set_title(f'{frame/25:.2f}s / source {mapping[frame]["source_pts_s"]:.3f}s' + unobserved, fontsize=8)
    for row, (region, centered) in enumerate((('brows', False), ('brows', True), ('eyes_expression', False), ('eyes_expression', True)), 1):
        ax = fig.add_subplot(grid[row, :]); indices = REGIONS[region]
        for j, ch in enumerate(indices):
            a, b = raw[:, ch].astype(float), final[:, ch].astype(float)
            if centered:
                a, b = a-a[valid].mean(), b-b[valid].mean()
            label = NAMES[j if region == 'brows' else j+5]
            color = f'C{j}'
            ax.plot(times, np.where(valid, a, np.nan), color=color, ls=':', lw=1.1, alpha=.7)
            ax.plot(times, np.where(valid, b, np.nan), color=color, lw=1.35, label=label)
        for frame in frame_ids: ax.axvline(frame/25, color='k', lw=.5, alpha=.18)
        if crop_start is not None: ax.axvspan(crop_start/25, crop_stop/25, color='#88aaff', alpha=.08, label='current train crop')
        ax.set_title(f'{region}: {"observed temporal mean removed" if centered else "absolute coefficients"}; dotted=raw, solid=final', fontsize=10)
        ax.legend(ncol=6, fontsize=8, loc='upper right'); ax.grid(alpha=.2); ax.set_xlim(0, times[-1]); ax.set_xlabel('source clock seconds')
        if not centered: ax.set_ylim(-.02, 1.02)
    fig.suptitle(cid + ' | fixed uniform-time source samples; no fitted lag/gain; current-fit only', fontsize=13)
    fig.savefig(out / 'source_and_coefficients.png', dpi=130, bbox_inches='tight'); plt.close(fig)


def inspect_clip(row, metadata, media, extract, args):
    cid = row['clip_id']; out = args.output/cid; out.mkdir(exist_ok=True)
    _, speaker, emotion, level, number = cid.split('_')
    video = args.video_root/speaker/'video/front'/emotion/f'level_{level[1:]}'/f'{number}.mp4'
    paths = {stage: args.data_root/f'processed/coeffs_{stage}/mead'/f'{cid}{".npy" if stage == "smooth" else ".npz"}' for stage in ('raw', 'smooth', 'final')}
    with np.load(paths['raw'], allow_pickle=False) as src: raw, valid = src['coeffs'].copy(), src['valid'].astype(bool)
    smooth = np.load(paths['smooth'], allow_pickle=False)
    with np.load(paths['final'], allow_pickle=False) as src: final = src['coeffs'].copy()
    if raw.shape != final.shape or smooth.shape != raw.shape or raw.shape != (len(valid), 52) or not valid.any():
        raise ValueError(f'Unexpected upstream shapes {cid}')
    interpolated = raw.copy(); good = np.flatnonzero(valid)
    for ch in range(52): interpolated[:, ch] = np.interp(np.arange(len(raw)), good, raw[good, ch])
    reproduced = np.clip(savgol_filter(interpolated, 5, 2, axis=0, mode='interp').astype(np.float32), 0, 1)
    native = args.output/'native'/f'{cid}.npz'; crop_start = crop_stop = None; train_valid = None
    native_info = {'available': native.exists(), 'metadata': metadata, 'metadata_sha_verified': False}
    if native.exists():
        digest = sha(native)
        if digest != metadata['artifact_sha256']: raise ValueError('Current native SHA differs from locked metadata')
        with np.load(native, allow_pickle=False) as src:
            motion, mask, times = src['motion'], src['mask'].astype(bool), src['times']
            provenance = json.loads(str(src['provenance'].item()))
        if not np.array_equal(motion, final): raise ValueError('Current native motion differs from upstream final')
        if provenance['raw_sha256'] != sha(paths['raw']) or provenance['bs_sha256'] != sha(paths['final']):
            raise ValueError('Current native source hashes differ from audited files')
        if not np.allclose(times, np.arange(len(raw))/25, rtol=0, atol=1e-8): raise ValueError('Native time clock differs')
        starts = np.arange(max(1, len(mask)-96+1)); prefix = np.r_[0, np.cumsum(mask)]
        candidates = starts[prefix[np.minimum(starts+96, len(mask))]-prefix[starts] >= 32]
        crop_start = int(candidates[np.argmin(abs(candidates-max(0, len(mask)-96)/2))]); crop_stop = min(crop_start+96, len(mask))
        train_valid = mask.copy()
        train_valid[:crop_start] = False
        train_valid[crop_stop:] = False
        native_info.update(metadata_sha_verified=True, sha256=digest, motion_bitwise_equals_final=True,
                           source_hashes_verified=True, provenance=provenance, valid_frames=int(mask.sum()),
                           train_crop_start=crop_start, train_crop_stop=crop_stop,
                           train_observed_frames=int(train_valid.sum()),
                           train_regions={name:statistics(raw, final, train_valid, indices) for name, indices in REGIONS.items()})
    clock = frame_clock(video, args.ffmpeg, out)
    if len(clock['mapping']) != len(raw): raise ValueError('Reproduced fps count differs from upstream coefficients')
    left, right = (crop_start, crop_stop-1) if crop_start is not None else (0, len(raw)-1)
    frame_ids = np.rint(np.linspace(left, right, 8)).astype(int).tolist()
    source_ids = [clock['mapping'][index]['source_index'] for index in frame_ids]
    crops = source_crops(video, source_ids)
    for index, crop in zip(frame_ids, crops):
        cv2.imwrite(str(out/f'source_sample_{index:04d}.jpg'), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    make_plot(cid, raw, final, valid, frame_ids, crops, clock['mapping'], out, crop_start, crop_stop, train_valid)
    result = {'clip_id': cid, 'scope': 'current fit only', 'paths': {**{stage:str(path) for stage,path in paths.items()}, 'video':str(video)},
              'hashes': {**{stage:sha(path) for stage,path in paths.items()}, 'video':sha(video)}, 'frames':len(raw),
              'raw_valid_frames':int(valid.sum()), 'raw_invalid_indices':np.flatnonzero(~valid).tolist(),
              'smooth_final_max_abs_difference':float(np.max(abs(smooth-final))),
              'sg5_poly2_reproduction_max_abs_difference':float(np.max(abs(reproduced-final))),
              'native':native_info, 'media_manifest':media, 'extract_manifest':extract,
              'regions':{name:statistics(raw, final, valid, indices) for name, indices in REGIONS.items()},
              'clock':{key:value for key,value in clock.items() if key!='mapping'}, 'sample_frame_ids':frame_ids,
              'sample_source_frame_ids':source_ids, 'sample_raw_valid':[bool(valid[i]) for i in frame_ids],
              'sample_training_observed':[bool(train_valid[i]) for i in frame_ids] if train_valid is not None else None,
              'crop_rectangle_normalized_xyxy':[.30,.31,.67,.64]}
    save(out/'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allowlist', type=Path, default=Path('artifacts/history_context_20260917/fit_allowlist.json'))
    parser.add_argument('--native-metadata', type=Path, default=Path('artifacts/formal_readiness/native_metadata/train.jsonl'))
    parser.add_argument('--data-root', type=Path, default=Path('D:/kinetalk_data'))
    parser.add_argument('--video-root', type=Path, default=Path('E:/mead'))
    parser.add_argument('--source-root', type=Path, default=Path('D:/实验室项目/新实验/数据集'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/history_context_20260917/tracking'))
    parser.add_argument('--ffmpeg', default='C:/Users/zhh/AppData/Local/com.minimax.hub/current/resources/ffmpeg/ffmpeg.exe')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    selection = choose(args.allowlist); locked = args.output/'selection.json'
    if locked.exists() and json.loads(locked.read_text(encoding='utf-8')) != selection:
        raise ValueError('Existing pre-read selection differs')
    save(locked, selection)
    metadata = metadata_subset(args.native_metadata, selection['clips']); wanted = set(metadata)
    media = manifest_subset(args.data_root/'manifests/02_media.jsonl', wanted)
    extract = manifest_subset(args.data_root/'manifests/03_landmarks.jsonl', wanted)
    static = static_audit(args.source_root, args.ffmpeg)
    results = []
    for row in selection['clips']:
        cid = row['clip_id']; result = inspect_clip(row, metadata[cid], media.get(cid), extract.get(cid), args)
        results.append(result); print(json.dumps({'clip_id':cid, 'native':result['native']['available'], 'regions':result['regions']}), flush=True)
    pooled = {}; training_pooled = {}
    for region in REGIONS:
        energies = {key: sum(row['regions'][region][key] for row in results) for key in
                    ('raw_centered_energy','final_centered_energy','raw_velocity_energy','final_velocity_energy')}
        pooled[region] = {'temporal_rms_retention':(energies['final_centered_energy']/energies['raw_centered_energy'])**.5,
                          'frame_displacement_rms_retention':(energies['final_velocity_energy']/energies['raw_velocity_energy'])**.5, **energies}
        training = [row['native']['train_regions'][region] for row in results if row['native'].get('metadata_sha_verified')]
        if training:
            energies = {key:sum(row[key] for row in training) for key in
                        ('raw_centered_energy','final_centered_energy','raw_velocity_energy','final_velocity_energy')}
            training_pooled[region] = {'clips':len(training),
                                      'temporal_rms_retention':(energies['final_centered_energy']/energies['raw_centered_energy'])**.5,
                                      'frame_displacement_rms_retention':(energies['final_velocity_energy']/energies['raw_velocity_energy'])**.5, **energies}
    summary = {'schema':'fit_tracking_audit_v1', 'selection':selection, 'script_sha256':sha(__file__), 'static':static,
               'clips':results, 'pooled_regions':pooled, 'pooled_current_training_regions':training_pooled, 'test_motion_loaded':False,
               'limits':['Eight deterministic fit examples, one shared sentence; not a dataset prevalence estimate.',
                         'Raw coefficients are tracker pseudo labels, not visual ground truth.',
                         'No tracker re-inference or quantitative independent visual-label validation.',
                         'Native source linkage certified only for available SHA-matched native artifacts.']}
    save(args.output/'summary.json', summary)
    board = Image.new('RGB', (1280, 143*len(results)), 'white'); draw = ImageDraw.Draw(board)
    for row_number, row in enumerate(results):
        cid = row['clip_id']; draw.text((5, row_number*143), cid, fill='black')
        for column, frame in enumerate(row['sample_frame_ids']):
            with Image.open(args.output/cid/f'source_sample_{frame:04d}.jpg') as source:
                source.thumbnail((160, 116)); board.paste(source, (column*160, row_number*143+23))
    board.save(args.output/'all_source_contact_sheet.jpg')
    cards = ''.join(f'<section><h2>{html.escape(row["clip_id"])}</h2><img src="{row["clip_id"]}/source_and_coefficients.png"><p><a href="{row["clip_id"]}/audit.json">audit + hashes</a></p></section>' for row in results)
    (args.output/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Fit tracking audit</title><style>body{font-family:sans-serif;max-width:1800px;margin:auto;padding:24px}img{width:100%}section{margin:36px 0}p{line-height:1.6}</style><h1>原视频与眉眼伪标签核验：固定8条fit样本</h1><p>按元数据预选，不按动作效果挑选。虚线raw，实线SG5/poly2最终系数；原帧依照ffmpeg fps=25精确映射。蓝色区间为可校验的当前96帧训练crop。相机动作、光照和tracker噪声仍须人工辨别。</p>'+cards, encoding='utf-8')
    print(json.dumps({'output':str(args.output.resolve()), 'pooled':pooled}), flush=True)


if __name__ == '__main__':
    main()
