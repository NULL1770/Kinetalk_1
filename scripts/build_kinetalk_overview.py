"""Original evidence-based KineTalk teaser: one clock and a reference intervention.

This composes existing verified renders. It does not synthesize or retouch faces,
rerun predictions, select new examples, or claim a successful Phase58 model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image
from scipy.io import wavfile

INK = '#233348'
GREY = '#667386'
BLUE = '#3a75a5'
TEAL = '#218778'
GOLD = '#b9852d'
CROP = (140, 70, 880, 810)  # One identical square crop, never a per-face warp.
NAME = '04_overview_kinetalk_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_overview(root):
    root = Path(root).resolve()
    repo = Path(__file__).resolve().parents[1]
    out = root/'figures'
    out.mkdir(exist_ok=True)
    manifest = json.loads((root/'stills_manifest.json').read_text())
    registry = {r['name']: r for r in manifest['stills']}
    receipt = json.loads((root/'stills_transparent/render_receipt.json').read_text())
    render_hashes = {r['name']: r['image_sha256'] for r in receipt['stills']}
    metadata = json.loads((root/'data/metadata.json').read_text())
    source = root/'data/comparison_happy.npz'
    assert sha(source) == metadata['display']['happy']['comparison_sha256']
    z = np.load(source, allow_pickle=False)
    cid = str(z['clip_id'].item())
    assert cid == 'mead_M025_happy_L3_005'
    assert list(z['mode_names'])[:2] == ['GT', 'Neutral B0']
    assert str(z['mode_names'][5]) == 'KineTalk Phase53 joint'
    frames = [registry[f'happy_sequence_{p}_m1']['frame'] for p in (20, 40, 60, 80)]
    times = z['times']
    selected_times = times[frames]
    assert np.all(z['valid'][frames]) and z['channel_mask'][17]
    images, sources = {}, {}
    wanted = [f'happy_sequence_{p}_m{m}' for p in (20, 40, 60, 80) for m in (1, 5)]
    wanted += [f'style_happy_m{m}' for m in (1, 2, 3)]
    style_source = repo/'final_experiment/evaluation/diagnostics/phase53_receiver_20261009/joint/style_audit/render_inputs/candidate_happy.npz'
    for name in wanted:
        row = registry[name]
        path = root/'stills_transparent'/(name+'.png')
        assert sha(path) == render_hashes[name]
        assert row['clip_id'] == cid
        expected_source = style_source if name.startswith('style_') else source
        assert row['source_sha256'] == sha(expected_source)
        im = Image.open(path).convert('RGBA')
        assert im.size == (1024, 1024)
        images[name] = im.crop(CROP)
        sources[name] = dict(image_sha256=render_hashes[name], source=str(expected_source),
            source_sha256=sha(expected_source), frame=row['frame'], time_seconds=row['time_seconds'], method=row['method'])
    style_time = registry['style_happy_m1']['time_seconds']
    assert all(registry[f'style_happy_m{m}']['time_seconds'] == style_time for m in (1, 2, 3))
    audio_path = repo/'final_experiment/evaluation/render_inputs'/f'{cid}.wav'
    rate, wave = wavfile.read(audio_path)
    wave = wave.astype(np.float64)
    if wave.ndim == 2:
        wave = wave.mean(1)
    start, stop = int(round(times[0]*rate)), int(round((times[-1]+.04)*rate))
    assert 0 <= start < stop <= len(wave)
    pieces = np.array_split(wave[start:stop], 360)
    envelope = np.array([np.sqrt(np.mean(v*v)) for v in pieces])
    envelope /= max(float(envelope.max()), 1e-12)
    audio_times = np.linspace(times[0], times[-1]+.04, len(envelope), endpoint=False)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(15.2, 8.48))
    ax.set(xlim=(0, 19), ylim=(0, 10.6), aspect='equal')
    ax.axis('off')
    # Minimal page furniture. Faces and native time, rather than a large
    # generic network box, carry the figure's central argument.
    ax.text(.3, 10.2, 'KineTalk', fontsize=26, fontweight='bold', color=INK)
    ax.text(2.75, 10.2, 'Speech motion from a neutral scaffold', fontsize=23, color=INK)
    ax.text(.32, 9.74, 'Neutral articulation', fontsize=13, color=BLUE)
    ax.text(3.5, 9.74, 'Audio-inferred expression', fontsize=13, color=TEAL)
    ax.text(7.68, 9.74, 'Reference motion style', fontsize=13, color=GOLD)
    ax.plot([.3, 18.7], [9.49, 9.49], color='#dce3e8', lw=.9)

    ax.text(.32, 9.09, '(a) One utterance, one native clock', fontsize=14, color=INK, fontweight='bold')
    ax.text(13.75, 9.09, '(b) Change the reference', fontsize=14, color=INK, fontweight='bold')
    scale = (10.6-3.7)/(selected_times[-1]-selected_times[0])
    origin = 3.7-scale*selected_times[0]
    tx = lambda values: origin+scale*np.asarray(values)
    centers = tx(selected_times)
    ax.text(.33, 8.47, 'Query\naudio', fontsize=12, va='center', color=GREY)
    ax.vlines(tx(audio_times), 8.48-.23*envelope, 8.48+.23*envelope, color=GREY, lw=.8)
    for x, time in zip(centers, selected_times):
        ax.plot([x, x], [2.1, 8.18], color='#e5e9ee', lw=.8, zorder=0)
        ax.plot(x, 8.18, 'o', color=GREY, ms=3)
        ax.text(x, 7.88, f'{time:.2f} s', ha='center', fontsize=12, color=GREY)

    def face(name, x, y, size):
        ax.imshow(images[name], extent=(x-size/2, x+size/2, y-size/2, y+size/2),
            aspect='equal', interpolation='lanczos', zorder=2)

    def arrow(start, end, color=TEAL, dashed=False, width=1.4):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12,
            lw=width, color=color, linestyle='--' if dashed else '-', zorder=3))

    ax.plot([.33, .33], [5.8, 7.32], color=BLUE, lw=3)
    ax.text(.56, 6.93, 'Neutral B0', fontsize=15, color=BLUE, fontweight='bold')
    ax.text(.56, 6.52, 'Speech scaffold', fontsize=12, color=INK)
    ax.text(.56, 6.07, 'Frozen content path', fontsize=10.5, color=GREY)
    ax.plot([.33, .33], [3.22, 4.75], color=TEAL, lw=3)
    ax.text(.56, 4.38, 'Final motion', fontsize=15, color=TEAL, fontweight='bold')
    ax.text(.56, 3.95, 'Conditioned response', fontsize=11.5, color=INK)
    ax.text(.56, 3.51, r'$B_0 + R_\theta(B_0,g,u,s)$', fontsize=11.5, color=GREY)
    for x, percent in zip(centers, (20, 40, 60, 80)):
        face(f'happy_sequence_{percent}_m1', x, 6.52, 1.95)
        face(f'happy_sequence_{percent}_m5', x, 3.99, 1.95)
        arrow((x, 5.5), (x, 5.02))

    # The conditions are a label on the receiver, not invented feature values
    # or an unverified manually controllable emotion slider.
    ax.text(.57, 5.6, 'g: global affect', fontsize=10.5, color=TEAL, va='center')
    ax.text(.57, 5.14, 'u(t): local expression', fontsize=10.5, color=TEAL, va='center')
    ax.plot([centers[0], centers[-1]], [5.28, 5.28], color=TEAL, lw=.8, alpha=.65)
    arrow((2.96, 5.28), (centers[0], 5.28), TEAL, width=.9)
    arrow((12.86, 5.28), (centers[-1], 5.28), GOLD, width=.9)
    ax.text(13.02, 5.62, 'Reference  s', fontsize=10.5, color=GOLD, ha='right',
        bbox=dict(facecolor='white', edgecolor='none', pad=2))

    # Actual jaw trajectories, same time transform as the four portraits.
    pred = np.clip(z['motions'][5, :, 17], 0., 1.)
    base = np.clip(z['motions'][1, :, 17], 0., 1.)
    valid = z['valid'] & np.isfinite(pred) & np.isfinite(base)
    maximum = max(.1, float(max(base[valid].max(), pred[valid].max())))
    top = np.ceil(maximum/.1)*.1
    y0, height = 1.96, .82
    ax.text(.32, 2.98, 'Jaw opening', fontsize=11, color=GREY)
    ax.plot([tx(times[0]), tx(times[-1])], [y0, y0], color='#d5dce2', lw=.7)
    for tick in (0., top):
        yy = y0+tick/top*height
        ax.text(tx(times[0])-.12, yy, f'{tick:.1f}', fontsize=9, color=GREY, ha='right', va='center')
    for value, color, label in ((base, BLUE, 'B0'), (pred, TEAL, 'Final')):
        v = np.where(valid, value, np.nan)
        ax.plot(tx(times), y0+v/top*height, lw=1.45, color=color, label=label)
    ax.plot([10.2, 10.6], [2.96, 2.96], color=BLUE, lw=1.5)
    ax.text(10.71, 2.96, 'B0', fontsize=10, va='center', color=BLUE)
    ax.plot([11.4, 11.8], [2.96, 2.96], color=TEAL, lw=1.5)
    ax.text(11.91, 2.96, 'Final', fontsize=10, va='center', color=TEAL)
    ax.text(6.9, 1.67, 'Original frame times; no time warping', fontsize=10.5, color=GREY, ha='center')

    ax.add_patch(FancyBboxPatch((13.55, 1.63), 5.17, 7.15,
        boxstyle='round,pad=0.015,rounding_size=.12', facecolor='#fcf9f1', edgecolor='#e5d8b8', lw=.9, zorder=0))
    ax.text(16.1, 8.36, f'Audio + g, u fixed  |  t = {style_time:.2f} s',
        fontsize=11.4, ha='center', color=INK)
    for y, mode, speaker in zip((7.09, 5.08, 3.07), (1, 2, 3), ('M025', 'M037', 'M039')):
        ax.add_patch(FancyBboxPatch((13.88, y-.37), 1.57, .78,
            boxstyle='round,pad=0.025,rounding_size=.09', fc='white', ec=GOLD, lw=.9))
        ax.text(14.665, y+.11, speaker, color=GOLD, fontsize=13, ha='center', fontweight='bold')
        ax.text(14.665, y-.17, '2 neutral clips', color=GREY, fontsize=9.3, ha='center')
        arrow((15.54, y), (16.24, y), GOLD)
        face(f'style_happy_m{mode}', 17.39, y, 1.94)
    ax.text(16.15, 1.83, 'Shared geometry; reference-driven motion', ha='center', fontsize=10.3, color=GREY)

    ax.plot([.3, 18.7], [1.28, 1.28], color='#dce3e8', lw=.9)
    ax.text(.33, .88, 'TRAINING ONLY', color='#80728e', fontsize=10, fontweight='bold', va='center')
    ax.text(2.52, .88, 'Motion posterior q', color=INK, fontsize=11.5, va='center')
    arrow((5.12, .88), (5.62, .88), '#80728e', True)
    ax.text(5.85, .88, 'Distribution matching', color='#80728e', fontsize=11, va='center')
    arrow((8.92, .88), (9.5, .88), '#80728e', True)
    ax.text(9.7, .88, 'Audio prior p', color=INK, fontsize=11.5, va='center')
    ax.text(13.75, .94, 'Inference: audio + independent neutral references', fontsize=9.9, color=GREY, va='center')
    ax.text(.33, .27, 'Phase53 joint  |  Same query throughout: MEAD M025 happy L3 005', color=GREY, fontsize=9.6)
    ax.text(18.7, .27, 'Actual predictions, common rig and camera', color=GREY, fontsize=9.6, ha='right')
    fig.subplots_adjust(left=.015, right=.995, bottom=.02, top=.995)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out/(NAME+'.'+ext), dpi=300, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    trace_file = out/(NAME+'_traces.npz')
    np.savez_compressed(trace_file, times=times, valid=valid, jaw_base=base, jaw_final=pred,
        display_frames=np.asarray(frames), display_times=selected_times)
    result = dict(schema='kinetalk_original_overview_v2', model='Phase53 joint', clip_id=cid,
        plot_source_sha256=sha(source), audio_sha256=sha(audio_path), checkpoint_sha256=metadata['checkpoint_sha256'],
        selected_native_times=selected_times.tolist(), style_native_time=style_time,
        crop_xyxy=list(CROP), same_crop_for_all_faces=True, retouching=False, new_predictions=False,
        design='Shared-clock neutral/final storyboard plus fixed-query reference intervention; no external figure template',
        source_stills=sources, files={p.name:sha(p) for p in [*(out/(NAME+'.'+e) for e in ('png','pdf','svg')), trace_file]},
        claims_not_established=['complete disentanglement','phonetic timing correctness','dynamic-u accuracy','SOTA'],
        phase58_used=False)
    (out/(NAME+'_receipt.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(output=str(out/NAME), stills=len(sources), times=selected_times.tolist()), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    build_overview(p.parse_args().root)
