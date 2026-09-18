"""Package fixed sparse teacher diagnostics and exactly five native pixels/event.

No fitting, teacher modification, or independent-truth claim. Every decoded RGB
frame is checked against the previously audited fresh-forward frame hash.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

SCHEMA = 'sparse_brow_teacher_review_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def checked(path, record):
    if not Path(path).is_file() or Path(path).stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        raise ValueError('Source hash/size mismatch: '+str(path))


def href(path, output):
    return os.path.relpath(Path(path).resolve(), Path(output).resolve()).replace('\\', '/')


def selected_frames(video, frame_ids, width, height, ffmpeg):
    """Decode only selected output frames, using exactly the audited fps filter."""
    ids = sorted(set(map(int, frame_ids)))
    expr = '+'.join(f'eq(n\\,{i})' for i in ids)
    command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-i', str(video),
        '-vf', 'fps=25,select='+expr, '-vsync', '0', '-an', '-sn', '-dn',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[-1000:])
    count = len(ids); size = width*height*3
    if len(result.stdout) != count*size:
        raise ValueError(f'Expected {count} complete RGB frames, got bytes={len(result.stdout)}')
    return {i: np.frombuffer(result.stdout[n*size:(n+1)*size], np.uint8).reshape(height, width, 3)
            for n, i in enumerate(ids)}


def chart(clip, arrays, destination):
    raw, smooth = arrays['raw5'], arrays['smooth5']
    valid = arrays['valid']; time = np.arange(len(raw))/25
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True)
    for ax, (group, channels) in zip(axes, [('raise', [2, 3, 4]), ('down', [0, 1])]):
        for c in clip['candidates']:
            if c['group'] != group: continue
            if not c['accepted']:
                ax.axvspan(c['start']/25, c['end']/25, color='#b8bec9', alpha=.08)
                if c['left_censored'] or c['right_censored']:
                    ax.plot([c['start']/25, c['end']/25], [.985, .985], transform=ax.get_xaxis_transform(), color='#ab5560', alpha=.23)
        yraw, y = raw[:, channels].mean(1), smooth[:, channels].mean(1)
        yraw = np.where(valid, yraw, np.nan); y = np.where(valid, y, np.nan)
        ax.plot(time, yraw, color='#79879a', alpha=.7, lw=.8, label='fresh raw group mean')
        ax.plot(time, y, color='#183f6d', lw=1.3, label='SG5/2 within valid run')
        for index, e in enumerate(clip['events']):
            if not any(m['group'] == group for m in e['groups']): continue
            ax.axvspan(e['start']/25, e['end']/25, color='#20a57b', alpha=.2)
            ax.plot(e['peak']/25, y[e['peak']], marker='o', ms=4, color='#d87920')
            ax.text(e['peak']/25, y[e['peak']]+.004, f'E{index+1}', fontsize=8)
        ax.set(ylabel=group+' coefficient', ylim=(-.01, 1.01))
        ax.grid(alpha=.2); ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Native time (seconds, 25 Hz); fixed coefficient axis [0,1]')
    fig.suptitle(clip['source_id']+' | green=accepted; grey=unsupported; top red=censored/support missing', fontsize=10)
    fig.tight_layout(); fig.savefig(destination, dpi=130); plt.close(fig)


def make_strip(frames, ids, labels, box, destination):
    left, top, right, bottom = box
    tile_w = 400
    tile_h = round((bottom-top)*tile_w/(right-left))
    canvas = Image.new('RGB', (tile_w*5, tile_h+42), '#111827')
    draw = ImageDraw.Draw(canvas)
    for n, (frame_id, label) in enumerate(zip(ids, labels)):
        cropped = Image.fromarray(frames[frame_id]).crop(box)
        cropped.resize((tile_w, tile_h), Image.Resampling.LANCZOS).save(destination.parent/f'{destination.stem}_{n}.png')
        canvas.paste(cropped.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (n*tile_w, 42))
        draw.text((n*tile_w+8, 7), f'{label}: frame {frame_id} / {frame_id/25:.2f}s', fill='white')
    canvas.save(destination)


def package(v1, v2, audit, output, ffmpeg):
    v1, v2, audit, output, ffmpeg = map(lambda p: Path(p).resolve(), (v1, v2, audit, output, ffmpeg))
    if output.exists() and any(output.iterdir()): raise FileExistsError('Fresh output required')
    output.mkdir(parents=True, exist_ok=True)
    (output/'curves').mkdir(); (output/'windows').mkdir()
    d1, d2, selection, provenance = read(v1/'teacher.json'), read(v2/'teacher.json'), read(audit/'selection.json'), read(audit/'provenance.json')
    if d1['schema'] != 'fixed_brow_excursion_teacher_v1' or d2['schema'] != 'fixed_brow_excursion_teacher_v2':
        raise ValueError('Pinned teacher versions required')
    if sha(ffmpeg) != provenance['ffmpeg_sha256']: raise ValueError('FFmpeg binary differs from RGB audit')
    am, m1, m2 = read(audit/'manifest.json'), read(v1/'manifest.json'), read(v2/'manifest.json')
    for root, manifest in [(v1, m1), (v2, m2)]:
        for name, record in manifest.items(): checked(root/name, record)
    selected = {r['clip_id']: r for r in selection['clips']}
    if {c['source_id'] for c in d2['clips']} != set(selected): raise ValueError('Locked selection membership differs')
    known_waits = sum(w['supported_hold'] and not w['right_censored'] and not w['left_censored']
                      for c in d2['clips'] for w in c['waits'])
    sections, windows, source_hashes = [], [], {}
    for clip in d2['clips']:
        cid = clip['source_id']; row = selected[cid]
        ap = audit/'arrays'/(cid+'_fresh_forward.npz'); cp = audit/'visual'/(cid+'_contact.jpg')
        checked(ap, am['arrays/'+ap.name]); checked(cp, am['visual/'+cp.name])
        with np.load(ap, allow_pickle=False) as a: observed = {k: a[k].copy() for k in ('head_pose', 'landmarks', 'valid', 'rgb_sha256')}
        with np.load(v2/'arrays'/(cid+'.npz'), allow_pickle=False) as a: arrays = {k: a[k].copy() for k in a.files}
        curve = output/'curves'/(cid+'.png'); chart(clip, arrays, curve)
        cards = []
        if clip['events']:
            video = Path(row['video'])
            if sha(video) != row['video_sha256']: raise ValueError('Original video hash changed: '+cid)
            source_hashes[cid] = row['video_sha256']
            import cv2
            capture = cv2.VideoCapture(str(video)); width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)); capture.release()
            event_ids = [[e['start'], (e['start']+e['peak'])//2, e['peak'],
                          (e['release']+e['end'])//2, e['end']] for e in clip['events']]
            ids = sorted(set(i for five in event_ids for i in five))
            frames = selected_frames(video, ids, width, height, ffmpeg)
            for i, rgb in frames.items():
                if hashlib.sha256(rgb.tobytes()).hexdigest() != str(observed['rgb_sha256'][i]):
                    raise ValueError(f'Original RGB frame hash mismatch: {cid} frame {i}')
            # One crop per clip, all valid landmarks: crop affects framing only,
            # pixels stay original; it is not stabilization or independent truth.
            lm = observed['landmarks'][observed['valid'], :, :2]
            mins, maxs = np.nanmin(lm, (0, 1)), np.nanmax(lm, (0, 1))
            span = maxs-mins; low = np.maximum(0, mins-span*.12); high = np.minimum(1, maxs+span*.12)
            box = (int(low[0]*width), int(low[1]*height), int(high[0]*width), int(high[1]*height))
            for index, (e, five) in enumerate(zip(clip['events'], event_ids), 1):
                strip = output/'windows'/f'{cid}_event{index:02d}.png'
                make_strip(frames, five, ['start', 'onset mid', 'peak', 'release mid', 'end'], box, strip)
                pose = observed['head_pose'][e['start']:e['end']+1]
                finite = np.isfinite(pose).all(1)
                pose_range = np.ptp(pose[finite], 0).tolist() if finite.any() else None
                record = {'source_id': cid, 'event_index': index, 'frames': five,
                    'frame_sha256': [str(observed['rgb_sha256'][i]) for i in five],
                    'event': e, 'head_pose_yaw_pitch_roll_range_degrees': pose_range,
                    'head_pose_is_independent_truth': False, 'crop_xyxy_original_pixels': box,
                    'pixel_strip': href(strip, output), 'video_sha256': row['video_sha256']}
                windows.append(record)
                groups = ', '.join(m['group']+(' +' if m['direction'] > 0 else ' -') for m in e['groups'])
                pose_text = '无有效头姿' if pose_range is None else '/'.join(f'{x:.2f}' for x in pose_range)+'°'
                cards.append(f'<div class="event" id="{cid}_event{index}"><h3>E{index} · {html.escape(groups)} · {e["start"]}→{e["peak"]}→{e["release"]}→{e["end"]}</h3>'
                    f'<a href="{href(strip,output)}"><img src="{href(strip,output)}" loading="lazy"></a>'
                    f'<p>窗口内 yaw/pitch/roll 极差：{pose_text}。这是同一跟踪器的代理量，不能独立验证眉部动作。点击图片查看原像素裁剪条。</p></div>')
        diag = clip['diagnostics']
        sections.append(f'<section id="{cid}"><h2>{html.escape(cid)}</h2>'
            f'<p>完整事件 {len(clip["events"])}；候选 {diag["candidate_count"]}；缺支撑／删失候选 {diag["censored_candidate_count"]}。'
            f'<a href="{href(cp,output)}">已锁定的原视频接触表</a> · <a href="{href(v2/"clips"/(cid+".json"),output)}">完整候选与原因 JSON</a></p>'
            f'<a href="{href(curve,output)}"><img src="{href(curve,output)}" loading="lazy"></a>'+''.join(cards)+'</section>')
        print(f'packaged {cid}: {len(clip["events"])} event windows', flush=True)
    if len(windows) != d2['summary']['complete_event_count']: raise ValueError('Missing event windows')
    intro = '''<h1>稀疏眉部动作教师核验</h1><p><strong>这页核验候选监督，不是新模型效果，也没有正式拟合。</strong>固定阈值和时长未降低；v2 修复了缺支撑候选遮盖完整候选的门控，并允许异步眉组分开保留。v1/v2来自同一批开发数据，不能把保留率变化当质量提升。</p>
    <p>32片原视频；每个事件固定5帧：开始、起势中点、峰值、回落中点、结束。原视频SHA与每帧完整RGB哈希均对照既有审计验证。照片只做固定裁剪和显示缩放，无跟踪稳定或生成修改。跨组事件可能时间重叠；每条仍保存5维位移，当前表示不能证明联合形状正确。</p>
    <p>灰色为未支持候选区间，顶部红线为缺支撑／删失，不意味着整段真实动作被裁掉。纵轴固定[0,1]；小波动不会靠逐图放大伪装为明显动作。头姿来自同一MediaPipe，只能提示耦合风险。独立人工盲评仍缺失。</p>'''
    s = d2['summary']
    intro += f'<p>v1：{d1["summary"]["complete_event_count"]}事件；v2：{s["complete_event_count"]}事件／{s["clips_with_complete_events"]}片。v2候选事件覆盖运动能量{s["event_energy_fraction"]:.1%}，未支持区间{s["unsupported_energy_fraction"]:.1%}。这些是系数差分能量覆盖，不能解释为自然度或准确率。</p>'
    intro += f'<p><strong>可信且起终点均观察到的等待区间：{known_waits}。因此没有拟合正式节律／等待分布。</strong>尚无证据说明音频动作时机已学会，也不能证明静止区间自然。</p>'
    intro += '<p><a href="../engineering_controls/index.html">合成工程控制演示（独立入口，若尚未生成则暂不可用）</a> · <a href="review_summary.json">本页核验记录</a></p>'
    links = '<nav>'+''.join(f'<a href="#{html.escape(c["source_id"])}">{html.escape(c["source_id"])}</a>' for c in d2['clips'])+'</nav>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>稀疏眉部教师核验</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f3f6fa;color:#1c293c;margin:0 auto;max-width:1450px;padding:30px}h1,h2,h3{line-height:1.3}section,.event{background:white;padding:22px;margin:24px 0;border-radius:10px}section img{width:100%;height:auto}p{max-width:1250px}a{color:#1556b8}nav{display:flex;flex-wrap:wrap;gap:8px}nav a{font-size:12px;background:white;padding:5px}.event{border:1px solid #d5dce6}</style><body>'+intro+links+''.join(sections)+'</body></html>'
    (output/'index.html').write_text(page, encoding='utf-8')
    summary = {'schema': SCHEMA, 'status': 'complete', 'clips': len(d2['clips']), 'event_windows': len(windows),
        'displayed_frame_slots': len(windows)*5, 'known_ended_supported_waits': known_waits,
        'formal_fit_started': False, 'verified_semantic_ground_truth': False, 'teacher_versions_unchanged': True,
        'v1_manifest_sha256': sha(v1/'manifest.json'), 'v2_manifest_sha256': sha(v2/'manifest.json'),
        'v1_teacher_sha256': sha(v1/'teacher.json'), 'v2_teacher_sha256': sha(v2/'teacher.json'),
        'selection_sha256': sha(audit/'selection.json'), 'ffmpeg_sha256': sha(ffmpeg),
        'packager_sha256': sha(__file__), 'source_video_sha256': source_hashes, 'windows': windows}
    write(output/'review_summary.json', summary)
    write(output/'manifest.json', {str(p.relative_to(output)).replace('\\','/'):
        {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('v1', 'v2', 'audit', 'output', 'ffmpeg'): parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    result = package(args.v1, args.v2, args.audit, args.output, args.ffmpeg)
    print(json.dumps({k: result[k] for k in ('schema', 'status', 'clips', 'event_windows', 'known_ended_supported_waits')}, ensure_ascii=False))


if __name__ == '__main__': main()
