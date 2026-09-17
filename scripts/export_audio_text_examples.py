"""Export the pre-existing nine-clip lock for the audio/text intensity pilot.

CPU-only: read saved epoch000/epoch008 curves, never perform model inference,
choose examples using output values, align time, or rescale coefficients.
The NPZ/audio exports are portable inputs to render_dynamic_rig_comparison.py.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import html
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_renderer_capacity_examples as common
from scripts.plot_projection_schedule_examples import make_lock, validate_query_metadata
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES, inspect_input
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_predictable_renderer import sha


SCHEMA = 'audio_text_visual_export_v1'
ARM_SCHEMA = 'audio_text_dynamics_v1'
SEEDS = [42, 123, 2026, 7, 19, 73, 211, 997]
AUDIT_MODES = ('full', 'zero', 'no_text', 'static_intensity', 'shuffled_text',
               'reverse_audio', 'oracle_intensity')
MODES = ('GT', 'source', 'text full', 'no-text trained', 'text static I', 'text oracle I')
COLORS = dict(zip(MODES, ('#111111', '#b66b10', '#2166ac', '#18834b', '#7651a8', '#cc4678')))
MODE_DEFINITIONS = {
    'GT': 'Internal validation query motion; diagnostic reference.',
    'source': 'Shared epoch000 seed42/full from this experiment before training; '
              'new local encoder is zero-initialized, so this is the old zero-local source, not old full-local.',
    'text full': 'Text-trained epoch008 seed42/full; ASR text and predicted intensity, no target motion.',
    'no-text trained': 'Matched no-text-trained epoch008 seed42/full; predicted intensity, no target motion.',
    'text static I': 'Text-trained epoch008; predicted intensity replaced by its valid-frame mean; '
                     'direct acoustic/text local features remain dynamic.',
    'text oracle I': 'Text-trained epoch008; target-derived intensity replaces prediction where valid. '
                     'Reads GT motion, diagnostic only; not deployable generation.',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')


def require_inventory(run, inventory, names):
    for name in names:
        path = run / name
        if not path.is_file() or inventory.get(name) != sha(path):
            raise ValueError('Run inventory differs: ' + str(path))


def load_arm(run, arm, cache_hash):
    """Verify fixed final epoch and hash bindings before exposing any curves."""
    run = Path(run).resolve()
    provenance, summary, inventory = (read(run / name) for name in
                                      ('provenance.json', 'summary.json', 'output_hashes.json'))
    recipe = provenance['recipe']
    digest = canonical_hash(recipe)
    if (recipe.get('schema') != ARM_SCHEMA or summary.get('schema') != ARM_SCHEMA
            or recipe.get('arm') != arm or summary.get('arm') != arm
            or provenance.get('recipe_sha256') != digest or summary.get('recipe_sha256') != digest):
        raise ValueError('Audio/text recipe/schema/arm binding differs')
    required = {'epochs': 8, 'batch_size': 16, 'decode_steps': 12, 'noise_seeds': SEEDS,
                'smoke_steps': 0, 'teacher_intensity_probability': 0.,
                'checkpoint_selection_performed': False}
    if any(recipe.get(key) != value for key, value in required.items()):
        raise ValueError('Audio/text fixed-epoch/deployment training protocol differs')
    if (recipe.get('input_sha256', {}).get('cache') != cache_hash
            or summary.get('completed_epochs') != 8 or summary.get('optimizer_steps') != 1160
            or summary.get('protected_unchanged') is not True):
        raise ValueError('Audio/text cache/epoch/protected contract differs')
    for record in (recipe, summary):
        for key in ('test_loaded', 'default_replaced'):
            if record.get(key) is not False:
                raise ValueError('Forbidden test/default exposure')
    require_inventory(run, inventory, ('provenance.json', 'summary.json', 'epoch008.json',
        'epoch000.pt', 'epoch000_curves.pt', 'epoch000_curves.provenance.json',
        'epoch008.pt', 'final_curves.pt', 'final_curves.provenance.json'))
    sidecar_path = run / 'final_curves.provenance.json'
    expected = {'schema': 'projection_schedule_curves_provenance_v1',
        'curve_sha256': sha(run / 'final_curves.pt'), 'checkpoint_sha256': sha(run / 'epoch008.pt'),
        'recipe_sha256': digest, 'cache_sha256': cache_hash}
    if (read(sidecar_path) != expected
            or summary.get('curve_provenance', {}).get('sha256') != sha(sidecar_path)):
        raise ValueError('Audio/text final curve/checkpoint/recipe/cache binding differs')
    initial_sidecar = run / 'epoch000_curves.provenance.json'
    initial_expected = {'schema': 'projection_schedule_curves_provenance_v1',
        'curve_sha256': sha(run / 'epoch000_curves.pt'), 'checkpoint_sha256': sha(run / 'epoch000.pt'),
        'recipe_sha256': digest, 'cache_sha256': cache_hash}
    if read(initial_sidecar) != initial_expected:
        raise ValueError('Audio/text initial curve/checkpoint/recipe/cache binding differs')
    epoch = read(run / 'epoch008.json')
    if epoch.get('epoch') != 8 or epoch.get('step') != 1160:
        raise ValueError('Final epoch draw record differs')
    loaded = {name: torch.load(run / filename, map_location='cpu', weights_only=False, mmap=True)
              for name, filename in (('initial', 'epoch000_curves.pt'), ('final', 'final_curves.pt'))}
    for name, curves in loaded.items():
        validate_protocol(curves, arm, initial=name == 'initial')
    return {'run': run, 'recipe': recipe, 'epoch': epoch, **loaded,
        'binding': {'run': str(run), 'recipe_sha256': digest, 'final': expected,
            'final_sidecar_sha256': sha(sidecar_path), 'provenance_sha256': sha(run / 'provenance.json'),
            'summary_sha256': sha(run / 'summary.json'), 'inventory_sha256': sha(run / 'output_hashes.json'),
            'initial': initial_expected, 'initial_sidecar_sha256': sha(initial_sidecar)}}


def validate_protocol(curves, arm, *, initial=False):
    seeds = SEEDS
    if (curves.get('schema') != ARM_SCHEMA or curves.get('arm') != arm
            or curves.get('noise_seeds') != seeds or curves.get('decode_steps') != 12
            or curves.get('oracle_is_target_conditioned') is not True
            or set(curves.get('motion', {})) != set(map(str, seeds))):
        raise ValueError('Saved curve sampling protocol differs')
    for seed in seeds:
        expected = {'full', 'zero'} if initial or seed != 42 else set(AUDIT_MODES)
        if set(curves['motion'][str(seed)]) != expected:
            raise ValueError('Saved curve seed/condition set differs')
    expected_i = {'full', 'zero'} if initial else set(AUDIT_MODES)
    if set(curves.get('intensity', {})) != expected_i:
        raise ValueError('Saved intensity condition set differs')


def validate_pair(text, no_text, query):
    a, b = (dict(meta['recipe']) for meta in (text, no_text))
    a.pop('arm'); b.pop('arm')
    if a != b:
        raise ValueError('Text/no-text matched recipes differ beyond arm')
    for key in ('minibatch_sha256', 'noise_time_sha256'):
        if not text['epoch'].get(key) or text['epoch'][key] != no_text['epoch'].get(key):
            raise ValueError('Text/no-text matched training draws differ: ' + key)
    shape, valid = query['motion'].shape, query['valid']
    for meta in (text, no_text):
        for phase in ('initial', 'final'):
            curves = meta[phase]
            if list(curves.get('clip_id', [])) != list(query['clip_id']):
                raise ValueError('Saved curve clip order differs')
            for values in curves['motion'].values():
                for value in values.values():
                    if value.shape != shape or not torch.isfinite(value[valid]).all():
                        raise ValueError('Saved curve native shape/values differ')
            for row in curves['intensity'].values():
                if set(row) != {'predicted', 'driving'}:
                    raise ValueError('Saved intensity fields differ')
                for value in row.values():
                    if value.shape != (*valid.shape, 1) or not torch.isfinite(value[valid]).all():
                        raise ValueError('Saved intensity native shape/values differ')
    for seed in SEEDS:
        for mode in ('full', 'zero'):
            if not torch.equal(text['initial']['motion'][str(seed)][mode], no_text['initial']['motion'][str(seed)][mode]):
                raise ValueError('Text/no-text initial curves differ')


def assemble_fields(query, text, no_text):
    tf, nf = text['final']['motion']['42'], no_text['final']['motion']['42']
    return dict(zip(MODES, (query['motion'], text['initial']['motion']['42']['full'],
        tf['full'], nf['full'], tf['static_intensity'], tf['oracle_intensity'])))


@contextmanager
def plotting_modes(modes=MODES, colors=COLORS):
    previous = common.DISPLAY_MODES, common.COLORS
    common.DISPLAY_MODES, common.COLORS = modes, colors
    try:
        yield
    finally:
        common.DISPLAY_MODES, common.COLORS = previous


def load_targets(text, query, cache_hash):
    path = Path(text['recipe']['derived_paths']['targets'])
    if sha(path) != text['recipe']['derived_sha256']['targets']:
        raise ValueError('Intensity target cache hash differs')
    targets = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    prov = targets['provenance']
    binding = prov.get('renderer_cache_sha256', prov.get('cache_sha256', prov.get('source_sha256', {}).get('cache')))
    if targets.get('schema') != 'expression_intensity_targets_v1' or binding != cache_hash:
        raise ValueError('Intensity target cache lineage differs')
    row = targets['splits']['validation']
    if (list(row['clip_id']) != list(query['clip_id']) or row['intensity'].shape != (*query['valid'].shape, 1)
            or row['valid'].shape != row['intensity'].shape or row['valid'].dtype != torch.bool
            or not torch.isfinite(row['intensity'][row['valid']]).all()):
        raise ValueError('Intensity target clock/order/values differ')
    return row, {'path': str(path), 'sha256': sha(path)}


def intensity_fields(text, no_text, targets, valid):
    tf, nf = text['final']['intensity'], no_text['final']['intensity']
    return {
        'GT intensity': (targets['intensity'], targets['valid'], COLORS['GT']),
        'text full': (tf['full']['driving'], valid[..., None], COLORS['text full']),
        'no-text trained': (nf['full']['driving'], valid[..., None], COLORS['no-text trained']),
        'text static I': (tf['static_intensity']['driving'], valid[..., None], COLORS['text static I']),
        'text oracle I': (tf['oracle_intensity']['driving'], valid[..., None], COLORS['text oracle I']),
    }


def plot_intensity(query, picks, fields, output):
    fig, axes = plt.subplots(3, 3, figsize=(17, 9), squeeze=False, sharey=True)
    for ax, pick in zip(axes.flat, picks):
        i = pick['index']
        times = query['times'][i].double().numpy().copy()
        times -= times[0]
        for name, (values, mask, color) in fields.items():
            y = values[i, :, 0].double().numpy().copy()
            y[~mask[i, :, 0].numpy()] = np.nan
            ax.plot(times, y, label=name, color=color,
                    linestyle='--' if name in ('text static I', 'text oracle I') else '-',
                    linewidth=2 if name == 'GT intensity' else 1.2)
        ax.set_title(pick['speaker'] + ' / ' + pick['emotion'])
        ax.set_xlabel('Native time from crop start (s)')
        ax.set_ylabel('Upper-face intensity proxy')
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(.5, .96))
    fig.suptitle('Fixed metadata lock | seed42 | oracle uses target motion; static I leaves direct features dynamic')
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def write_html(output, picks, *, prior_source=False):
    options = ''.join('<option value="' + html.escape(p['speaker'], quote=True) + '">'
                      + html.escape(p['clip_id']) + '</option>' for p in picks)
    initial = html.escape(picks[0]['speaker'], quote=True) + '/comparison.mp4'
    content = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Audio + ASR text · 眉眼动态固定对照</title>
<style>body{background:#12171f;color:#eee;font:15px system-ui;margin:18px}p{max-width:1200px;line-height:1.6}
video{max-width:100%;max-height:76vh;display:block;background:#000}button,select{font:inherit;padding:7px;margin:3px}
a{color:#9bc7ff}img{max-width:100%;background:white}summary{cursor:pointer;margin:15px 0}strong{color:#ffd89a}</style>
<h2>固定九片、三人连续六格对照</h2>
<p>六格顺序：GT / source / text full / no-text trained / text static I / text oracle I。
GT 是真实动作，source 是本实验共同初始化（新局部分支初始为零，即旧 zero-local，不是旧 full-local）。
text 与 no-text 各训练 8 epochs；seed42、同一原生时间轴、原始幅度。
只有显示时将系数 clamp 到 [0,1]，无效帧按最近有效帧补显示。每人的视频始终取原九片锁中的第一片，未按效果选择。</p>
<p><strong>text oracle I 使用真实动作计算的强度，是 GT 条件替换诊断，不是可部署结果。</strong>
text static I 只固定强度为窗口均值，直接声学与文本特征仍可变化；它不代表完全静态条件。
强度是眉毛和眼部表情通道的中性参考相对幅度，不等于心理情感强度。以下为内部验证展示，不能单独证明泛化或身份、口型质量。</p>
<select id="clip" aria-label="固定样例">__OPTIONS__</select>
<button id="normal">从头正常播放</button><button id="slow">从头 0.25 倍播放</button><button id="pause">暂停</button>
<video id="video" controls preload="metadata" src="__INITIAL__"></video><p id="status"></p>
<p><a href="provenance.json">数据与曲线来源</a> · <a href="render_jobs.json">固定渲染任务</a> ·
<a href="nine_clip_curves.npz">九片原始曲线数据</a></p>
<details open><summary>逐帧强度：GT 与预测、静态及 oracle</summary><img src="intensity_seed42.png" alt="固定九片逐帧强度曲线"></details>
<details><summary>原始系数：包含均值与幅度</summary><img src="raw_seed42.png" alt="固定九片原始系数"></details>
<details><summary>中心化残差：减去 B0 与身份基线后，仅用于观察动态</summary><img src="centered_seed42.png" alt="固定九片中心化残差"></details>
__PRIOR_SOURCE__
<script>
const v=document.getElementById('video'),s=document.getElementById('status');
document.getElementById('clip').onchange=()=>{v.src=document.getElementById('clip').value+'/comparison.mp4';v.load()};
document.getElementById('normal').onclick=()=>{v.currentTime=0;v.playbackRate=1;v.play()};
document.getElementById('slow').onclick=()=>{v.currentTime=0;v.playbackRate=.25;v.play()};
document.getElementById('pause').onclick=()=>v.pause();
v.ontimeupdate=()=>s.textContent=`${v.currentTime.toFixed(2)} / ${v.duration.toFixed(2)} 秒 · ${v.playbackRate} 倍 · ${v.paused?'暂停':'连续播放'}`;
v.onerror=()=>s.textContent='该固定样例视频尚未渲染或文件未找到；请按 render_jobs.json 完成对应渲染。';
</script></html>
'''
    prior_html = ('<details><summary>额外历史对照：旧 full-local 与本实验 zero-local 初始化</summary>'
                  '<p>旧 full-local 仅作额外来源对照，没有替换六格中的共同初始化。</p>'
                  '<img src="prior_source_raw_seed42.png" alt="旧 full-local 原始系数对照">'
                  '<img src="prior_source_centered_seed42.png" alt="旧 full-local 中心化残差对照"></details>') if prior_source else ''
    (output / 'review.html').write_text(content.replace('__OPTIONS__', options).replace('__INITIAL__', initial)
                                       .replace('__PRIOR_SOURCE__', prior_html), encoding='utf8')


def export(root, selection, native_root, output, *, prior_source_run=None):
    root, selection, output = Path(root), Path(selection).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Fresh output required')
    lock = read(selection)
    metadata = Path(lock['metadata_dir'])
    if lock != make_lock(metadata) or len(lock['clips']) != 9:
        raise ValueError('Existing metadata-only nine-lock differs')
    picks = common.video_picks(lock['clips'])
    cache_path = metadata / 'renderer_cache.pt'
    cache_hash = sha(cache_path)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False, mmap=True)
    if (cache.get('schema') != 'predictable_renderer_cache_v1'
            or cache['provenance']['internal_split_lock_sha256'] != lock['internal_split_lock_sha256']
            or cache['provenance']['manifest_hashes']['validation'] != lock['source_sha256']['validation_manifest']):
        raise ValueError('Cache metadata lineage differs')
    split, query = cache['splits']['validation'], cache['splits']['validation']['q']
    validate_query_metadata(query, lock)
    text, no_text = (load_arm(root / arm, arm, cache_hash) for arm in ('text', 'no_text'))
    validate_pair(text, no_text, query)
    if text['recipe']['input_sha256']['split_lock'] != lock['source_sha256']['split_lock']:
        raise ValueError('Audio/text split differs from existing lock')
    targets, target_binding = load_targets(text, query, cache_hash)
    fields = assemble_fields(query, text, no_text)
    intensities = intensity_fields(text, no_text, targets, query['valid'])
    prior_source, prior_binding = None, None
    if prior_source_run is not None:
        from scripts.export_direct_audio_examples import bound_curves
        prior, prior_binding, prior_recipe = bound_curves(Path(prior_source_run), 'epoch000', cache_hash)
        prior_source = prior['full']
        if (prior_recipe['input_sha256']['split_lock'] != lock['source_sha256']['split_lock']
                or prior_source.shape != query['motion'].shape
                or not torch.isfinite(prior_source[query['valid']]).all()):
            raise ValueError('Historical full-local source split/curves differ')
    # Complete clock/value/audio validation before creating the output directory.
    for pick in lock['clips']:
        common.native_arrays(query, pick['index'], torch.stack([v[pick['index']] for v in fields.values()]).float())
    rows = [json.loads(line) for line in (metadata / 'validation.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    audio = {pick['clip_id']: common.bind_native_audio(query, pick, rows[pick['index']], native_root) for pick in picks}
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(selection, output / 'selection.json')
    with plotting_modes():
        for centered in (False, True):
            common.plot_nine(split, lock['clips'], fields,
                output / ('centered_seed42.png' if centered else 'raw_seed42.png'), centered=centered)
    plot_intensity(query, lock['clips'], intensities, output / 'intensity_seed42.png')
    ids = [pick['index'] for pick in lock['clips']]
    np.savez_compressed(output / 'nine_clip_curves.npz', channels=np.asarray(ARKIT_NAMES),
        clip_id=np.asarray([pick['clip_id'] for pick in lock['clips']]), mode_names=np.asarray(MODES),
        times=query['times'][ids].numpy(), valid=query['valid'][ids].numpy(),
        channel_mask=query['channel_mask'][ids].numpy(), motions=torch.stack([v[ids] for v in fields.values()]).numpy(),
        cached_b0=split['base']['b0'][ids].numpy(), cached_identity_baseline=split['identity']['baseline'][ids].numpy(),
        intensity_names=np.asarray(list(intensities)),
        intensities=torch.stack([v[0][ids] for v in intensities.values()]).numpy(),
        intensity_valid=torch.stack([v[1][ids] for v in intensities.values()]).numpy(), noise_seed=np.asarray(42))
    if prior_source is not None:
        extra = {name: fields[name] for name in ('GT', 'source', 'text full', 'no-text trained')}
        extra['old full-local'] = prior_source
        with plotting_modes(tuple(extra), {**COLORS, 'old full-local': '#d24a27'}):
            for centered in (False, True):
                common.plot_nine(split, lock['clips'], extra,
                    output / ('prior_source_centered_seed42.png' if centered else 'prior_source_raw_seed42.png'), centered=centered)
        np.savez_compressed(output / 'prior_source_seed42.npz', channels=np.asarray(ARKIT_NAMES),
            clip_id=np.asarray([pick['clip_id'] for pick in lock['clips']]),
            times=query['times'][ids].numpy(), valid=query['valid'][ids].numpy(),
            channel_mask=query['channel_mask'][ids].numpy(), motions=prior_source[ids].numpy(), noise_seed=np.asarray(42))
    (output / 'video_npz').mkdir(); (output / 'audio').mkdir()
    videos, jobs = [], []
    for pick in picks:
        index = pick['index']
        times, valid, channel_mask, motions = common.native_arrays(query, index,
            torch.stack([v[index] for v in fields.values()]).float())
        binding = audio[pick['clip_id']]
        waveform = Path(binding['audio_path'])
        audio_relative = Path('audio') / (pick['clip_id'] + waveform.suffix)
        shutil.copy2(waveform, output / audio_relative)
        if sha(output / audio_relative) != binding['audio_sha256']:
            raise ValueError('Exported audio copy hash differs')
        relative = Path('video_npz') / (pick['speaker'] + '_first_locked.npz')
        path = output / relative
        np.savez_compressed(path, channels=np.asarray(ARKIT_NAMES), times=times, valid=valid,
            channel_mask=channel_mask, mode_names=np.asarray(MODES), motions=motions,
            clip_id=np.asarray(pick['clip_id']), noise_seed=np.asarray(42),
            audio_path=np.asarray(binding['audio_path']), audio_relative_path=np.asarray(audio_relative.as_posix()),
            audio_sha256=np.asarray(binding['audio_sha256']), audio_offset_seconds=np.asarray(binding['audio_offset_s']))
        videos.append({'path': relative.as_posix(), 'sha256': sha(path), 'clip': pick,
            'audio_binding': binding, 'audio_copy': audio_relative.as_posix(), 'display_report': inspect_input(path, 25)[-1]})
        jobs.append({'input': relative.as_posix(), 'output': pick['speaker'], 'audio': audio_relative.as_posix(),
            'audio_sha256': binding['audio_sha256'], 'audio_offset_seconds': binding['audio_offset_s'],
            'fps': 25, 'columns': 3, 'tile_size': 480, 'samples': 32, 'max_frames': 0,
            'expected_video': pick['speaker'] + '/comparison.mp4'})
    write_json(output / 'render_jobs.json', {'schema': 'audio_text_render_jobs_v1',
        'paths_relative_to': 'directory containing this JSON', 'driver': 'scripts/render_dynamic_rig_comparison.py',
        'layout': '2 rows x 3 columns; same six ordered conditions in every frame', 'jobs': jobs})
    write_html(output, picks, prior_source=prior_source is not None)
    provenance = {'schema': SCHEMA, 'epoch': 8, 'noise_seed': 42,
        'selection_path': str(selection), 'selection_sha256': sha(selection),
        'selection_uses_metadata_only': True, 'outcome_based_selection': False,
        'video_selection_rule': 'first locked clip per speaker, original lock order',
        'mode_names': list(MODES), 'mode_definitions': MODE_DEFINITIONS,
        'nine_plot_clips': lock['clips'], 'cache_path': str(cache_path), 'cache_sha256': cache_hash,
        'bindings': {'text': text['binding'], 'no_text': no_text['binding'], 'targets': target_binding},
        'prior_source_full_local_binding': prior_binding,
        'initial_curves_equal_exactly': True, 'training_draws_equal_exactly': True,
        'amplitude_rescaling': False, 'lag_alignment': False, 'videos': videos,
        'oracle_is_target_conditioned': True, 'test_loaded': False, 'default_replaced': False,
        'script_sha256': sha(__file__),
        'limits': 'Internal validation405 only; source B0/global have historical exposure. '
                  'Fixed display examples do not establish generalization, identity, lip-sync or perceived emotion quality. '
                  'Oracle reads GT intensity. Static intensity retains dynamic direct features.'}
    write_json(output / 'provenance.json', provenance)
    write_json(output / 'export_hashes.json', {str(p.relative_to(output).as_posix()): sha(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'export_hashes.json'})
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'selection', 'native-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prior-source-run', type=Path,
                        help='Optional old full-local source, separate plots only; never replaces six-panel source')
    args = parser.parse_args()
    report = export(args.root, args.selection, args.native_root, args.output, prior_source_run=args.prior_source_run)
    print(json.dumps({'output': str(args.output), 'videos': [v['path'] for v in report['videos']],
                      'render_jobs': str(args.output / 'render_jobs.json')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
