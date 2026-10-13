import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/v5_final_20261014'
EVID = ROOT / 'docs/v5_rebuild_20261013/evidence/factor_repairs'
OUT = DOC / 'paper_evidence'
OUT.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def mean(values):
    values = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(np.mean(values)) if values else None


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


main = []
for row in csv.DictReader((DOC / 'metrics.csv').open(encoding='utf8')):
    main.append({key: (value if key in ('role', 'mode') else float(value)) for key, value in row.items()})
write_csv('table_main_metrics.csv', main)

joint = read(EVID / 'joint_eval_j/report.json')['results']
ablation = []
comparison_path = EVID / 'joint_eval_j/comparison.csv'
if comparison_path.exists():
    # This is the flat, auditable export of the same nested report.  Reading it
    # avoids silently treating nested ``independent_probes`` as flat fields.
    for row in csv.DictReader(comparison_path.open(encoding='utf8')):
        num = {k: (float(v) if k not in ('arm', 'role') else v) for k, v in row.items()}
        num['probe_mean'] = mean([num[f'mean_probe_{i}'] for i in range(4)])
        num['style_preserved_mean'] = mean([num[f'style_preserved_probe_{i}'] for i in range(4)])
        ablation.append(num)
else:
    for variant, result in joint.items():
        held, unseen = result['held_sentence'], result['unseen_speaker']
        for role, row in [('held_sentence', held), ('unseen_speaker', unseen)]:
            ablation.append({'arm': variant, 'role': role,
                             'mean_mouth': row['mean_mouth_mae'],
                             'sample_mouth': row['sample_mouth_mae'],
                             'probe_mean': mean([row['independent_probes'][str(i)]['mean'] for i in range(4)])})
write_csv('table_factor_ablation.csv', ablation)

style_report = read(EVID / 'style_lips_o/report.json')
style_rows = []
for role, result in style_report['results'].items():
    for swap, values in result.items():
        if isinstance(values, dict):
            style_rows.append({'role': role, 'swap': swap, **values})
write_csv('table_style_swap.csv', style_rows)

direction = read(EVID / 'personal_direction_k/report.json')['results']
direction_rows = []
for variant, result in direction.items():
    values = result['summary']
    direction_rows.append({'variant': variant, **values})
write_csv('table_style_direction.csv', direction_rows)

semantic = read(EVID / 'semantic_control_m2/report.json')['results']
dynamic_rows = []
emotion_rows = []
for role, result in semantic.items():
    for mode in ('automatic', 'declared_emotion'):
        values = result[mode]
        dynamics = values.get('upper_dynamics', {})
        dynamic_rows.append({'role': role, 'mode': mode, 'mouth_mae': values['mouth_mae'],
                             'probe_mean': mean([values[f'probe{i}_correct'] for i in range(4)]),
                             'upper_seed_diversity': values.get('upper_seed_diversity'),
                             'mouth_seed_diversity': values.get('mouth_seed_diversity'),
                             'upper_gt_amplitude': dynamics.get('amplitude', {}).get('gt'),
                             'upper_sample_amplitude': dynamics.get('amplitude', {}).get('sample'),
                             'upper_gt_speed': dynamics.get('speed', {}).get('gt'),
                             'upper_sample_speed': dynamics.get('speed', {}).get('sample'),
                             'upper_gt_acceleration': dynamics.get('acceleration', {}).get('gt'),
                             'upper_sample_acceleration': dynamics.get('acceleration', {}).get('sample')})
        for emotion, item in values.get('per_emotion', {}).items():
            emotion_rows.append({'role': role, 'mode': mode, 'emotion': emotion, **item,
                                 'probe_mean': mean([item.get(f'probe{i}_correct') for i in range(4)])})
write_csv('table_dynamic_control.csv', dynamic_rows)
write_csv('table_emotion_breakdown.csv', emotion_rows)

# Booktabs-friendly fragments are generated from the same CSVs, so paper tables
# cannot drift from the machine-readable evidence.
def tex_escape(x):
    return str(x).replace('_', r'\_').replace('%', r'\%')

def write_tex(name, csv_name, columns, caption):
    rows = list(csv.DictReader((OUT / csv_name).open(encoding='utf8')))
    body = ['\\begin{table}[t]', '\\centering', '\\caption{' + caption + '}',
            '\\begin{tabular}{' + 'l' * len(columns) + '}', '\\toprule',
            ' & '.join(tex_escape(c) for c in columns) + r' \\', '\\midrule']
    for row in rows:
        body.append(' & '.join(tex_escape(row.get(c, '')) for c in columns) + r' \\')
    body += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    (OUT / name).write_text('\n'.join(body), encoding='utf8')

write_tex('table_main_metrics.tex', 'table_main_metrics.csv',
          ['role', 'mode', 'mouth_mae', 'probe0_accuracy', 'probe1_accuracy', 'probe2_accuracy', 'probe3_accuracy'],
          'Frozen v5 reconstruction and independent emotion-probe results.')
write_tex('table_factor_ablation.tex', 'table_factor_ablation.csv',
          ['arm', 'role', 'mean_mouth', 'mean_full', 'probe_mean', 'style_preserved_mean'],
          'Reference-response weight ablation on the development protocol.')
write_tex('table_style_swap.tex', 'table_style_swap.csv',
          ['role', 'swap', 'aperture_correlation', 'velocity_cosine', 'amplitude_ratio', 'width_change_mm'],
          'Reference-swap stability and response statistics.')
write_tex('table_dynamic_control.tex', 'table_dynamic_control.csv',
          ['role', 'mode', 'mouth_mae', 'probe_mean', 'upper_gt_amplitude', 'upper_sample_amplitude', 'upper_gt_speed', 'upper_sample_speed'],
          'Automatic and explicitly controlled affect conditions.')

claims = {
    'style_definition': 'Speaking style is limited to cross-utterance and cross-emotion articulation habits: mouth aperture scale, mouth width/bias, pouting and lip/jaw coordination. It is not query phonetic content or query emotion.',
    'mimic_protocol': [
        'Mimic uses mouth opening/closing amplitude and pouting as style examples.',
        'Mimic evaluates lip dynamics deviation (LDD) and style cosine similarity (SCS) with an independent motion classifier.',
        'Mimic removes style classifier, inverse identity classifier, content contrastive loss and latent cycles in ablations.',
        'The current v5 audit has no independent SCS classifier, official HDTF mesh split or sealed test, so it is proxy evidence only.'
    ],
    'supported': [
        'Same-person cross-emotion reference: aperture correlation 0.99849 and velocity cosine 0.99656.',
        'Cross-person reference: aperture correlation 0.90939, velocity cosine 0.84034, amplitude ratio 1.23727 and mouth-width absolute change 2.15616 mm.',
        'Matched cross-person style direction: target descriptor error changes from 0.47667 to 0.36295 for frozen start, with direction cosine 0.61283.',
        'Automatic development emotion probes are .688/.625/.641/.672 with mouth MAE 0.036635; upper-face sampled amplitude is 0.0480 versus GT 0.0826.'
    ],
    'not_established': [
        'Three original contributions are not established by the current development audits.',
        'Complete emotion/style disentanglement is not established; reference swaps retain emotion leakage.',
        'Teacher deployment improvement is not established; the new StateFlow is disabled in the frozen release.',
        'Dynamic intensity is under-recovered and Angry upper-face tension remains weak.'
    ],
    'gates': [
        {'candidate': 'reference-conditioned articulation style', 'status': 'partial proxy evidence', 'missing': 'independent SCS/LDD, unseen-speaker reference swap and component ablation'},
        {'candidate': 'teacher/student affect trend plus stochastic dynamics', 'status': 'not established', 'missing': 'teacher/audio/student ablation, multi-sample coverage and intensity-stratified event metrics'},
        {'candidate': 'joint content-affect-style decoder', 'status': 'partial development evidence', 'missing': 'counterfactual swaps with lip closure, sync and emotion preservation on a sealed split'}
    ]
}
(OUT / 'style_and_contribution_evidence.json').write_text(json.dumps(claims, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
(OUT / 'style_and_contribution_evidence.md').write_text(
    '# 风格论证与创新点验收\n\n'
    '## 风格定义\n\n' + claims['style_definition'] + '\n\n'
    '## Mimic 对照\n\n' + '\n'.join('- ' + x for x in claims['mimic_protocol']) + '\n\n'
    '## 当前证据\n\n' + '\n'.join('- ' + x for x in claims['supported']) + '\n\n'
    '|候选贡献|当前状态|缺口|\n|---|---|---|\n' +
    ''.join(f"|{x['candidate']}|{x['status']}|{x['missing']}|\n" for x in claims['gates']) +
    '\n## 不能写成已成立的结论\n\n' + '\n'.join('- ' + x for x in claims['not_established']) + '\n', encoding='utf8')

(OUT / 'README.md').write_text(
    '# v5 论文证据包\n\n'
    '所有 CSV/TeX 均从冻结 v5 报告生成。`metrics.csv`、`joint_eval_j/comparison.csv`、'
    '`semantic_control_m2/report.json` 是当前开发协议的来源。表格不得与旧 checkpoint 的跨方法数字混排；'
    '需要新的公开基准或 sealed test 时，应另行运行实验。\n', encoding='utf8')

print(json.dumps({'output': str(OUT), 'table_count': 4}, ensure_ascii=False))
