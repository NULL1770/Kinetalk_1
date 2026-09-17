"""Plot the completed OOF diagnostic in native units; no new evaluations."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    args = p.parse_args()
    source = args.study / 'paired_audit.json'
    report = json.loads(source.read_text(encoding='utf8'))
    groups = ['brows', 'eyes_expression', 'mouth', 'jaw17']
    labels = ['Brows', 'Eyes', 'Mouth', 'Jaw']
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), layout='constrained')
    for ax, mode, title in zip(axes, ['full', 'metric_projection_oracle'],
                               ['Audio prediction', 'Real-motion projection (oracle)']):
        for delta, arm, color in [(-.17, 'native', '#536b87'), (.17, 'scaled', '#dd9044')]:
            values = report['comparisons'][f'{arm}_{mode}__vs__zero']['populations']['nonneutral']
            means = np.array([values[g]['r2_improvement'] for g in groups])
            intervals = np.array([values[g]['r2_improvement_ci95'] for g in groups]).T
            ax.bar(np.arange(4) + delta, means, width=.31, color=color,
                   label=arm, yerr=np.stack([means - intervals[0], intervals[1] - means]),
                   capsize=3, error_kw={'lw': 1})
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(4), labels)
        ax.set_ylabel('Native-unit R2 vs zero residual')
        ax.set_ylim(0, 1.02)
        ax.spines[['right', 'top']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle('Better brow projection did not improve audio-driven brow prediction', fontsize=13)
    fig.supxlabel('19 identities / 1,893 nonneutral OOF clips / 52 sentence clusters\n'
                  'Fixed rank 8, alpha 1; 95% sentence bootstrap; existing-training diagnostic', fontsize=9)
    fig.savefig(args.study / 'summary.png', dpi=190)
    fig.savefig(args.study / 'summary.svg')
    (args.study / 'plot_provenance.json').write_text(json.dumps({
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'raw_units': True, 'population': 'nonneutral', 'shared_ylim': [0, 1.02],
        'selection': 'all predefined regions except overlapping upper/all expression',
        'oracle_note': 'Target-conditioned metric projection, not audio inference or strict native-MSE bound'
    }, indent=2))


if __name__ == '__main__':
    main()
