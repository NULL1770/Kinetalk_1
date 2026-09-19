"""Shared ARKit report used by native generation and archive reevaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_arkit_literature_metrics import (
    LIP23, UPPER9, SOURCE, literature_coefficient_metrics, pending_external_metrics,
    pending_metric,
)
from scripts.render_dynamic_rig_comparison import ARKIT_NAMES
from scripts import evaluate_arkit_literature_metrics as formula_module


def as_array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def score_fullface(samples, clip):
    """Score raw GT only; an absent mask must never turn display fills into GT."""
    metadata = clip.get('metadata', {})
    row = {key: clip.get(key, metadata.get(key))
           for key in ('clip_id', 'sentence', 'speaker', 'emotion')}
    if 'channel_mask' not in clip:
        reason = 'Raw channel_mask[T,52] missing; display or finite-value masks are insufficient.'
        row['metrics'] = {name: pending_metric(reason, ['raw channel_mask']) for name in (
            'arkit_mbe', 'arkit_lbe', 'arkit_fdd_signed', 'arkit_fdd_absolute',
            'supp_lip23_lbe', 'supp_upper9_fdd_signed', 'supp_upper9_fdd_absolute')}
        row['metrics'].update(pending_external_metrics())
        return row
    target = as_array(clip['target52'])
    channel = as_array(clip['channel_mask'])
    native = as_array(clip['valid'])
    if channel.shape != target.shape or channel.dtype != bool:
        raise ValueError('Raw Boolean channel_mask[T,52] required')
    if native.shape != (len(target),) or native.dtype != bool:
        raise ValueError('Raw Boolean native valid[T] required')
    mask = channel & native[:, None]
    # Observed nonfinite targets are corrupt data, not silently missing values.
    clock = as_array(clip.get('times', np.arange(len(target)) / 25))
    primary = literature_coefficient_metrics(samples, target, mask, clock)
    supplement = literature_coefficient_metrics(samples, target, mask, clock,
                                               lip_channels=LIP23, upper_channels=UPPER9)
    row.update({k: primary[k] for k in ('metrics', 'samples', 'frames', 'mask_protocol')})
    for dest, src in [('supp_lip23_lbe', 'arkit_lbe'),
                      ('supp_upper9_fdd_signed', 'arkit_fdd_signed'),
                      ('supp_upper9_fdd_absolute', 'arkit_fdd_absolute')]:
        row['metrics'][dest] = supplement['metrics'][src]
    row['observed_channel_names'] = [name for c, name in enumerate(ARKIT_NAMES) if mask[:, c].any()]
    return row


def build_report(rows, *, scope, sources=None):
    names = sorted(set(pending_external_metrics()) | {k for r in rows for k in r['metrics']})
    summary = {}
    for name in names:
        records = [r['metrics'][name] for r in rows if name in r['metrics']]
        good = [r['value'] for r in records if r['status'] == 'computed']
        if good:
            summary[name] = {'status': 'computed' if len(good) == len(rows) else 'partial',
                             'value': float(np.mean(good)), 'clips_scored': len(good),
                             'clips_total': len(rows)}
        else:
            summary[name] = records[0].copy() if records else pending_external_metrics()[name]
    from scripts.evaluate_arkit_literature_metrics import BEAT_LIP12, BEAT_UPPER16
    return {'schema': 'arkit_benchmark_report_v1', 'scope': scope,
            'official_dataset_benchmark_reproduced': False,
            'definition_source': SOURCE,
            'region_names': {'main_lbe': [ARKIT_NAMES[c] for c in BEAT_LIP12],
                             'main_fdd': [ARKIT_NAMES[c] for c in BEAT_UPPER16],
                             'supp_lip23': [ARKIT_NAMES[c] for c in LIP23],
                             'supp_upper9': [ARKIT_NAMES[c] for c in UPPER9]},
            'aggregation': 'metric per draw, mean draws per clip, equal mean clips; no best draw',
            'fdd_caution': 'Main official region excludes eyebrows. Energy-std FDD is invariant to frame permutations and does not measure timing.',
            'sources': sources or {}, 'summary': summary, 'per_clip': rows,
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'formula_implementation_sha256': hashlib.sha256(Path(formula_module.__file__).read_bytes()).hexdigest()}


def write_report(path, report):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
