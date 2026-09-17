import copy

import numpy as np
import pytest
import torch

from tests.test_audio_conditioned_flow_audit import fixture_curves
from scripts.audit_renderer_capacity_probe import analyze_capacity_curves, compact_report
from scripts.audit_audio_conditioned_flow_probe import NOISE_SEEDS


def test_oracle_success_cannot_make_failed_audio_pass_and_compact_preserves_signs():
    reference, source = fixture_curves()
    curves = {k: copy.deepcopy(source) for k in ('frozen', 'oracle_local', 'audio_local', 'zero_local')}
    for seed in NOISE_SEEDS:
        curves['oracle_local']['motion'][str(seed)]['oracle'] = reference['q']['motion'].clone()
    report = analyze_capacity_curves(curves, reference, samples=30)
    assert not report['paired_gt_dynamic_pass']
    assert not report['generative_distribution_pass']
    row = report['capacity_diagnostics']['trained_oracle_vs_frozen_oracle']['paired_gt_dynamic']['nonneutral']['brows']
    assert row['r2_improvement'] > 0
    assert row['r2_improvement_ci95'][0] > 0
    compact = compact_report(report)
    assert compact['comparisons']['matched_adapted_zero']['nonneutral']['brows']['r2_gain'] == 0
    assert compact['capacity_diagnostics']['trained_oracle_vs_frozen_oracle']['paired_gt_dynamic']['nonneutral']['brows'] == row
    assert 'Motion oracle is not deployable' in compact['interpretation']


def test_capacity_audit_rejects_missing_control_arm():
    reference, source = fixture_curves()
    with pytest.raises(ValueError, match='all three'):
        analyze_capacity_curves({'frozen': source, 'audio_local': source, 'zero_local': source}, reference, samples=30)
