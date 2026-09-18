import json

import numpy as np
import pytest
import torch

from scripts import audit_semantic_student_transfer as audit
from scripts.joint_motion_metrics import score_clip, summarize


def test_all_ten_arms_recomputed_and_nonzero_paired_sentence_intervals(tmp_path):
    run = tmp_path/'run'; run.mkdir(); clips = {}
    for i, sentence in enumerate(('a', 'a', 'b', 'c')):
        frames = 12; native = np.ones(frames, bool); native[6] = False
        valid = native.copy(); valid[-1] = False
        target52 = np.full((frames, 52), .4)
        target = .4 + .12*np.sin(np.arange(frames)[:, None]*.55+np.arange(9)[None, :]*.2)
        target52[:, audit.UPPER] = target
        baseline = np.full((frames, 52), .4); samples = {}
        for kind in audit.KINDS:
            for mode in audit.MODES:
                # New is perfect; all four comparison conditions are constant
                # and biased, so raw and centered new-minus-control ES < 0.
                values = target if mode == 'new' else np.full_like(target, .7 + .01*i)
                values = values.copy(); values[~native] = baseline[~native][:, audit.UPPER]
                samples[kind+'_'+mode] = np.broadcast_to(values, (4, frames, 9)).copy()
        clips[f'c{i}'] = {'metadata': {'clip_id': f'c{i}', 'sentence': sentence, 'split': 'holdout' if i < 3 else 'train',
                                      'emotion': 'test', 'speaker': 'test'},
                         'valid': valid, 'native_valid': native, 'times': np.arange(frames)/25,
                         'target': target, 'target52': target52, 'baseline52': baseline,
                         'samples': samples, 'seeds': audit.SEEDS}
    protocol = {'sample_seeds': audit.SEEDS, 'modes': list(audit.MODES)}
    torch.save({'schema': audit.SOURCE_SCHEMA, 'protocol': protocol, 'clips': clips}, run/'predictions.pt')
    reports = {}
    for kind in audit.KINDS:
        scale = np.linspace(.07, .12, 9) * (1. if kind == 'va' else 1.2)
        torch.save({'metric_scale': torch.from_numpy(scale)}, run/(kind+'_scales.pt'))
        for mode in audit.MODES:
            arm = kind+'_'+mode; reports[arm] = {}
            for role, split in (('holdout', 'holdout'), ('fit_examples', 'train')):
                rows = [{'clip_id': cid, 'sentence': clip['metadata']['sentence'],
                         **score_clip(clip['samples'][arm], clip['target'], clip['valid'], scale)}
                        for cid, clip in clips.items() if clip['metadata']['split'] == split]
                reports[arm][role] = {'rows': rows, 'summary': summarize(rows)}
    (run/'reports.json').write_text(json.dumps(reports), encoding='utf8')
    result = audit.audit(run, tmp_path/'audit.json', bootstrap_draws=200)
    assert result['max_report_difference'] < 1e-12
    assert set(result['independent_primary']) == set(audit.ARMS)
    assert result['clip_counts'] == {'holdout': 3, 'fit_examples': 1}
    for comparison in result['paired']['holdout'].values():
        for metric in ('raw', 'centered'):
            b = comparison[metric]
            assert b['mean'] < 0 and b['q025'] <= b['q975'] < 0
            assert b['sentences'] == 2 and b['clips'] == 3
            assert b == audit.sentence_bootstrap(comparison['clip_differences'], metric, draws=200)
    # Unequal cluster sizes preserve clip-equal, not sentence-equal, mean.
    b = audit.sentence_bootstrap([{'sentence': 'a', 'raw': 1.}, {'sentence': 'a', 'raw': 3.},
                                 {'sentence': 'b', 'raw': 8.}], 'raw', draws=200)
    assert b['mean'] == pytest.approx(4.)
    reports['va_new']['holdout']['summary']['joint_fair_es']['raw'] += .01
    (run/'reports.json').write_text(json.dumps(reports), encoding='utf8')
    with pytest.raises(ValueError, match='primary report mismatch'):
        audit.audit(run, tmp_path/'corrupt_audit.json', bootstrap_draws=10)
