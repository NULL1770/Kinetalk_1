import numpy as np
import pytest
from scipy.stats import multivariate_t
from scripts import fit_sparse_brow_prior as prior


def events():
    return [{'source_id': 'clip'+str(i//4), 'p5': [.06+.002*i, .05, 0, 0, 0],
             'r5': [0, 0, 0, 0, 0], 'd_onset': 5+i%3, 'd_apex': i%2,
             'd_release': 7+i%2} for i in range(16)]


def test_student_density_matches_independent_scipy():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(13, 13)); cov = a@a.T+np.eye(13)
    loc, x = rng.normal(size=13), rng.normal(size=(8, 13))
    actual, _ = prior.t_logpdf(x, loc, cov)
    np.testing.assert_allclose(actual, multivariate_t.logpdf(x, loc=loc, shape=cov, df=5), atol=1e-11)


def test_complete_shapes_fit_positive_definite_joint_components():
    components, diag = prior.fit_marks(events())
    assert diag['clips'] == 4 and diag['events'] == 16 and diag['iterations'] == 20
    assert len(components) == 1
    assert np.isfinite(diag['fit_clip_equal_nll'])
    for c in components:
        assert min(np.linalg.eigvalsh(c['scale'])) > 0


def test_clip_balancing_not_length_balancing():
    sample = events(); sample.extend([dict(sample[0]) for _ in range(40)])
    weights = prior.clip_weights(sample)
    for cid in {e['source_id'] for e in sample}:
        assert sum(w for w, e in zip(weights, sample) if e['source_id'] == cid) == pytest.approx(.25)


def test_refuses_sparse_or_censored_shapes():
    with pytest.raises(ValueError, match='Insufficient'):
        prior.fit_marks(events()[:7])
    sample = events(); sample[0]['left_censored'] = True
    with pytest.raises(ValueError, match='Complete'):
        prior.fit_marks(sample)


def test_unknown_waits_are_not_static_supervision():
    bad = [{'source_id': 'clip1', 'duration_frames': 200, 'supported_hold': False,
            'left_censored': False, 'right_censored': False}]*100
    hazards, diag = prior.fit_waits(bad)
    assert hazards is None and not diag['learned_waiting']
    with pytest.raises(ValueError, match='waiting'):
        prior.fit(events(), bad)


def test_right_censoring_changes_survival_without_fake_event():
    good = [{'source_id': 'clip'+str(i), 'duration_frames': 35+i,
             'supported_hold': True, 'left_censored': False, 'right_censored': False} for i in range(4)]
    _, before = prior.fit_waits(good)
    good.append({'source_id': 'censored', 'duration_frames': 150, 'supported_hold': True,
                 'left_censored': False, 'right_censored': True})
    hazards, after = prior.fit_waits(good)
    assert after['observed'] == before['observed'] == 4
    assert after['right_censored'] == 1
    assert sum(after['endings_clip_balanced']) == 4
    assert sum(after['exposure_clip_balanced_frames']) > sum(before['exposure_clip_balanced_frames'])
    assert all(0 < h < 1 for h in hazards)
