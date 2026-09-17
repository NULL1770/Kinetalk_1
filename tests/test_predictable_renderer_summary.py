import torch
from scripts.summarize_predictable_renderer import averaged_statistics


def test_seeds_average_errors_not_predictions_or_independent_samples():
    target=torch.zeros(2,3,1)
    weight=torch.ones(2,3)
    stats=averaged_statistics([torch.ones_like(target),-torch.ones_like(target)],target,weight,[0])
    assert stats.shape==(2,7)
    assert (stats[:,0]==3).all()  # average prediction is zero, but noise error is 1.
    assert (stats[:,3]==3).all()  # two seeds do not double the sample denominator.
