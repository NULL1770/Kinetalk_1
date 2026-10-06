"""Ensure isolated B0 target studies have exact and compatible update budgets."""
import pytest
import torch
from scripts.train_full_staged import articulation_update_plan, articulation_epoch_batches


def test_default_keeps_full_epochs_and_batches():
    assert articulation_update_plan(3298,16,2)==(2,207,414)
    order=torch.arange(23)
    old=order.split(4)
    new=articulation_epoch_batches(order,4,0,None)
    assert all(torch.equal(a,b) for a,b in zip(old,new)) and len(old)==len(new)


@pytest.mark.parametrize('clips,expected_epochs,samples',[(3298,8,24990),(12536,2,25072)])
def test_target_scopes_match_exact_updates(clips,expected_epochs,samples):
    epochs,per_epoch,updates=articulation_update_plan(clips,16,2,1568)
    assert epochs==expected_epochs and updates==1568
    completed=0;seen=0
    for _ in range(epochs):
        batches=articulation_epoch_batches(torch.arange(clips),16,completed,updates)
        completed+=len(batches);seen+=sum(len(b) for b in batches)
    assert completed==1568 and seen==samples


def test_resume_preserves_remaining_budget():
    batches=articulation_epoch_batches(torch.arange(3298),16,1449,1568)
    assert len(batches)==119 and sum(len(b) for b in batches)==1904
    with pytest.raises(ValueError,match='exhausted'):
        articulation_epoch_batches(torch.arange(3),2,1568,1568)


@pytest.mark.parametrize('budget',[0,-1,True,3.5])
def test_budget_requires_positive_integer(budget):
    with pytest.raises(ValueError):articulation_update_plan(3298,16,2,budget)


def test_budget_smaller_than_one_epoch():
    assert articulation_update_plan(12536,16,12,2)==(1,784,2)
    assert len(articulation_epoch_batches(torch.arange(12536),16,0,2))==2
