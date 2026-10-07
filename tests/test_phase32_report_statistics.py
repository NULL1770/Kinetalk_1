"""Protect score aggregation and pairing; no generator fitting in this audit."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from kinetalk_b0.emotion_probe import classification_metrics

spec=importlib.util.spec_from_file_location('phase32_statistics',Path(__file__).resolve().parents[1]/'.codex-finalizer/phase32_statistics.py')
stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)


def test_declared_classes_match_original_probe_f1():
    labels=np.array([0,0,1,1,2,2,3]);prediction=np.array([0,1,1,2,2,0,3])
    names=[str(i) for i in range(8)]
    original=classification_metrics(torch.from_numpy(labels),torch.from_numpy(prediction),names)
    np.testing.assert_array_equal(stats.class_f1(labels,prediction,8),
                                  [original['per_class'][n]['f1'] for n in names])
    assert stats.class_f1(labels,prediction,8).mean()==original['macro_f1']


def test_cached_cluster_bootstrap_matches_expanded_speakers():
    rng=np.random.default_rng(74)
    speaker=np.repeat([10,22,33],8);labels=np.tile(np.arange(8),3)
    control=rng.integers(8,size=(3,4,24));candidate=rng.integers(8,size=(3,4,24))
    metric=rng.normal(size=(24,2));region=rng.normal(size=(24,2,2))
    result=stats.paired_ci(metric,region,control,candidate,labels,speaker,8)
    choices=np.random.default_rng(20261005).choice(np.unique(speaker),(2000,3),replace=True)
    all_m=[];all_r=[];all_f=[]
    for row in choices:
        ix=np.concatenate([np.flatnonzero(speaker==s) for s in row])
        all_m.append(metric[ix].mean(0));all_r.append(region[ix].mean(0))
        all_f.append(np.asarray([[stats.class_f1(labels[ix],candidate[d,p,ix],8)
            -stats.class_f1(labels[ix],control[d,p,ix],8) for p in range(4)] for d in range(3)]).mean(0))
    np.testing.assert_allclose(result['metrics_ci95'],np.quantile(all_m,[.025,.975],axis=0),atol=1e-14)
    np.testing.assert_allclose(result['region_ci95'],np.quantile(all_r,[.025,.975],axis=0),atol=1e-14)
    np.testing.assert_allclose(result['probe_class_f1_ci95'],np.quantile(all_f,[.025,.975],axis=0),atol=1e-14)
    np.testing.assert_allclose(result['probe_f1_ci95'],np.quantile(np.asarray(all_f).mean(-1),[.025,.975],axis=0),atol=1e-14)


@pytest.mark.parametrize('failure', ['none','mbe','geometry','original_f1','neutral','fear'])
def test_gate_requires_improvement_and_blocks_clear_regression(failure):
    columns=['arkit_mbe','arkit_lbe','lve_mean_mm_mean','vertex_lve_sqrt_mean',
             'eve_mean_mm_mean','vertex_fdd_absolute_mm2_mean']
    classes=['neutral','happy','sad','angry','fear','disgust','surprise','contempt']
    ci=dict(metrics_ci95=np.full((2,6),-.1),region_ci95=np.zeros((2,2,1)),
            probe_f1_ci95=np.zeros((2,4)),probe_class_f1_ci95=np.zeros((2,4,8)))
    if failure=='mbe':ci['metrics_ci95'][:]=0
    if failure=='geometry':ci['metrics_ci95'][:,2]=.1
    if failure=='original_f1':ci['probe_f1_ci95'][:,1]=-.01
    if failure in ('neutral','fear'):ci['probe_class_f1_ci95'][:,0,classes.index(failure)]=-.01
    gate=stats.pilot_gate({}, {}, ci, columns,['displacement_mse'],['jawOpen','mouth_jaw'],classes)
    assert gate['passed']==(failure=='none')
