"""Reject corrupted compact archives and verify native endpoint recomputation."""
import copy
from pathlib import Path

import pytest
import torch

from scripts import audit_native_context_results as a


@pytest.fixture(autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def fixture():
    generator = torch.Generator().manual_seed(135)
    lengths, starts = torch.tensor([123, 70, 150]), torch.tensor([13, 0, 27])
    valid = torch.arange(96)[None]+starts[:, None] < lengths[:, None]
    valid[0, 29:32] = False
    target = (torch.randn(3,96,52,generator=generator)*.03+.4).half()
    q = {'clip_id':['a','b','c'],'sentence_id':['s0','s1','s2'],'motion':target,
        'valid':valid,'times':(torch.arange(96)[None]+starts[:,None]).double()*.04,
        'channel_mask':torch.ones(3,52,dtype=torch.bool),
        'emotion_id':torch.tensor([0,1,2]),'speaker_id':torch.tensor([0,1,2])}
    static = torch.full((3,9),.4)
    predictions = {key:torch.where(valid[...,None],torch.randn(3,96,9,generator=generator)*.025+.4,0.)
                   for key in a.KEYS}
    curves = {k:v for k,v in q.items() if k not in ('motion','channel_mask')}
    curves.update(schema=a.EVAL_SCHEMA,mode='full',target_upper9=torch.where(valid[...,None],target[...,a.p.CC].float(),0.),
        channel_mask_upper=q['channel_mask'][:,a.p.CC],native_lengths=lengths,center_starts=starts,
        population_noise_draw_frames=160,upper_predictions9=predictions,upper_channel_indices=list(a.p.CC),
        noise_seeds=list(a.p.r.SEEDS),decode_steps=12,GT_was_input=False,oracle_modes=[],dc_saved=False,
        fullface_deployment_evaluation=False,static_used=static,static_upper=static.clone(),
        baseline_sha256='bound',baseline_path='base.pt',nonupper_protection_checked=True)
    report = {k:curves[k] for k in ('schema','mode','population_noise_draw_frames','noise_seeds','decode_steps',
        'GT_was_input','oracle_modes','fullface_deployment_evaluation')}
    report.update(test_loaded=False,teacher_readout_scored=False,nonupper_scored=False,clips=3,sentence_count=3,
                  per_clip_order=q['clip_id'],primary_composition='raw',secondary_composition='dc')
    dc = {key:a.compose_dc_upper(value,static,valid) for key,value in predictions.items()}
    report['dc_invariants'] = {key:a._composition_invariants(value,dc[key],static,valid)
                               for key,value in predictions.items()}
    report['compositions'] = {}
    for name, samples in (('raw',predictions),('dc',dc)):
        score = a._score_predictions(samples,curves['target_upper9'],valid,curves['channel_mask_upper'],q['emotion_id'])
        for key,value in samples.items():
            score['modes'][key]['actual_decoder_seams'] = a.actual_decoder_seams(
                value,curves['target_upper9'],valid,starts,mode='full')
        report['compositions'][name] = score
    base = torch.randn(3,96,52,generator=generator)
    base[~valid] = -0.
    bases = {**q,'target':target.float(), 'predictions':{str(seed)+'/base':base for seed in a.p.r.SEEDS}}
    entries = {cid:{'native_frames':int(n),'center_start':int(s)} for cid,n,s in zip(q['clip_id'],lengths,starts)}
    return curves,report,q,entries,bases,dc


def test_recomputes_all_raw_dc_scores_and_offset_decoder_seams():
    curves,report,q,entries,_,_ = fixture()
    a.validate_curves(curves,report,q,entries,mode='full',expected_n=3)
    summary,_ = a.rescore(curves,report,'fixture')
    assert set(summary) == {'raw','dc'}
    assert summary['raw']['actual_decoder_seams_seed42']['observed_seam_pairs'] > 0


def test_recomputation_rejects_changed_report_metric():
    curves,report,*_ = fixture()
    report['compositions']['raw']['modes']['42/full']['paired']['brows']['raw_mse'] += .001
    with pytest.raises(ValueError,match='Value differs'):
        a.rescore(curves,report,'fixture')


@pytest.mark.parametrize('field',['target_upper9','times','center_starts'])
def test_source_native_metadata_corruption_rejected(field):
    curves,report,q,entries,*_ = fixture()
    curves=copy.deepcopy(curves)
    curves[field].reshape(-1)[0] += 1
    with pytest.raises(ValueError,match='differ'):
        a.validate_curves(curves,report,q,entries,mode='full',expected_n=3)


def test_fullface_restore_preserves_signed_invalid_and_43_channels():
    curves,_,_,_,bases,dc = fixture()
    before = bases['predictions']['42/base'].clone()
    restored,checks = a.restore_compact(curves,bases,Path('base.pt'),'bound',dc)
    for value in restored.values():
        assert a._same_bits(value[~curves['valid']],before[~curves['valid']])
        assert torch.signbit(value[~curves['valid']]).all()
        assert a._same_bits(value[...,list(a.p.r.NOT_UPPER)],before[...,list(a.p.r.NOT_UPPER)])
    assert a._same_bits(before,bases['predictions']['42/base'])
    assert len(checks) == 10


def test_invalid_upper_corruption_cannot_be_discarded_during_restoration():
    curves,_,_,_,bases,dc = fixture()
    curves['upper_predictions9']['42/full'][~curves['valid']] = .5
    with pytest.raises(ValueError,match='invalid placeholders'):
        a.restore_compact(curves,bases,Path('base.pt'),'bound',dc)


def test_pair_metadata_does_not_force_different_audio_means_equal():
    curves,*_ = fixture()
    other = copy.deepcopy(curves)
    other['static_upper'] += .2
    a.equal_metadata(curves,other)
    other['valid'][0,0] = False
    with pytest.raises(ValueError,match='Common tensor metadata'):
        a.equal_metadata(curves,other)


def test_draw_replay_is_reproducible_and_detects_native_clock_change():
    first = a.replay_draws(torch.tensor([123,70,150]),torch.tensor([13,0,27]),2)
    assert first == a.replay_draws(torch.tensor([123,70,150]),torch.tensor([13,0,27]),2)
    assert first != a.replay_draws(torch.tensor([123,70,170]),torch.tensor([13,0,27]),2)
