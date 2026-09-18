import copy
import json

import numpy as np
import pytest
import torch

from kinetalk_b0.models.continuous_upper_motion import ContinuousLatentFlow, ContinuousUpperAE
from scripts import train_continuous_motion_latent as runner


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def items():
    generator = torch.Generator().manual_seed(124)
    return [{'residual': torch.randn(n, 9, generator=generator),
             'audio': torch.randn(n, 6, generator=generator), 'context': torch.randn(7, generator=generator),
             'metadata': {'clip_id': 'c'+str(i//2)}} for i, n in enumerate((11, 8, 15, 6))]


def stage(model, data, stats, output, **kwargs):
    return runner._train_stage(model, data, stats, stage='ae', budget=4, batch_size=3,
                               device=torch.device('cpu'), output=output, binding={'data': 'locked'}, seed=95, **kwargs)


def test_interrupted_ae_two_plus_two_updates_exactly_matches_four(tmp_path):
    torch.manual_seed(947)
    initial = ContinuousUpperAE(hidden=16, latent_dim=8, depth=2)
    full, first, resumed = copy.deepcopy(initial), copy.deepcopy(initial), copy.deepcopy(initial)
    data = items()
    a = stage(full, data, {}, tmp_path/'full')
    partial = stage(first, data, {}, tmp_path/'split', stop_after=2)
    assert partial['step'] == 2 and not (tmp_path/'split/ae_final.pt').exists()
    # Unrelated global RNG consumption cannot change resumed sampling.
    torch.randn(39)
    b = stage(resumed, data, {}, tmp_path/'split', resume=True)
    assert a['losses'] == b['losses'] and a['order_sha256'] == b['order_sha256']
    for key in full.state_dict(): torch.testing.assert_close(full.state_dict()[key], resumed.state_dict()[key], rtol=0, atol=0)
    final_mtime = (tmp_path/'split/ae_final.pt').stat().st_mtime_ns
    skipped = stage(copy.deepcopy(initial), data, {}, tmp_path/'split', resume=True)
    assert skipped['step'] == 4 and final_mtime == (tmp_path/'split/ae_final.pt').stat().st_mtime_ns


@pytest.mark.parametrize('change', ['binding', 'budget', 'initial'])
def test_resume_rejects_changed_protocol_budget_or_initial_model(tmp_path, change):
    torch.manual_seed(45); initial = ContinuousUpperAE(hidden=8, latent_dim=4, depth=1)
    stage(copy.deepcopy(initial), items(), {}, tmp_path, stop_after=2)
    model = copy.deepcopy(initial); binding = {'data': 'locked'}; budget = 4
    if change == 'binding': binding = {'data': 'different'}
    if change == 'budget': budget = 5
    if change == 'initial': next(model.parameters()).data.add_(1)
    with pytest.raises(ValueError, match='Resume mismatch'):
        runner._train_stage(model, items(), {}, stage='ae', budget=budget, batch_size=3, device=torch.device('cpu'),
                            output=tmp_path, binding=binding, seed=95, resume=True)


def test_paired_audio_starts_same_function_and_receives_same_random_training(tmp_path):
    torch.manual_seed(157)
    ae = ContinuousUpperAE(hidden=16, latent_dim=8, depth=2)
    encoded = runner._encode_segments(ae, items(), torch.device('cpu'))
    stats = runner._latent_stats(encoded)
    base = ContinuousLatentFlow(context_dim=7, audio_dim=6, latent_dim=8, hidden=16, depth=2)
    torch.nn.init.zeros_(base.audio_block_projection.weight); torch.nn.init.zeros_(base.audio_block_projection.bias)
    target, valid, context, audio, noise, fraction, frames = runner._flow_batch(base, encoded, stats, torch.device('cpu'), 44, 1)
    no_audio = base.velocity(noise, fraction, valid, context, audio, False, audio_frame_valid=frames)
    with_audio = base.velocity(noise, fraction, valid, context, audio, True, audio_frame_valid=frames)
    torch.testing.assert_close(no_audio, with_audio, rtol=0, atol=0)
    results = []
    for name, enabled in [('matched_global', False), ('audio', True)]:
        results.append(runner._train_stage(copy.deepcopy(base), encoded, stats, stage=name,
            budget=4, batch_size=3, device=torch.device('cpu'), output=tmp_path,
            binding={'data':'same'}, seed=85, use_audio=enabled))
    assert results[0]['initial_state_sha256'] == results[1]['initial_state_sha256']
    assert results[0]['order_sha256'] == results[1]['order_sha256']
    assert results[0]['losses'][0] == results[1]['losses'][0]
    assert not torch.equal(results[0]['state']['audio_block_projection.weight'], results[1]['state']['audio_block_projection.weight'])


def test_flow_two_plus_two_resumes_exact_noise_and_optimizer(tmp_path):
    torch.manual_seed(823)
    ae = ContinuousUpperAE(hidden=8, latent_dim=4, depth=1)
    encoded = runner._encode_segments(ae, items(), torch.device('cpu')); stats = runner._latent_stats(encoded)
    initial = ContinuousLatentFlow(7, audio_dim=6, latent_dim=4, hidden=12, depth=1)
    kwargs = dict(stage='audio', budget=4, batch_size=3, device=torch.device('cpu'),
                  binding={'locked':True}, seed=24, use_audio=True)
    full = runner._train_stage(copy.deepcopy(initial), encoded, stats, output=tmp_path/'full', **kwargs)
    runner._train_stage(copy.deepcopy(initial), encoded, stats, output=tmp_path/'split', stop_after=2, **kwargs)
    split = runner._train_stage(copy.deepcopy(initial), encoded, stats, output=tmp_path/'split', resume=True, **kwargs)
    assert full['losses'] == split['losses']
    for key in full['state']: torch.testing.assert_close(full['state'][key], split['state'][key], rtol=0, atol=0)


def test_segment_sampling_is_clip_equal_and_diagnostics_never_mix_splits():
    groups = [[0], list(range(1, 20))]
    sampled = runner._sample_indices(groups, 10000, np.random.default_rng(58))
    assert .47 < sampled.count(0)/len(sampled) < .53
    clips = [{'clip_id':f'{role}{i:03}', 'split':role, 'speaker':i%3, 'emotion':i%4}
             for role, count in [('train',80),('holdout',100)] for i in range(count)]
    selected = runner._diagnostic_members(clips)
    assert len(selected['train_diagnostic']) == 16 and len(selected['holdout']) == 64
    assert {c['split'] for c in selected['train_diagnostic']} == {'train'}
    assert {c['split'] for c in selected['holdout']} == {'holdout'}


def test_prior_gate_rejects_bad_motion_without_certifying_good_motion():
    result = {'numerical_gate':{'passed':True},
        'summary':{'rms_ratio':[1,1,1,1], 'speed':{'all':{'rms':1}, 'reference_all':{'rms':1}}},
        'numerics':[{'native_generated_values':100, 'raw_upper_out_of_range_count':0,
                     'raw_upper_min':0, 'raw_upper_max':1}]}
    good = runner._prior_gate(result)
    assert good['passed'] and good['needs_visual_review'] and not good['naturalness_certified']
    bad = copy.deepcopy(result); bad['summary']['speed']['all']['rms'] = 6
    assert not runner._prior_gate(bad)['passed']
    bad = copy.deepcopy(result); bad['numerics'][0]['raw_upper_out_of_range_count'] = 36
    assert not runner._prior_gate(bad)['passed']
    bad = copy.deepcopy(result); bad['summary']['rms_ratio'] = [.01]*4
    assert not runner._prior_gate(bad)['passed']


def test_ae_gate_checks_variation_and_absolute_error_against_constant(tmp_path):
    target = np.linspace(.1,.8,10)[:,None].repeat(9,1)
    def evaluate(sample):
        torch.save({'clips':{'a':{'score_mask':np.ones(10,bool),'target':target,'samples':sample[None]}}},tmp_path/'curves.pt')
        return runner._ae_gate({'numerical_gate':{'passed':True}},tmp_path)
    assert evaluate(target)['passed']
    assert not evaluate(np.broadcast_to(target.mean(0),target.shape))['passed']
    assert not evaluate(target+1)['passed']  # centered R2 alone would miss level failure


def test_gates_cannot_be_disabled_for_production(tmp_path):
    args = runner.parse_args(['--dataset',str(tmp_path/'missing'), '--output',str(tmp_path/'out'),
                              '--skip-quality-gates'])
    with pytest.raises(ValueError, match='requires --smoke'): runner.run(args)


def test_tiny_complete_pipeline_launcher_directory_and_final_resume(tmp_path):
    from scripts.prepare_continuous_motion_dataset import fit_statistics, SCHEMA
    torch.manual_seed(113)
    clips = []
    upper = [41,42,43,44,45,5,6,12,13]
    for index, split in enumerate(('train','train','holdout','holdout')):
        n = 11; motion = torch.full((n,52),.25)
        motion[:,upper] += torch.sin(torch.linspace(0,4,n))[:,None]*.05
        baseline = torch.full_like(motion,.25)
        clips.append({'clip_id':f'clip{index}', 'split':split, 'sentence':f'sentence{index}',
            'speaker':0, 'emotion':0, 'features':torch.randn(n,1540), 'context':torch.randn(202),
            'valid':torch.ones(n,dtype=torch.bool),'motion_mask':torch.ones(n,9,dtype=torch.bool),
            'motion9':motion[:,upper], 'b9':torch.full((9,),.25),'baseline52':baseline,
            'target52':motion,'times':torch.arange(n)/25})
    dataset = tmp_path/'dataset.pt'; torch.save({'schema':SCHEMA,'clips':clips,'stats':fit_statistics(clips)},dataset)
    output = tmp_path/'output'; output.mkdir(); (output/'launch.json').write_text('{}')
    args = runner.parse_args(['--dataset',str(dataset),'--output',str(output),'--device','cpu',
        '--ae-steps','1','--prior-steps','1','--adapt-steps','1','--batch-size','1','--smoke','--skip-quality-gates'])
    assert runner.run(args) == 0
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'complete' and status['total_updates'] == 4
    assert (output/'ae_reconstruction/train_diagnostic/result.json').exists()
    assert (output/'ae_reconstruction/holdout/result.json').exists()
    final_time = (output/'audio_final.pt').stat().st_mtime_ns
    args.resume = True
    assert runner.run(args) == 0 and (output/'audio_final.pt').stat().st_mtime_ns == final_time
