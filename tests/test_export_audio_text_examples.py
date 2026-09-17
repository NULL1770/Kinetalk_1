"""Saved curve provenance and fixed display condition contracts."""
import copy

import pytest
import torch

from scripts import export_audio_text_examples as exporter


def curves(arm, *, initial=False):
    modes = ('full', 'zero') if initial else exporter.AUDIT_MODES
    return {'schema': exporter.ARM_SCHEMA, 'arm': arm, 'noise_seeds': exporter.SEEDS,
        'decode_steps': 12, 'oracle_is_target_conditioned': True, 'clip_id': ['a', 'b'],
        'motion': {str(seed): {mode: torch.full((2, 4, 52), float(i) / 10)
            for i, mode in enumerate(modes if seed == 42 else ('full', 'zero'))} for seed in exporter.SEEDS},
        'intensity': {mode: {'predicted': torch.ones(2, 4, 1), 'driving': torch.ones(2, 4, 1)} for mode in modes}}


def make_run(tmp_path, arm):
    run = tmp_path / arm
    run.mkdir()
    recipe = {'schema': exporter.ARM_SCHEMA, 'arm': arm, 'epochs': 8, 'batch_size': 16,
        'decode_steps': 12, 'noise_seeds': exporter.SEEDS, 'smoke_steps': 0,
        'teacher_intensity_probability': 0., 'checkpoint_selection_performed': False,
        'input_sha256': {'cache': 'cache', 'split_lock': 'split'}, 'test_loaded': False, 'default_replaced': False}
    digest = exporter.canonical_hash(recipe)
    exporter.write_json(run / 'provenance.json', {'recipe': recipe, 'recipe_sha256': digest})
    for stem, initial in (('epoch000', True), ('final', False)):
        checkpoint = run / ('epoch000.pt' if initial else 'epoch008.pt')
        torch.save({'renderer': {}, 'epoch': 0 if initial else 8}, checkpoint)
        path = run / (stem + '_curves.pt')
        torch.save(curves(arm, initial=initial), path)
        exporter.write_json(run / (stem + '_curves.provenance.json'), {
            'schema': 'projection_schedule_curves_provenance_v1', 'curve_sha256': exporter.sha(path),
            'checkpoint_sha256': exporter.sha(checkpoint), 'recipe_sha256': digest, 'cache_sha256': 'cache'})
    exporter.write_json(run / 'epoch008.json', {'epoch': 8, 'step': 1160,
        'minibatch_sha256': 'batch', 'noise_time_sha256': 'noise'})
    exporter.write_json(run / 'summary.json', {'schema': exporter.ARM_SCHEMA, 'arm': arm,
        'recipe_sha256': digest, 'completed_epochs': 8, 'optimizer_steps': 1160,
        'protected_unchanged': True, 'test_loaded': False, 'default_replaced': False,
        'curve_provenance': {'sha256': exporter.sha(run / 'final_curves.provenance.json')}})
    exporter.write_json(run / 'output_hashes.json', {p.name: exporter.sha(p) for p in run.iterdir()})
    return run


def query():
    return {'motion': torch.zeros(2, 4, 52), 'valid': torch.ones(2, 4, dtype=torch.bool), 'clip_id': ['a', 'b']}


def test_arm_validates_both_initial_and_final_checkpoint_bindings(tmp_path):
    run = make_run(tmp_path, 'text')
    loaded = exporter.load_arm(run, 'text', 'cache')
    assert loaded['binding']['initial']['curve_sha256'] == exporter.sha(run / 'epoch000_curves.pt')
    sidecar = run / 'epoch000_curves.provenance.json'
    changed = exporter.read(sidecar)
    changed['checkpoint_sha256'] = 'substituted-checkpoint'
    exporter.write_json(sidecar, changed)
    inventory = exporter.read(run / 'output_hashes.json')
    inventory[sidecar.name] = exporter.sha(sidecar)
    exporter.write_json(run / 'output_hashes.json', inventory)
    with pytest.raises(ValueError, match='initial curve/checkpoint'):
        exporter.load_arm(run, 'text', 'cache')


def test_pair_requires_equal_draws_all_initial_seeds_and_exact_clip_order(tmp_path):
    text = exporter.load_arm(make_run(tmp_path, 'text'), 'text', 'cache')
    no_text = exporter.load_arm(make_run(tmp_path, 'no_text'), 'no_text', 'cache')
    exporter.validate_pair(text, no_text, query())
    bad = copy.deepcopy(no_text)
    bad['initial']['motion']['997']['full'][0, 0, 0] += .1
    with pytest.raises(ValueError, match='initial curves differ'):
        exporter.validate_pair(text, bad, query())
    bad = copy.deepcopy(no_text)
    bad['epoch']['noise_time_sha256'] = 'different'
    with pytest.raises(ValueError, match='training draws differ'):
        exporter.validate_pair(text, bad, query())
    bad = copy.deepcopy(no_text)
    bad['final']['clip_id'].reverse()
    with pytest.raises(ValueError, match='clip order differs'):
        exporter.validate_pair(text, bad, query())


def test_protocol_allows_only_first_seed_full_audit_and_rejects_missing_oracle():
    value = curves('text')
    exporter.validate_protocol(value, 'text')
    del value['motion']['42']['oracle_intensity']
    with pytest.raises(ValueError, match='condition set differs'):
        exporter.validate_protocol(value, 'text')
    value = curves('text', initial=True)
    value['noise_seeds'] = [42]
    with pytest.raises(ValueError, match='sampling protocol differs'):
        exporter.validate_protocol(value, 'text', initial=True)


def test_six_panel_mapping_preserves_values_and_discloses_oracle(tmp_path):
    text = {'initial': curves('text', initial=True), 'final': curves('text')}
    no_text = {'final': curves('no_text')}
    no_text['final']['motion']['42']['full'].fill_(.29)
    fields = exporter.assemble_fields(query(), text, no_text)
    assert tuple(fields) == exporter.MODES
    assert torch.equal(fields['no-text trained'], no_text['final']['motion']['42']['full'])
    assert torch.equal(fields['text static I'], text['final']['motion']['42']['static_intensity'])
    assert torch.equal(fields['text oracle I'], text['final']['motion']['42']['oracle_intensity'])
    picks = [{'speaker': 'M023', 'clip_id': 'fixed<&clip'}]
    exporter.write_html(tmp_path, picks, prior_source=True)
    document = (tmp_path / 'review.html').read_text(encoding='utf8')
    assert '不是可部署结果' in document and '旧 zero-local' in document
    assert 'fixed&lt;&amp;clip' in document and 'prior_source_raw_seed42.png' in document
    previous = exporter.common.DISPLAY_MODES, exporter.common.COLORS
    with exporter.plotting_modes():
        assert exporter.common.DISPLAY_MODES == exporter.MODES
    assert (exporter.common.DISPLAY_MODES, exporter.common.COLORS) == previous
