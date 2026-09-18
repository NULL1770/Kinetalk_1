"""Source binding and full-clock, motion-free frozen acoustic extraction."""
import copy
import json

import pytest
import torch

from kinetalk_b0.models.slow_state_affect import SlowStateAffect
from scripts import joint_prior_audio_context as context


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(784)
        model = SlowStateAffect(torch.randn(7), torch.rand(7)+.5,
            hidden=12, global_dim=5, local_dim=3, num_emotions=4, num_levels=4)
        # A trained local head is nonzero; exercise actual temporal outputs.
        torch.nn.init.normal_(model.local_head.weight, std=.2)
        torch.nn.init.normal_(model.local_head.bias, std=.1)
    return model.eval().requires_grad_(False)


def clip_fixture():
    rng = torch.Generator().manual_seed(818)
    result = []
    for index, length in enumerate((23, 137, 65)):
        valid = torch.ones(length, dtype=torch.bool)
        valid[4:7] = False
        valid[-2:] = False
        features = torch.randn(length, 7, generator=rng)
        features[~valid] = float('nan')
        result.append({'clip_id': 'clip'+str(index), 'features': features, 'valid': valid})
    return result


def test_full_native_outputs_equal_original_frozen_forward():
    model, clips = model_fixture(), clip_fixture()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    result = context.encode_clips(model, clips, batch_size=2, device='cpu')
    assert [r['clip_id'] for r in result] == [c['clip_id'] for c in clips]
    for clip, encoded in zip(clips, result):
        expected = model(clip['features'][None], clip['valid'][None])
        for key, source in (('global', 'global'), ('intensity', 'intensity_value'), ('local', 'local')):
            torch.testing.assert_close(encoded[key], expected[source][0], rtol=2e-6, atol=2e-6)
            assert not encoded[key].requires_grad
            assert encoded[key].device.type == 'cpu'
        assert encoded['local'].shape == (len(clip['valid']), 3)
        assert encoded['local'][~clip['valid']].count_nonzero() == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
    assert all(p.grad is None for p in model.parameters())


def test_batch_padding_and_gap_poison_are_invariant():
    model, clips = model_fixture(), clip_fixture()
    together = context.encode_clips(model, clips, batch_size=3)
    singles = context.encode_clips(model, clips, batch_size=1)
    clean = copy.deepcopy(clips)
    for clip in clean:
        clip['features'][~clip['valid']] = 0.
    cleaned = context.encode_clips(model, clean, batch_size=2)
    for a, b, c in zip(together, singles, cleaned):
        for key in ('global', 'intensity', 'local'):
            torch.testing.assert_close(a[key], b[key], rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(a[key], c[key], rtol=2e-6, atol=2e-6)


def test_long_native_tail_contributes_and_no_96_crop_occurs():
    model = model_fixture()
    clips = [clip_fixture()[1]]
    changed = copy.deepcopy(clips)
    changed[0]['features'][110:125] += 7.
    a = context.encode_clips(model, clips)[0]
    b = context.encode_clips(model, changed)[0]
    assert a['local'].shape[0] == 137
    assert not torch.allclose(a['global'], b['global'])
    assert not torch.allclose(a['local'][110:125], b['local'][110:125])
    torch.testing.assert_close(a['local'][:90], b['local'][:90], rtol=0, atol=0)


def test_encoder_never_accesses_motion_labels_anchor_or_teacher_fields(monkeypatch):
    class AcousticOnly(dict):
        def __getitem__(self, key):
            assert key in ('clip_id', 'features', 'valid'), 'Forbidden query field accessed: '+key
            return super().__getitem__(key)

        def get(self, key, default=None):
            raise AssertionError('Only explicit three-key acoustic access is allowed')

        def items(self):
            raise AssertionError('Do not enumerate mixed acoustic and target fields')

    model = model_fixture()
    clips = [AcousticOnly(**clip, motion=object(), emotion=object(), anchor=object(),
                          state=object(), upper=object()) for clip in clip_fixture()]
    # The state head and motion-derived override path must never execute.
    monkeypatch.setattr(model.state_head, 'forward', lambda *_: pytest.fail('Unused state head ran'))
    result = context.encode_clips(model, clips)
    assert len(result) == 3


@pytest.mark.parametrize('kind', ['trainable', 'training', 'empty', 'all_invalid',
                                  'bad_observed', 'bad_mask', 'duplicate', 'batch_zero', 'device'])
def test_rejects_invalid_frozen_or_acoustic_contract(kind):
    model, clips = model_fixture(), clip_fixture()
    kwargs = {}
    if kind == 'trainable':
        model.local_head.requires_grad_(True)
    elif kind == 'training':
        model.train()
    elif kind == 'empty':
        clips = []
    elif kind == 'all_invalid':
        clips[0]['valid'].fill_(False)
    elif kind == 'bad_observed':
        clips[0]['features'][0, 0] = float('inf')
    elif kind == 'bad_mask':
        clips[0]['valid'] = clips[0]['valid'].float()
    elif kind == 'duplicate':
        clips[1]['clip_id'] = clips[0]['clip_id']
    elif kind == 'batch_zero':
        kwargs['batch_size'] = 0
    elif kind == 'device':
        kwargs['device'] = 'cuda'
    with pytest.raises(ValueError):
        context.encode_clips(model, clips, **kwargs)


def checkpoint_fixture(tmp_path, monkeypatch):
    model = model_fixture()
    root = tmp_path/'run12'
    folder = root/'audio'
    folder.mkdir(parents=True)
    recipe = {'schema': context.SOURCE_SCHEMA, 'epochs_per_stage': 12, 'stride_frames': 16,
              'data_provenance': {'input_sha256': dict(context.EXPECTED_INPUT_SHA256)}}
    recipe_hash = context._canonical_hash(recipe)
    path = folder/'final.pt'
    payload = {'schema': context.SOURCE_SCHEMA, 'stage': 'audio', 'completed_epochs': 12,
        'recipe_sha256': recipe_hash, 'inference_only': True, 'audio': model.state_dict()}
    torch.save(payload, path)
    digest = context._sha(path)
    (root/'provenance.json').write_text(json.dumps({'recipe': recipe, 'recipe_sha256': recipe_hash}), encoding='utf8')
    (folder/'complete.json').write_text(json.dumps({'stage': 'audio', 'completed_epochs': 12,
                                                 'final_sha256': digest}), encoding='utf8')
    monkeypatch.setattr(context, 'EXPECTED_CHECKPOINT_SHA256', digest)
    monkeypatch.setattr(context, 'EXPECTED_RECIPE_SHA256', recipe_hash)
    return path, model


def test_frozen_loader_binds_source_infers_dims_and_preserves_rng(tmp_path, monkeypatch):
    path, original = checkpoint_fixture(tmp_path, monkeypatch)
    rng_before = torch.get_rng_state().clone()
    model = context.load_frozen_audio(path, 'cpu')
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(value, original.state_dict()[key])
    binding = context.assert_source_binding(model, dict(context.EXPECTED_INPUT_SHA256))
    assert binding['dimensions'] == {'features': 7, 'global': 5, 'intensity': 1, 'local': 3}
    assert binding['query_motion_used'] is False
    assert binding['full_native_clock'] is True


@pytest.mark.parametrize('kind', ['checkpoint', 'complete', 'recipe', 'epoch', 'inputs', 'checkpoint_stage'])
def test_loader_rejects_forged_or_mismatched_source(tmp_path, monkeypatch, kind):
    path, _ = checkpoint_fixture(tmp_path, monkeypatch)
    if kind == 'checkpoint':
        with path.open('ab') as handle:
            handle.write(b'changed')
    elif kind == 'complete':
        complete = json.loads(path.with_name('complete.json').read_text())
        complete['final_sha256'] = '0'*64
        path.with_name('complete.json').write_text(json.dumps(complete))
    elif kind in ('recipe', 'epoch', 'inputs'):
        provenance_path = path.parent.parent/'provenance.json'
        value = json.loads(provenance_path.read_text())
        if kind == 'recipe':
            value['recipe']['extra'] = 'changed'
        elif kind == 'epoch':
            value['recipe']['epochs_per_stage'] = 4
        else:
            value['recipe']['data_provenance']['input_sha256']['audio'] = '0'*64
        # Isolate the subsequent semantic/data contract after recipe binding.
        if kind != 'recipe':
            digest = context._canonical_hash(value['recipe'])
            value['recipe_sha256'] = digest
            monkeypatch.setattr(context, 'EXPECTED_RECIPE_SHA256', digest)
        provenance_path.write_text(json.dumps(value))
    else:
        value = torch.load(path, weights_only=False)
        value['stage'] = 'teacher'
        torch.save(value, path)
        digest = context._sha(path)
        complete = json.loads(path.with_name('complete.json').read_text())
        complete['final_sha256'] = digest
        path.with_name('complete.json').write_text(json.dumps(complete))
        monkeypatch.setattr(context, 'EXPECTED_CHECKPOINT_SHA256', digest)
    with pytest.raises(ValueError):
        context.load_frozen_audio(path)


def test_live_data_source_binding_rejects_other_audio_targets_or_unvalidated_model(tmp_path, monkeypatch):
    path, _ = checkpoint_fixture(tmp_path, monkeypatch)
    model = context.load_frozen_audio(path)
    for key in context.EXPECTED_INPUT_SHA256:
        lineage = {**context.EXPECTED_INPUT_SHA256, key: '0'*64}
        with pytest.raises(ValueError, match='differs'):
            context.assert_source_binding(model, lineage)
    with pytest.raises(ValueError, match='Validated'):
        context.assert_source_binding(model_fixture(), dict(context.EXPECTED_INPUT_SHA256))
