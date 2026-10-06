import torch

from scripts.faceformer_arkit_model import FaceFormerARKit, FaceFormerARKitConfig, masked_mse


def _batch(n=2, t=6):
    g = torch.Generator().manual_seed(7)
    return (torch.randn(n, t, 1540, generator=g),
            torch.randn(n, t, 52, generator=g),
            torch.randn(n, 52, generator=g),
            torch.ones(n, t, dtype=torch.bool))


def test_faceformer_forward_and_autoregressive_shapes():
    audio, target, anchor, valid = _batch()
    model = FaceFormerARKit(FaceFormerARKitConfig(feature_dim=16, heads=4,
                                                  max_seq_len=32, dropout=0.0))
    model.eval()
    teacher = model(audio, anchor, target, valid, teacher_forcing=True)
    generated = model.predict(audio, anchor)
    assert teacher.shape == target.shape
    assert generated.shape == target.shape
    assert torch.isfinite(teacher).all() and torch.isfinite(generated).all()


def test_zero_initialized_head_preserves_anchor_before_training():
    audio, target, anchor, valid = _batch(1, 4)
    model = FaceFormerARKit(FaceFormerARKitConfig(feature_dim=16, heads=4,
                                                  max_seq_len=32, dropout=0.0))
    prediction = model(audio, anchor, target, valid, teacher_forcing=True)
    assert torch.allclose(prediction, anchor[:, None].expand_as(prediction))
    assert masked_mse(prediction, target, valid).item() >= 0


def test_autoregression_matches_teacher_forcing_on_own_predictions():
    torch.set_num_threads(1)
    torch.manual_seed(91)
    audio, _, anchor, valid = _batch(1, 5)
    model = FaceFormerARKit(FaceFormerARKitConfig(feature_dim=16, heads=4,
                                                  max_seq_len=32, dropout=0.0)).eval()
    torch.nn.init.normal_(model.motion_map_r.weight, std=.1)
    with torch.no_grad():
        generated = model.predict(audio, anchor)
        forced = model(audio, anchor, generated, valid)
    torch.testing.assert_close(generated, forced, atol=1e-6, rtol=1e-5)
