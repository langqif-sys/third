"""Structural/regression checks; these do not establish forecast accuracy."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.TimeEmb import Model as Original
from models.TimeEmb_FRCA import Model, ReferenceAttention
from utils.forecast_loss import forecast_loss

torch.set_num_threads(2)


def config(**overrides):
    options = dict(seq_len=33, pred_len=13, enc_in=3, d_model=32, use_revin=1,
                   use_hour_index=1, use_day_index=1, hour_length=24, day_length=7,
                   frca_dim=16, frca_heads=4, frca_patch_len=8, frca_stride=4,
                   frca_dropout=0.0, frca_mode='complement', frca_beta_init=0.1,
                   frca_beta_fixed=-1.0, frca_eta_init=0.01, frca_detach_reference=1,
                   frca_reference='timeemb', frca_ma_kernel=5, frca_inverted=1)
    options.update(overrides)
    return SimpleNamespace(**options)


@pytest.mark.parametrize('use_revin', [0, 1])
@pytest.mark.parametrize('length', [32, 33])
def test_frequency_path_matches_original(use_revin, length):
    torch.manual_seed(10)
    cfg = config(use_revin=use_revin, seq_len=length)
    original, hybrid = Original(cfg).eval(), Model(cfg).eval()
    with torch.no_grad():
        original.emb_hour.normal_(std=0.1)
        original.emb_day.normal_(std=0.1)
    hybrid.load_state_dict(original.state_dict(), strict=False)
    hybrid.eta.data.zero_()
    x = torch.randn(3, length, 3)
    hour, day = torch.tensor([0, 23, 25]), torch.tensor([0, 6, 8])
    with torch.no_grad():
        torch.testing.assert_close(hybrid(x, hour, day), original(x, hour, day), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('mode', ['complement', 'plain', 'residual'])
@pytest.mark.parametrize('length, channels', [(7, 1), (33, 3), (96, 7)])
def test_forward_backward_shapes_and_finite_values(mode, length, channels):
    torch.manual_seed(11)
    model = Model(config(frca_mode=mode, seq_len=length, enc_in=channels))
    x = torch.randn(3, length, channels)
    y = model(x, torch.arange(3), torch.arange(3))
    assert y.shape == (3, 13, channels)
    y.square().mean().backward()
    assert torch.isfinite(y).all()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
    assert model.w.grad.abs().sum() > 0
    assert model.eta.grad.abs().sum() > 0
    if mode == 'complement':
        assert model.attention.beta_logits.shape == (4,)
        assert (model.attention.beta_logits.grad.abs() > 0).all()


def test_reference_changes_values_but_not_relations():
    torch.manual_seed(12)
    attn = Model(config()).attention
    zx, zs = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    pos = torch.randn(1, 5, 16)
    q1, k1, v1, _ = attn.project_qkv(zx, zs, pos)
    q2, k2, v2, _ = attn.project_qkv(zx, 2 * zs, pos)
    torch.testing.assert_close(q1, q2, atol=0, rtol=0)
    torch.testing.assert_close(k1, k2, atol=0, rtol=0)
    assert not torch.allclose(v1, v2)


def test_beta_zero_equals_unreferenced_attention():
    torch.manual_seed(13)
    complement = ReferenceAttention(16, 4, 'complement', 0.1, 0.0, 0.0, True).eval()
    plain = ReferenceAttention(16, 4, 'plain', 0.1, -1, 0.0, True).eval()
    plain.load_state_dict(complement.state_dict())
    zx, zs, pos = torch.randn(2, 5, 16), torch.randn(2, 5, 16), torch.randn(1, 5, 16)
    torch.testing.assert_close(complement(zx, zs, pos), plain(zx, zs, pos), atol=0, rtol=0)


def test_equal_reference_cancels_values_not_position():
    attn = ReferenceAttention(16, 4, 'complement', 0.1, 1.0, 0.0, True)
    z, pos = torch.randn(2, 5, 16), torch.randn(1, 5, 16)
    q, k, v, _ = attn.project_qkv(z, z, pos)
    assert torch.count_nonzero(v) == 0
    assert torch.count_nonzero(q) > 0 and torch.count_nonzero(k) > 0


@pytest.mark.parametrize('detach', [0, 1])
def test_reference_stop_gradient_does_not_freeze_frequency_path(detach):
    torch.manual_seed(14)
    model = Model(config(frca_detach_reference=detach))
    x, hours, days = torch.randn(2, 33, 3), torch.arange(2), torch.arange(2)
    _, parts = model(x, hours, days, return_parts=True)
    parts['correction'].square().mean().backward()
    if detach:
        assert model.w.grad is None
    else:
        assert model.w.grad is not None and model.w.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model(x, hours, days).square().mean().backward()
    assert model.w.grad.abs().sum() > 0


def test_no_batch_or_channel_leakage():
    torch.manual_seed(15)
    model = Model(config()).eval()
    x = torch.randn(3, 33, 3)
    changed = x.clone()
    changed[0, :, 0] += torch.randn(33) * 5
    with torch.no_grad():
        a = model(x, torch.arange(3), torch.arange(3))
        b = model(changed, torch.arange(3), torch.arange(3))
    torch.testing.assert_close(a[1:], b[1:])
    torch.testing.assert_close(a[0, :, 1:], b[0, :, 1:])


def test_cpu_autocast_non_power_of_two_sequence():
    model = Model(config())
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = model(torch.randn(2, 33, 3), torch.arange(2), torch.arange(2))
        loss = out.square().mean()
    loss.backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(model.attention.beta_logits.grad).all()


def test_short_optimization_updates_betas_and_reduces_training_loss():
    torch.manual_seed(16)
    model = Model(config(use_revin=0, frca_eta_init=0.1))
    x = torch.randn(4, 33, 3)
    target = x[:, -13:, :] * 0.5
    hours, days = torch.arange(4), torch.arange(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial_beta = model.attention.beta().detach().clone()
    initial_loss = (model(x, hours, days) - target).square().mean().item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = (model(x, hours, days) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert loss.item() < initial_loss * 0.5
    assert not torch.equal(initial_beta, model.attention.beta().detach())


def test_moving_average_reference_and_standard_attention_ablation():
    model = Model(config(frca_reference='moving_average', frca_inverted=0))
    out = model(torch.randn(2, 33, 3), torch.arange(2), torch.arange(2))
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert model.attention.beta_logits.grad.abs().sum() > 0


@pytest.mark.parametrize('mode', ['fft', 'rfft', 'rfft-D', 'rfft-2D'])
def test_shared_loss_matches_original_formula(mode):
    torch.manual_seed(17)
    x, y = torch.randn(2, 13, 3, requires_grad=True), torch.randn(2, 13, 3)
    args = SimpleNamespace(rec_lambda=0.25, auxi_lambda=0.75, auxi_mode=mode,
                           auxi_type='complex', auxi_loss='MAE', module_first=1)
    if mode == 'fft':
        a, b = torch.fft.fft(x, dim=1), torch.fft.fft(y, dim=1)
    elif mode == 'rfft':
        a, b = torch.fft.rfft(x, dim=1), torch.fft.rfft(y, dim=1)
    elif mode == 'rfft-D':
        a, b = torch.fft.rfft(x, dim=-1), torch.fft.rfft(y, dim=-1)
    else:
        a, b = torch.fft.rfft2(x), torch.fft.rfft2(y)
    expected = 0.25 * (x - y).square().mean() + 0.75 * (a - b).abs().mean()
    actual = forecast_loss(x, y, args)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(x.grad).all()


def test_invalid_heads_and_missing_day_index_fail_clearly():
    with pytest.raises(ValueError, match='divisible'):
        Model(config(frca_dim=16, frca_heads=3))
    model = Model(config())
    with pytest.raises(ValueError, match='Day bank'):
        model(torch.randn(2, 33, 3), torch.arange(2), torch.full((2,), -1))
