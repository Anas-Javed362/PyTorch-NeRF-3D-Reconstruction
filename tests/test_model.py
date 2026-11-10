"""
Unit tests for NeRF neural network model (src/model.py).
"""

import pytest
import torch
from src.model import NeRF, count_parameters, create_nerf
from src.encoding import PositionalEncoder


def test_model_output_shape():
    """Verify rgb shape (B, 3) and sigma shape (B, 1) for batch input."""
    pos_enc_dim = 63
    dir_enc_dim = 27
    B = 64

    model = NeRF(pos_enc_dim=pos_enc_dim, dir_enc_dim=dir_enc_dim, use_viewdirs=True)
    x = torch.randn(B, pos_enc_dim)
    d = torch.randn(B, dir_enc_dim)

    out = model(x, d)
    assert "rgb" in out and "sigma" in out
    assert out["rgb"].shape == (B, 3)
    assert out["sigma"].shape == (B, 1)


def test_rgb_range():
    """Verify rgb values are in [0, 1] after sigmoid."""
    pos_enc_dim = 63
    dir_enc_dim = 27
    model = NeRF(pos_enc_dim=pos_enc_dim, dir_enc_dim=dir_enc_dim, use_viewdirs=True)

    x = torch.randn(128, pos_enc_dim) * 10.0
    d = torch.randn(128, dir_enc_dim) * 10.0

    out = model(x, d)
    rgb = out["rgb"]
    assert (rgb >= 0.0).all()
    assert (rgb <= 1.0).all()


def test_density_nonnegative():
    """Verify sigma >= 0 after ReLU."""
    pos_enc_dim = 63
    dir_enc_dim = 27
    model = NeRF(pos_enc_dim=pos_enc_dim, dir_enc_dim=dir_enc_dim, use_viewdirs=True)

    x = torch.randn(128, pos_enc_dim) * 10.0
    d = torch.randn(128, dir_enc_dim) * 10.0

    out = model(x, d)
    sigma = out["sigma"]
    assert (sigma >= 0.0).all()


def test_with_viewdirs():
    """Test model with use_viewdirs=True produces correct shapes and requires d_encoded."""
    pos_enc_dim = 63
    dir_enc_dim = 27
    model = NeRF(pos_enc_dim=pos_enc_dim, dir_enc_dim=dir_enc_dim, use_viewdirs=True)

    x = torch.randn(32, pos_enc_dim)
    d = torch.randn(32, dir_enc_dim)

    out = model(x, d)
    assert out["rgb"].shape == (32, 3)
    assert out["sigma"].shape == (32, 1)

    # Calling without d_encoded when use_viewdirs=True should raise ValueError
    with pytest.raises(ValueError):
        model(x, None)


def test_without_viewdirs():
    """Test model with use_viewdirs=False works without direction input."""
    pos_enc_dim = 63
    model = NeRF(pos_enc_dim=pos_enc_dim, use_viewdirs=False)

    x = torch.randn(32, pos_enc_dim)
    out = model(x)
    assert out["rgb"].shape == (32, 3)
    assert out["sigma"].shape == (32, 1)


def test_gradient_flow():
    """Verify gradients flow back through the entire model to all parameters."""
    pos_enc_dim = 63
    dir_enc_dim = 27
    model = NeRF(pos_enc_dim=pos_enc_dim, dir_enc_dim=dir_enc_dim, use_viewdirs=True)

    x = torch.randn(16, pos_enc_dim, requires_grad=True)
    d = torch.randn(16, dir_enc_dim, requires_grad=True)

    out = model(x, d)
    loss = out["rgb"].sum() + out["sigma"].sum()
    loss.backward()

    # Check that every trainable parameter has received a gradient
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    # Also input gradients
    assert x.grad is not None
    assert d.grad is not None


def test_parameter_count():
    """Verify model has a reasonable number of parameters (>100k and <10M)."""
    pos_enc_dim = 63
    dir_enc_dim = 27
    model = NeRF(
        pos_enc_dim=pos_enc_dim,
        dir_enc_dim=dir_enc_dim,
        num_layers=8,
        hidden_dim=256,
        skip_connect=(4,),
        use_viewdirs=True,
    )
    total_params = count_parameters(model)
    # Expected around 500,000 to 1,200,000 parameters
    assert 100_000 < total_params < 2_000_000


def test_create_nerf_factory():
    """Verify create_nerf function builds model from config dict."""
    config = {
        "pos_enc_freqs": 10,
        "dir_enc_freqs": 4,
        "num_layers": 4,
        "hidden_dim": 128,
        "skip_connect": [2],
        "use_viewdirs": True,
    }
    pos_enc, dir_enc, model = create_nerf(config)
    assert pos_enc is not None
    assert dir_enc is not None
    assert isinstance(model, NeRF)
    assert model.num_layers == 4
    assert model.hidden_dim == 128
    assert model.use_viewdirs is True

    # Test create_nerf with use_viewdirs=False
    config_no_dir = {
        "pos_enc_freqs": 6,
        "use_viewdirs": False,
    }
    pos_enc2, dir_enc2, model2 = create_nerf(config_no_dir)
    assert pos_enc2 is not None
    assert dir_enc2 is None
    assert model2.use_viewdirs is False
