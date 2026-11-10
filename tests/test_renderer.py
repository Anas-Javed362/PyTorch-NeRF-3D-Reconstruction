"""
Unit tests for volume rendering (src/renderer.py).
"""

import math
import pytest
import torch
from src.renderer import VolumeRenderer


def test_render_output_keys():
    """Verify output dict has 'rgb', 'depth', 'weights', 'opacity' keys."""
    renderer = VolumeRenderer()
    batch_size = 4
    num_samples = 16

    rgb_sigma = torch.rand(batch_size, num_samples, 4)
    # sigma >= 0
    rgb_sigma[..., 3] = torch.abs(rgb_sigma[..., 3])
    z_vals = torch.linspace(2.0, 6.0, num_samples).unsqueeze(0).expand(batch_size, -1)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(batch_size, 3)

    out = renderer.render_rays(rgb_sigma, z_vals, rays_d)

    assert isinstance(out, dict)
    for key in ("rgb", "depth", "weights", "opacity"):
        assert key in out, f"Expected key '{key}' in render_rays output"

    assert out["rgb"].shape == (batch_size, 3)
    assert out["depth"].shape == (batch_size,)
    assert out["weights"].shape == (batch_size, num_samples)
    assert out["opacity"].shape == (batch_size,)


def test_rgb_output_range():
    """Verify rendered RGB is in [0, 1] with both white and non-white background."""
    renderer = VolumeRenderer()
    batch_size = 10
    num_samples = 32

    for white_bkgd in (True, False):
        rgb_sigma = torch.rand(batch_size, num_samples, 4)
        rgb_sigma[..., 3] = torch.abs(rgb_sigma[..., 3]) * 5.0
        z_vals = (
            torch.linspace(2.0, 6.0, num_samples).unsqueeze(0).expand(batch_size, -1)
        )
        rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(batch_size, 3)

        out = renderer.render_rays(rgb_sigma, z_vals, rays_d, white_bkgd=white_bkgd)
        rgb = out["rgb"]
        assert (rgb >= 0.0 - 1e-6).all(), f"RGB < 0 found (white_bkgd={white_bkgd})"
        assert (rgb <= 1.0 + 1e-6).all(), f"RGB > 1 found (white_bkgd={white_bkgd})"


def test_weights_sum_to_at_most_one():
    """Sum of weights along ray <= 1."""
    renderer = VolumeRenderer()
    batch_size = 8
    num_samples = 32

    rgb_sigma = torch.rand(batch_size, num_samples, 4)
    rgb_sigma[..., 3] = torch.abs(rgb_sigma[..., 3]) * 10.0
    z_vals = torch.linspace(1.0, 5.0, num_samples).unsqueeze(0).expand(batch_size, -1)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(batch_size, 3)

    out = renderer.render_rays(rgb_sigma, z_vals, rays_d)
    weights_sum = torch.sum(out["weights"], dim=-1)
    assert (weights_sum <= 1.0 + 1e-5).all()


def test_opacity_range():
    """Opacity (accumulated alpha) is in [0, 1]."""
    renderer = VolumeRenderer()
    batch_size = 8
    num_samples = 24

    rgb_sigma = torch.rand(batch_size, num_samples, 4)
    rgb_sigma[..., 3] = torch.abs(rgb_sigma[..., 3]) * 2.0
    z_vals = torch.linspace(2.0, 6.0, num_samples).unsqueeze(0).expand(batch_size, -1)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(batch_size, 3)

    out = renderer.render_rays(rgb_sigma, z_vals, rays_d)
    opacity = out["opacity"]
    assert (opacity >= 0.0 - 1e-6).all()
    assert (opacity <= 1.0 + 1e-5).all()


def test_weights_nonnegative():
    """All weights >= 0."""
    renderer = VolumeRenderer()
    batch_size = 6
    num_samples = 20

    rgb_sigma = torch.rand(batch_size, num_samples, 4)
    rgb_sigma[..., 3] = torch.abs(rgb_sigma[..., 3])
    z_vals = torch.linspace(2.0, 6.0, num_samples).unsqueeze(0).expand(batch_size, -1)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(batch_size, 3)

    out = renderer.render_rays(rgb_sigma, z_vals, rays_d)
    assert (out["weights"] >= 0.0).all()


def test_white_background():
    """With high density uniform rgba, RGB should be close to object color."""
    renderer = VolumeRenderer()
    num_samples = 32

    # High density object, color = pure green [0.0, 1.0, 0.0]
    rgb_sigma = torch.zeros(1, num_samples, 4)
    rgb_sigma[..., 1] = 1.0  # Green channel
    rgb_sigma[..., 3] = 100.0  # Extremely high density

    z_vals = torch.linspace(2.0, 6.0, num_samples).unsqueeze(0)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]])

    out = renderer.render_rays(rgb_sigma, z_vals, rays_d, white_bkgd=True)
    expected_green = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(out["rgb"], expected_green, atol=1e-3)


def test_black_background():
    """With zero density everywhere, with black bkgd, result should be black."""
    renderer = VolumeRenderer()
    num_samples = 32

    # Zero density everywhere, non-zero colors
    rgb_sigma = torch.ones(2, num_samples, 4)
    rgb_sigma[..., 3] = 0.0  # Density = 0

    z_vals = torch.linspace(2.0, 6.0, num_samples).unsqueeze(0).expand(2, -1)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(2, 3)

    out = renderer.render_rays(rgb_sigma, z_vals, rays_d, white_bkgd=False)
    zeros = torch.zeros_like(out["rgb"])
    assert torch.allclose(out["rgb"], zeros, atol=1e-6)
    assert torch.allclose(out["opacity"], torch.zeros(2), atol=1e-6)


def test_gradient_flows():
    """Verify gradient flows from rendered output back to sigma inputs."""
    renderer = VolumeRenderer()
    batch_size = 4
    num_samples = 16

    rgb = torch.rand(batch_size, num_samples, 3)
    sigma = torch.rand(batch_size, num_samples, requires_grad=True)

    rgb_sigma = torch.cat([rgb, sigma.unsqueeze(-1)], dim=-1)
    z_vals = torch.linspace(2.0, 6.0, num_samples).unsqueeze(0).expand(batch_size, -1)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(batch_size, 3)

    out = renderer.render_rays(rgb_sigma, z_vals, rays_d)
    loss = out["rgb"].sum() + out["depth"].sum()
    loss.backward()

    assert sigma.grad is not None
    assert not torch.isnan(sigma.grad).any()
    assert not torch.all(sigma.grad == 0)


def test_transmittance_decreasing():
    """Transmittance T_i should be non-increasing along ray."""
    renderer = VolumeRenderer()
    num_samples = 32

    sigma = torch.abs(torch.randn(5, num_samples)) + 0.1
    deltas = torch.full((5, num_samples), 0.1)

    alpha = renderer.compute_alpha(sigma, deltas)
    transmittance = renderer.compute_transmittance(alpha)

    # T_0 must be 1.0
    assert torch.allclose(transmittance[:, 0], torch.ones(5), atol=1e-6)

    # T_{i+1} <= T_i + eps
    diffs = transmittance[:, 1:] - transmittance[:, :-1]
    assert (diffs <= 1e-6).all()


def test_rendering_equation_manual():
    """Manually compute expected result for 2-sample case and verify."""
    renderer = VolumeRenderer()

    # 1 ray, 2 depth samples
    z_vals = torch.tensor([[1.0, 3.0]])
    rays_d = torch.tensor([[0.0, 0.0, -1.0]])

    # Colors: c0 = [0.8, 0.2, 0.4], c1 = [0.1, 0.9, 0.3]
    c0 = [0.8, 0.2, 0.4]
    c1 = [0.1, 0.9, 0.3]
    s0 = 0.5
    s1 = 0.3

    rgb_sigma = torch.tensor([[[c0[0], c0[1], c0[2], s0], [c1[0], c1[1], c1[2], s1]]])

    # Compute deltas:
    # delta_0 = 3.0 - 1.0 = 2.0
    # delta_1 = 1e10
    delta0 = 2.0
    delta1 = 1e10

    alpha0 = 1.0 - math.exp(-s0 * delta0)
    alpha1 = 1.0 - math.exp(-s1 * delta1)  # ~ 1.0

    T0 = 1.0
    T1 = 1.0 - alpha0

    w0 = T0 * alpha0
    w1 = T1 * alpha1

    # Expected RGB (without white bkgd)
    rgb_expected = torch.tensor(
        [
            [
                w0 * c0[0] + w1 * c1[0],
                w0 * c0[1] + w1 * c1[1],
                w0 * c0[2] + w1 * c1[2],
            ]
        ]
    )

    expected_depth = torch.tensor([w0 * 1.0 + w1 * 3.0])
    expected_weights = torch.tensor([[w0, w1]])

    out = renderer.render_rays(rgb_sigma, z_vals, rays_d, white_bkgd=False)

    assert torch.allclose(out["weights"], expected_weights, atol=1e-5)
    assert torch.allclose(out["rgb"], rgb_expected, atol=1e-5)
    assert torch.allclose(out["depth"], expected_depth, atol=1e-5)
