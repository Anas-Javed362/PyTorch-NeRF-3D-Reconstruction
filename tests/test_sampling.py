"""
Unit tests for ray point sampling (src/sampling.py).
"""

import pytest
import torch
from src.sampling import sample_points_along_rays, compute_sample_distances


def test_sample_points_shape():
    """Verify pts shape (..., num_samples, 3) for 1D and 2D batch shapes."""
    N = 32
    num_samples = 64
    rays_o = torch.zeros(N, 3)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(N, 3)

    pts, _ = sample_points_along_rays(
        rays_o, rays_d, near=2.0, far=6.0, num_samples=num_samples, stratified=False
    )
    assert pts.shape == (N, num_samples, 3)

    # 2D batch shape (H, W, 3)
    H, W = 10, 12
    rays_o_2d = torch.zeros(H, W, 3)
    rays_d_2d = torch.tensor([[0.0, 0.0, -1.0]]).expand(H, W, 3)

    pts_2d, _ = sample_points_along_rays(
        rays_o_2d, rays_d_2d, near=1.0, far=4.0, num_samples=num_samples, stratified=True
    )
    assert pts_2d.shape == (H, W, num_samples, 3)


def test_z_vals_shape():
    """Verify z_vals shape (..., num_samples)."""
    N = 16
    num_samples = 32
    rays_o = torch.zeros(N, 3)
    rays_d = torch.randn(N, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    _, z_vals = sample_points_along_rays(
        rays_o, rays_d, near=1.5, far=5.5, num_samples=num_samples
    )
    assert z_vals.shape == (N, num_samples)


def test_z_vals_bounds():
    """Verify all z_vals are within [near, far] with and without stratification."""
    near, far = 2.0, 6.0
    num_samples = 64
    rays_o = torch.zeros(20, 3)
    rays_d = torch.randn(20, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Stratified = False
    _, z_vals_det = sample_points_along_rays(
        rays_o, rays_d, near=near, far=far, num_samples=num_samples, stratified=False
    )
    assert (z_vals_det >= near - 1e-6).all()
    assert (z_vals_det <= far + 1e-6).all()

    # Stratified = True
    _, z_vals_rand = sample_points_along_rays(
        rays_o, rays_d, near=near, far=far, num_samples=num_samples, stratified=True
    )
    assert (z_vals_rand >= near - 1e-6).all()
    assert (z_vals_rand <= far + 1e-6).all()


def test_z_vals_ordered_no_stratified():
    """Without stratification, z_vals should be strictly increasing (non-decreasing)."""
    near, far = 1.0, 5.0
    num_samples = 48
    rays_o = torch.zeros(10, 3)
    rays_d = torch.randn(10, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    _, z_vals = sample_points_along_rays(
        rays_o, rays_d, near=near, far=far, num_samples=num_samples, stratified=False
    )
    diffs = z_vals[..., 1:] - z_vals[..., :-1]
    assert (diffs > 0).all()


def test_points_on_ray():
    """Verify pts = rays_o + z_vals * rays_d."""
    near, far = 2.0, 7.0
    num_samples = 16
    rays_o = torch.randn(8, 3)
    rays_d = torch.randn(8, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    pts, z_vals = sample_points_along_rays(
        rays_o, rays_d, near=near, far=far, num_samples=num_samples, stratified=True
    )

    expected_pts = rays_o.unsqueeze(1) + z_vals.unsqueeze(-1) * rays_d.unsqueeze(1)
    assert torch.allclose(pts, expected_pts, atol=1e-5)


def test_compute_deltas_shape():
    """Verify deltas shape (..., num_samples)."""
    N = 14
    num_samples = 32
    rays_o = torch.zeros(N, 3)
    rays_d = torch.randn(N, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    _, z_vals = sample_points_along_rays(
        rays_o, rays_d, near=2.0, far=6.0, num_samples=num_samples, stratified=False
    )
    deltas = compute_sample_distances(z_vals, rays_d)
    assert deltas.shape == (N, num_samples)


def test_deltas_positive():
    """Verify all deltas > 0 for unstratified sampling."""
    num_samples = 40
    rays_o = torch.zeros(12, 3)
    rays_d = torch.randn(12, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    _, z_vals = sample_points_along_rays(
        rays_o, rays_d, near=2.0, far=6.0, num_samples=num_samples, stratified=False
    )
    deltas = compute_sample_distances(z_vals, rays_d)
    assert (deltas > 0).all()
