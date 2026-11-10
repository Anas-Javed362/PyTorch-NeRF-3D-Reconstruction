"""
Unit tests for camera geometry and ray generation (src/camera.py).
"""

import pytest
import torch
from src.camera import (
    get_pixel_coords,
    pixel_to_camera_coords,
    get_rays,
    get_rays_from_poses,
    get_rays_flat,
)


def test_get_pixel_coords():
    """Verify shape (H, W) for each of i, j, and verify values at corners."""
    H, W = 100, 150
    i, j = get_pixel_coords(H, W)

    assert i.shape == (H, W)
    assert j.shape == (H, W)
    assert i.dtype == torch.float32
    assert j.dtype == torch.float32

    # Corner checks
    # Top-left (row 0, col 0)
    assert torch.isclose(i[0, 0], torch.tensor(0.0))
    assert torch.isclose(j[0, 0], torch.tensor(0.0))

    # Top-right (row 0, col W-1)
    assert torch.isclose(i[0, W - 1], torch.tensor(float(W - 1)))
    assert torch.isclose(j[0, W - 1], torch.tensor(0.0))

    # Bottom-left (row H-1, col 0)
    assert torch.isclose(i[H - 1, 0], torch.tensor(0.0))
    assert torch.isclose(j[H - 1, 0], torch.tensor(float(H - 1)))

    # Bottom-right (row H-1, col W-1)
    assert torch.isclose(i[H - 1, W - 1], torch.tensor(float(W - 1)))
    assert torch.isclose(j[H - 1, W - 1], torch.tensor(float(H - 1)))


def test_pixel_to_camera_coords_shape():
    """Verify output shape (..., 3) for various input shapes."""
    H, W = 64, 64
    focal = 50.0
    i, j = get_pixel_coords(H, W)

    dirs = pixel_to_camera_coords(i, j, focal, H, W)
    assert dirs.shape == (H, W, 3)

    # Test with 1D flat coordinates
    i_flat = i.flatten()
    j_flat = j.flatten()
    dirs_flat = pixel_to_camera_coords(i_flat, j_flat, focal, H, W)
    assert dirs_flat.shape == (H * W, 3)

    # Test with single point
    dir_single = pixel_to_camera_coords(
        torch.tensor(32.0), torch.tensor(32.0), focal, H, W
    )
    assert dir_single.shape == (3,)


def test_pixel_center_direction():
    """For principal pixel (cx, cy), direction should be (0, 0, -1)."""
    H, W = 80, 100
    focal = 120.0
    cx = W / 2.0
    cy = H / 2.0

    i_center = torch.tensor(cx)
    j_center = torch.tensor(cy)

    dir_center = pixel_to_camera_coords(i_center, j_center, focal, H, W, cx=cx, cy=cy)
    expected = torch.tensor([0.0, 0.0, -1.0])
    assert torch.allclose(dir_center, expected, atol=1e-6)


def test_rays_origin_shape():
    """Verify get_rays returns (H, W, 3) shapes for origins and directions."""
    H, W = 32, 48
    focal = 40.0
    c2w = torch.eye(4)

    rays_o, rays_d = get_rays(H, W, focal, c2w)
    assert rays_o.shape == (H, W, 3)
    assert rays_d.shape == (H, W, 3)


def test_rays_direction_normalized():
    """Verify all directions have unit length (norm ≈ 1)."""
    H, W = 40, 40
    focal = 50.0
    c2w = torch.eye(4)

    _, rays_d = get_rays(H, W, focal, c2w)
    norms = torch.norm(rays_d, dim=-1)
    ones = torch.ones_like(norms)
    assert torch.allclose(norms, ones, atol=1e-5)


def test_rays_identity_pose():
    """With identity c2w, ray origins should be (0, 0, 0) for all pixels."""
    H, W = 30, 30
    focal = 35.0
    c2w = torch.eye(4)

    rays_o, _ = get_rays(H, W, focal, c2w)
    zeros = torch.zeros_like(rays_o)
    assert torch.allclose(rays_o, zeros, atol=1e-6)

    # Also test with non-zero translation
    c2w_translated = torch.eye(4)
    c2w_translated[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    rays_o_trans, _ = get_rays(H, W, focal, c2w_translated)
    expected_origin = torch.tensor([1.0, 2.0, 3.0]).expand_as(rays_o_trans)
    assert torch.allclose(rays_o_trans, expected_origin, atol=1e-6)


def test_get_rays_flat():
    """Verify flattened output shapes (H*W, 3)."""
    H, W = 20, 25
    focal = 30.0
    c2w = torch.eye(4)

    rays_o_flat, rays_d_flat = get_rays_flat(H, W, focal, c2w)
    assert rays_o_flat.shape == (H * W, 3)
    assert rays_d_flat.shape == (H * W, 3)
