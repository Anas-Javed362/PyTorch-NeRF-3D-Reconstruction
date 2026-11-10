"""
Camera Geometry for NeRF.

Implements camera ray generation from pixel coordinates and camera poses.

Conventions:
- Camera coordinate system: x right, y up, z backward (OpenGL/NeRF convention)
- Pixel coordinates: (0,0) at top-left, (W-1, H-1) at bottom-right
- Rays: origin (3,) + direction (3,) normalized unit vectors
- Camera-to-world (c2w) matrix: (4,4) or (3,4) transform from camera to world space
- Focal length in pixels
"""

import torch
import numpy as np
from typing import Tuple, Optional


def get_pixel_coords(H: int, W: int, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate pixel grid coordinates.
    
    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        device: PyTorch device.
        
    Returns:
        i: Column indices, shape (H, W), float32.
        j: Row indices, shape (H, W), float32.
    """
    j, i = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    return i, j


def pixel_to_camera_coords(
    i: torch.Tensor,
    j: torch.Tensor,
    focal: float,
    H: int,
    W: int,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> torch.Tensor:
    """
    Convert pixel coordinates to camera-space ray directions.
    
    Uses the pinhole camera model. The principal point defaults to image center.
    
    Args:
        i: Column pixel coords, any shape.
        j: Row pixel coords, same shape as i.
        focal: Focal length in pixels.
        H: Image height.
        W: Image width.
        cx: Principal point x (column). Defaults to W/2.
        cy: Principal point y (row). Defaults to H/2.
        
    Returns:
        dirs: Camera-space directions, shape (*i.shape, 3), float32.
              Convention: x right, y up (negate row), z backward (negate).
    """
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0
    
    # Pixel to normalized image coordinates, then to camera space
    # x: (col - cx) / focal
    # y: -(row - cy) / focal  (negate because image y is down, camera y is up)
    # z: -1 (looking down -z in OpenGL convention)
    x = (i - cx) / focal
    y = -(j - cy) / focal
    z = -torch.ones_like(x)
    
    dirs = torch.stack([x, y, z], dim=-1)  # (..., 3)
    return dirs


def get_rays(
    H: int,
    W: int,
    focal: float,
    c2w: torch.Tensor,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate camera rays for all pixels in an image.
    
    Args:
        H: Image height.
        W: Image width.
        focal: Focal length in pixels.
        c2w: Camera-to-world transformation matrix, shape (3,4) or (4,4).
        cx: Principal point x. Defaults to W/2.
        cy: Principal point y. Defaults to H/2.
        
    Returns:
        rays_o: Ray origins in world space, shape (H, W, 3).
        rays_d: Ray directions in world space (unit vectors), shape (H, W, 3).
    """
    device = c2w.device
    i, j = get_pixel_coords(H, W, device=device)
    
    # Camera-space directions: shape (H, W, 3)
    dirs = pixel_to_camera_coords(i, j, focal, H, W, cx, cy)
    
    # Rotate directions from camera to world space using the rotation part of c2w
    # dirs @ R^T = (R @ dirs^T)^T
    # c2w[:3, :3] is the rotation matrix
    R = c2w[:3, :3]  # (3, 3)
    rays_d = dirs @ R.T  # (H, W, 3)
    
    # Normalize to unit vectors
    rays_d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-10)
    
    # Ray origins: translation part of c2w, broadcast to all pixels
    rays_o = c2w[:3, 3].expand_as(rays_d)  # (H, W, 3)
    
    return rays_o, rays_d


def get_rays_from_poses(
    H: int,
    W: int,
    focal: float,
    c2w_batch: torch.Tensor,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for a batch of camera poses.
    
    Args:
        H: Image height.
        W: Image width.
        focal: Focal length in pixels.
        c2w_batch: Camera-to-world matrices, shape (N, 4, 4) or (N, 3, 4).
        cx: Principal point x.
        cy: Principal point y.
        
    Returns:
        rays_o: Ray origins, shape (N, H, W, 3).
        rays_d: Ray directions (unit vectors), shape (N, H, W, 3).
    """
    N = c2w_batch.shape[0]
    all_rays_o = []
    all_rays_d = []
    
    for n in range(N):
        rays_o, rays_d = get_rays(H, W, focal, c2w_batch[n], cx, cy)
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
    
    rays_o = torch.stack(all_rays_o, dim=0)  # (N, H, W, 3)
    rays_d = torch.stack(all_rays_d, dim=0)  # (N, H, W, 3)
    return rays_o, rays_d


def get_rays_flat(
    H: int,
    W: int,
    focal: float,
    c2w: torch.Tensor,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate flattened rays for a single camera pose.
    
    Returns:
        rays_o: Ray origins, shape (H*W, 3).
        rays_d: Ray directions, shape (H*W, 3).
    """
    rays_o, rays_d = get_rays(H, W, focal, c2w, cx, cy)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
