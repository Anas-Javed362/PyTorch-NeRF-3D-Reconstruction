"""
Ray Sampling for NeRF.

Implements stratified sampling of 3D points along camera rays between
configurable near and far bounds.

The sampling strategy:
1. Divide [near, far] into N evenly-spaced bins.
2. Optionally sample uniformly within each bin (stratified/jittered sampling).
3. During training use stratified=True to reduce aliasing.
4. During evaluation use stratified=False for determinism.
"""

import torch
from typing import Tuple


def sample_points_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    stratified: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 3D points along camera rays.
    
    Args:
        rays_o: Ray origins, shape (..., 3).
        rays_d: Ray directions (unit vectors), shape (..., 3).
        near: Near bound for ray sampling (distance from origin).
        far: Far bound for ray sampling.
        num_samples: Number of sample points per ray.
        stratified: If True, add uniform noise within each bin (training).
                    If False, use bin midpoints (evaluation).
        
    Returns:
        pts: 3D sample points in world space, shape (..., num_samples, 3).
        z_vals: Depth values along rays, shape (..., num_samples).
    """
    batch_shape = rays_o.shape[:-1]  # Everything except the last dim (3)
    device = rays_o.device
    
    # Create evenly spaced depth values in [near, far]
    # z_vals_base: (num_samples,) linspace from near to far
    t = torch.linspace(0.0, 1.0, num_samples, device=device)
    z_vals = near * (1.0 - t) + far * t  # (num_samples,)
    
    # Expand to match batch dimensions: (..., num_samples)
    z_vals = z_vals.expand(*batch_shape, num_samples)
    
    if stratified:
        # Add uniform noise within each bin to get stratified samples
        # Bin width
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        
        # Uniform noise in [0, 1) for each sample
        noise = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * noise
    
    # Clamp to [near, far] for safety
    z_vals = z_vals.clamp(near, far)
    
    # Compute 3D sample positions: origin + depth * direction
    # rays_o: (..., 3), rays_d: (..., 3), z_vals: (..., num_samples)
    # Expand for broadcasting: (..., num_samples, 3)
    pts = rays_o[..., None, :] + z_vals[..., :, None] * rays_d[..., None, :]
    # pts shape: (..., num_samples, 3)
    
    return pts, z_vals


def compute_sample_distances(
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    last_delta: float = 1e10,
) -> torch.Tensor:
    """
    Compute distances between adjacent sample points along each ray.
    
    These distances (deltas) are used in the volume rendering equation.
    
    Args:
        z_vals: Depth values, shape (..., num_samples).
        rays_d: Ray directions (unit vectors), shape (..., 3).
        last_delta: Distance assigned to the last sample (effectively infinity).
        
    Returns:
        deltas: Distances between adjacent samples, shape (..., num_samples).
                Scaled by the ray direction magnitude (for non-unit rays).
    """
    # Differences between adjacent depth values
    diffs = z_vals[..., 1:] - z_vals[..., :-1]  # (..., num_samples-1)
    
    # Pad with a large value for the last sample (open-ended)
    last = torch.full((*z_vals.shape[:-1], 1), last_delta, device=z_vals.device)
    diffs = torch.cat([diffs, last], dim=-1)  # (..., num_samples)
    
    # Scale by the norm of the ray direction
    # For unit direction vectors this is 1, but we keep it general
    ray_norm = torch.norm(rays_d, dim=-1, keepdim=True)  # (..., 1)
    deltas = diffs * ray_norm  # (..., num_samples)
    
    return deltas
