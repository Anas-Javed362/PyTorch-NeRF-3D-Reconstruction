"""
Volume Renderer for NeRF.

Implements the differentiable volume rendering equation:
  C(r) = Σ_i T_i * alpha_i * c_i

Where:
  alpha_i = 1 - exp(-sigma_i * delta_i)
  T_i = exp(-Σ_{j<i} sigma_j * delta_j)   (transmittance, exclusive prefix product)
  delta_i = distance between adjacent samples
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Callable


class VolumeRenderer:
    """Differentiable volume renderer for NeRF."""

    def compute_deltas(self, z_vals: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between adjacent sample points along each ray.

        Args:
            z_vals: Depth values along rays, shape (batch, num_samples).
            rays_d: Ray directions in world space, shape (batch, 3).
                    Used to convert depth deltas to real-space distances.

        Returns:
            deltas: Distances between adjacent samples, shape (batch, num_samples).
                    The last delta is set to 1e10 (infinity) to represent the
                    open-ended last interval.
        """
        # Differences between consecutive depth values: (batch, num_samples-1)
        diffs = z_vals[..., 1:] - z_vals[..., :-1]

        # Append a large value for the last (open) interval
        large_val = torch.full(
            (*z_vals.shape[:-1], 1), 1e10, dtype=z_vals.dtype, device=z_vals.device
        )
        diffs = torch.cat([diffs, large_val], dim=-1)  # (batch, num_samples)

        # Scale by the norm of ray direction to get world-space distances.
        # rays_d may already be normalised (unit), but we handle both cases.
        rays_d_norm = torch.norm(rays_d, dim=-1, keepdim=True)  # (batch, 1)
        deltas = diffs * rays_d_norm  # (batch, num_samples)
        return deltas

    def compute_alpha(self, sigma: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample alpha (opacity) values.

        alpha_i = 1 - exp(-sigma_i * delta_i), clamped to [0, 1].

        Args:
            sigma: Volume density, shape (batch, num_samples) or (batch, num_samples, 1).
                   Must be >= 0.
            deltas: Sample interval lengths, shape (batch, num_samples).

        Returns:
            alpha: Per-sample opacity, shape (batch, num_samples), in [0, 1].
        """
        if sigma.dim() == deltas.dim() + 1 and sigma.shape[-1] == 1:
            sigma = sigma.squeeze(-1)
        elif deltas.dim() == sigma.dim() + 1 and deltas.shape[-1] == 1:
            deltas = deltas.squeeze(-1)

        alpha = 1.0 - torch.exp(-sigma * deltas)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        return alpha

    def compute_transmittance(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute transmittance T_i via exclusive cumulative product.

        T_0 = 1
        T_i = prod_{j=0}^{i-1} (1 - alpha_j)

        We compute an out-of-place exclusive prefix product by prepending 1.0
        along the sample dimension and dropping the last product, ensuring full
        gradient flow with zero in-place mutations.

        Args:
            alpha: Per-sample opacity, shape (batch, num_samples).

        Returns:
            transmittance: Shape (batch, num_samples), T_i values.
        """
        eps = 1e-10
        # (1 - alpha), clamped for numerical stability
        one_minus_alpha = torch.clamp(1.0 - alpha + eps, 0.0, 1.0)

        ones = torch.ones((*alpha.shape[:-1], 1), dtype=alpha.dtype, device=alpha.device)
        # Exclusive cumprod: prepend 1.0 and slice [..., :-1]
        transmittance = torch.cumprod(
            torch.cat([ones, one_minus_alpha], dim=-1), dim=-1
        )[..., :-1]
        return transmittance

    def compute_weights(
        self, alpha: torch.Tensor, transmittance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rendering weights: w_i = T_i * alpha_i.

        Args:
            alpha: Per-sample opacity, shape (batch, num_samples).
            transmittance: Transmittance T_i, shape (batch, num_samples).

        Returns:
            weights: Rendering weights, shape (batch, num_samples).
                     Weights sum to <= 1 along the sample dimension.
        """
        weights = transmittance * alpha
        return weights

    def composite_rgb(
        self, weights: torch.Tensor, rgb: torch.Tensor
    ) -> torch.Tensor:
        """
        Composite RGB colors using rendering weights.

        C(r) = Σ_i w_i * c_i

        Args:
            weights: Rendering weights, shape (batch, num_samples).
            rgb: Per-sample colors, shape (batch, num_samples, 3).

        Returns:
            rgb_map: Rendered RGB image, shape (batch, 3).
        """
        # weights: (batch, S) -> (batch, S, 1) for broadcasting
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)  # (batch, 3)
        return rgb_map

    def composite_depth(
        self, weights: torch.Tensor, z_vals: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute expected depth map (weighted sum of depth values).

        depth(r) = Σ_i w_i * z_i

        Args:
            weights: Rendering weights, shape (batch, num_samples).
            z_vals: Depth sample values, shape (batch, num_samples).

        Returns:
            depth_map: Expected depth, shape (batch,).
        """
        depth_map = torch.sum(weights * z_vals, dim=-1)  # (batch,)
        return depth_map

    def render_rays(
        self,
        rgb_sigma: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        white_bkgd: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform volume rendering for a batch of rays.

        Args:
            rgb_sigma: Model output, shape (batch, num_samples, 4).
                       rgb_sigma[..., :3] = RGB colors (after sigmoid, in [0,1])
                       rgb_sigma[..., 3:] = volume density sigma (after ReLU, >= 0)
            z_vals: Depth values along rays, shape (batch, num_samples).
            rays_d: Ray directions in world space, shape (batch, 3).
            white_bkgd: If True, composite against a white background.

        Returns:
            dict with keys:
                'rgb':     Rendered RGB colors, shape (batch, 3).
                'depth':   Expected depth map, shape (batch,).
                'weights': Per-sample rendering weights, shape (batch, num_samples).
                'opacity': Accumulated opacity (alpha), shape (batch,).
        """
        # Split model output
        rgb = rgb_sigma[..., :3]      # (batch, num_samples, 3)
        sigma = rgb_sigma[..., 3]     # (batch, num_samples)

        # 1. Compute distances between samples
        deltas = self.compute_deltas(z_vals, rays_d)      # (batch, num_samples)

        # 2. Compute per-sample alpha
        alpha = self.compute_alpha(sigma, deltas)          # (batch, num_samples)

        # 3. Compute transmittance (exclusive prefix product)
        transmittance = self.compute_transmittance(alpha)  # (batch, num_samples)

        # 4. Compute rendering weights
        weights = self.compute_weights(alpha, transmittance)  # (batch, num_samples)

        # 5. Composite RGB
        rgb_map = self.composite_rgb(weights, rgb)         # (batch, 3)

        # 6. Compute expected depth
        depth_map = self.composite_depth(weights, z_vals)  # (batch,)

        # 7. Accumulated opacity
        opacity = torch.sum(weights, dim=-1)               # (batch,)

        # 8. White background compositing
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - opacity.unsqueeze(-1))

        return {
            "rgb": rgb_map,
            "depth": depth_map,
            "weights": weights,
            "opacity": opacity,
        }


# ---------------------------------------------------------------------------
# Chunked rendering helper
# ---------------------------------------------------------------------------

def render_rays_chunked(
    model: torch.nn.Module,
    encoder_pos: Callable,
    encoder_dir: Callable,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    config: dict,
    chunk_size: int = 1024,
) -> Dict[str, torch.Tensor]:
    """
    Render a large batch of rays in smaller chunks to avoid OOM errors.

    This function samples points along each ray, encodes positions and
    directions, queries the NeRF model, and applies volume rendering —
    all in chunks of `chunk_size` rays.

    Args:
        model:       NeRF network (nn.Module). Called as model(pos_enc, dir_enc).
        encoder_pos: Positional encoder for 3-D sample positions.
                     Called as encoder_pos(pts) -> encoded tensor.
        encoder_dir: Directional encoder for view directions.
                     Called as encoder_dir(dirs) -> encoded tensor.
        rays_o:      Ray origins, shape (N, 3).
        rays_d:      Ray directions, shape (N, 3).
        config:      Dict with keys:
                       'near'        (float) – near plane depth.
                       'far'         (float) – far plane depth.
                       'num_samples' (int)   – number of depth samples per ray.
                       'white_bkgd'  (bool)  – white background compositing.
                       'perturb'     (bool)  – add random perturbation to samples.
        chunk_size:  Number of rays to process at once.

    Returns:
        Dict with keys 'rgb', 'depth', 'weights', 'opacity', each concatenated
        over all chunks (shapes consistent with render_rays output).
    """
    near = config.get("near", 2.0)
    far = config.get("far", 6.0)
    num_samples = config.get("num_samples", 64)
    white_bkgd = config.get("white_bkgd", True)
    perturb = config.get("perturb", False)

    renderer = VolumeRenderer()
    device = rays_o.device
    N = rays_o.shape[0]

    all_rgb: list = []
    all_depth: list = []
    all_weights: list = []
    all_opacity: list = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        rays_o_chunk = rays_o[start:end]   # (C, 3)
        rays_d_chunk = rays_d[start:end]   # (C, 3)
        C = rays_o_chunk.shape[0]

        # ---- Sample depths ------------------------------------------------
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
        z_vals = near * (1.0 - t_vals) + far * t_vals   # (num_samples,)
        z_vals = z_vals.unsqueeze(0).expand(C, -1)       # (C, num_samples)

        if perturb:
            # Stratified sampling: add uniform noise within each bin
            mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mid, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mid], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

        # ---- Compute 3-D sample positions ---------------------------------
        # pts: (C, num_samples, 3)
        pts = rays_o_chunk.unsqueeze(1) + rays_d_chunk.unsqueeze(1) * z_vals.unsqueeze(-1)

        # ---- Encode positions ---------------------------------------------
        pts_flat = pts.reshape(-1, 3)                    # (C*S, 3)
        pos_enc = encoder_pos(pts_flat)                  # (C*S, pos_dim)

        # ---- Encode directions --------------------------------------------
        if encoder_dir is not None:
            dirs = rays_d_chunk / (torch.norm(rays_d_chunk, dim=-1, keepdim=True) + 1e-10)
            dirs_expanded = dirs.unsqueeze(1).expand_as(pts) # (C, S, 3)
            dirs_flat = dirs_expanded.reshape(-1, 3)         # (C*S, 3)
            dir_enc = encoder_dir(dirs_flat)                 # (C*S, dir_dim)
        else:
            dir_enc = None

        # ---- Query model --------------------------------------------------
        with torch.no_grad() if not model.training else torch.enable_grad():
            if dir_enc is not None:
                raw = model(pos_enc, dir_enc)
            else:
                raw = model(pos_enc)

        if isinstance(raw, dict):
            rgb = raw["rgb"].reshape(C, num_samples, 3)
            sigma = raw["sigma"].reshape(C, num_samples, 1)
            rgb_sigma = torch.cat([rgb, sigma], dim=-1)
        else:
            raw = raw.reshape(C, num_samples, 4)
            rgb_sigma = torch.cat(
                [torch.sigmoid(raw[..., :3]), F.relu(raw[..., 3:])], dim=-1
            )

        # ---- Volume render ------------------------------------------------
        out = renderer.render_rays(rgb_sigma, z_vals, rays_d_chunk, white_bkgd)

        all_rgb.append(out["rgb"])
        all_depth.append(out["depth"])
        all_weights.append(out["weights"])
        all_opacity.append(out["opacity"])

    return {
        "rgb": torch.cat(all_rgb, dim=0),
        "depth": torch.cat(all_depth, dim=0),
        "weights": torch.cat(all_weights, dim=0),
        "opacity": torch.cat(all_opacity, dim=0),
    }
