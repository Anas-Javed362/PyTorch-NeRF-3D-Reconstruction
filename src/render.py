"""
Novel View Synthesis Rendering for NeRF.

Generates rendered images from novel camera viewpoints using a trained NeRF model.
Supports spherical camera trajectories and GIF/PNG output.

Usage:
    python -m src.render --checkpoint outputs/checkpoints/best.pth
    python -m src.render --checkpoint outputs/checkpoints/best.pth --num_frames 60 --output_dir outputs/renders
"""

import os
import sys
import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

# ---------------------------------------------------------------------------
# Camera trajectory generation
# ---------------------------------------------------------------------------

def generate_spherical_poses(
    num_frames: int,
    radius: float = 4.0,
    elevation: float = -30.0,
) -> List[np.ndarray]:
    """
    Generate camera-to-world matrices for a spherical camera trajectory.

    Cameras are placed on the surface of a sphere of given *radius* looking
    at the world origin.  The azimuth sweeps 0–360° evenly across
    *num_frames* frames; *elevation* (degrees, negative = below the equator)
    sets the fixed latitude.

    Args:
        num_frames:  Number of frames (poses) to generate.
        radius:      Distance from scene centre to each camera.
        elevation:   Camera elevation in degrees.  Negative means below the
                     horizontal plane.

    Returns:
        List of (4, 4) float32 camera-to-world matrices in the OpenGL / NeRF
        convention (x right, y up, z backward).
    """
    poses: List[np.ndarray] = []

    # Convert elevation to radians
    phi = math.radians(elevation)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    for i in range(num_frames):
        # Azimuth angle in radians
        theta = 2.0 * math.pi * i / num_frames

        # Camera centre in world coordinates
        cx = radius * cos_phi * math.cos(theta)
        cy = radius * sin_phi              # elevation component (y-up)
        cz = radius * cos_phi * math.sin(theta)
        cam_pos = np.array([cx, cy, cz], dtype=np.float32)

        # Build orthonormal basis: forward = -cam_pos (looking at origin)
        forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)

        # World up vector; handle degenerate case near poles
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right) + 1e-8

        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-8

        # Build c2w: columns are right, up, -forward (OpenGL convention)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward   # z-axis points *away* from scene (backward)
        c2w[:3, 3] = cam_pos

        poses.append(c2w)

    return poses


# ---------------------------------------------------------------------------
# Single-image rendering
# ---------------------------------------------------------------------------

def render_image(
    model,
    encoder_pos,
    encoder_dir,
    pose: np.ndarray,
    H: int,
    W: int,
    focal: float,
    config: dict,
    device: torch.device,
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    Render a single image from a given camera pose.

    Generates pixel rays, queries the NeRF model in chunks to avoid OOM,
    and composites the final RGB image.

    Args:
        model:        Trained NeRF network (nn.Module).
        encoder_pos:  Positional encoder for 3-D positions.
        encoder_dir:  Positional encoder for viewing directions.
        pose:         (4, 4) camera-to-world matrix (numpy or tensor).
        H, W:         Image height and width in pixels.
        focal:        Focal length in pixels.
        config:       Configuration dict with keys ``near``, ``far``,
                      ``num_samples``, ``perturb``.
        device:       Torch device.
        chunk_size:   Number of rays processed per forward pass.

    Returns:
        Rendered RGB image as a (H, W, 3) float32 numpy array in [0, 1].
    """
    model.eval()

    # Convert pose to tensor
    if not isinstance(pose, torch.Tensor):
        pose_t = torch.tensor(pose, dtype=torch.float32, device=device)
    else:
        pose_t = pose.to(device)

    # -- Build pixel grid --------------------------------------------------
    i_coords = torch.arange(W, dtype=torch.float32, device=device)
    j_coords = torch.arange(H, dtype=torch.float32, device=device)
    j_grid, i_grid = torch.meshgrid(j_coords, i_coords, indexing='ij')  # (H, W)

    # Direction in camera space (OpenGL: x right, y up, z backward)
    dirs_x = (i_grid - W * 0.5) / focal
    dirs_y = -(j_grid - H * 0.5) / focal
    dirs_z = -torch.ones_like(dirs_x)
    dirs_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)

    # Rotate to world space
    rot = pose_t[:3, :3]                             # (3, 3)
    dirs_world = (dirs_cam.reshape(-1, 3) @ rot.T)  # (H*W, 3)
    dirs_world = dirs_world / (dirs_world.norm(dim=-1, keepdim=True) + 1e-8)

    # Ray origins = camera position (same for every pixel)
    origins = pose_t[:3, 3].unsqueeze(0).expand(H * W, -1)  # (H*W, 3)

    # -- Volume rendering parameters ---------------------------------------
    near = float(config.get('near', 2.0))
    far = float(config.get('far', 6.0))
    num_samples = int(config.get('num_samples', 64))

    # Sample depths along each ray (stratified, no perturbation at eval time)
    t_vals = torch.linspace(near, far, num_samples, device=device)
    t_vals = t_vals.unsqueeze(0).expand(H * W, -1)  # (N_rays, N_samples)

    # 3-D sample positions
    pts = origins.unsqueeze(1) + dirs_world.unsqueeze(1) * t_vals.unsqueeze(2)
    # pts: (N_rays, N_samples, 3)

    # Flatten for batched encoder/model calls
    pts_flat = pts.reshape(-1, 3)                                  # (N_rays*N_s, 3)
    dirs_flat = dirs_world.unsqueeze(1).expand_as(pts).reshape(-1, 3)

    # -- Chunked forward pass ---------------------------------------------
    all_rgb = []
    all_sigma = []
    n_pts = pts_flat.shape[0]

    with torch.no_grad():
        for start in range(0, n_pts, chunk_size):
            end = min(start + chunk_size, n_pts)
            pts_chunk = pts_flat[start:end]
            dirs_chunk = dirs_flat[start:end]

            enc_pos = encoder_pos(pts_chunk)
            enc_dir = encoder_dir(dirs_chunk) if encoder_dir is not None else None

            out = model(enc_pos, enc_dir)   # (chunk, 4) or dict with 'rgb', 'sigma'

            if isinstance(out, dict):
                rgb_chunk = out['rgb']
                sigma_chunk = out['sigma']
                if sigma_chunk.dim() > 1 and sigma_chunk.shape[-1] == 1:
                    sigma_chunk = sigma_chunk.squeeze(-1)
            else:
                rgb_chunk = torch.sigmoid(out[..., :3])
                sigma_chunk = F.relu(out[..., 3])

            all_rgb.append(rgb_chunk)
            all_sigma.append(sigma_chunk)

    rgb_flat = torch.cat(all_rgb, dim=0)    # (N_rays*N_s, 3)
    sigma_flat = torch.cat(all_sigma, dim=0)

    # Reshape back
    n_rays = H * W
    rgb_vol = rgb_flat.reshape(n_rays, num_samples, 3)
    sigma_vol = sigma_flat.reshape(n_rays, num_samples)

    # -- Volume composite (alpha compositing) -----------------------------
    deltas = torch.zeros_like(t_vals)
    deltas[:, :-1] = t_vals[:, 1:] - t_vals[:, :-1]
    deltas[:, -1] = 1e10  # last segment extends to infinity

    alpha = 1.0 - torch.exp(-sigma_vol * deltas)
    trans = torch.cumprod(
        torch.cat([torch.ones((n_rays, 1), device=device),
                   1.0 - alpha + 1e-10], dim=1),
        dim=1
    )[:, :-1]
    weights = alpha * trans                # (N_rays, N_samples)

    rgb_map = (weights.unsqueeze(-1) * rgb_vol).sum(dim=1)  # (N_rays, 3)
    rgb_map = rgb_map.reshape(H, W, 3)

    return rgb_map.cpu().numpy().clip(0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------

def save_image_sequence(
    images: List[np.ndarray],
    output_dir: str,
    prefix: str = 'frame',
) -> List[str]:
    """
    Save a list of float RGB images as numbered PNG files.

    Args:
        images:     List of (H, W, 3) float32 arrays in [0, 1].
        output_dir: Directory to write PNG files into (created if absent).
        prefix:     Filename prefix; files are named ``{prefix}_{i:04d}.png``.

    Returns:
        List of absolute paths to the saved PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    for i, img in enumerate(images):
        # Convert float [0,1] to uint8
        img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        path = output_dir / f"{prefix}_{i:04d}.png"
        pil_img.save(str(path))
        saved_paths.append(str(path))

    return saved_paths


def create_gif(
    images: List[np.ndarray],
    output_path: str,
    fps: int = 10,
) -> bool:
    """
    Create an animated GIF from a list of float RGB images using imageio.

    Args:
        images:      List of (H, W, 3) float32 arrays in [0, 1].
        output_path: Path where the GIF will be written.
        fps:         Frames per second for the animation.

    Returns:
        True on success, False if imageio is unavailable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_uint8 = [(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) for img in images]

    try:
        import imageio
        duration = 1.0 / fps  # seconds per frame
        imageio.mimsave(str(output_path), frames_uint8, duration=duration, loop=0)
        print(f"[render] GIF saved (via imageio) → {output_path}")
        return True
    except ImportError:
        # Fallback to PIL Image saving
        try:
            pil_frames = [Image.fromarray(f) for f in frames_uint8]
            duration_ms = int(1000.0 / fps)
            pil_frames[0].save(
                str(output_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
            )
            print(f"[render] GIF saved (via PIL fallback) → {output_path}")
            return True
        except Exception as e:
            print(f"[render] Failed to create GIF: {e}")
            return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render novel views from a trained NeRF checkpoint."
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help="Path to the .pth checkpoint file."
    )
    parser.add_argument(
        '--config', type=str, default='configs/lego_baseline.yaml',
        help="YAML configuration file (default: configs/lego_baseline.yaml)."
    )
    parser.add_argument(
        '--num_frames', type=int, default=40,
        help="Number of frames in the 360° orbit (default: 40)."
    )
    parser.add_argument(
        '--radius', type=float, default=4.0,
        help="Sphere radius for camera orbit (default: 4.0)."
    )
    parser.add_argument(
        '--elevation', type=float, default=-30.0,
        help="Camera elevation in degrees (default: -30.0)."
    )
    parser.add_argument(
        '--output_dir', type=str, default='outputs/renders',
        help="Directory for rendered frames (default: outputs/renders)."
    )
    parser.add_argument(
        '--H', type=int, default=100,
        help="Render height in pixels (default: 100)."
    )
    parser.add_argument(
        '--W', type=int, default=100,
        help="Render width in pixels (default: 100)."
    )
    parser.add_argument(
        '--focal', type=float, default=138.9,
        help="Focal length in pixels (default: 138.9)."
    )
    parser.add_argument(
        '--chunk_size', type=int, default=1024,
        help="Rays per forward pass (default: 1024)."
    )
    parser.add_argument(
        '--gif', action='store_true', default=True,
        help="Also produce an animated GIF (requires imageio)."
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help="Device: 'cpu', 'cuda', or 'auto' (default: auto)."
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # -- Device selection --------------------------------------------------
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[render] Using device: {device}")

    # -- Load config -------------------------------------------------------
    try:
        import yaml
        with open(args.config, 'r') as f:
            full_cfg = yaml.safe_load(f)
        sampling_cfg = full_cfg.get('sampling', {})
        config = {
            'near': sampling_cfg.get('near', 2.0),
            'far': sampling_cfg.get('far', 6.0),
            'num_samples': sampling_cfg.get('num_samples', 64),
        }
    except Exception as e:
        print(f"[render] Warning – could not load config '{args.config}': {e}")
        config = {'near': 2.0, 'far': 6.0, 'num_samples': 64}

    # -- Load checkpoint ---------------------------------------------------
    if not os.path.isfile(args.checkpoint):
        print(f"[render] Error: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"[render] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Import NeRF model and encoders (defined in src/model.py)
    try:
        from src.model import create_nerf

        model_cfg = ckpt.get('config', {}).get('model', {})
        encoder_pos, encoder_dir, model = create_nerf(model_cfg)
        model = model.to(device)

        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print("[render] Model loaded successfully.")

    except ImportError as exc:
        print(f"[render] Cannot import model: {exc}")
        sys.exit(1)

    # -- Generate camera trajectory ----------------------------------------
    print(f"[render] Generating {args.num_frames} spherical poses "
          f"(radius={args.radius}, elevation={args.elevation}°) …")
    poses = generate_spherical_poses(
        num_frames=args.num_frames,
        radius=args.radius,
        elevation=args.elevation,
    )

    # -- Render frames -----------------------------------------------------
    H, W, focal = args.H, args.W, args.focal
    print(f"[render] Rendering {args.num_frames} frames at {H}×{W} px …")

    images: List[np.ndarray] = []
    for pose in tqdm(poses, desc="Rendering", unit="frame"):
        img = render_image(
            model=model,
            encoder_pos=encoder_pos,
            encoder_dir=encoder_dir,
            pose=pose,
            H=H, W=W, focal=focal,
            config=config,
            device=device,
            chunk_size=args.chunk_size,
        )
        images.append(img)

    # -- Save PNG sequence -------------------------------------------------
    output_dir = Path(args.output_dir)
    saved = save_image_sequence(images, str(output_dir), prefix='frame')
    print(f"[render] Saved {len(saved)} PNG frames → {output_dir}/")

    # -- Create GIF --------------------------------------------------------
    if args.gif:
        gif_path = output_dir / 'orbit.gif'
        create_gif(images, str(gif_path), fps=10)

    print("[render] Done.")


if __name__ == '__main__':
    main()
