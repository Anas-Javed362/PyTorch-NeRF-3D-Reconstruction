"""
NeRF Evaluation Script.

Evaluates a trained NeRF model on held-out test (or validation) views.

Usage
-----
Run from the project root:

    python -m src.evaluate --checkpoint outputs/checkpoints/best.pth
    python -m src.evaluate --checkpoint outputs/checkpoints/best.pth --config configs/lego_baseline.yaml --split test

Steps:
1. Load checkpoint and configuration.
2. Load evaluation dataset (test or val split).
3. For each image:
   a. Generate camera rays.
   b. Render view using chunked rendering to prevent memory issues.
   c. Compute PSNR and SSIM against ground-truth image.
   d. Save rendered view to outputs/renders/.
4. Compute dataset-wide mean PSNR and mean SSIM.
5. Save results to:
   - outputs/metrics/eval_results.json
   - outputs/metrics/eval_results.csv
6. Print summary table to console.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.utils import (
    load_config,
    merge_config,
    get_device,
    load_checkpoint,
    compute_psnr,
    img_to_uint8,
    make_dirs,
    format_time,
    Timer,
)
from src.model import create_nerf
from src.camera import get_rays_flat
from src.sampling import sample_points_along_rays
from src.renderer import VolumeRenderer


# ---------------------------------------------------------------------------
# SSIM Metric Helper
# ---------------------------------------------------------------------------

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
    """Compute Structural Similarity Index (SSIM) between two float images in [0, 1].

    Args:
        img1: Predicted image array of shape (H, W, 3), float in [0, 1].
        img2: Ground-truth image array of shape (H, W, 3), float in [0, 1].

    Returns:
        SSIM value as float in [-1, 1], or raises ImportError if scikit-image
        is unavailable.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for SSIM calculation, but is not installed.\n"
            "Please install it using: pip install scikit-image"
        ) from exc

    # Convert to float64 for skimage precision
    im1 = np.clip(img1, 0.0, 1.0).astype(np.float64)
    im2 = np.clip(img2, 0.0, 1.0).astype(np.float64)

    try:
        # skimage >= 0.19 uses channel_axis
        return float(structural_similarity(im1, im2, channel_axis=-1, data_range=1.0))
    except TypeError:
        # Older versions use multichannel
        return float(structural_similarity(im1, im2, multichannel=True, data_range=1.0))


# ---------------------------------------------------------------------------
# Robust Chunked Rendering for Evaluation
# ---------------------------------------------------------------------------

def render_rays_chunked_eval(
    model: torch.nn.Module,
    encoder_pos,
    encoder_dir,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    config: Dict[str, Any],
    chunk_size: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Render rays in chunks to avoid out-of-memory errors during evaluation.

    Handles model outputs whether returned as a dict ({"rgb": ..., "sigma": ...})
    or as a raw 4-channel tensor.

    Args:
        model: Trained NeRF MLP.
        encoder_pos: Positional encoder for 3-D points.
        encoder_dir: Directional encoder (or None if use_viewdirs=False).
        rays_o: Ray origins in world space, shape (N, 3).
        rays_d: Ray directions in world space, shape (N, 3).
        config: Dict containing 'near', 'far', 'num_samples', 'white_bkgd'.
        chunk_size: Batch size of rays processed at once.
        device: Device where model and tensors reside.

    Returns:
        Dict with keys: 'rgb' (N, 3), 'depth' (N,), 'opacity' (N,).
    """
    near: float = config.get("near", 2.0)
    far: float = config.get("far", 6.0)
    num_samples: int = config.get("num_samples", 64)
    white_bkgd: bool = config.get("white_bkgd", True)

    renderer = VolumeRenderer()
    N = rays_o.shape[0]

    all_rgb = []
    all_depth = []
    all_opacity = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        r_o = rays_o[start:end].to(device)  # (C, 3)
        r_d = rays_d[start:end].to(device)  # (C, 3)
        C = r_o.shape[0]

        # Sample points along rays (stratified=False for deterministic evaluation)
        pts, z_vals = sample_points_along_rays(
            rays_o=r_o,
            rays_d=r_d,
            near=near,
            far=far,
            num_samples=num_samples,
            stratified=False,
        )  # pts: (C, S, 3), z_vals: (C, S)

        # Encode positions
        pts_flat = pts.reshape(-1, 3)      # (C*S, 3)
        pos_enc = encoder_pos(pts_flat)    # (C*S, pos_dim)

        # Encode view directions if enabled
        if encoder_dir is not None:
            dirs = r_d / (torch.norm(r_d, dim=-1, keepdim=True) + 1e-10)
            dirs_exp = dirs.unsqueeze(1).expand(C, num_samples, 3)
            dirs_flat = dirs_exp.reshape(-1, 3)
            dir_enc = encoder_dir(dirs_flat)
        else:
            dir_enc = None

        with torch.no_grad():
            raw = model(pos_enc, dir_enc)

        # Accommodate dict output or raw tensor output
        if isinstance(raw, dict):
            rgb_raw = raw["rgb"]      # (C*S, 3)
            sigma_raw = raw["sigma"]  # (C*S, 1)
            rgb_rs = rgb_raw.reshape(C, num_samples, 3)
            sigma_rs = sigma_raw.reshape(C, num_samples, 1)
            rgb_sigma = torch.cat([rgb_rs, sigma_rs], dim=-1)  # (C, S, 4)
        else:
            raw_rs = raw.reshape(C, num_samples, 4)
            rgb_sigma = torch.cat(
                [torch.sigmoid(raw_rs[..., :3]), F.relu(raw_rs[..., 3:])], dim=-1
            )

        render_out = renderer.render_rays(
            rgb_sigma=rgb_sigma,
            z_vals=z_vals,
            rays_d=r_d,
            white_bkgd=white_bkgd,
        )

        all_rgb.append(render_out["rgb"].cpu())
        all_depth.append(render_out["depth"].cpu())
        all_opacity.append(render_out["opacity"].cpu())

    return {
        "rgb": torch.cat(all_rgb, dim=0),
        "depth": torch.cat(all_depth, dim=0),
        "opacity": torch.cat(all_opacity, dim=0),
    }


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    split: str = "test",
    chunk_size: int = 1024,
    device_str: str = "auto",
    val_skip: int = 1,
) -> Dict[str, Any]:
    """Evaluate a trained NeRF checkpoint on a chosen dataset split.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        config_path:     Optional YAML config file path.
        split:           Dataset split ('test', 'val', or 'train').
        chunk_size:      Ray chunk size during rendering.
        device_str:      Device specification ('auto', 'cpu', 'cuda').
        val_skip:        Skip factor for val/test images.

    Returns:
        Dict containing full evaluation metrics.
    """
    device = get_device(device_str)
    print(f"Evaluation device: {device}")

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    map_location = device if device is not None else torch.device("cpu")
    ckpt_data = torch.load(checkpoint_path, map_location=map_location)
    ckpt_config = ckpt_data.get("config", {})

    # If external config provided, merge it
    if config_path is not None:
        file_config = load_config(config_path)
        config = merge_config(ckpt_config, file_config)
    else:
        config = ckpt_config

    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    sampling_cfg = config.get("sampling", {})
    output_cfg = config.get("output", {})

    # Create directories
    render_dir = output_cfg.get("render_dir", "outputs/renders")
    metrics_dir = output_cfg.get("metrics_dir", "outputs/metrics")
    make_dirs(render_dir, metrics_dir)

    # Initialize model
    pos_encoder, dir_encoder, model = create_nerf(model_cfg)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model = model.to(device)
    pos_encoder = pos_encoder.to(device)
    if dir_encoder is not None:
        dir_encoder = dir_encoder.to(device)
    model.eval()

    # Load dataset
    from src.dataset import BlenderDataset

    data_dir = dataset_cfg.get("data_dir", "data/lego")
    img_wh = tuple(dataset_cfg.get("img_wh", [100, 100]))
    white_bkgd = dataset_cfg.get("white_bkgd", True)

    print(f"Loading '{split}' dataset from '{data_dir}' (resolution: {img_wh}) …")
    dataset = BlenderDataset(
        data_dir=data_dir,
        split=split,
        img_wh=img_wh,
        white_bkgd=white_bkgd,
        val_skip=val_skip,
    )
    num_images = len(dataset)
    print(f"Found {num_images} images in '{split}' split.")

    render_config = {
        "near": sampling_cfg.get("near", 2.0),
        "far": sampling_cfg.get("far", 6.0),
        "num_samples": sampling_cfg.get("num_samples", 64),
        "white_bkgd": white_bkgd,
        "perturb": False,
    }

    per_image_results: List[Dict[str, Any]] = []
    total_start = time.time()

    print("\n" + "=" * 65)
    print(f"{'Idx':>5} | {'PSNR (dB)':>10} | {'SSIM':>10} | {'Time (s)':>10}")
    print("-" * 65)

    for idx in range(num_images):
        img_start = time.time()
        sample = dataset[idx]
        img_gt: np.ndarray = sample["img"].cpu().numpy()  # (H, W, 3)
        pose: torch.Tensor = sample["pose"]               # (4, 4)

        H, W = dataset.H, dataset.W
        focal = dataset.focal

        rays_o, rays_d = get_rays_flat(H, W, focal, pose)

        out = render_rays_chunked_eval(
            model=model,
            encoder_pos=pos_encoder,
            encoder_dir=dir_encoder,
            rays_o=rays_o,
            rays_d=rays_d,
            config=render_config,
            chunk_size=chunk_size,
            device=device,
        )

        rgb_pred = out["rgb"].numpy().reshape(H, W, 3)
        img_elapsed = time.time() - img_start

        # Compute MSE & PSNR
        mse = float(np.mean((rgb_pred.astype(np.float32) - img_gt.astype(np.float32)) ** 2))
        psnr = compute_psnr(mse)

        # Compute SSIM
        try:
            ssim = compute_ssim(rgb_pred, img_gt)
            ssim_str = f"{ssim:.4f}"
        except ImportError as e:
            ssim = None
            ssim_str = "N/A (no skimage)"

        # Save rendered view
        pred_uint8 = img_to_uint8(rgb_pred)
        render_filename = f"{split}_{idx:03d}.png"
        render_path = os.path.join(render_dir, render_filename)
        Image.fromarray(pred_uint8).save(render_path)

        per_image_results.append({
            "idx": idx,
            "psnr": round(psnr, 4),
            "ssim": round(ssim, 4) if ssim is not None else None,
            "time_s": round(img_elapsed, 2),
            "render_file": render_filename,
        })

        print(f"{idx:5d} | {psnr:10.2f} | {ssim_str:>10} | {img_elapsed:10.2f}")

    total_time = time.time() - total_start
    mean_psnr = float(np.mean([r["psnr"] for r in per_image_results]))

    valid_ssims = [r["ssim"] for r in per_image_results if r["ssim"] is not None]
    mean_ssim = float(np.mean(valid_ssims)) if valid_ssims else None

    print("=" * 65)
    mean_ssim_str = f"{mean_ssim:.4f}" if mean_ssim is not None else "N/A"
    print(f"{'Mean':>5} | {mean_psnr:10.2f} | {mean_ssim_str:>10} | {total_time:10.2f}")
    print("=" * 65)
    print(f"Total evaluation time: {format_time(total_time)}")

    # ------------------------------------------------------------------
    # Save outputs/metrics/eval_results.json
    # ------------------------------------------------------------------
    results_json = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "num_images": num_images,
        "mean_psnr": round(mean_psnr, 4),
        "mean_ssim": round(mean_ssim, 4) if mean_ssim is not None else None,
        "per_image": per_image_results,
        "total_time_s": round(total_time, 2),
        "config": config,
    }

    json_path = os.path.join(metrics_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved evaluation JSON to: {json_path}")

    # ------------------------------------------------------------------
    # Save outputs/metrics/eval_results.csv
    # ------------------------------------------------------------------
    csv_path = os.path.join(metrics_dir, "eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "psnr", "ssim", "time_s", "render_file"])
        for r in per_image_results:
            writer.writerow([r["idx"], r["psnr"], r["ssim"] if r["ssim"] is not None else "", r["time_s"], r["render_file"]])
        writer.writerow(["mean", round(mean_psnr, 4), round(mean_ssim, 4) if mean_ssim is not None else "", round(total_time, 2), ""])
    print(f"Saved evaluation CSV to: {csv_path}")

    return results_json


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NeRF model on test/val views.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best.pth",
        help="Path to NeRF checkpoint file (.pth).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config YAML (overrides checkpoint config).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Ray chunk size for rendering.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--val_skip",
        type=int,
        default=1,
        help="Subsampling factor for evaluation images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        chunk_size=args.chunk_size,
        device_str=args.device,
        val_skip=args.val_skip,
    )
