"""
Research-Quality Visualization Utilities for PyTorch NeRF.

Provides plotting functions for:
- Training progress (loss and PSNR curves)
- Qualitative rendering comparisons (side-by-side, error maps, grids)
- Scene geometry (camera pose trajectories, depth maps)
- Model mechanics (positional encoding representations)
- Ablation experiment summaries

All figures are styled for publication-quality output (dpi=150) and can accept
either PyTorch tensors or NumPy arrays as inputs.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

# Set writable matplotlib config dir
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless / server environments
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helper: Tensor / Array conversions and styling
# ---------------------------------------------------------------------------

def _to_numpy(val: Union[torch.Tensor, np.ndarray, Sequence]) -> np.ndarray:
    """Convert a PyTorch tensor or sequence to a NumPy ndarray."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return np.asarray(val)


def _prepare_image(img: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Ensure image is a float32 NumPy array in [0, 1] with shape (H, W, C) or (H, W)."""
    arr = _to_numpy(img).astype(np.float32)
    # If channels-first (C, H, W) where C in (1, 3, 4), transpose to (H, W, C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.max() > 1.0 + 1e-4:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _save_and_finish(fig: plt.Figure, save_path: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    """Helper to save figure to disk if save_path is provided, ensuring directories exist."""
    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_p), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 1. Training Loss Curve
# ---------------------------------------------------------------------------

def plot_training_loss(
    loss_history: Union[Sequence[float], np.ndarray, torch.Tensor],
    save_path: str = "outputs/figures/training_loss.png",
    log_scale: bool = False,
    window_size: int = 50,
) -> plt.Figure:
    """
    Plot training loss curve over iterations, optionally with moving average and log-scale.

    Args:
        loss_history: Sequence or 1D array of loss values per iteration.
        save_path: Destination file path.
        log_scale: If True, y-axis uses logarithmic scale.
        window_size: Window size for smoothed rolling average.

    Returns:
        matplotlib.figure.Figure
    """
    losses = _to_numpy(loss_history).flatten()
    iterations = np.arange(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(iterations, losses, alpha=0.35, color='#1f77b4', label='Raw Loss')

    # Rolling average if enough points
    if len(losses) >= window_size and window_size > 1:
        kernel = np.ones(window_size) / window_size
        smooth = np.convolve(losses, kernel, mode='valid')
        smooth_iters = np.arange(window_size, len(losses) + 1)
        ax.plot(smooth_iters, smooth, color='#d62728', linewidth=1.8, label=f'{window_size}-iter Moving Avg')

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')

    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_title('NeRF Training Loss Curve', fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    return _save_and_finish(fig, save_path)


# ---------------------------------------------------------------------------
# 2. PSNR Curve
# ---------------------------------------------------------------------------

def plot_psnr_curve(
    psnr_history: Union[Sequence[float], np.ndarray, torch.Tensor],
    save_path: str = "outputs/figures/psnr_curve.png",
    iterations: Optional[Sequence[int]] = None,
) -> plt.Figure:
    """
    Plot PSNR (Peak Signal-to-Noise Ratio) progression across training iterations.

    Args:
        psnr_history: Sequence of PSNR values (dB).
        save_path: Destination file path.
        iterations: Optional explicit iteration indices corresponding to psnr values.

    Returns:
        matplotlib.figure.Figure
    """
    psnr_vals = _to_numpy(psnr_history).flatten()
    if iterations is None:
        iters = np.arange(1, len(psnr_vals) + 1)
    else:
        iters = _to_numpy(iterations).flatten()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(iters, psnr_vals, marker='o', markersize=4, color='#2ca02c', linewidth=2.0, label='Validation PSNR')

    if len(psnr_vals) > 0:
        max_idx = np.argmax(psnr_vals)
        ax.scatter([iters[max_idx]], [psnr_vals[max_idx]], color='#d62728', s=80, zorder=5,
                   label=f'Best: {psnr_vals[max_idx]:.2f} dB (iter {iters[max_idx]})')

    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('Validation PSNR Progression', fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    return _save_and_finish(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Ground Truth vs Rendered Comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    gt_img: Union[torch.Tensor, np.ndarray],
    pred_img: Union[torch.Tensor, np.ndarray],
    save_path: str = "outputs/figures/comparison.png",
    title: str = "Ground Truth vs. Rendered View",
    psnr: Optional[float] = None,
) -> plt.Figure:
    """
    Side-by-side visual comparison between ground truth and NeRF-rendered image.

    Args:
        gt_img: Ground truth image (H, W, 3).
        pred_img: Synthesized image (H, W, 3).
        save_path: Destination file path.
        title: Overall plot title.
        psnr: Optional PSNR score to show in subtitle.

    Returns:
        matplotlib.figure.Figure
    """
    gt = _prepare_image(gt_img)
    pred = _prepare_image(pred_img)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

    axes[0].imshow(gt)
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    pred_title = 'Rendered NeRF' if psnr is None else f'Rendered NeRF (PSNR: {psnr:.2f} dB)'
    axes[1].imshow(pred)
    axes[1].set_title(pred_title, fontsize=12, fontweight='bold')
    axes[1].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    return _save_and_finish(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Error Map
# ---------------------------------------------------------------------------

def plot_error_map(
    gt_img: Union[torch.Tensor, np.ndarray],
    pred_img: Union[torch.Tensor, np.ndarray],
    save_path: str = "outputs/figures/error_map.png",
) -> plt.Figure:
    """
    Plot ground truth, rendered image, and absolute per-pixel error heat map with colorbar.

    Args:
        gt_img: Ground truth image (H, W, 3).
        pred_img: Rendered image (H, W, 3).
        save_path: Destination file path.

    Returns:
        matplotlib.figure.Figure
    """
    gt = _prepare_image(gt_img)
    pred = _prepare_image(pred_img)

    # Compute absolute error per channel and average across RGB
    err = np.abs(gt - pred)
    if err.ndim == 3 and err.shape[-1] >= 3:
        err_gray = np.mean(err[..., :3], axis=-1)
    else:
        err_gray = err.squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    axes[0].imshow(gt)
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(pred)
    axes[1].set_title('Rendered View', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    im = axes[2].imshow(err_gray, cmap='inferno', vmin=0.0, vmax=max(0.2, float(np.max(err_gray))))
    axes[2].set_title(f'Mean Abs Error (MAE: {np.mean(err_gray):.4f})', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Error', fontsize=10)

    fig.suptitle('Novel View Reconstruction Error Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    return _save_and_finish(fig, save_path)


# ---------------------------------------------------------------------------
# 5. 3D Camera Poses
# ---------------------------------------------------------------------------

def plot_camera_poses(
    poses: Union[torch.Tensor, np.ndarray, Sequence[np.ndarray]],
    save_path: str = "outputs/figures/camera_poses.png",
    scale: float = 0.4,
) -> plt.Figure:
    """
    3D visualization of camera positions and optical axis viewing directions in world coordinates.

    Args:
        poses: Array or list of (4, 4) or (3, 4) camera-to-world (c2w) matrices.
        save_path: Destination file path.
        scale: Length of the viewing direction vector arrows.

    Returns:
        matplotlib.figure.Figure
    """
    poses_arr = _to_numpy(poses)
    if poses_arr.ndim == 2:
        poses_arr = poses_arr[np.newaxis, ...]  # (1, 4, 4)

    fig = plt.figure(figsize=(9, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    origins = []
    directions = []

    for pose in poses_arr:
        c2w = np.asarray(pose)
        cam_pos = c2w[:3, 3]
        # In OpenGL/NeRF convention, camera looks down -z
        view_dir = -c2w[:3, 2]
        view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)

        origins.append(cam_pos)
        directions.append(view_dir)

    origins = np.array(origins)
    directions = np.array(directions)

    # Plot camera centres
    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c='#1f77b4', s=35, label='Camera Centers', alpha=0.85)

    # Plot orientation arrows
    ax.quiver(
        origins[:, 0], origins[:, 1], origins[:, 2],
        directions[:, 0] * scale, directions[:, 1] * scale, directions[:, 2] * scale,
        color='#d62728', arrow_length_ratio=0.25, linewidth=1.2, label='Viewing Vector (-z)'
    )

    # World origin marker
    ax.scatter([0], [0], [0], c='green', marker='*', s=150, label='Scene Origin (0,0,0)')

    ax.set_xlabel('X (World)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Y (World)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_zlabel('Z (World)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Camera Trajectory & View Directions (World Space)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper right', frameon=True, fontsize=9)

    # Auto aspect ratio
    max_range = np.array([
        origins[:, 0].max() - origins[:, 0].min(),
        origins[:, 1].max() - origins[:, 1].min(),
        origins[:, 2].max() - origins[:, 2].min()
    ]).max() / 2.0
    max_range = max(float(max_range), 1.0)
    mid_x = (origins[:, 0].max() + origins[:, 0].min()) * 0.5
    mid_y = (origins[:, 1].max() + origins[:, 1].min()) * 0.5
    mid_z = (origins[:, 2].max() + origins[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return _save_and_finish(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Ablation Results Bar Chart
# ---------------------------------------------------------------------------

def plot_ablation_results(
    results_dict: Dict[str, Union[float, Dict[str, float]]],
    metric: str = 'psnr',
    save_path: str = "outputs/figures/ablation_results.png",
) -> plt.Figure:
    """
    Bar chart comparing experiment configurations across a metric (e.g., PSNR or SSIM).

    Args:
        results_dict: Dictionary mapping experiment names to metric values (or dicts containing metric).
                      Example: {'Freq=4': 24.2, 'Freq=10': 27.5, 'Freq=16': 27.1}
                      or {'Exp1': {'psnr': 25.1, 'loss': 0.003}}
        metric: Name of the metric to plot if values are dicts (default: 'psnr').
        save_path: Destination file path.

    Returns:
        matplotlib.figure.Figure
    """
    labels = []
    values = []

    for k, v in results_dict.items():
        labels.append(str(k))
        if isinstance(v, dict):
            values.append(float(v.get(metric, 0.0)))
        else:
            values.append(float(v))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bars = ax.bar(labels, values, color='#4575b4', edgecolor='#2c5282', width=0.55, alpha=0.9)

    # Add numeric labels on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
    ax.set_title(f'Ablation Study: Model Performance by {metric.upper()}', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Set bottom limit slightly below minimum for better visual separation
    if values and min(values) > 0:
        ax.set_ylim(bottom=max(0.0, min(values) * 0.85))

    plt.tight_layout()
    return _save_and_finish(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Depth Map Visualization
# ---------------------------------------------------------------------------

def plot_depth_map(
    depth: Union[torch.Tensor, np.ndarray],
    save_path: str = "outputs/figures/depth_map.png",
    cmap: str = 'turbo',
) -> plt.Figure:
    """
    Visualize estimated depth map from NeRF volume rendering with colorbar.

    Args:
        depth: 2D array of depth values (H, W).
        save_path: Destination file path.
        cmap: Matplotlib colormap name (e.g., 'turbo', 'viridis', 'plasma').

    Returns:
        matplotlib.figure.Figure
    """
    depth_arr = _to_numpy(depth).squeeze().astype(np.float32)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(depth_arr, cmap=cmap)
    ax.set_title('Accumulated Expected Depth Map', fontsize=13, fontweight='bold', pad=10)
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (world units)', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return _save_and_finish(fig, save_path)


# ---------------------------------------------------------------------------
# 8. Positional Encoding Visualization
# ---------------------------------------------------------------------------

def plot_positional_encoding(
    encoder,
    save_path: str = "outputs/figures/positional_encoding.png",
    num_points: int = 200,
) -> plt.Figure:
    """
    Visualize Fourier positional encoding curves across frequencies for 1D inputs in [-1, 1].

    Args:
        encoder: Instance of PositionalEncoder or similar module.
        save_path: Destination file path.
        num_points: Number of evaluation points along the 1D domain.

    Returns:
        matplotlib.figure.Figure
    """
    x = torch.linspace(-1.0, 1.0, num_points)
    # Form 3D points [x, 0, 0] to feed to encoder
    pts = torch.zeros(num_points, 3)
    pts[:, 0] = x

    with torch.no_grad():
        encoded = encoder(pts)  # (num_points, out_dims)
    enc_np = _to_numpy(encoded)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Plot the first few frequency components corresponding to the x coordinate
    # Typically encoding alternates sin/cos for 2^0, 2^1, ...
    cols_to_plot = min(8, enc_np.shape[1])
    cmap = plt.cm.tab10(np.linspace(0, 1, cols_to_plot))

    for k in range(cols_to_plot):
        ax.plot(_to_numpy(x), enc_np[:, k], label=f'Dim {k}', color=cmap[k], linewidth=1.5)

    ax.set_xlabel(r'Coordinate Input $x \in [-1, 1]$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\gamma(x)$ Value', fontsize=12, fontweight='bold')
    ax.set_title('Positional Encoding Harmonic Functions', fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    return _save_and_finish(fig, save_path)


# ---------------------------------------------------------------------------
# 9. Grid of Training Images
# ---------------------------------------------------------------------------

def plot_training_images(
    imgs: Union[torch.Tensor, np.ndarray, Sequence],
    save_path: str = "outputs/figures/training_samples.png",
    num_show: int = 9,
) -> plt.Figure:
    """
    Plot a grid of sample training camera images.

    Args:
        imgs: Sequence or tensor/array of images (N, H, W, 3).
        save_path: Destination file path.
        num_show: Number of images to display (default: 9 in a 3x3 grid).

    Returns:
        matplotlib.figure.Figure
    """
    total = len(imgs)
    k = min(total, num_show)

    # Determine grid rows and cols
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), dpi=150)
    axes_flat = np.array(axes).flatten()

    # Evenly pick indices across dataset
    indices = np.linspace(0, total - 1, k, dtype=int)

    for i, idx in enumerate(indices):
        img = _prepare_image(imgs[idx])
        axes_flat[i].imshow(img)
        axes_flat[i].set_title(f'View #{idx}', fontsize=10, fontweight='bold')
        axes_flat[i].axis('off')

    # Turn off unused subplots
    for j in range(k, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle('Multi-View Dataset Sample Images', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    return _save_and_finish(fig, save_path)


# ---------------------------------------------------------------------------
# 10. Multi-view Comparison Grid
# ---------------------------------------------------------------------------

def create_comparison_grid(
    gt_imgs: Union[torch.Tensor, np.ndarray, Sequence],
    pred_imgs: Union[torch.Tensor, np.ndarray, Sequence],
    save_path: str = "outputs/figures/comparison_grid.png",
    num_show: int = 4,
) -> plt.Figure:
    """
    Create a 2 x N comparison grid (Top row: Ground Truth, Bottom row: NeRF Synthesized).

    Args:
        gt_imgs: Sequence of ground truth images.
        pred_imgs: Sequence of synthesized images.
        save_path: Destination file path.
        num_show: Number of views to display (columns).

    Returns:
        matplotlib.figure.Figure
    """
    k = min(len(gt_imgs), len(pred_imgs), num_show)

    fig, axes = plt.subplots(2, k, figsize=(k * 3.5, 7), dpi=150)
    if k == 1:
        axes = np.expand_dims(axes, axis=1)

    for col in range(k):
        gt = _prepare_image(gt_imgs[col])
        pred = _prepare_image(pred_imgs[col])

        # Top row: Ground truth
        axes[0, col].imshow(gt)
        axes[0, col].set_title(f'GT View #{col+1}', fontsize=11, fontweight='bold')
        axes[0, col].axis('off')

        # Bottom row: Rendered
        axes[1, col].imshow(pred)
        axes[1, col].set_title(f'NeRF View #{col+1}', fontsize=11, fontweight='bold')
        axes[1, col].axis('off')

    fig.suptitle('Multi-View Reconstruction Comparison (Top: GT, Bottom: NeRF)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    return _save_and_finish(fig, save_path)
