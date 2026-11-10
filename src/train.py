"""
NeRF Training Script.

Trains a Neural Radiance Field (NeRF) model on the synthetic Blender dataset.

Usage
-----
Run from the project root:

    python -m src.train --config configs/lego_baseline.yaml
    python -m src.train --config configs/lego_baseline.yaml --resume outputs/checkpoints/latest.pth

The script:
1. Loads the config and dataset.
2. Pre-generates all training rays (rays_o, rays_d, rgb_gt) at startup.
3. Trains the NeRF MLP in mini-batches of rays with exponential LR decay.
4. Logs loss / PSNR to TensorBoard and the console.
5. Saves checkpoints ('latest.pth', 'best.pth', 'iter_{N}.pth').
6. Renders a validation image every eval_interval iterations.
7. Handles KeyboardInterrupt gracefully (saves checkpoint before exiting).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.utils import (
    load_config,
    set_seed,
    get_device,
    save_checkpoint,
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
from src.renderer import VolumeRenderer, render_rays_chunked


# ---------------------------------------------------------------------------
# Validation / rendering helper
# ---------------------------------------------------------------------------

def render_validation_image(
    model: torch.nn.Module,
    encoder_pos,
    encoder_dir,
    dataset,
    img_idx: int,
    config: Dict[str, Any],
    device: torch.device,
) -> tuple:
    """Render a full validation image for a single pose from *dataset*.

    The model is set to eval mode during rendering and restored afterwards.

    Args:
        model:       Trained NeRF MLP.
        encoder_pos: Positional encoder for 3-D positions.
        encoder_dir: Directional encoder for view directions.
        dataset:     BlenderDataset instance with .focal, .H, .W, .poses, .imgs.
        img_idx:     Index of the image in the dataset to render.
        config:      Full training config dict (must contain 'sampling' sub-dict).
        device:      Device to run inference on.

    Returns:
        Tuple of ``(rgb_pred, rgb_gt)`` where both are float NumPy arrays
        of shape (H, W, 3) in [0, 1].
    """
    was_training = model.training
    model.eval()

    sampling_cfg = config.get("sampling", {})
    dataset_cfg = config.get("dataset", {})

    H: int = dataset.H
    W: int = dataset.W
    focal: float = dataset.focal
    white_bkgd: bool = dataset_cfg.get("white_bkgd", True)

    render_config = {
        "near": sampling_cfg.get("near", 2.0),
        "far": sampling_cfg.get("far", 6.0),
        "num_samples": sampling_cfg.get("num_samples", 64),
        "white_bkgd": white_bkgd,
        "perturb": False,  # deterministic at eval time
    }

    pose: torch.Tensor = dataset.poses[img_idx].to(device)  # (4,4)
    img_gt: torch.Tensor = dataset.imgs[img_idx]             # (H,W,3), [0,1]

    with torch.no_grad():
        rays_o, rays_d = get_rays_flat(H, W, focal, pose)   # (H*W, 3) each
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)

        chunk_size: int = config.get("training", {}).get("chunk_size", 1024)
        out = render_rays_chunked(
            model=model,
            encoder_pos=encoder_pos,
            encoder_dir=encoder_dir,
            rays_o=rays_o,
            rays_d=rays_d,
            config=render_config,
            chunk_size=chunk_size,
        )

    rgb_pred = out["rgb"].cpu().numpy().reshape(H, W, 3)   # (H,W,3)
    rgb_gt = img_gt.numpy()                                 # (H,W,3)

    if was_training:
        model.train()

    return rgb_pred, rgb_gt


# ---------------------------------------------------------------------------
# Pre-generate all training rays
# ---------------------------------------------------------------------------

def build_training_rays(
    dataset, device: torch.device, cpu: bool = True
) -> tuple:
    """Generate (rays_o, rays_d, rgb_gt) tensors for every training pixel.

    Returns tensors of shape (N*H*W, 3) stored on CPU (then moved to GPU in
    mini-batches during training).

    Args:
        dataset: BlenderDataset with .poses, .imgs, .focal, .H, .W.
        device:  Unused here (kept for signature consistency); rays are on CPU.
        cpu:     Always keep rays on CPU to avoid OOM for large scenes.

    Returns:
        Tuple of ``(rays_o, rays_d, rgb_gt)`` each shape (N*H*W, 3), float32.
    """
    H: int = dataset.H
    W: int = dataset.W
    focal: float = dataset.focal
    N: int = len(dataset)

    all_rays_o = []
    all_rays_d = []
    all_rgb = []

    print(f"Pre-generating {N} × {H}×{W} = {N * H * W:,} training rays …")
    for idx in range(N):
        pose: torch.Tensor = dataset.poses[idx]   # (4,4) on CPU
        img: torch.Tensor = dataset.imgs[idx]     # (H,W,3) on CPU

        rays_o, rays_d = get_rays_flat(H, W, focal, pose)  # (H*W, 3) CPU

        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_rgb.append(img.reshape(-1, 3))

    rays_o = torch.cat(all_rays_o, dim=0)  # (N*H*W, 3)
    rays_d = torch.cat(all_rays_d, dim=0)
    rgb_gt = torch.cat(all_rgb, dim=0)

    print(f"Training rays ready: {rays_o.shape}")
    return rays_o, rays_d, rgb_gt


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config_path: str, resume_path: Optional[str] = None) -> None:
    """Train a NeRF model from a YAML config file.

    Args:
        config_path: Path to the YAML configuration file.
        resume_path: Optional path to a checkpoint to resume from.
    """
    # ------------------------------------------------------------------
    # 1. Load config and setup
    # ------------------------------------------------------------------
    config = load_config(config_path)

    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    sampling_cfg = config.get("sampling", {})
    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})

    seed: int = config.get("seed", 42)
    set_seed(seed)

    device_str: str = training_cfg.get("device", "auto")
    device: torch.device = get_device(device_str)
    print(f"Using device: {device}")

    # Directories
    ckpt_dir: str = output_cfg.get("checkpoint_dir", "outputs/checkpoints")
    render_dir: str = output_cfg.get("render_dir", "outputs/renders")
    log_dir: str = output_cfg.get("log_dir", "logs")
    make_dirs(ckpt_dir, render_dir, log_dir)

    exp_name: str = config.get("experiment_name", "nerf_experiment")
    tb_log_dir = os.path.join(log_dir, exp_name)
    make_dirs(tb_log_dir)

    # TensorBoard writer (optional)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logs → {tb_log_dir}")
    except Exception as exc:
        writer = None
        print(f"[Warning] TensorBoard unavailable ({type(exc).__name__}); continuing without TensorBoard logging.")

    # ------------------------------------------------------------------
    # 2. Dataset
    # ------------------------------------------------------------------
    # Import here to allow graceful failure if dataset.py is not yet written
    try:
        from src.dataset import BlenderDataset
    except ImportError as exc:
        raise ImportError(
            "src/dataset.py not found. Ensure the dataset module has been created."
        ) from exc

    data_dir: str = dataset_cfg.get("data_dir", "data/lego")
    img_wh = tuple(dataset_cfg.get("img_wh", [100, 100]))
    white_bkgd: bool = dataset_cfg.get("white_bkgd", True)
    val_skip: int = dataset_cfg.get("val_skip", 8)

    print(f"Loading training dataset from '{data_dir}' at {img_wh} …")
    train_dataset = BlenderDataset(
        data_dir=data_dir,
        split="train",
        img_wh=img_wh,
        white_bkgd=white_bkgd,
    )
    val_dataset = BlenderDataset(
        data_dir=data_dir,
        split="val",
        img_wh=img_wh,
        white_bkgd=white_bkgd,
        val_skip=val_skip,
    )
    print(
        f"Train images: {len(train_dataset)}  |  "
        f"Val images: {len(val_dataset)}  |  "
        f"H={train_dataset.H}, W={train_dataset.W}, focal={train_dataset.focal:.2f}"
    )

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    pos_encoder, dir_encoder, model = create_nerf(model_cfg)
    model = model.to(device)
    pos_encoder = pos_encoder.to(device)
    if dir_encoder is not None:
        dir_encoder = dir_encoder.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"NeRF parameters: {num_params:,}")

    # ------------------------------------------------------------------
    # 4. Optimizer
    # ------------------------------------------------------------------
    lr: float = training_cfg.get("lr", 5e-4)
    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(pos_encoder.parameters())
        + (list(dir_encoder.parameters()) if dir_encoder is not None else []),
        lr=lr,
    )

    # ------------------------------------------------------------------
    # 5. Resume from checkpoint (optional)
    # ------------------------------------------------------------------
    start_iter: int = 0
    best_psnr: float = -float("inf")

    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        start_iter, _, ckpt_config = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_iter += 1  # continue from the next iteration
        print(f"  Resumed at iteration {start_iter - 1}")

    # ------------------------------------------------------------------
    # 6. Pre-generate all training rays
    # ------------------------------------------------------------------
    rays_o_all, rays_d_all, rgb_gt_all = build_training_rays(train_dataset, device)
    total_rays: int = rays_o_all.shape[0]

    # ------------------------------------------------------------------
    # 7. Training hyper-parameters
    # ------------------------------------------------------------------
    num_iters: int = training_cfg.get("num_iters", 50_000)
    batch_size: int = training_cfg.get("batch_size", 1024)
    lr_decay: float = training_cfg.get("lr_decay", 0.1)
    lr_decay_steps: int = training_cfg.get("lr_decay_steps", 25_000)
    checkpoint_interval: int = training_cfg.get("checkpoint_interval", 5_000)
    eval_interval: int = training_cfg.get("eval_interval", 5_000)
    log_interval: int = training_cfg.get("log_interval", 100)

    near: float = sampling_cfg.get("near", 2.0)
    far: float = sampling_cfg.get("far", 6.0)
    num_samples: int = sampling_cfg.get("num_samples", 64)
    perturb: bool = sampling_cfg.get("perturb", True)

    use_viewdirs: bool = model_cfg.get("use_viewdirs", True)

    renderer = VolumeRenderer()

    # Render config forwarded to chunked renderer during validation
    render_config = {
        "near": near,
        "far": far,
        "num_samples": num_samples,
        "white_bkgd": white_bkgd,
        "perturb": False,
    }

    # ------------------------------------------------------------------
    # 8. Shuffled ray indices for epoch-based batching
    # ------------------------------------------------------------------
    perm = torch.randperm(total_rays)
    perm_ptr: int = 0  # pointer into the current permutation

    def next_batch_indices(bsz: int):
        """Return *bsz* random ray indices, reshuffling when exhausted."""
        nonlocal perm, perm_ptr
        if perm_ptr + bsz > total_rays:
            perm = torch.randperm(total_rays)
            perm_ptr = 0
        idx = perm[perm_ptr : perm_ptr + bsz]
        perm_ptr += bsz
        return idx

    # ------------------------------------------------------------------
    # 9. Training loop
    # ------------------------------------------------------------------
    model.train()
    total_start = time.time()
    loss_val: float = float("nan")

    if start_iter >= num_iters:
        print(
            f"Checkpoint is already at iteration {start_iter} >= target {num_iters}. "
            f"Training already complete. Exiting."
        )
        return

    print(
        f"\n{'='*60}\n"
        f"  Starting training  |  iters: {start_iter}→{num_iters}  |  batch: {batch_size}\n"
        f"{'='*60}\n"
    )

    pbar = tqdm(
        range(start_iter, num_iters),
        initial=start_iter,
        total=num_iters,
        desc="Training",
        dynamic_ncols=True,
    )

    try:
        for iteration in pbar:
            iter_start = time.time()

            # ---- Exponential LR decay --------------------------------
            # lr(t) = lr_0 * decay ^ (t / decay_steps)
            decay_rate = lr_decay ** (iteration / lr_decay_steps)
            new_lr = lr * decay_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

            # ---- Sample a mini-batch of rays -------------------------
            idx = next_batch_indices(batch_size)
            rays_o_batch = rays_o_all[idx].to(device)   # (B, 3)
            rays_d_batch = rays_d_all[idx].to(device)   # (B, 3)
            rgb_target   = rgb_gt_all[idx].to(device)   # (B, 3)

            # ---- Sample points along rays ----------------------------
            # pts: (B, S, 3)   z_vals: (B, S)
            pts, z_vals = sample_points_along_rays(
                rays_o=rays_o_batch,
                rays_d=rays_d_batch,
                near=near,
                far=far,
                num_samples=num_samples,
                stratified=perturb,
            )

            B, S, _ = pts.shape

            # ---- Encode positions ------------------------------------
            pts_flat = pts.reshape(-1, 3)                     # (B*S, 3)
            pos_enc = pos_encoder(pts_flat)                   # (B*S, pos_dim)

            # ---- Encode directions -----------------------------------
            if use_viewdirs and dir_encoder is not None:
                dirs = rays_d_batch / (
                    torch.norm(rays_d_batch, dim=-1, keepdim=True) + 1e-10
                )                                              # (B, 3) unit
                dirs_exp = dirs.unsqueeze(1).expand(B, S, 3)  # (B, S, 3)
                dirs_flat = dirs_exp.reshape(-1, 3)            # (B*S, 3)
                dir_enc = dir_encoder(dirs_flat)               # (B*S, dir_dim)
            else:
                dir_enc = None

            # ---- Forward pass ----------------------------------------
            raw_out = model(pos_enc, dir_enc)   # dict: {'rgb': (B*S,3), 'sigma': (B*S,1)}
            rgb_raw   = raw_out["rgb"]          # (B*S, 3) already sigmoid-activated
            sigma_raw = raw_out["sigma"]        # (B*S, 1) already relu-activated

            # Reshape and assemble (B, S, 4) for the renderer
            rgb_rs    = rgb_raw.reshape(B, S, 3)
            sigma_rs  = sigma_raw.reshape(B, S, 1)
            rgb_sigma = torch.cat([rgb_rs, sigma_rs], dim=-1)  # (B, S, 4)

            # ---- Volume render ----------------------------------------
            render_out = renderer.render_rays(
                rgb_sigma=rgb_sigma,
                z_vals=z_vals,
                rays_d=rays_d_batch,
                white_bkgd=white_bkgd,
            )
            rgb_pred = render_out["rgb"]   # (B, 3)

            # ---- MSE loss & optimise ---------------------------------
            loss = F.mse_loss(rgb_pred, rgb_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val: float = loss.item()
            psnr_val: float = compute_psnr(loss_val)
            iter_time: float = time.time() - iter_start

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                psnr=f"{psnr_val:.2f}dB",
                lr=f"{new_lr:.2e}",
            )

            # ---- Logging -----------------------------------------
            if iteration % log_interval == 0:
                elapsed = format_time(time.time() - total_start)
                log_line = (
                    f"[{iteration:>6d}/{num_iters}] "
                    f"loss={loss_val:.4f}  psnr={psnr_val:.2f}dB  "
                    f"lr={new_lr:.2e}  elapsed={elapsed}"
                )
                # GPU memory info
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated(device) / 1e6
                    log_line += f"  gpu_mem={mem_alloc:.0f}MB"
                tqdm.write(log_line)

                if writer is not None:
                    writer.add_scalar("train/loss", loss_val, iteration)
                    writer.add_scalar("train/psnr", psnr_val, iteration)
                    writer.add_scalar("train/lr", new_lr, iteration)

            # ---- Checkpoint saving --------------------------------
            if iteration > 0 and iteration % checkpoint_interval == 0:
                latest_path = os.path.join(ckpt_dir, "latest.pth")
                iter_path   = os.path.join(ckpt_dir, f"iter_{iteration:06d}.pth")
                save_checkpoint(
                    path=latest_path,
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                    loss=loss_val,
                    config=config,
                )
                save_checkpoint(
                    path=iter_path,
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                    loss=loss_val,
                    config=config,
                )
                tqdm.write(f"  [ckpt] Saved latest.pth and iter_{iteration:06d}.pth")

            # ---- Validation rendering ----------------------------
            if iteration > 0 and iteration % eval_interval == 0:
                val_idx = iteration // eval_interval % len(val_dataset)
                tqdm.write(f"  [eval] Rendering validation image idx={val_idx} …")
                with Timer() as t:
                    rgb_pred_img, rgb_gt_img = render_validation_image(
                        model=model,
                        encoder_pos=pos_encoder,
                        encoder_dir=dir_encoder,
                        dataset=val_dataset,
                        img_idx=val_idx,
                        config=config,
                        device=device,
                    )
                # Compute val PSNR
                mse_val = float(
                    np.mean((rgb_pred_img.astype(np.float32) - rgb_gt_img.astype(np.float32)) ** 2)
                )
                val_psnr = compute_psnr(mse_val)
                tqdm.write(
                    f"  [eval] iter={iteration}  val_psnr={val_psnr:.2f}dB  "
                    f"render_time={t.elapsed:.1f}s"
                )

                if writer is not None:
                    writer.add_scalar("val/psnr", val_psnr, iteration)
                    # Log rendered image to TensorBoard (H,W,3) → (3,H,W)
                    img_uint8 = img_to_uint8(rgb_pred_img)
                    img_tensor = torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0)
                    writer.add_images("val/rendered", img_tensor / 255.0, iteration)

                # Save rendered image to disk
                try:
                    from PIL import Image
                    H_v, W_v = rgb_pred_img.shape[:2]
                    render_path = os.path.join(render_dir, f"val_iter{iteration:06d}.png")
                    Image.fromarray(img_to_uint8(rgb_pred_img)).save(render_path)
                except ImportError:
                    pass  # Pillow not available

                # Save best checkpoint
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    best_path = os.path.join(ckpt_dir, "best.pth")
                    save_checkpoint(
                        path=best_path,
                        model=model,
                        optimizer=optimizer,
                        iteration=iteration,
                        loss=loss_val,
                        config=config,
                        extra={"val_psnr": val_psnr},
                    )
                    tqdm.write(
                        f"  [ckpt] New best val_psnr={val_psnr:.2f}dB → saved best.pth"
                    )

                model.train()  # restore training mode

    except KeyboardInterrupt:
        print("\n[!] Training interrupted. Saving checkpoint before exit …")
        interrupt_path = os.path.join(ckpt_dir, "latest.pth")
        save_checkpoint(
            path=interrupt_path,
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            loss=loss_val if "loss_val" in dir() else float("nan"),
            config=config,
        )
        print(f"[!] Checkpoint saved to {interrupt_path}")
        if writer is not None:
            writer.close()
        return

    # ------------------------------------------------------------------
    # 10. Final checkpoint
    # ------------------------------------------------------------------
    final_path = os.path.join(ckpt_dir, "latest.pth")
    save_checkpoint(
        path=final_path,
        model=model,
        optimizer=optimizer,
        iteration=num_iters - 1,
        loss=loss_val,
        config=config,
    )
    total_time = format_time(time.time() - total_start)
    print(f"\nTraining complete in {total_time}. Final checkpoint: {final_path}")

    if writer is not None:
        writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train a NeRF model on the synthetic Blender dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lego_baseline.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(config_path=args.config, resume_path=args.resume)
