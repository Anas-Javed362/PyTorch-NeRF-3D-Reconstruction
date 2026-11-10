"""
End-to-end smoke test script for PyTorch NeRF pipeline.

Creates a minimal synthetic multi-view scene (train, val, test),
runs training for a few iterations, verifies checkpointing,
runs novel-view rendering, and runs evaluation.
"""

import os
import sys
import json
import math
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, load_checkpoint
from src.dataset import BlenderDataset
from src.model import create_nerf
from src.camera import get_rays_flat
from src.sampling import sample_points_along_rays
from src.renderer import VolumeRenderer, render_rays_chunked
import subprocess


def create_synthetic_scene(scene_dir: Path, img_size: int = 16):
    """Create a minimal synthetic multi-view dataset."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": 4,
        "val": 2,
        "test": 2,
    }
    
    camera_angle_x = 0.6911112070083618  # ~39.6 deg
    
    for split_name, num_frames in splits.items():
        split_dir = scene_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        frames = []
        for i in range(num_frames):
            # Create a simple colored test pattern with an alpha circle in center
            arr = np.zeros((img_size, img_size, 4), dtype=np.uint8)
            y, x = np.ogrid[:img_size, :img_size]
            center = img_size / 2.0
            dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
            mask = dist_from_center < (img_size / 3.0)
            
            # RGB object in center
            arr[mask, 0] = int(120 + 100 * math.sin(i))
            arr[mask, 1] = int(120 + 100 * math.cos(i))
            arr[mask, 2] = 200
            arr[mask, 3] = 255  # fully opaque in center
            
            # Transparent background
            arr[~mask, 3] = 0
            
            img_path = split_dir / f"r_{i}.png"
            Image.fromarray(arr).save(img_path)
            
            # Camera pose on a circle around origin
            angle = 2.0 * math.pi * i / max(num_frames, 1)
            radius = 4.0
            cam_x = radius * math.cos(angle)
            cam_y = 1.0
            cam_z = radius * math.sin(angle)
            
            # Forward vector looking at origin
            forward = np.array([-cam_x, -cam_y, -cam_z], dtype=np.float32)
            forward /= np.linalg.norm(forward) + 1e-8
            
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, up)
            right /= np.linalg.norm(right) + 1e-8
            up = np.cross(right, forward)
            
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = -forward  # OpenGL convention: looking down -z
            c2w[:3, 3] = [cam_x, cam_y, cam_z]
            
            frames.append({
                "file_path": f"./{split_name}/r_{i}",
                "transform_matrix": c2w.tolist()
            })
            
        transforms_data = {
            "camera_angle_x": camera_angle_x,
            "frames": frames
        }
        with open(scene_dir / f"transforms_{split_name}.json", "w") as f:
            json.dump(transforms_data, f, indent=2)


def main():
    print("=" * 60)
    print("  PyTorch NeRF: End-to-End Smoke Test")
    print("=" * 60)
    
    # Step 1: Create synthetic scene
    print("\n[Step 1/9] Creating synthetic multi-view dataset...")
    synth_dir = PROJECT_ROOT / "data" / "synthetic_smoke"
    create_synthetic_scene(synth_dir, img_size=16)
    print(f"  Created synthetic scene at: {synth_dir}")

    # Step 2: Create smoke test config
    print("\n[Step 2/9] Generating smoke_test.yaml config...")
    smoke_config_path = PROJECT_ROOT / "configs" / "smoke_test.yaml"
    smoke_config = {
        "experiment_name": "smoke_test_run",
        "seed": 42,
        "dataset": {
            "scene": "synthetic_smoke",
            "data_dir": "data/synthetic_smoke",
            "img_wh": [16, 16],
            "white_bkgd": True,
            "val_skip": 1,
        },
        "model": {
            "num_layers": 4,
            "hidden_dim": 64,
            "skip_connect": [2],
            "use_viewdirs": True,
            "pos_enc_freqs": 4,
            "dir_enc_freqs": 2,
        },
        "sampling": {
            "num_samples": 16,
            "near": 2.0,
            "far": 6.0,
            "perturb": True,
        },
        "training": {
            "num_iters": 20,
            "batch_size": 128,
            "lr": 1.0e-3,
            "lr_decay": 0.5,
            "lr_decay_steps": 10,
            "checkpoint_interval": 10,
            "eval_interval": 10,
            "log_interval": 5,
            "device": "cpu",
        },
        "output": {
            "checkpoint_dir": "outputs/checkpoints",
            "render_dir": "outputs/renders",
            "figures_dir": "outputs/figures",
            "metrics_dir": "outputs/metrics",
            "log_dir": "logs",
        }
    }
    
    import yaml
    with open(smoke_config_path, "w") as f:
        yaml.dump(smoke_config, f, default_flow_style=False)
    print(f"  Wrote config to: {smoke_config_path}")

    # Step 3: Test dataset loading directly
    print("\n[Step 3/9] Testing BlenderDataset loading...")
    dataset = BlenderDataset(str(synth_dir), split="train", img_wh=(16, 16))
    assert len(dataset) == 4, f"Expected 4 train images, got {len(dataset)}"
    sample = dataset[0]
    assert sample["img"].shape == (16, 16, 3), f"Bad img shape: {sample['img'].shape}"
    assert sample["pose"].shape == (4, 4), f"Bad pose shape: {sample['pose'].shape}"
    print("  BlenderDataset loading OK.")

    # Step 4: Test Forward pass, loss calculation & backprop
    print("\n[Step 4/9] Testing forward pass, loss, and backpropagation...")
    pos_enc, dir_enc, model = create_nerf(smoke_config["model"])
    rays_o, rays_d = get_rays_flat(16, 16, dataset.focal, dataset.poses[0])
    rays_o = rays_o[:32]
    rays_d = rays_d[:32]
    pts, z_vals = sample_points_along_rays(rays_o, rays_d, near=2.0, far=6.0, num_samples=16, stratified=True)
    
    # Forward pass
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = rays_d[:, None, :].expand_as(pts).reshape(-1, 3)
    pts_enc = pos_enc(pts_flat)
    dirs_enc = dir_enc(dirs_flat)
    raw = model(pts_enc, dirs_enc)
    
    rgb = raw["rgb"].reshape(32, 16, 3)
    sigma = raw["sigma"].reshape(32, 16, 1)
    rgb_sigma = torch.cat([rgb, sigma], dim=-1)
    
    renderer = VolumeRenderer()
    rendered = renderer.render_rays(rgb_sigma, z_vals, rays_d, white_bkgd=True)
    assert rendered["rgb"].shape == (32, 3), f"Bad render shape: {rendered['rgb'].shape}"
    
    # Loss & backprop
    gt_rgb = torch.rand(32, 3)
    loss = torch.nn.functional.mse_loss(rendered["rgb"], gt_rgb)
    loss.backward()
    
    grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert grad_exists, "Gradients failed to propagate to model parameters!"
    print(f"  Forward pass, loss ({loss.item():.4f}), and backprop OK.")

    # Step 5: Test training script
    print("\n[Step 5/9] Running training pipeline via CLI (20 iterations)...")
    res = subprocess.run(
        [sys.executable, "-m", "src.train", "--config", str(smoke_config_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(res.stdout[-600:] if len(res.stdout) > 600 else res.stdout)
    if res.returncode != 0:
        print("STDERR:", res.stderr)
        raise RuntimeError("Training script failed!")
    print("  Training script completed successfully.")

    # Step 6: Verify checkpoint saving
    print("\n[Step 6/9] Verifying checkpoint creation...")
    ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "latest.pth"
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} was not created!"
    print(f"  Checkpoint exists at {ckpt_path} ({ckpt_path.stat().st_size / 1024:.1f} KB).")

    # Step 7: Verify checkpoint loading and resumption
    print("\n[Step 7/9] Testing checkpoint resumption...")
    res = subprocess.run(
        [sys.executable, "-m", "src.train", "--config", str(smoke_config_path), "--resume", str(ckpt_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print("STDERR:", res.stderr)
        raise RuntimeError("Resume training failed!")
    print("  Resume training completed successfully.")

    # Step 8: Test Novel View Rendering
    print("\n[Step 8/9] Testing novel view rendering via src.render...")
    render_out_dir = PROJECT_ROOT / "outputs" / "renders" / "smoke_renders"
    res = subprocess.run(
        [
            sys.executable, "-m", "src.render",
            "--checkpoint", str(ckpt_path),
            "--config", str(smoke_config_path),
            "--num_frames", "4",
            "--output_dir", str(render_out_dir)
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(res.stdout[-400:] if len(res.stdout) > 400 else res.stdout)
    if res.returncode != 0:
        print("STDERR:", res.stderr)
        raise RuntimeError("Rendering failed!")
    
    rendered_frames = list(render_out_dir.glob("*.png"))
    assert len(rendered_frames) >= 4, f"Expected >= 4 rendered frames, got {len(rendered_frames)}"
    print(f"  Novel view rendering OK ({len(rendered_frames)} frames generated).")

    # Step 9: Test Evaluation
    print("\n[Step 9/9] Testing evaluation via src.evaluate...")
    res = subprocess.run(
        [
            sys.executable, "-m", "src.evaluate",
            "--checkpoint", str(ckpt_path),
            "--config", str(smoke_config_path),
            "--split", "val",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(res.stdout[-500:] if len(res.stdout) > 500 else res.stdout)
    if res.returncode != 0:
        print("STDERR:", res.stderr)
        raise RuntimeError("Evaluation failed!")
    
    eval_json = PROJECT_ROOT / "outputs" / "metrics" / "eval_results.json"
    assert eval_json.exists(), f"Evaluation JSON {eval_json} not found!"
    print("  Evaluation script completed OK.")

    print("\n" + "=" * 60)
    print("  ALL 9 END-TO-END VERIFICATION CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

