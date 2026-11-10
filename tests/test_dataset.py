"""
Unit tests for NeRF synthetic Blender dataset loader (src/dataset.py).
Tests use synthetic/mock data generated via pytest tmp_path (no real dataset download needed).
"""

import json
import math
import numpy as np
import pytest
import torch
from PIL import Image
from src.dataset import BlenderDataset


@pytest.fixture
def fake_dataset_dir(tmp_path):
    """
    Generate a minimal valid NeRF synthetic Blender dataset structure in tmp_path.
    Contains:
      - transforms_train.json with 2 frames
      - train/r_0.png and train/r_1.png (small 10x10 RGBA images)
    """
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Frame 0: Semi-transparent red (128 alpha)
    img0 = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    img0.save(train_dir / "r_0.png")

    # Frame 1: Completely transparent blue (0 alpha)
    img1 = Image.new("RGBA", (10, 10), (0, 0, 255, 0))
    img1.save(train_dir / "r_1.png")

    # transforms_train.json
    transforms_data = {
        "camera_angle_x": 0.6911112070083618,  # ~39.6 degrees
        "frames": [
            {
                "file_path": "./train/r_0",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            {
                "file_path": "./train/r_1",
                "transform_matrix": [
                    [0.0, -1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        ],
    }

    with open(tmp_path / "transforms_train.json", "w") as f:
        json.dump(transforms_data, f)

    return tmp_path


def test_focal_computation(fake_dataset_dir):
    """Verify focal length formula: f = W / (2 * tan(camera_angle_x / 2))."""
    W = 10
    camera_angle_x = 0.6911112070083618
    expected_focal = W / (2.0 * math.tan(camera_angle_x / 2.0))

    dataset = BlenderDataset(
        data_dir=str(fake_dataset_dir),
        split="train",
        img_wh=(W, W),
    )

    assert math.isclose(dataset.focal, expected_focal, rel_tol=1e-5)


def test_dataset_loading(fake_dataset_dir):
    """Verify dataset loads successfully from the synthetic structure."""
    dataset = BlenderDataset(
        data_dir=str(fake_dataset_dir),
        split="train",
        img_wh=(10, 10),
    )
    assert dataset is not None
    assert hasattr(dataset, "imgs")
    assert hasattr(dataset, "poses")
    assert hasattr(dataset, "focal")


def test_image_shape(fake_dataset_dir):
    """Verify loaded images have correct shape (N, H, W, 3)."""
    dataset = BlenderDataset(
        data_dir=str(fake_dataset_dir),
        split="train",
        img_wh=(10, 10),
    )
    assert dataset.imgs.shape == (2, 10, 10, 3)
    assert dataset.imgs.dtype == torch.float32


def test_pose_shape(fake_dataset_dir):
    """Verify camera poses have shape (N, 4, 4)."""
    dataset = BlenderDataset(
        data_dir=str(fake_dataset_dir),
        split="train",
        img_wh=(10, 10),
    )
    assert dataset.poses.shape == (2, 4, 4)
    assert dataset.poses.dtype == torch.float32


def test_white_background(fake_dataset_dir):
    """Verify white vs black background compositing for transparent pixels."""
    # Frame 1 has alpha = 0.0 (completely transparent)
    # With white_bkgd=True, pixel should be [1.0, 1.0, 1.0]
    ds_white = BlenderDataset(
        data_dir=str(fake_dataset_dir),
        split="train",
        img_wh=(10, 10),
        white_bkgd=True,
    )
    transparent_pixel_white = ds_white.imgs[1, 0, 0]
    assert torch.allclose(transparent_pixel_white, torch.tensor([1.0, 1.0, 1.0]), atol=1e-3)

    # With white_bkgd=False, pixel should be [0.0, 0.0, 0.0]
    ds_black = BlenderDataset(
        data_dir=str(fake_dataset_dir),
        split="train",
        img_wh=(10, 10),
        white_bkgd=False,
    )
    transparent_pixel_black = ds_black.imgs[1, 0, 0]
    assert torch.allclose(transparent_pixel_black, torch.tensor([0.0, 0.0, 0.0]), atol=1e-3)


def test_dataset_len(fake_dataset_dir):
    """Verify __len__ returns correct frame count."""
    dataset = BlenderDataset(
        data_dir=str(fake_dataset_dir),
        split="train",
        img_wh=(10, 10),
    )
    assert len(dataset) == 2


def test_dataset_getitem(fake_dataset_dir):
    """Verify __getitem__ returns dict with correct keys and shapes."""
    dataset = BlenderDataset(
        data_dir=str(fake_dataset_dir),
        split="train",
        img_wh=(10, 10),
    )
    sample0 = dataset[0]
    assert isinstance(sample0, dict)
    assert "img" in sample0
    assert "pose" in sample0
    assert "idx" in sample0

    assert sample0["img"].shape == (10, 10, 3)
    assert sample0["pose"].shape == (4, 4)
    assert sample0["idx"] == 0
