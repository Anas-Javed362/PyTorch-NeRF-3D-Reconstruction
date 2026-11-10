"""
NeRF Synthetic Blender Dataset Loader.

Loads the standard NeRF synthetic dataset produced by Blender rendering:
- transforms_{split}.json  – camera intrinsics + per-frame extrinsics
- {split}/r_*.png          – RGBA PNG images (800×800 originally)

Supports train / val / test splits with optional subsampling, image resizing,
and RGBA-to-RGB compositing against a configurable background colour.
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# BlenderDataset
# ---------------------------------------------------------------------------

class BlenderDataset(Dataset):
    """
    PyTorch Dataset for the NeRF synthetic Blender dataset.

    After construction the following attributes are available:
      self.focal  – focal length in pixels (scaled to img_wh)
      self.imgs   – (N, H, W, 3) float32 tensor, RGB in [0, 1]
      self.poses  – (N, 4, 4) float32 tensor, camera-to-world matrices
      self.H      – image height (pixels)
      self.W      – image width  (pixels)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_wh: Tuple[int, int] = (100, 100),
        white_bkgd: bool = True,
        val_skip: int = 1,
    ) -> None:
        """
        Load the NeRF synthetic Blender dataset.

        Args:
            data_dir:   Path to scene directory (e.g. 'data/lego').
            split:      One of 'train', 'val', or 'test'.
            img_wh:     (width, height) to resize images to.
            white_bkgd: If True, composite RGBA against a white background;
                        otherwise composite against black.
            val_skip:   For val/test splits, use every val_skip-th image to
                        keep the validation/test set manageable.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.img_wh = img_wh
        self.white_bkgd = white_bkgd
        self.val_skip = val_skip if split != "train" else 1

        self.W, self.H = img_wh

        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read the transforms JSON, load images, build tensor arrays."""
        transforms_path = self.data_dir / f"transforms_{self.split}.json"
        if not transforms_path.exists():
            raise FileNotFoundError(
                f"Transforms file not found: {transforms_path}. "
                f"Make sure the dataset is downloaded and placed in {self.data_dir}."
            )

        try:
            with transforms_path.open("r") as f:
                meta = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in {transforms_path}: {exc}"
            ) from exc

        # Validate required keys
        if "camera_angle_x" not in meta:
            raise ValueError(
                f"{transforms_path} is missing required key 'camera_angle_x'."
            )
        if "frames" not in meta or not isinstance(meta["frames"], list):
            raise ValueError(
                f"{transforms_path} is missing required key 'frames' (list)."
            )

        camera_angle_x: float = float(meta["camera_angle_x"])
        frames: List[dict] = meta["frames"]

        # Subsample frames for val/test
        if self.val_skip > 1:
            frames = frames[:: self.val_skip]

        # We need at least one frame
        if len(frames) == 0:
            raise ValueError(
                f"No frames found in {transforms_path} after subsampling "
                f"(val_skip={self.val_skip})."
            )

        # ---- Load the first image to discover original resolution ---------
        first_path = self._resolve_img_path(frames[0]["file_path"])
        with Image.open(first_path) as img0:
            W_orig, H_orig = img0.size

        # ---- Compute focal length ----------------------------------------
        # From the original image width and horizontal FOV angle:
        focal_orig = W_orig / (2.0 * math.tan(camera_angle_x / 2.0))

        # Scale proportionally to the new width
        self.focal = focal_orig * (self.W / W_orig)

        # ---- Load images and poses ---------------------------------------
        imgs_list: List[np.ndarray] = []
        poses_list: List[np.ndarray] = []
        self.file_paths: List[Path] = []

        for frame in frames:
            # --- Pose ---
            if "transform_matrix" not in frame:
                raise ValueError(
                    f"Frame missing 'transform_matrix': {frame.get('file_path', '?')}"
                )
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            if c2w.shape != (4, 4):
                raise ValueError(
                    f"transform_matrix must be (4,4), got {c2w.shape} for "
                    f"frame {frame.get('file_path', '?')}"
                )
            poses_list.append(c2w)

            # --- Image ---
            img_path = self._resolve_img_path(frame["file_path"])
            self.file_paths.append(img_path)
            rgb = self._load_image(img_path)
            imgs_list.append(rgb)

        # Stack into tensors
        self.imgs = torch.from_numpy(
            np.stack(imgs_list, axis=0)
        ).float()   # (N, H, W, 3)

        self.poses = torch.from_numpy(
            np.stack(poses_list, axis=0)
        ).float()   # (N, 4, 4)

    def _resolve_img_path(self, file_path: str) -> Path:
        """
        Resolve a frame file_path (possibly without extension) to an
        absolute PNG path.

        The transforms JSON stores paths like './train/r_0' (no extension).
        We normalise the path and append '.png' if necessary.
        """
        # Normalise separators and strip leading './'
        clean = file_path.replace("\\", "/").lstrip("./")
        p = self.data_dir / clean
        # Add .png extension if not already present
        if p.suffix == "":
            p = p.with_suffix(".png")
        elif p.suffix.lower() != ".png":
            p = p.with_suffix(".png")

        if not p.exists():
            raise FileNotFoundError(
                f"Image not found: {p}. "
                f"Original file_path in JSON: {file_path!r}"
            )
        return p

    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load a PNG image, resize to self.img_wh, composite RGBA to RGB.

        Returns:
            np.ndarray of shape (H, W, 3), float32 in [0, 1].
        """
        with Image.open(path) as img:
            img = img.convert("RGBA")
            if (img.width, img.height) != (self.W, self.H):
                img = img.resize((self.W, self.H), Image.LANCZOS)
            img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, 4)

        rgb = img_np[..., :3]    # (H, W, 3)
        alpha = img_np[..., 3:]  # (H, W, 1)

        if self.white_bkgd:
            background = np.ones_like(rgb)
        else:
            background = np.zeros_like(rgb)

        # Alpha compositing: rgb_out = rgb_src * alpha + bg * (1 - alpha)
        rgb_composited = rgb * alpha + background * (1.0 - alpha)
        return rgb_composited.astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single image and its camera pose.

        Returns:
            dict with keys:
                'img':  Tensor of shape (H, W, 3), float32 in [0, 1].
                'pose': Tensor of shape (4, 4), float32 camera-to-world matrix.
                'idx':  Integer index of this sample.
        """
        return {
            "img": self.imgs[idx],
            "pose": self.poses[idx],
            "idx": idx,
        }


# ---------------------------------------------------------------------------
# get_all_rays
# ---------------------------------------------------------------------------

def get_all_rays(
    dataset: BlenderDataset,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate all camera rays for every image in the dataset.

    Uses :func:`src.camera.get_rays` to cast rays for each image,
    then concatenates across the whole dataset.

    Args:
        dataset: A :class:`BlenderDataset` instance.
        device:  PyTorch device to place tensors on.

    Returns:
        Tuple ``(rays_o, rays_d, rgb_gt)`` each of shape ``(N*H*W, 3)``:
          - rays_o  – ray origins in world space.
          - rays_d  – ray directions in world space (not normalised; normalisation
                      is performed inside the renderer).
          - rgb_gt  – ground-truth RGB values in [0, 1].
    """
    # Lazy import to avoid circular dependency issues at module load time
    from src.camera import get_rays  # type: ignore[import]

    H, W = dataset.H, dataset.W
    focal = dataset.focal

    rays_o_list: List[torch.Tensor] = []
    rays_d_list: List[torch.Tensor] = []
    rgb_gt_list: List[torch.Tensor] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        pose = sample["pose"].to(device)   # (4, 4)
        img = sample["img"].to(device)     # (H, W, 3)

        # get_rays returns (H, W, 3) each
        rays_o, rays_d = get_rays(H, W, focal, pose)

        rays_o_list.append(rays_o.reshape(-1, 3))   # (H*W, 3)
        rays_d_list.append(rays_d.reshape(-1, 3))   # (H*W, 3)
        rgb_gt_list.append(img.reshape(-1, 3))       # (H*W, 3)

    rays_o = torch.cat(rays_o_list, dim=0)   # (N*H*W, 3)
    rays_d = torch.cat(rays_d_list, dim=0)   # (N*H*W, 3)
    rgb_gt = torch.cat(rgb_gt_list, dim=0)   # (N*H*W, 3)

    return rays_o, rays_d, rgb_gt


# ---------------------------------------------------------------------------
# SyntheticDataModule
# ---------------------------------------------------------------------------

class SyntheticDataModule:
    """
    Plain Python data module that owns train, val, and test datasets.

    Not a PyTorch Lightning module – just a convenient container that
    reads a config dict and instantiates the three :class:`BlenderDataset`
    objects.

    Config dict expected keys (all inside ``config['dataset']``):
      - data_dir   (str)        – scene directory, e.g. 'data/lego'
      - img_wh     (list[int])  – [width, height] to resize images to
      - white_bkgd (bool)       – composite RGBA against white background
      - val_skip   (int)        – subsample val/test sets (default 1)
    """

    def __init__(self, config: dict) -> None:
        ds_cfg = config.get("dataset", config)  # allow flat or nested config

        data_dir: str = ds_cfg["data_dir"]
        img_wh: Tuple[int, int] = tuple(ds_cfg.get("img_wh", (100, 100)))  # type: ignore[assignment]
        white_bkgd: bool = ds_cfg.get("white_bkgd", True)
        val_skip: int = int(ds_cfg.get("val_skip", 1))

        self._train_dataset = BlenderDataset(
            data_dir, split="train", img_wh=img_wh, white_bkgd=white_bkgd, val_skip=1
        )
        self._val_dataset = BlenderDataset(
            data_dir, split="val", img_wh=img_wh, white_bkgd=white_bkgd, val_skip=val_skip
        )
        self._test_dataset = BlenderDataset(
            data_dir, split="test", img_wh=img_wh, white_bkgd=white_bkgd, val_skip=val_skip
        )

        self._validate_splits()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def train_dataset(self) -> BlenderDataset:
        return self._train_dataset

    @property
    def val_dataset(self) -> BlenderDataset:
        return self._val_dataset

    @property
    def test_dataset(self) -> BlenderDataset:
        return self._test_dataset

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_splits(self) -> None:
        """
        Sanity-check that training and test images do not overlap.
        """
        # Verify focal lengths are consistent across splits (same camera)
        train_focal = self._train_dataset.focal
        test_focal = self._test_dataset.focal
        if not math.isclose(train_focal, test_focal, rel_tol=1e-4):
            raise ValueError(
                f"Train and test focal lengths differ substantially: "
                f"train={train_focal:.4f}, test={test_focal:.4f}. "
                f"Make sure both splits come from the same scene."
            )

        # Validate that train and test image files are completely disjoint
        train_files = {p.resolve() for p in self._train_dataset.file_paths}
        test_files = {p.resolve() for p in self._test_dataset.file_paths}
        overlap = train_files.intersection(test_files)
        if overlap:
            raise ValueError(
                f"Train and test datasets contain overlapping images ({len(overlap)} overlaps). "
                f"Train and test sets must be strictly disjoint."
            )

        n_train = len(self._train_dataset)
        n_test = len(self._test_dataset)
        if n_train == 0 or n_test == 0:
            raise ValueError(
                f"Train or test dataset is empty (train={n_train}, test={n_test}). Check your data_dir."
            )
