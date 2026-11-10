"""
Dataset Setup and Verification Utility for PyTorch NeRF.

This script guides downloading and verifying the Synthetic Blender NeRF dataset
(e.g., the standard 'lego' scene).

Dataset Source:
    Google Drive: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
    Official NeRF dataset mirror: https://people.eecs.berkeley.edu/~bmild/nerf/nerf_synthetic.zip

Usage:
    python scripts/download_dataset.py --scene lego
    python scripts/download_dataset.py --scene drums
"""

import argparse
import os
import sys
from pathlib import Path


SYNTHETIC_BLENDER_GDRIVE = "https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1"
SYNTHETIC_BLENDER_ZIP_URL = "https://people.eecs.berkeley.edu/~bmild/nerf/nerf_synthetic.zip"

AVAILABLE_SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship"
]


def print_instructions(scene: str, target_dir: Path) -> None:
    print("=" * 70)
    print(f"  PyTorch NeRF Dataset Setup: '{scene}'")
    print("=" * 70)
    print("\nThe NeRF Blender synthetic dataset contains multi-view synthetic renders")
    print("with known camera intrinsic parameters and extrinsic poses.")
    print("\n[OPTION 1: Direct Google Drive Download (Official Folder)]")
    print(f"  URL: {SYNTHETIC_BLENDER_GDRIVE}")
    print("  1. Open the URL above in your browser.")
    print(f"  2. Locate the archive for '{scene}' (or 'nerf_synthetic.zip') and download it.")
    print(f"  3. Extract the contents into:")
    print(f"     {target_dir.resolve()}")

    print("\n[OPTION 2: Direct Berkeley Mirror (Full Zip, ~5.2 GB)]")
    print(f"  URL: {SYNTHETIC_BLENDER_ZIP_URL}")
    print(f"  Download and extract 'nerf_synthetic/{scene}' into:")
    print(f"     {target_dir.resolve()}")

    print("\nExpected directory layout:")
    print(f"  {target_dir}/")
    print(f"  ├── transforms_train.json")
    print(f"  ├── transforms_val.json")
    print(f"  ├── transforms_test.json")
    print(f"  ├── train/")
    print(f"  ├── val/")
    print(f"  └── test/")
    print("=" * 70)


def verify_dataset(scene_dir: Path) -> bool:
    print(f"\nVerifying dataset structure at: {scene_dir.resolve()} ...\n")
    if not scene_dir.exists():
        print(f"[-] Directory does not exist: {scene_dir}")
        print("    Please follow the download and extraction steps above.")
        return False

    required_files = [
        "transforms_train.json",
        "transforms_val.json",
        "transforms_test.json",
    ]
    required_dirs = [
        "train",
        "val",
        "test",
    ]

    all_ok = True

    print("Checking JSON camera transform metadata:")
    for fname in required_files:
        fpath = scene_dir / fname
        if fpath.is_file():
            size_kb = fpath.stat().st_size / 1024
            print(f"  [+] {fname:<25} FOUND ({size_kb:.1f} KB)")
        else:
            print(f"  [-] {fname:<25} MISSING")
            all_ok = False

    print("\nChecking image split directories:")
    for dname in required_dirs:
        dpath = scene_dir / dname
        if dpath.is_dir():
            pngs = list(dpath.glob("*.png"))
            print(f"  [+] {dname + '/':<25} FOUND ({len(pngs)} PNG images)")
            if len(pngs) == 0:
                print(f"      [WARNING] Directory '{dname}' exists but contains no .png files!")
                all_ok = False
        else:
            print(f"  [-] {dname + '/':<25} MISSING")
            all_ok = False

    print("-" * 70)
    if all_ok:
        print("[SUCCESS] Dataset verification passed! You are ready to train.")
    else:
        print("[INCOMPLETE] Some required files/directories are missing.")
    print("=" * 70)
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset download instructions and integrity verifier for NeRF synthetic data."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        help="Scene name (e.g., lego, drums, chair, ship). Default: lego",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root data directory. Default: data",
    )
    args = parser.parse_args()

    scene = args.scene.lower()
    scene_dir = Path(args.data_root) / scene

    # Print download instructions
    print_instructions(scene, scene_dir)

    # Check if files already exist
    verify_dataset(scene_dir)


if __name__ == "__main__":
    main()
