"""
Environment and System Diagnostic Script for PyTorch NeRF.

Verifies:
1. Python version and Operating System
2. PyTorch installation, version, and CUDA/CPU device availability
3. Required Python dependencies and their installed versions
4. System hardware (CPU cores, RAM capacity, GPU if present)
5. Project directory structure (data/, outputs/, configs/)
6. End-to-end smoke test through a tiny NeRF forward and backward pass
"""

import os
import platform
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported regardless of execution cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure MPLCONFIGDIR is in a writable location
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
from typing import Dict, List, Tuple


def print_header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def check_python_and_os() -> Tuple[bool, str]:
    py_ver = sys.version.split()[0]
    os_name = f"{platform.system()} {platform.release()} ({platform.architecture()[0]})"
    major, minor = sys.version_info.major, sys.version_info.minor

    info = f"Python: {py_ver} | OS: {os_name}"
    passed = (major == 3 and minor >= 8)
    return passed, info


def check_pytorch() -> Tuple[bool, str]:
    try:
        import torch
        ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        cuda_info = f"CUDA Available: {cuda_avail}"
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            cuda_info += f" ({device_name})"
        else:
            cuda_info += " (Running in CPU mode)"
        info = f"PyTorch {ver} | {cuda_info}"
        return True, info
    except ImportError as e:
        return False, f"PyTorch not installed: {e}"


def check_packages() -> Tuple[bool, List[str]]:
    required_packages = [
        "torch",
        "numpy",
        "matplotlib",
        "PIL",  # Pillow
        "tqdm",
        "yaml",  # PyYAML
    ]
    optional_packages = [
        "cv2",       # opencv-python
        "skimage",   # scikit-image
        "tensorboard",
        "imageio",
        "pytest",
        "psutil",
    ]

    all_passed = True
    reports = []

    for pkg in required_packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            reports.append(f"  [PASS] {pkg:<15} : v{ver}")
        except ImportError:
            all_passed = False
            reports.append(f"  [FAIL] {pkg:<15} : NOT INSTALLED (Required)")

    for pkg in optional_packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            reports.append(f"  [PASS] {pkg:<15} : v{ver}")
        except ImportError:
            reports.append(f"  [WARN] {pkg:<15} : NOT INSTALLED (Optional)")

    return all_passed, reports


def get_hardware_info() -> Tuple[bool, str]:
    info_lines = []
    # CPU info
    cores_physical = os.cpu_count() or "Unknown"
    info_lines.append(f"CPU Cores: {cores_physical} logical cores")

    # RAM and Disk info via psutil if available
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024 ** 3)
        avail_ram_gb = vm.available / (1024 ** 3)
        info_lines.append(f"System RAM: {total_ram_gb:.2f} GB total, {avail_ram_gb:.2f} GB available")
    except ImportError:
        info_lines.append("RAM info: psutil not installed")

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            device_cnt = torch.cuda.device_count()
            for d in range(device_cnt):
                name = torch.cuda.get_device_name(d)
                mem = torch.cuda.get_device_properties(d).total_memory / (1024 ** 3)
                info_lines.append(f"GPU {d}: {name} ({mem:.2f} GB VRAM)")
        else:
            info_lines.append("GPU: No CUDA-capable GPU detected (CPU mode active)")
    except Exception:
        pass

    return True, "\n  ".join(info_lines)


def check_directories() -> Tuple[bool, List[str]]:
    dirs_to_check = [
        ("configs", True),
        ("src", True),
        ("scripts", True),
        ("data", False),
        ("outputs/checkpoints", False),
        ("outputs/renders", False),
        ("outputs/figures", False),
        ("outputs/metrics", False),
        ("logs", False),
    ]

    all_passed = True
    reports = []
    root = Path(__file__).resolve().parent.parent

    for rel_path, required in dirs_to_check:
        p = root / rel_path
        if p.exists() and p.is_dir():
            reports.append(f"  [PASS] {rel_path:<22} : exists")
        else:
            if required:
                all_passed = False
                reports.append(f"  [FAIL] {rel_path:<22} : MISSING (Required)")
            else:
                reports.append(f"  [INFO] {rel_path:<22} : not yet created (will be created automatically)")

    return all_passed, reports


def run_smoke_test() -> Tuple[bool, str]:
    """Run a small dummy tensor forward and backward pass through NeRF modules."""
    try:
        import torch
        from src.encoding import PositionalEncoder
        from src.model import NeRF

        device = torch.device("cpu")
        pos_enc = PositionalEncoder(num_freqs=4)
        dir_enc = PositionalEncoder(num_freqs=2)

        # Output dims: 3 + 2*4*3 = 27, 3 + 2*2*3 = 15
        pos_dim = 3 + 2 * 4 * 3
        dir_dim = 3 + 2 * 2 * 3

        model = NeRF(
            pos_enc_dim=pos_dim,
            dir_enc_dim=dir_dim,
            num_layers=4,
            hidden_dim=64,
            skip_connect=(2,),
            use_viewdirs=True,
        ).to(device)

        batch_size = 16
        pts = torch.randn(batch_size, 3, device=device)
        dirs = torch.randn(batch_size, 3, device=device)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

        pts_encoded = pos_enc(pts)
        dirs_encoded = dir_enc(dirs)

        out = model(pts_encoded, dirs_encoded)

        # Check outputs
        rgb = out["rgb"]
        sigma = out["sigma"]

        if rgb.shape != (batch_size, 3):
            return False, f"Unexpected RGB shape: {rgb.shape}"
        if sigma.shape != (batch_size, 1):
            return False, f"Unexpected sigma shape: {sigma.shape}"

        # Test backward pass
        loss = rgb.sum() + sigma.sum()
        loss.backward()

        return True, "Smoke test passed successfully (Forward + Backward pass OK)"
    except Exception as e:
        return False, f"Smoke test failed: {e}"


def main() -> None:
    print_header("PyTorch NeRF: Environment Diagnostic Check")
    summary: Dict[str, bool] = {}

    # 1. Python & OS
    ok, info = check_python_and_os()
    summary["Python & OS"] = ok
    status = "PASS" if ok else "FAIL"
    print(f"\n[1] Platform & Python: [{status}]")
    print(f"  {info}")

    # 2. PyTorch & CUDA
    ok, info = check_pytorch()
    summary["PyTorch"] = ok
    status = "PASS" if ok else "FAIL"
    print(f"\n[2] PyTorch & Device: [{status}]")
    print(f"  {info}")

    # 3. Hardware Info
    ok, info = get_hardware_info()
    summary["Hardware"] = ok
    print(f"\n[3] Hardware Resources:")
    print(f"  {info}")

    # 4. Packages
    ok, reports = check_packages()
    summary["Required Packages"] = ok
    status = "PASS" if ok else "FAIL"
    print(f"\n[4] Dependencies: [{status}]")
    for r in reports:
        print(r)

    # 5. Directories
    ok, dir_reports = check_directories()
    summary["Project Structure"] = ok
    status = "PASS" if ok else "FAIL"
    print(f"\n[5] Directory Structure: [{status}]")
    for r in dir_reports:
        print(r)

    # 6. NeRF Smoke Test
    ok, test_info = run_smoke_test()
    summary["NeRF Smoke Test"] = ok
    status = "PASS" if ok else "FAIL"
    print(f"\n[6] Functional Smoke Test: [{status}]")
    print(f"  {test_info}")

    # Overall Summary
    print_header("DIAGNOSTIC SUMMARY")
    all_ok = True
    for item, passed in summary.items():
        tag = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_ok = False
        print(f"  {tag:<8} : {item}")

    print("-" * 65)
    if all_ok:
        print("  ALL CORE CHECKS PASSED: Environment is ready for NeRF experiments.")
    else:
        print("  SOME CHECKS FAILED: Please resolve the failed items above.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
