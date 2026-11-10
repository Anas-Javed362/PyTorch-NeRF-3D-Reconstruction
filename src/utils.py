"""Shared utilities for the PyTorch NeRF project."""

import os
import json
import yaml
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file and return it as a nested dict.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config as a nested Python dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def merge_config(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into a copy of *base*.

    For every key in *override*:
    - If both values are dicts, recurse.
    - Otherwise the override value wins.

    Args:
        base:     The base configuration dict.
        override: Overrides to apply on top of base.

    Returns:
        A new dict with merged settings.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic ops where possible (may slow things down on GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(device_str: str = "auto") -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Args:
        device_str: ``'auto'`` selects CUDA when available, else CPU.
                    ``'cuda'`` forces CUDA (raises if unavailable).
                    ``'cpu'`` forces CPU.

    Returns:
        A resolved :class:`torch.device`.

    Raises:
        RuntimeError: If ``device_str='cuda'`` but CUDA is not available.
        ValueError: If *device_str* is not recognised.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but CUDA is not available on this system."
            )
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    # Allow explicit 'cuda:0', 'cuda:1', etc.
    try:
        return torch.device(device_str)
    except RuntimeError as exc:
        raise ValueError(f"Unrecognised device string: '{device_str}'") from exc


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    loss: float,
    config: Dict,
    extra: Optional[Dict] = None,
) -> None:
    """Save a training checkpoint to disk.

    The checkpoint contains the model state dict, optimizer state dict,
    current iteration, latest loss, the config used for training, and any
    *extra* information supplied by the caller.

    Args:
        path:      Destination file path (will be created / overwritten).
        model:     NeRF model whose weights should be saved.
        optimizer: Optimizer whose state should be saved.
        iteration: Current training iteration (0-based).
        loss:      Most recent training loss value.
        config:    Full training config dict.
        extra:     Optional dict of additional data to store in the checkpoint.
    """
    make_dirs(str(Path(path).parent))
    payload: Dict[str, Any] = {
        "iteration": iteration,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Args:
        path:      Path to the checkpoint file.
        model:     Model to load weights into.
        optimizer: If provided, optimizer state is restored as well.
        device:    Device to map tensors to. Defaults to CPU.

    Returns:
        Tuple of ``(iteration, loss, config)`` where *config* is the dict
        embedded in the checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    iteration: int = checkpoint.get("iteration", 0)
    loss: float = checkpoint.get("loss", float("inf"))
    config: Dict = checkpoint.get("config", {})

    return iteration, loss, config


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(mse: float) -> float:
    """Compute Peak Signal-to-Noise Ratio from Mean Squared Error.

    PSNR = 10 * log10(1 / mse)

    Args:
        mse: Mean squared error (must be >= 0).

    Returns:
        PSNR in dB. Returns ``float('inf')`` when *mse* == 0.
    """
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def img_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8 in [0, 255].

    Values are clipped to [0, 1] before scaling to handle minor floating-point
    overflows.

    Args:
        img: NumPy array with float values expected in [0, 1].

    Returns:
        NumPy array with dtype uint8.
    """
    img_clipped = np.clip(img, 0.0, 1.0)
    return (img_clipped * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# File system
# ---------------------------------------------------------------------------

def make_dirs(*paths: str) -> None:
    """Create one or more directories (and their parents) if they don't exist.

    Args:
        *paths: Paths to create.
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format a duration in seconds as a human-readable ``'Xh Ym Zs'`` string.

    Args:
        seconds: Duration in seconds (non-negative).

    Returns:
        String like ``'1h 23m 45s'``, ``'5m 3s'``, or ``'7s'``.

    Examples:
        >>> format_time(3661)
        '1h 1m 1s'
        >>> format_time(90)
        '1m 30s'
        >>> format_time(7)
        '7s'
    """
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for timing code blocks.

    Usage::

        with Timer() as t:
            do_something()
        print(f"Took {t.elapsed:.2f}s")

    Attributes:
        elapsed: Elapsed wall-clock time in seconds (set after ``__exit__``).
                 Before the block finishes, it holds the time since entry.
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:  # type: ignore[override]
        self.elapsed = time.perf_counter() - self._start
