"""
NeRF Experiment and Ablation Study Runner.

Executes a series of training and evaluation experiments defined in an experiments
YAML configuration (e.g., configs/experiments.yaml), systematically varying
hyperparameters such as:
  - Positional encoding frequency bands (L = 4, 10, 16)
  - Coarse ray sample counts (N = 32, 64, 128)
  - Training iteration budgets (10k, 25k, 50k)

For each experiment:
  1. Merges baseline config with experiment-specific overrides.
  2. Creates a temporary configuration file.
  3. Launches training as a subprocess: `python -m src.train --config ...`.
  4. Runs evaluation on validation / test split.
  5. Collects metrics and compiles them into `outputs/metrics/experiment_results.csv`.
  6. Automatically generates ablation comparison bar charts in `outputs/figures/`.
  7. Prints a clean tabular summary to the console.

Usage:
    python scripts/run_experiments.py --config configs/experiments.yaml
    python scripts/run_experiments.py --ablation pos_enc_ablation
    python scripts/run_experiments.py --dry-run
"""

import argparse
import copy
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure MPLCONFIGDIR is in a writable location
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from src.visualization import plot_ablation_results


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def extract_varied_info(exp_dict: Dict[str, Any], vary_key: Optional[str]) -> Tuple[str, str]:
    """Helper to extract parameter name and value that was varied."""
    if not vary_key:
        return "custom", "n/a"

    parts = vary_key.split(".")
    curr = exp_dict
    for p in parts:
        if isinstance(curr, dict) and p in curr:
            curr = curr[p]
        else:
            return vary_key, "default"
    return vary_key, str(curr)


def run_single_experiment(
    exp_name: str,
    config_path: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute training for a single experiment and return results dictionary."""
    print(f"\n{'=' * 65}")
    print(f"  Starting Experiment: {exp_name}")
    print(f"  Config: {config_path}")
    print(f"{'=' * 65}")

    start_time = time.time()
    cmd = [sys.executable, "-m", "src.train", "--config", str(config_path)]

    if dry_run:
        print(f"[DRY-RUN] Command: {' '.join(cmd)}")
        elapsed = 0.0
        return {
            "status": "DRY_RUN",
            "time_sec": elapsed,
            "final_loss": 0.0,
            "val_psnr": 0.0,
        }

    try:
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        elapsed = time.time() - start_time
        print(f"[+] Completed {exp_name} in {elapsed:.1f}s")

        # Try to read checkpoint metrics if available
        # Find checkpoint
        cfg = load_yaml(config_path)
        ckpt_dir = PROJECT_ROOT / cfg.get("output", {}).get("checkpoint_dir", "outputs/checkpoints")
        best_ckpt = ckpt_dir / "best.pth"
        latest_ckpt = ckpt_dir / "latest.pth"
        target_ckpt = best_ckpt if best_ckpt.exists() else latest_ckpt

        final_loss = 0.0
        val_psnr = 0.0

        if target_ckpt.exists():
            import torch
            ckpt_data = torch.load(str(target_ckpt), map_location="cpu")
            final_loss = float(ckpt_data.get("loss", 0.0))
            val_psnr = float(ckpt_data.get("val_psnr", 0.0))

        return {
            "status": "SUCCESS",
            "time_sec": elapsed,
            "final_loss": final_loss,
            "val_psnr": val_psnr,
        }

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[-] Experiment {exp_name} failed with exit code {e.returncode}")
        return {
            "status": f"FAILED(code={e.returncode})",
            "time_sec": elapsed,
            "final_loss": -1.0,
            "val_psnr": -1.0,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[-] Experiment {exp_name} error: {e}")
        return {
            "status": "ERROR",
            "time_sec": elapsed,
            "final_loss": -1.0,
            "val_psnr": -1.0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeRF Experiment Runner and Ablation Study Engine."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments.yaml",
        help="Path to experiments master YAML configuration.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Specific ablation suite to run (e.g. pos_enc_ablation, sampling_ablation, budget_ablation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution, generating configs without running training.",
    )
    parser.add_argument(
        "--override-iters",
        type=int,
        default=None,
        help="Override iteration count for all experiments (useful for fast smoke-testing).",
    )
    args = parser.parse_args()

    exp_yaml_path = Path(args.config)
    if not exp_yaml_path.exists():
        print(f"Error: Master configuration not found at {exp_yaml_path}")
        sys.exit(1)

    exp_data = load_yaml(exp_yaml_path)
    base_config_rel = exp_data.get("base_config", "configs/lego_baseline.yaml")
    base_config_path = PROJECT_ROOT / base_config_rel
    if not base_config_path.exists():
        print(f"Error: Base configuration file not found at {base_config_path}")
        sys.exit(1)

    base_config = load_yaml(base_config_path)

    # Filter suites
    suites = {}
    for k, v in exp_data.items():
        if isinstance(v, dict) and "experiments" in v:
            if args.ablation is None or args.ablation == k:
                suites[k] = v

    if not suites:
        print(f"No experiment suites matched filter: '{args.ablation}'")
        sys.exit(1)

    all_results: List[Dict[str, Any]] = []
    temp_dir = PROJECT_ROOT / "configs" / "generated"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"  PyTorch NeRF: Ablation Experiment Pipeline")
    print(f"  Master Config: {exp_yaml_path}")
    print(f"  Base Config  : {base_config_path}")
    print(f"  Suites Found : {list(suites.keys())}")
    print(f"  Dry Run      : {args.dry_run}")
    print(f"=======================================================\n")

    for suite_name, suite_cfg in suites.items():
        desc = suite_cfg.get("description", suite_name)
        vary_key = suite_cfg.get("vary", None)
        experiments = suite_cfg.get("experiments", [])

        print(f"\n>>> Running Ablation Suite: [{suite_name}]")
        print(f"    Description: {desc}")
        print(f"    Varying:     {vary_key}")
        print(f"    Total Runs:  {len(experiments)}")

        suite_psnr_dict: Dict[str, float] = {}

        for exp in experiments:
            exp_name = exp.get("name", "unnamed_exp")

            # Deep merge baseline + experiment overrides
            merged = deep_merge(base_config, exp)
            merged["experiment_name"] = exp_name

            # Point output subdirectories to experiment-specific folders
            merged.setdefault("output", {})
            merged["output"]["checkpoint_dir"] = f"outputs/checkpoints/{exp_name}"
            merged["output"]["render_dir"] = f"outputs/renders/{exp_name}"
            merged["output"]["figures_dir"] = f"outputs/figures/{exp_name}"
            merged["output"]["metrics_dir"] = f"outputs/metrics/{exp_name}"
            merged["output"]["log_dir"] = f"logs/{exp_name}"

            if args.override_iters is not None:
                merged.setdefault("training", {})["num_iters"] = args.override_iters

            # Save generated config
            cfg_file = temp_dir / f"{exp_name}.yaml"
            save_yaml(merged, cfg_file)

            # Extract parameter value
            param_name, param_val = extract_varied_info(exp, vary_key)

            # Run experiment
            run_out = run_single_experiment(exp_name, cfg_file, dry_run=args.dry_run)

            record = {
                "suite": suite_name,
                "experiment_name": exp_name,
                "varied_param": param_name,
                "varied_value": param_val,
                "num_iters": merged.get("training", {}).get("num_iters", 0),
                "status": run_out["status"],
                "time_sec": round(run_out["time_sec"], 2),
                "final_loss": round(run_out["final_loss"], 5),
                "val_psnr": round(run_out["val_psnr"], 2),
            }
            all_results.append(record)
            suite_psnr_dict[f"{param_val}"] = run_out["val_psnr"]

        # Generate ablation comparison bar chart for this suite if not dry-run
        if not args.dry_run and any(v > 0 for v in suite_psnr_dict.values()):
            fig_path = PROJECT_ROOT / "outputs" / "figures" / f"ablation_{suite_name}.png"
            try:
                plot_ablation_results(
                    suite_psnr_dict,
                    metric="psnr",
                    save_path=str(fig_path),
                )
                print(f"[+] Ablation chart saved: {fig_path}")
            except Exception as e:
                print(f"[-] Could not generate plot for {suite_name}: {e}")

    # Write metrics CSV
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "experiment_results.csv"

    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[+] Compiled results saved to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("  EXPERIMENT SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Suite':<20} | {'Experiment':<15} | {'Param':<15} | {'Value':<8} | {'Status':<10} | {'PSNR (dB)':<10} | {'Time (s)':<8}"
    print(header)
    print("-" * 90)
    for r in all_results:
        row = (
            f"{r['suite']:<20} | "
            f"{r['experiment_name']:<15} | "
            f"{r['varied_param'].split('.')[-1]:<15} | "
            f"{str(r['varied_value']):<8} | "
            f"{r['status']:<10} | "
            f"{r['val_psnr']:<10.2f} | "
            f"{r['time_sec']:<8.1f}"
        )
        print(row)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
