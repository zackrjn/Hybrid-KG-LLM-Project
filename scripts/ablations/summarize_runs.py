#!/usr/bin/env python3
"""
Summarize existing DPO ablation runs in a directory and generate base-level plots.

Usage:
  python scripts/ablations/summarize_runs.py --dir experiments/dpo/i4_smoke_text
  python scripts/ablations/summarize_runs.py --dir experiments/dpo/i4_smoke_imgs
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure repo root on path to import plotting util
REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    # Reuse plotting from ablation runner
    from scripts.ablations.dpo_ablate import plot_results  # type: ignore
except Exception:
    plot_results = None  # Fallback to minimal plotting if import fails


def collect_results(base_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for item in sorted(base_dir.iterdir()):
        if not item.is_dir():
            continue
        # Heuristic: run dir name contains 4 fields like ts_thr_beta_seed
        parts = item.name.split("_")
        if len(parts) < 4:
            # allow arbitrary names but require files
            pass
        cfg_path = item / "config.json"
        met_path = item / "metrics.json"
        if not (cfg_path.exists() and met_path.exists()):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            with open(met_path, "r", encoding="utf-8") as f:
                met = json.load(f)
            results.append({
                "run_id": item.name,
                "threshold": float(cfg.get("threshold")),
                "beta": float(cfg.get("beta")),
                "seed": int(cfg.get("seed")),
                "metrics": met,
                "run_dir": str(item),
            })
        except Exception:
            # Skip malformed entries
            continue
    return results


def write_summary(results: List[Dict[str, Any]], base_dir: Path) -> None:
    summary = {
        "num_runs": len(results),
        "thresholds": sorted({r["threshold"] for r in results}),
        "betas": sorted({r["beta"] for r in results}),
        "seeds": sorted({r["seed"] for r in results}),
        "runs": [],
    }
    for r in results:
        summary["runs"].append({
            "run_id": r["run_id"],
            "threshold": r["threshold"],
            "beta": r["beta"],
            "seed": r["seed"],
            "train_loss": r["metrics"].get("train_loss"),
            "eval_loss": r["metrics"].get("eval_loss"),
            "error": r["metrics"].get("error"),
        })
    with open(base_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def minimal_plot(results: List[Dict[str, Any]], base_dir: Path) -> None:
    # Fallback: create a scatter grid of available points
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np  # noqa: F401
    except Exception:
        return
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    xs, ys, cs = [], [], []
    for r in results:
        loss = r["metrics"].get("train_loss")
        if loss is None:
            continue
        xs.append(r["beta"])
        ys.append(r["threshold"])
        cs.append(loss)
    if xs:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis")
        ax.set_xlabel("Beta")
        ax.set_ylabel("Threshold")
        ax.set_title("Train Loss (scatter)")
        plt.colorbar(sc, ax=ax, label="Train Loss")
        plt.tight_layout()
        plt.savefig(plots_dir / "grid_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Base directory with ablation run subfolders")
    args = parser.parse_args()
    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f"Error: directory not found: {base_dir}")
        sys.exit(1)
    results = collect_results(base_dir)
    print(f"Found {len(results)} runs under {base_dir}")
    write_summary(results, base_dir)
    # Generate plots
    if len(results) >= 2:
        try:
            if plot_results is not None:
                plot_results(results, base_dir, dry_run=False)
            else:
                minimal_plot(results, base_dir)
        except Exception:
            minimal_plot(results, base_dir)
    print(f"Summary written to: {base_dir / 'summary.json'}")


if __name__ == "__main__":
    main()


