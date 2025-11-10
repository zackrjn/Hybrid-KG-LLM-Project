#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_csv_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def read_jsonl(path: str, max_items: int = 0) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if max_items and len(items) >= max_items:
                break
    return items


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plot_aggregates(base_out: str, results: List[Tuple[float, float, int]]):
    """results: list of (threshold, beta, train_examples) as a simple demo plot."""
    if len(results) < 2:
        return
    plots_dir = os.path.join(base_out, "plots")
    ensure_dir(plots_dir)
    # Scatter: threshold vs beta, size by train_examples (demo)
    xs = [r[0] for r in results]
    ys = [r[1] for r in results]
    sizes = [max(20, min(200, r[2] // 10)) for r in results]
    plt.figure(figsize=(5, 4))
    plt.scatter(xs, ys, s=sizes, c="tab:blue", alpha=0.7)
    plt.xlabel("threshold")
    plt.ylabel("beta")
    plt.title("DPO grid (demo sizes by train_examples)")
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(plots_dir, "grid_scatter.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="DPO ablation grid runner (lightweight scaffold).")
    parser.add_argument("--thresholds", type=str, required=True, help="CSV, e.g., 0.7,0.8,0.9")
    parser.add_argument("--betas", type=str, required=True, help="CSV, e.g., 0.3,0.5")
    parser.add_argument("--seeds", type=str, required=True, help="CSV, e.g., 1,2")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples per run (0 = all)")
    parser.add_argument("--provider", type=str, default=os.getenv("PROVIDER", "groq"), choices=["groq", "openai"])
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--train_jsonl", type=str, default="", help="Path to train.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="", help="Path to val.jsonl")
    parser.add_argument("--out_dir", type=str, default="experiments/dpo")
    parser.add_argument("--dry-run", action="store_true", help="Skip any heavy steps; just structure and metrics stubs")
    parser.add_argument("--force", action="store_true", help="Bypass guardrails")
    args = parser.parse_args()

    thresholds = parse_csv_floats(args.thresholds)
    betas = parse_csv_floats(args.betas)
    seeds = parse_csv_ints(args.seeds)

    # Guardrails
    total_runs = len(thresholds) * len(betas) * len(seeds)
    if not args.force and total_runs > 100:
        print(f"Refusing to run {total_runs} configs without --force (limit=100).", file=sys.stderr)
        sys.exit(2)

    # Provider sanity (no real API calls here; just check env if provider=groq)
    if args.provider == "groq" and not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not set; proceeding without provider calls.", file=sys.stderr)

    # Load data (for counts only; the script is a scaffold)
    if not args.train_jsonl or not args.val_jsonl:
        print("Error: --train_jsonl and --val_jsonl are required.", file=sys.stderr)
        sys.exit(2)
    train = read_jsonl(args.train_jsonl, max_items=args.max-samples if False else args.max_samples)
    val = read_jsonl(args.val_jsonl, max_items=args.max_samples)

    base_out = args.out_dir
    ensure_dir(base_out)

    results_for_plots: List[Tuple[float, float, int]] = []
    start_wall = time.time()
    for thr in thresholds:
        for beta in betas:
            for seed in seeds:
                run_ts = now_ts()
                run_dir = os.path.join(base_out, f"{run_ts}_{thr}_{beta}_{seed}")
                ensure_dir(run_dir)

                cfg = {
                    "threshold": thr,
                    "beta": beta,
                    "seed": seed,
                    "provider": args.provider,
                    "model": args.model,
                    "max_samples": args.max_samples,
                    "train_jsonl": args.train_jsonl,
                    "val_jsonl": args.val_jsonl,
                    "dry_run": args.dry_run,
                }
                save_json(os.path.join(run_dir, "config.json"), cfg)

                # Minimal placeholder "metrics": we record counts and timestamps.
                wall0 = time.time()
                # In a full implementation, call into training/eval here.
                wall1 = time.time()
                metrics = {
                    "train_examples": len(train),
                    "val_examples": len(val),
                    "elapsed_sec": round(wall1 - wall0, 3),
                    # Placeholders
                    "train_loss": None,
                    "eval_loss": None,
                }
                save_json(os.path.join(run_dir, "metrics.json"), metrics)
                with open(os.path.join(run_dir, "logs.txt"), "w", encoding="utf-8") as f:
                    f.write(f"[{run_ts}] thr={thr} beta={beta} seed={seed} "
                            f"train={len(train)} val={len(val)} elapsed={metrics['elapsed_sec']}s\n")

                results_for_plots.append((thr, beta, len(train)))

    # Aggregated plot
    plot_aggregates(base_out, results_for_plots)
    print(f"Completed {total_runs} runs in {round(time.time() - start_wall, 2)}s.")
    print(f"Outputs under: {base_out}")


if __name__ == "__main__":
    main()


