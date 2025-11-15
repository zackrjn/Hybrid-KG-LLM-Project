import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path


def get_git_commit_short() -> str:
    """Get first 8 chars of current git commit, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def normalize_subset_params(params: Dict[str, Any]) -> str:
    """Normalize params dict to deterministic JSON string (sorted keys)."""
    # Remove None values and sort keys
    filtered = {k: v for k, v in params.items() if v is not None}
    return json.dumps(filtered, sort_keys=True, ensure_ascii=True)


def compute_subset_id(params: Dict[str, Any]) -> str:
    """Compute subset_id as first 12 hex chars of SHA256 hash."""
    normalized = normalize_subset_params(params)
    hash_obj = hashlib.sha256(normalized.encode('utf-8'))
    return hash_obj.hexdigest()[:12]


def get_cache_dir(base_cache_dir: str, subset_id: str) -> str:
    """Get cache directory path for subset_id."""
    return os.path.join(base_cache_dir, subset_id)


def cache_exists(cache_dir: str, required_files: List[str]) -> bool:
    """Check if cache exists with all required files."""
    if not os.path.exists(cache_dir):
        return False
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in required_files)


def write_run_manifest(
    cache_dir: str,
    subset_id: str,
    args: Dict[str, Any],
    counts: Dict[str, int],
    timings: Dict[str, float],
    paths: Dict[str, str]
) -> None:
    """Write run manifest JSON to cache directory."""
    manifest = {
        "subset_id": subset_id,
        "args": args,
        "counts": counts,
        "timings": timings,
        "paths": paths,
        "created_at": time.time()
    }
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def write_flat_manifest(
    experiments_dir: str,
    subset_id: str,
    out_dir: str,
    train_count: int,
    val_count: int,
    args: Dict[str, Any],
) -> None:
    """Write flat run manifest to experiments/runs/run_manifest_<subset_id>.json"""
    os.makedirs(os.path.join(experiments_dir, "runs"), exist_ok=True)
    
    # Convert end_idx=None to -1 for "all"
    args_copy = args.copy()
    if args_copy.get("end_idx") is None:
        args_copy["end_idx"] = -1
    
    # Format timestamp as human-readable string
    created_at_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    flat_manifest = {
        "subset_id_temp": subset_id,
        "out_dir": out_dir,
        "train_count": train_count,
        "val_count": val_count,
        "args": args_copy,
        "created_at": created_at_str,
    }
    
    manifest_path = os.path.join(experiments_dir, "runs", f"run_manifest_{subset_id}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(flat_manifest, f, indent=2, ensure_ascii=False)


