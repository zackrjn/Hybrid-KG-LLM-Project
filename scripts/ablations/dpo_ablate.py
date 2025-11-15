#!/usr/bin/env python3
"""
DPO Ablation Runner

Grid search over DPO hyperparameters (thresholds, beta, seeds) with structured output.

Usage Examples:
    # Tiny dry-run target (10 samples, single config)
    python scripts/ablations/dpo_ablate.py --max-samples 10 --dry-run --thresholds 0.7 --betas 0.5 --seeds 42

    # Dry-run with sanity limits
    python scripts/ablations/dpo_ablate.py --max-samples 50 --dry-run --thresholds 0.7,0.8 --betas 0.5

    # Full ablation sweep
    python scripts/ablations/dpo_ablate.py --thresholds 0.7,0.8,0.9 --betas 0.1,0.3,0.5,0.7 --seeds 42,43,44

    # CI-friendly single config
    PROVIDER=groq GROQ_API_KEY=xxx python scripts/ablations/dpo_ablate.py --seeds 42 --betas 0.5 --thresholds 0.8

    # Plan-mode examples
    # Tiny smoke on PrimeKG smoke pairs (aliases supported)
    python scripts/ablations/dpo_ablate.py \
      --thresholds 0.9 --betas 0.5 --seeds 1 --max-samples 200 \
      --provider groq --model <groq-model> \
      --train_jsonl data/primekg/hybrid_minilm_smoke/train.jsonl \
      --val_jsonl   data/primekg/hybrid_minilm_smoke/val.jsonl \
      --out_dir     experiments/dpo/smoke

    # Larger Roar template (<=100 configs recommended; use --force to exceed)
    python scripts/ablations/dpo_ablate.py \
      --thresholds 0.7,0.8,0.9 --betas 0.1,0.3,0.5,0.7 --seeds 1,2,3 \
      --out_dir experiments/dpo/roar

    Resource Notes:
    - Each run spawns full training; use --max-samples for testing
    - Parallel runs possible via separate processes (not implemented)
    - Outputs ~1-5MB per run (config + metrics JSON)
    - Plots: ~100-500KB total
    - Sanity limits: <=100 runs total and <=1000 samples per run (override with --force)
"""
import argparse
import json
import itertools
import os
import sys
import re
import tempfile
import shutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set non-interactive backend for matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hybrid_dpo import train_hybrid_dpo
from src.config import HybridConfig


# Sanity limits
MAX_SAMPLES_PER_RUN = 1000
MAX_TOTAL_RUNS = 100


def parse_list_or_range(value: str, parse_type=float) -> List:
    """Parse comma-separated list or range specification."""
    if ',' in value:
        return [parse_type(x.strip()) for x in value.split(',')]
    return [parse_type(value)]


def slice_jsonl(input_path: str, output_path: str, max_samples: int) -> int:
    """Slice JSONL file to first max_samples lines."""
    count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            count += 1
            if count >= max_samples:
                break
    return count




def run_single_ablation(
    threshold: float,
    beta: float,
    seed: int,
    output_dir: Path,
    train_jsonl: str,
    eval_jsonl: str,
    max_samples: Optional[int],
    dry_run: bool,
    provider: str,
    model: Optional[str],
    base_config: Dict[str, Any],
    force: bool
) -> Dict[str, Any]:
    """Run a single DPO training with given hyperparameters."""
    run_id = f"{int(datetime.now().timestamp())}_{threshold}_{beta}_{seed}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare temporary JSONL files if slicing needed
    temp_train = None
    temp_eval = None
    actual_train_path = train_jsonl
    actual_eval_path = eval_jsonl
    
    if max_samples:
        # Sanity check
        if max_samples > MAX_SAMPLES_PER_RUN and not force:
            print(f"Warning: max_samples {max_samples} exceeds limit {MAX_SAMPLES_PER_RUN}, capping (use --force to override)")
            max_samples = MAX_SAMPLES_PER_RUN
        
        # Create temporary sliced files
        temp_dir = Path(tempfile.mkdtemp())
        temp_train = temp_dir / "train.jsonl"
        temp_eval = temp_dir / "eval.jsonl"
        
        train_count = slice_jsonl(train_jsonl, str(temp_train), max_samples)
        eval_count = slice_jsonl(eval_jsonl, str(temp_eval), max_samples)
        actual_train_path = str(temp_train)
        actual_eval_path = str(temp_eval)
    else:
        # Count lines in original files
        with open(train_jsonl, 'r') as f:
            train_count = sum(1 for line in f if line.strip())
        with open(eval_jsonl, 'r') as f:
            eval_count = sum(1 for line in f if line.strip())
    
    # Build config overrides
    config_overrides = base_config.copy()
    config_overrides.setdefault('sns', {})['similarity_threshold'] = threshold
    config_overrides.setdefault('dpo', {})['beta'] = beta
    config_overrides.setdefault('data', {})['seed'] = seed
    config_overrides.setdefault('data', {})['train_path'] = actual_train_path
    config_overrides.setdefault('data', {})['eval_path'] = actual_eval_path
    config_overrides.setdefault('dpo', {})['output_dir'] = str(run_dir / "checkpoint")
    
    if model:
        config_overrides.setdefault('model', {})['base_model_name_or_path'] = model
    
    if dry_run:
        # Minimal training for dry-run
        config_overrides.setdefault('dpo', {})['num_train_epochs'] = 1
        config_overrides.setdefault('dpo', {})['per_device_train_batch_size'] = 2
        config_overrides.setdefault('dpo', {})['per_device_eval_batch_size'] = 2
        config_overrides.setdefault('dpo', {})['gradient_accumulation_steps'] = 1
        config_overrides.setdefault('dpo', {})['logging_steps'] = 5
        config_overrides.setdefault('dpo', {})['save_steps'] = 1000  # Don't save during dry-run
    
    # Write config.json
    config_dict = {
        'threshold': threshold,
        'beta': beta,
        'seed': seed,
        'provider': provider,
        'model': model or base_config.get('model', {}).get('base_model_name_or_path', 'default'),
        'train_path': train_jsonl,
        'eval_path': eval_jsonl,
        'train_count': train_count,
        'eval_count': eval_count,
        'max_samples': max_samples,
        'dry_run': dry_run,
        'force': force,
        'config_overrides': config_overrides,
        'created_at': datetime.now().isoformat()
    }
    
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Run training with logs redirected
    log_file = run_dir / 'logs.txt'
    metrics = {}
    
    # Print summary to console
    print(f"  Starting training: threshold={threshold}, beta={beta}, seed={seed}")
    print(f"  Provider: {provider}, Model: {config_dict['model']}")
    print(f"  Train samples: {train_count}, Eval samples: {eval_count}")
    
    try:
        # Write detailed info to log file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"DPO Ablation Run\n")
            f.write(f"Threshold: {threshold}, Beta: {beta}, Seed: {seed}\n")
            f.write(f"Provider: {provider}, Model: {config_dict['model']}\n")
            f.write(f"Train samples: {train_count}, Eval samples: {eval_count}\n")
            f.write(f"{'='*60}\n\n")
        
        # Append training output to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = f
                sys.stderr = f
                train_hybrid_dpo(config_overrides)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\nTraining completed successfully\n")
    
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"  ERROR: {error_msg}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nERROR: {error_msg}\n")
            f.write(traceback.format_exc())
        metrics = {'error': error_msg, 'train_loss': None, 'eval_loss': None}
    finally:
        # Clean up temporary files
        if temp_train and temp_train.exists():
            shutil.rmtree(temp_train.parent, ignore_errors=True)
    
    # Extract metrics from logs or trainer state
    checkpoint_dir = run_dir / "checkpoint"
    if 'error' not in metrics:
        metrics = extract_metrics(log_file, checkpoint_dir)
    
    # Write metrics.json
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return {
        'run_id': run_id,
        'threshold': threshold,
        'beta': beta,
        'seed': seed,
        'metrics': metrics,
        'run_dir': str(run_dir)
    }


def extract_metrics(log_file: Path, checkpoint_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Extract metrics from training logs and trainer state."""
    metrics = {
        'train_loss': None,
        'eval_loss': None,
        'final_epoch': None,
        'total_steps': None
    }
    
    # Try to load trainer state first (most reliable)
    if checkpoint_dir and checkpoint_dir.exists():
        state_file = checkpoint_dir / 'trainer_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                log_history = state.get('log_history', [])
                if log_history:
                    # Get final training metrics
                    train_entries = [e for e in log_history if 'loss' in e and 'eval_loss' not in e]
                    eval_entries = [e for e in log_history if 'eval_loss' in e]
                    
                    if train_entries:
                        final_train = train_entries[-1]
                        metrics['train_loss'] = final_train.get('loss')
                        metrics['train_losses'] = [e.get('loss') for e in train_entries if 'loss' in e]
                    
                    if eval_entries:
                        final_eval = eval_entries[-1]
                        metrics['eval_loss'] = final_eval.get('eval_loss')
                        metrics['eval_losses'] = [e.get('eval_loss') for e in eval_entries if 'eval_loss' in e]
                    
                    # Get epoch/step info
                    if log_history:
                        final = log_history[-1]
                        metrics['final_epoch'] = final.get('epoch')
                        metrics['total_steps'] = final.get('step')
                    
                    return metrics
            except Exception as e:
                pass  # Fall back to log parsing
    
    # Fall back to log parsing
    if not log_file.exists():
        return metrics
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Look for loss patterns in logs
        # Format varies by trainer, but typically has "loss: X.XXX" or similar
        train_losses = []
        eval_losses = []
        
        for line in lines:
            # Try to find loss values
            if 'loss' in line.lower():
                # Try to extract float after loss
                loss_match = re.search(r'loss[:\s=]+([\d.]+)', line, re.IGNORECASE)
                if loss_match and 'eval' not in line.lower():
                    try:
                        train_losses.append(float(loss_match.group(1)))
                    except:
                        pass
            
            if 'eval_loss' in line.lower() or 'validation_loss' in line.lower():
                loss_match = re.search(r'loss[:\s=]+([\d.]+)', line, re.IGNORECASE)
                if loss_match:
                    try:
                        eval_losses.append(float(loss_match.group(1)))
                    except:
                        pass
        
        if train_losses:
            metrics['train_loss'] = train_losses[-1]  # Final loss
            metrics['train_losses'] = train_losses  # All losses
        
        if eval_losses:
            metrics['eval_loss'] = eval_losses[-1]
            metrics['eval_losses'] = eval_losses
        
        # If no losses found, set defaults for dry-run
        if not train_losses and not eval_losses:
            metrics['train_loss'] = 0.0  # Placeholder
            metrics['note'] = 'Metrics extracted from logs (may need trainer state for accurate values)'
    
    except Exception as e:
        metrics['extraction_error'] = str(e)
    
    return metrics


def plot_results(results: List[Dict[str, Any]], output_dir: Path, dry_run: bool):
    """Generate plots from ablation results."""
    if dry_run and len(results) < 2:
        print("Skipping plots for dry-run with < 2 runs")
        return
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    data = []
    for r in results:
        if 'error' not in r['metrics']:
            data.append({
                'threshold': r['threshold'],
                'beta': r['beta'],
                'seed': r['seed'],
                'train_loss': r['metrics'].get('train_loss'),
                'eval_loss': r['metrics'].get('eval_loss')
            })
    
    if not data:
        print("No valid data for plotting")
        return
    
    # Plot 1: Beta vs metric (for each threshold)
    thresholds = sorted(set(d['threshold'] for d in data))
    betas = sorted(set(d['beta'] for d in data))
    
    if len(betas) > 1:
        num_plots = max(1, len(thresholds))
        fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 4))
        if num_plots == 1:
            axes = [axes]
        
        for idx, thresh in enumerate(thresholds):
            ax = axes[idx]
            thresh_data = [d for d in data if d['threshold'] == thresh]
            
            # Group by beta and average across seeds
            beta_metrics = {}
            for d in thresh_data:
                beta = d['beta']
                if beta not in beta_metrics:
                    beta_metrics[beta] = []
                if d['train_loss'] is not None:
                    beta_metrics[beta].append(d['train_loss'])
            
            betas_sorted = sorted(beta_metrics.keys())
            avg_losses = [np.mean(beta_metrics[b]) for b in betas_sorted]
            
            ax.plot(betas_sorted, avg_losses, 'o-', label='Train Loss')
            ax.set_xlabel('Beta')
            ax.set_ylabel('Train Loss')
            ax.set_title(f'Threshold = {thresh}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'beta_vs_metric.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Threshold vs metric (for each beta)
    if len(thresholds) > 1:
        num_plots = max(1, len(betas))
        fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 4))
        if num_plots == 1:
            axes = [axes]
        
        for idx, beta in enumerate(betas):
            ax = axes[idx]
            beta_data = [d for d in data if d['beta'] == beta]
            
            # Group by threshold and average across seeds
            thresh_metrics = {}
            for d in beta_data:
                thresh = d['threshold']
                if thresh not in thresh_metrics:
                    thresh_metrics[thresh] = []
                if d['train_loss'] is not None:
                    thresh_metrics[thresh].append(d['train_loss'])
            
            thresh_sorted = sorted(thresh_metrics.keys())
            avg_losses = [np.mean(thresh_metrics[t]) for t in thresh_sorted]
            
            ax.plot(thresh_sorted, avg_losses, 's-', label='Train Loss')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Train Loss')
            ax.set_title(f'Beta = {beta}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'threshold_vs_metric.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Heatmap
    if len(thresholds) > 1 and len(betas) > 1:
        # Create heatmap data (average across seeds)
        heatmap_data = np.zeros((len(thresholds), len(betas)))
        
        for i, thresh in enumerate(sorted(thresholds)):
            for j, beta in enumerate(sorted(betas)):
                matching = [d['train_loss'] for d in data 
                           if d['threshold'] == thresh and d['beta'] == beta 
                           and d['train_loss'] is not None]
                if matching:
                    heatmap_data[i, j] = np.mean(matching)
                else:
                    heatmap_data[i, j] = np.nan
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
        
        ax.set_xticks(np.arange(len(betas)))
        ax.set_yticks(np.arange(len(thresholds)))
        ax.set_xticklabels([f'{b:.2f}' for b in sorted(betas)])
        ax.set_yticklabels([f'{t:.2f}' for t in sorted(thresholds)])
        
        ax.set_xlabel('Beta')
        ax.set_ylabel('Threshold')
        ax.set_title('Train Loss Heatmap')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(plots_dir / 'heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='DPO ablation runner: grid search over thresholds, beta, seeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Hyperparameters
    parser.add_argument('--thresholds', type=str, default='0.7,0.8,0.9',
                       help='Comma-separated thresholds or single value (default: 0.7,0.8,0.9)')
    parser.add_argument('--betas', type=str, default='0.1,0.3,0.5,0.7',
                       help='Comma-separated beta values or single value (default: 0.1,0.3,0.5,0.7)')
    parser.add_argument('--seeds', type=str, default='42,43,44',
                       help='Comma-separated seeds or single value (default: 42,43,44)')
    
    # Data
    parser.add_argument('--train-jsonl', '--train_jsonl', dest='train_jsonl', type=str, default=None,
                       help='Path to train JSONL (default: from config)')
    parser.add_argument('--eval-jsonl', '--val_jsonl', dest='eval_jsonl', type=str, default=None,
                       help='Path to eval JSONL (default: from config)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Cap samples per dataset (for dry-run/testing, max 1000)')
    
    # Output
    parser.add_argument('--output-dir', '--out_dir', dest='output_dir', type=str, default='experiments/dpo',
                       help='Base output directory (default: experiments/dpo)')
    
    # Model/Provider
    parser.add_argument('--provider', type=str, default=None, choices=['groq', 'openai'],
                       help='Provider (groq|openai). Default: groq (via PROVIDER env)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name/path (default: from MODEL env or config)')
    
    # Mode
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry-run mode: minimal training, skip plots if < 2 runs')
    parser.add_argument('--force', action='store_true',
                       help='Override safety limits on number of runs and samples per run')
    
    args = parser.parse_args()
    
    # Parse lists
    thresholds = parse_list_or_range(args.thresholds, float)
    betas = parse_list_or_range(args.betas, float)
    seeds = parse_list_or_range(args.seeds, int)
    
    # Provider/model from env or args
    provider = args.provider or os.environ.get('PROVIDER', 'groq')
    model = args.model or os.environ.get('MODEL', None)
    
    if provider == 'groq' and 'GROQ_API_KEY' not in os.environ:
        print("Warning: GROQ_API_KEY not set, but provider is groq")
    
    # Sanity checks
    total_runs = len(thresholds) * len(betas) * len(seeds)
    if total_runs > MAX_TOTAL_RUNS and not args.force:
        print(f"Error: Total runs {total_runs} exceeds limit {MAX_TOTAL_RUNS} (use --force to override)")
        sys.exit(1)
    elif total_runs > MAX_TOTAL_RUNS and args.force:
        print(f"Warning: Overriding total runs limit ({total_runs} > {MAX_TOTAL_RUNS}) due to --force")
    
    if args.max_samples and args.max_samples > MAX_SAMPLES_PER_RUN and not args.force:
        print(f"Error: max_samples {args.max_samples} exceeds limit {MAX_SAMPLES_PER_RUN} (use --force to override)")
        sys.exit(1)
    elif args.max_samples and args.max_samples > MAX_SAMPLES_PER_RUN and args.force:
        print(f"Warning: Overriding max_samples limit ({args.max_samples} > {MAX_SAMPLES_PER_RUN}) due to --force")
    
    # Get data paths
    base_config = HybridConfig()
    train_jsonl = args.train_jsonl or base_config.data.train_path
    eval_jsonl = args.eval_jsonl or base_config.data.eval_path
    
    if not Path(train_jsonl).exists():
        print(f"Error: Train JSONL not found: {train_jsonl}")
        sys.exit(1)
    if not Path(eval_jsonl).exists():
        print(f"Error: Eval JSONL not found: {eval_jsonl}")
        sys.exit(1)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base config dict
    base_config_dict = {}
    if model:
        base_config_dict['model'] = {'base_model_name_or_path': model}
    
    print(f"Starting DPO ablation:")
    print(f"  Thresholds: {thresholds}")
    print(f"  Betas: {betas}")
    print(f"  Seeds: {seeds}")
    print(f"  Total runs: {total_runs}")
    print(f"  Provider: {provider}")
    print(f"  Model: {model or 'default'}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print(f"  Force: {args.force}")
    print()
    
    # Run grid search
    results = []
    for threshold, beta, seed in itertools.product(thresholds, betas, seeds):
        print(f"\n{'='*60}")
        print(f"Run: threshold={threshold}, beta={beta}, seed={seed}")
        print(f"{'='*60}")
        
        result = run_single_ablation(
            threshold=threshold,
            beta=beta,
            seed=seed,
            output_dir=output_dir,
            train_jsonl=train_jsonl,
            eval_jsonl=eval_jsonl,
            max_samples=args.max_samples,
            dry_run=args.dry_run,
            provider=provider,
            model=model,
            base_config=base_config_dict,
            force=args.force
        )
        results.append(result)
    
    # Generate plots
    if not args.dry_run or len(results) >= 2:
        print("\nGenerating plots...")
        plot_results(results, output_dir, args.dry_run)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Ablation complete: {len(results)} runs")
    print(f"Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

