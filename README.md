## Hybrid-KG-LLM-Project

Hybrid multi-hop reasoning over knowledge graphs with LLM alignment and optional vision grounding. This repository combines ideas and utilities from SNS, GITA, and GraphWiz, and adds a tailored hybrid DPO training and data tooling for KG reasoning.

References:
- [SNS](https://github.com/ruili33/SNS)
- [GITA](https://github.com/WEIYanbin1999/GITA)
- [GraphWiz](https://github.com/Graph-Reasoner/Graph-Reasoning-LLM)

## Overview

- **Goal**: Train and evaluate an LLM (with optional visual cues) to perform multi-hop reasoning over a KG using a hybrid dataset and preference-alignment (DPO).
- **Data tooling**: Build synthetic and subset datasets from PRIMEKG, render small graph neighborhoods, and create DPO pairs.
- **Training**: A simple entrypoint for DPO fine-tuning on hybrid KG reasoning examples.
- **Evaluation**: Scripts for link prediction and multi-hop QA on generated/test splits.

## High-level architecture

1. Data is prepared from raw KG triples and entity texts using `scripts/prepare_hybrid_dataset.py`.
2. Optional neighbor ranking with SimCSE (see `src/sns_ranker.py`) to focus candidate paths.
3. DPO training with `src/hybrid_dpo.py` aligns the model on positive vs negative reasoning chains.
4. Evaluation (`scripts/eval_link_prediction.py`, `scripts/eval_multihop_qa.py`) measures performance.
5. Visualization (`src/kg_visualize.py`) renders small subgraphs for qualitative analysis.

## Directory guide

- `src/`
  - `config.py`: Centralized configuration helper and defaults for paths, model names, and training knobs.
  - `hybrid_dpo.py`: DPO training entrypoints/utilities for hybrid KG reasoning datasets.
  - `kg_data.py`: Lightweight KG data utilities: loading triples/entity texts, sampling neighborhoods, batching.
  - `prompting.py`: Prompt templates and formatting utilities for reasoning over KG facts.
  - `graphwiz_integration.py`: Hooks to generate or verify reasoning paths with GraphWiz-style methods when desired.
  - `sns_ranker.py`: SimCSE-based neighbor ranking to prioritize graph expansions.
  - `kg_visualize.py`: Small utilities to draw subgraphs used in datasets/evals.

- `scripts/`
  - `prepare_hybrid_dataset.py`: Build hybrid datasets (JSONL + rendered images) from triples; can sub-sample for demos.
  - `primekg_download.py`: Download PRIMEKG data (raw); large files are not pushed to the repo.
  - `primekg_subset.py`: Create smaller subsets from PRIMEKG for quick experiments.
  - `train_hybrid_dpo.sh` / `train_hybrid_dpo.ps1`: Convenience launchers for training.
  - `eval_link_prediction.py`: Evaluate link prediction on held-out edges/splits.
  - `eval_multihop_qa.py`: Evaluate multi-hop QA style tasks generated from subgraphs.

- `data/`
  - Tracked example datasets live here: `hybrid/`, `hybrid_simcse/`, `hybrid_simcse_default/`, plus small samples like `entity_texts.jsonl` and `sample_triples.jsonl`.
  - Note: Large raw datasets (`data/primekg_raw/`) are intentionally ignored in Git. Use the download and subset scripts below to reproduce.

- `graphwiz_module/`, `gita_module/`, `sns_module/`, `third_party/`
  - Upstream references and scripts. See their own READMEs for deeper details. This repo only relies on a small subset of utilities to keep the pipeline pragmatic.

- `third_party_licenses/`: Collected third-party license references.

## Setup

1. Create and activate Python 3.10+.
2. Install PyTorch matching your CUDA (`https://pytorch.org/get-started/locally/`).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional (Windows line-endings):

```bash
git config core.autocrlf true
```

## Data: download, subset, and prepare

Download PRIMEKG (raw, large; will NOT be committed):

```bash
python scripts/primekg_download.py --out_dir data/primekg_raw
```

Create a smaller subset for quick experiments:

```bash
python scripts/primekg_subset.py --in_dir data/primekg_raw --out_dir data --max_edges 50000
```

Prepare a demo hybrid dataset (small sample with images):

```bash
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid \
  --limit 50
```

SimCSE-assisted variants are controlled via flags (see `--help`) and implemented in `src/sns_ranker.py`.

## Training (Hybrid DPO)

Python API:

```python
from src.hybrid_dpo import train_hybrid_dpo

train_hybrid_dpo({
    "dataset_dir": "data/hybrid",
    "output_dir": "outputs/hybrid-dpo",
    # add model/training args as needed
})
```

CLI launcher:

```bash
# Linux/macOS
bash scripts/train_hybrid_dpo.sh

# Windows PowerShell
pwsh scripts/train_hybrid_dpo.ps1
```

Artifacts (checkpoints, logs) are written under `outputs/`.

## Evaluation

Link prediction:

```bash
python scripts/eval_link_prediction.py --dataset_dir data/hybrid --ckpt outputs/hybrid-dpo
```

Multi-hop QA:

```bash
python scripts/eval_multihop_qa.py --dataset_dir data/hybrid --ckpt outputs/hybrid-dpo
```

## Visualization

Render small subgraphs for inspection:

```bash
python -c "from src.kg_visualize import render_example; render_example('data/hybrid/train.jsonl', 0)"
```

## Configuration

- Start with `src/config.py` for default knobs and path conventions.
- Prompts and formatting live in `src/prompting.py`.
- Neighbor ranking options in `src/sns_ranker.py`.

## For readers (and GPT-5) exploring the code

Recommended order to read:
- `src/config.py` → how configuration is passed.
- `src/kg_data.py` → how triples and entity texts are loaded and batched.
- `src/prompting.py` → how inputs are formatted for the model.
- `src/hybrid_dpo.py` → main training and DPO setup.
- `src/graphwiz_integration.py` → optional path-finding hooks.
- `src/sns_ranker.py` → SimCSE ranking.
- `src/kg_visualize.py` → diagnostics and plots.

## Large files and Git LFS

- `data/primekg_raw/` is ignored to avoid exceeding GitHub’s 100MB file limit.
- If you need to track large artifacts, consider Git LFS (`https://git-lfs.github.com`).
- Otherwise, use the provided download/subset scripts to reproduce datasets locally.

## Licenses and acknowledgements

- Original third-party licenses are retained in `third_party_licenses/`.
- This repo builds upon SNS, GITA, and GraphWiz; please cite and follow their licenses.
