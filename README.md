Hybrid-KG-LLM-Project

A hybrid vision-textual reasoning model on knowledge graphs.

Extends SNS, GITA, and GraphWiz with novel KG-specific contributions.

References:
- SNS: https://github.com/ruili33/SNS
- GITA: https://github.com/WEIYanbin1999/GITA
- GraphWiz: https://github.com/Graph-Reasoner/Graph-Reasoning-LLM

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install PyTorch matching your CUDA: https://pytorch.org/get-started/locally/
3. pip install -r requirements.txt

## Quickstart (Hybrid Pipeline)

1) Prepare a tiny KG and build demo DPO pairs with rendered images:

```bash
python scripts/prepare_hybrid_dataset.py --triples_jsonl data/sample_triples.jsonl --out_dir data/hybrid --limit 50
```

2) (Optional) Rank neighbors via SimCSE (default: `princeton-nlp/sup-simcse-bert-base-uncased`, fallback to `sentence-transformers/all-MiniLM-L6-v2`) in your data prep: see `src/sns_ranker.py`.

3) Train Hybrid DPO on the demo data:

```python
from src.hybrid_dpo import train_hybrid_dpo

train_hybrid_dpo({})
```

Artifacts will be saved under `outputs/hybrid-dpo/`.

## Repository structure
- sns_module: SNS components (main.py, utils.py, call_api.py)
- gita_module: GITA scripts (finetune_lora_loop.sh, eval_loop.sh) and dataset handlers
- graphwiz_module: GraphWiz scripts (generate_all_train_datasets.sh, rft.sh), find_paths, and training mains
- src: hybrid code including kg_visualize.py, sns_ranker.py, prompting.py, graphwiz_integration.py, hybrid_dpo.py
- data: datasets (e.g., merged GVLQA + KG subsets)

## Licenses
Original licenses retained in third_party_licenses/

## Next steps
- Integrate KG data loaders and path finders
- Unify training loop with multi-modal DPO
- Add deepspeed configs and PyG wheels guidance
