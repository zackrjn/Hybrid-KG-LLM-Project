# Hybrid KG LLM Pipeline: Implementation and PrimeKG Application

## Objectives

- Build an end-to-end, reproducible hybrid pipeline (visual + textual + SNS) on demo data, then scale to PrimeKG (medicine).
- Produce training/eval scripts, metrics, ablations, and paper-ready artifacts.

## Portions & Methodology

- Data: Acquire/subset KGs; normalize triples and texts; SNS neighbor selection; render subgraphs; construct SFT and DPO pairs; governance (licenses, PII-free).
- Modeling: Base LLM (Mistral) and optional VLM (LLaVA-1.5); LoRA adapters; prompting templates; GraphWiz-style path diversity; negative sampling for link prediction.
- Training: Two-stage SFT→DPO. SFT on hybrid instructions, then DPO on chosen/rejected pairs; DeepSpeed ZeRO-3 optional; Windows/Linux launchers.
- Evaluation: Link prediction (MRR/Hits@10), multi-hop QA (Acc/F1), qualitative visual grounding; compare to baselines.
- Ablations: Similarity threshold/top-k, layout strategy, DPO β, SFT vs. no-SFT, visual on/off, SNS on/off.
- Medicine Application: PrimeKG subset (drug–disease–gene) tasks for link prediction and QA; adaptive visuals.
- Reproducibility: README quickstarts, configs, scripts, metrics JSONs, figures.

## Part A — Demo Pipeline (ready now)

1) Wire SNS selection into dataset prep

- Update `scripts/prepare_hybrid_dataset.py` to optionally:
- load entity texts (JSONL: `{id, text}`), compute SimCSE embeddings (default: `princeton-nlp/sup-simcse-bert-base-uncased`, fallback to `sentence-transformers/all-MiniLM-L6-v2`), select top-k neighbors (`--sns_top_k`, `--sns_threshold`)
- construct chosen/rejected pairs using neighbor quality; render subgraph images via `src/kg_visualize.py`
- Leverage `src/sns_ranker.py` and `src/config.py` defaults.

2) Training launchers

- Add `scripts/train_hybrid_dpo.sh` (Linux) and `scripts/train_hybrid_dpo.ps1` (Windows) to run minimal DPO with optional DeepSpeed (`gita_module/zero3.json`).
- Allow overriding paths/hyperparams via CLI/env.

3) Smoke test

- Generate demo data: `python scripts/prepare_hybrid_dataset.py --triples_jsonl data/sample_triples.jsonl --out_dir data/hybrid --limit 50`
- Run training: `python -c "from src.hybrid_dpo import train_hybrid_dpo; train_hybrid_dpo({})"`
- Expected: model saves to `outputs/hybrid-dpo/` and logs train/val loss.

## Part B — Evaluation Harness

- Implement `scripts/eval_link_prediction.py` (MRR, Hits@10) for demo triples.
- Implement `scripts/eval_multihop_qa.py` (Accuracy/F1) on simple QA pairs.
- Save metrics under `outputs/eval/` with JSON summaries.

## Part C — PrimeKG (medicine) Integration

1) Acquisition & Subset

- Fork `mims-harvard/PrimeKG` to `zackrjn/PrimeKG`; clone under `third_party/PrimeKG/` (ignored by Git).
- Scripts:
- `scripts/primekg_download.py` (pull nodes/edges CSVs)
- `scripts/primekg_subset.py` (focus on drug–disease–gene; 10k–50k nodes; export `triples.jsonl` and `entity_texts.jsonl`)
- Normalize fields to `{head, relation, tail}` and `{id, text}` (drug/disease/gene names + descriptions).

2) Hybrid dataset construction

- `scripts/prepare_primekg_hybrid.py` (build 10k–20k SFT+DPO examples):
- SNS-filtered neighbors (SimCSE default with MiniLM fallback), render adaptive layouts (force-directed for sparse, dot for pathway-like)
- Prompting via `src/prompting.py` with medical tasks (link prediction, multi-hop QA)
- 80/10/10 split; store under `data/primekg/hybrid/`

3) Training & Config

- Stage 1 SFT: 1–2 epochs, BSZ 4, LR 5e-6 on SFT prompts; output `outputs/primekg-sft/`.
- Stage 2 DPO: 2–3 epochs, BSZ 4, LR 5e-6 on chosen/rejected pairs; init from SFT; `gita_module/zero3.json` optional.
- Add config overrides (e.g., `configs/primekg_hybrid.yaml`) or CLI args for paths/hyperparams.

## Part D — Ablations & Baselines

- Grid: similarity threshold {0.7, 0.8, 0.9}, top-k {3,5,8}, layout {spring, dot, force}, DPO β {0.3, 0.5}, SFT {on, off}.
- Baselines: text-only, visual-only, SNS-off; external: GraphWiz-DPO (text), GPT-4V (where allowed).
- Scripts: `scripts/run_ablation_grid.sh` (Linux), `.ps1` (Windows) to schedule runs and collect JSON metrics.

## Part E — Paper Artifacts & Reproducibility

- Update `README.md` with PrimeKG quickstart; add dataset/model cards under `docs/`.
- Checkpoint registry under `outputs/` with metrics plots; add `scripts/plot_results.py`.
- Draft methodology figures: hybrid pipeline diagram, example prompts, ablation summaries.
- Add medical safety statement (non-clinical use), licenses and data sources table.

## Review Gates (checkpoints)

- Gate A (Demo): demo data + training completes; eval scripts run on demo; metrics JSON produced.
- Gate B (PrimeKG Subset): subset + texts exported; sample renders verified.
- Gate C (PrimeKG Training): SFT→DPO completes; link prediction MRR/Hits@10 improves over baselines by ≥5%.
- Gate D (Ablations): complete grid; identify best settings.
- Gate E (Artifacts): docs updated; figures generated.

## Risks & Mitigations

- PyGraphviz install (Windows): provide wheels/docs; fallback to `networkx` layouts.
- Memory on PrimeKG: subgraph sampling; gradient checkpointing; DeepSpeed ZeRO-3.
- LFS weights: keep out of repo; document optional downloads.
- Licensing & Safety: retain upstream LICENSEs; attribute PrimeKG; non-clinical-use disclaimer.

## Provider Fallback (Temporary: GPT-5 → Groq)

- Until org verification enables direct GPT-5 use in Cursor settings, all GPT-5–assigned experiment tasks (e.g., `dpo-grid`) will run via Groq using their OpenAI-compatible endpoint.
- Worktrees:
  - Keep existing: `agent/dpo-grid-gpt5-...` (reserved for when verification completes)
  - New temporary: `agent/dpo-grid-groq-...` (active for ablation work now)
- Keys:
  - Ensure `GROQ_API_KEY` is set locally and on Roar (for any code paths needing API access)
- Revert plan:
  - When org is verified, resume using the GPT-5 worktree and provider by switching the active agent window back to `agent/dpo-grid-gpt5-...`