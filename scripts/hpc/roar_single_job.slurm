#!/bin/bash
#SBATCH -J hybrid_prep
#SBATCH -p open
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -t 02:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

module load anaconda
conda activate hybridkg
export MPLBACKEND=Agg

python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl /scratch/$USER/primekg/triples.jsonl \
  --out_dir /scratch/$USER/primekg/hybrid_minilm \
  --limit 5000 \
  --use_sns \
  --entity_texts_jsonl /scratch/$USER/primekg/entity_texts.jsonl \
  --sns_model sentence-transformers/all-MiniLM-L6-v2 \
  --sns_device cuda \
  --sns_batch_size 1024 \
  --sns_top_k 3 \
  --sns_threshold 0.0 \
  --emb_cache_npz /scratch/$USER/primekg/emb_cache_minilm.npz \
  --num_workers 24 \
  --max_images 5000


