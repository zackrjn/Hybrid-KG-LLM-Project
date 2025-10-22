#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for Hybrid DPO training.
# Env overrides:
#   PYTHON: python executable (default: python)

PYTHON_BIN=${PYTHON:-python}

echo "[train_hybrid_dpo.sh] Starting training via ${PYTHON_BIN}"
${PYTHON_BIN} - <<'PYCODE'
from src.hybrid_dpo import train_hybrid_dpo

# Uses default HybridConfig; edit src/config.py or extend train_hybrid_dpo to pass overrides
train_hybrid_dpo({})
PYCODE

echo "[train_hybrid_dpo.sh] Done"


