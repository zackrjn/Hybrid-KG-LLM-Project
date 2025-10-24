#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for Hybrid DPO training.
# Env overrides:
#   PYTHON: python executable (default: python)
#   MODEL: base model path/name (default: from src/config.py)
#   TRAIN_JSONL: train data path
#   EVAL_JSONL: eval data path
#   OUTPUT_DIR: output directory
#   EPOCHS, BSZ, LR, GAS, LOG_STEPS, SAVE_STEPS, DEEPSPEED

PYTHON_BIN=${PYTHON:-python}

echo "[train_hybrid_dpo.sh] Starting training via ${PYTHON_BIN}"
MODEL_VAL=${MODEL:-}
TRAIN_JSONL_VAL=${TRAIN_JSONL:-}
EVAL_JSONL_VAL=${EVAL_JSONL:-}
OUTPUT_DIR_VAL=${OUTPUT_DIR:-}
EPOCHS_VAL=${EPOCHS:-}
BSZ_VAL=${BSZ:-}
LR_VAL=${LR:-}
GAS_VAL=${GAS:-}
LOG_STEPS_VAL=${LOG_STEPS:-}
SAVE_STEPS_VAL=${SAVE_STEPS:-}
DEEPSPEED_VAL=${DEEPSPEED:-}

${PYTHON_BIN} - <<'PYCODE'
import os
from src.hybrid_dpo import train_hybrid_dpo

overrides = {}
model = os.environ.get("MODEL_VAL")
train = os.environ.get("TRAIN_JSONL_VAL")
val = os.environ.get("EVAL_JSONL_VAL")
outd = os.environ.get("OUTPUT_DIR_VAL")
epochs = os.environ.get("EPOCHS_VAL")
bsz = os.environ.get("BSZ_VAL")
lr = os.environ.get("LR_VAL")
gas = os.environ.get("GAS_VAL")
log_steps = os.environ.get("LOG_STEPS_VAL")
save_steps = os.environ.get("SAVE_STEPS_VAL")
ds = os.environ.get("DEEPSPEED_VAL")

if model:
    overrides.setdefault("model", {})["base_model_name_or_path"] = model
if train or val:
    overrides.setdefault("data", {})
    if train:
        overrides["data"]["train_path"] = train
    if val:
        overrides["data"]["eval_path"] = val
if outd or epochs or bsz or lr or gas or log_steps or save_steps or ds:
    overrides.setdefault("dpo", {})
    if outd:
        overrides["dpo"]["output_dir"] = outd
    if epochs:
        overrides["dpo"]["num_train_epochs"] = float(epochs)
    if bsz:
        overrides["dpo"]["per_device_train_batch_size"] = int(bsz)
        overrides["dpo"]["per_device_eval_batch_size"] = int(bsz)
    if lr:
        overrides["dpo"]["learning_rate"] = float(lr)
    if gas:
        overrides["dpo"]["gradient_accumulation_steps"] = int(gas)
    if log_steps:
        overrides["dpo"]["logging_steps"] = int(log_steps)
    if save_steps:
        overrides["dpo"]["save_steps"] = int(save_steps)
    if ds:
        overrides["dpo"]["deepspeed"] = ds

train_hybrid_dpo(overrides)
PYCODE

echo "[train_hybrid_dpo.sh] Done"


