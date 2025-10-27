Param(
  [string]$Python = "python",
  [string]$Model = "",
  [string]$TrainJsonl = "",
  [string]$EvalJsonl = "",
  [string]$OutputDir = "",
  [double]$Epochs = 0,
  [int]$Bsz = 0,
  [double]$Lr = 0,
  [int]$Gas = 0,
  [int]$LogSteps = 0,
  [int]$SaveSteps = 0,
  [string]$DeepSpeed = ""
)

Write-Host "[train_hybrid_dpo.ps1] Starting training via $Python"

$env:MODEL_VAL = $Model
$env:TRAIN_JSONL_VAL = $TrainJsonl
$env:EVAL_JSONL_VAL = $EvalJsonl
$env:OUTPUT_DIR_VAL = $OutputDir
if ($Epochs -gt 0) { $env:EPOCHS_VAL = "$Epochs" } else { Remove-Item Env:EPOCHS_VAL -ErrorAction SilentlyContinue }
if ($Bsz -gt 0) { $env:BSZ_VAL = "$Bsz" } else { Remove-Item Env:BSZ_VAL -ErrorAction SilentlyContinue }
if ($Lr -gt 0) { $env:LR_VAL = "$Lr" } else { Remove-Item Env:LR_VAL -ErrorAction SilentlyContinue }
if ($Gas -gt 0) { $env:GAS_VAL = "$Gas" } else { Remove-Item Env:GAS_VAL -ErrorAction SilentlyContinue }
if ($LogSteps -gt 0) { $env:LOG_STEPS_VAL = "$LogSteps" } else { Remove-Item Env:LOG_STEPS_VAL -ErrorAction SilentlyContinue }
if ($SaveSteps -gt 0) { $env:SAVE_STEPS_VAL = "$SaveSteps" } else { Remove-Item Env:SAVE_STEPS_VAL -ErrorAction SilentlyContinue }
$env:DEEPSPEED_VAL = $DeepSpeed

$code = @'
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
'@

& $Python -c $code

Write-Host "[train_hybrid_dpo.ps1] Done"


