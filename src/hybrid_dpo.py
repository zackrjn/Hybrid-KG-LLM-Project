from typing import Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import DPOTrainer, DPOConfig as TRLDPOConfig

from .config import HybridConfig, DPOConfig as LocalDPOConfig


def _prepare_pairs(example: Dict[str, Any]) -> Dict[str, Any]:
    # expects fields: prompt, chosen, rejected
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def _apply_overrides(cfg: HybridConfig, overrides: Dict[str, Any]) -> HybridConfig:
    # Supports nested dicts matching dataclass structure (model, data, sns, visual, dpo, paths)
    for section, values in overrides.items():
        if not hasattr(cfg, section):
            continue
        section_obj = getattr(cfg, section)
        if isinstance(values, dict):
            for k, v in values.items():
                if hasattr(section_obj, k):
                    setattr(section_obj, k, v)
        else:
            setattr(cfg, section, values)
    return cfg


def train_hybrid_dpo(config: Dict[str, Any]) -> None:
    cfg = HybridConfig()
    if config:
        cfg = _apply_overrides(cfg, config)
    dpo_cfg: LocalDPOConfig = cfg.dpo

    # Ensure reproducibility across runs when seed varies in ablations
    try:
        set_seed(cfg.data.seed)
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name_or_path, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_name_or_path,
        torch_dtype="auto",
    )
    if getattr(cfg.model, "gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if getattr(tokenizer, "pad_token_id", None) is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("json", data_files={
        "train": cfg.data.train_path,
        "validation": cfg.data.eval_path,
    })
    train_ds = dataset["train"].map(_prepare_pairs)
    eval_ds = dataset["validation"].map(_prepare_pairs)

    training_args = TRLDPOConfig(
        output_dir=dpo_cfg.output_dir,
        per_device_train_batch_size=dpo_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=dpo_cfg.per_device_eval_batch_size,
        num_train_epochs=dpo_cfg.num_train_epochs,
        learning_rate=dpo_cfg.learning_rate,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        logging_steps=dpo_cfg.logging_steps,
        save_steps=dpo_cfg.save_steps,
        warmup_ratio=dpo_cfg.warmup_ratio,
        lr_scheduler_type=dpo_cfg.lr_scheduler_type,
        deepspeed=dpo_cfg.deepspeed,
        beta=dpo_cfg.beta,
        max_length=128,
        max_prompt_length=64,
        reference_free=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(dpo_cfg.output_dir)
    # Save trainer state so downstream tools can extract metrics (trainer_state.json)
    try:
        trainer.save_state()
    except Exception:
        pass

