from typing import Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer

from .config import HybridConfig, DPOConfig


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
    dpo_cfg: DPOConfig = cfg.dpo

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name_or_path, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(cfg.model.base_model_name_or_path)

    dataset = load_dataset("json", data_files={
        "train": cfg.data.train_path,
        "validation": cfg.data.eval_path,
    })
    train_ds = dataset["train"].map(_prepare_pairs)
    eval_ds = dataset["validation"].map(_prepare_pairs)

    training_args = TrainingArguments(
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
        report_to=["none"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        beta=dpo_cfg.beta,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(dpo_cfg.output_dir)

