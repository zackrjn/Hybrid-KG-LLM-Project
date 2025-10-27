from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    base_model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2"
    vision_model_name_or_path: Optional[str] = None  # e.g., "liuhaotian/llava-v1.5-7b"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_lora: bool = True
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    train_path: str = "data/hybrid/train.jsonl"
    eval_path: str = "data/hybrid/val.jsonl"
    image_root: Optional[str] = "data/hybrid/images"
    triples_path: Optional[str] = None
    format: str = "jsonl"  # jsonl|tsv
    text_field: str = "text"
    relation_field: str = "relation"
    head_field: str = "head"
    tail_field: str = "tail"
    max_prompt_length: int = 2048
    seed: int = 42


@dataclass
class SNSConfig:
    simcse_model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased"
    top_k_neighbors: int = 5
    similarity_threshold: float = 0.0


@dataclass
class VisualConfig:
    enable_visuals: bool = True
    layout: str = "spring"  # spring|dot|neato|fdp
    dpi: int = 200
    layout_aug: Optional[str] = None  # e.g., "random", "force_directed"


@dataclass
class DPOConfig:
    beta: float = 0.5
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-6
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    save_steps: int = 500
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    output_dir: str = "outputs/hybrid-dpo"
    deepspeed: Optional[str] = None  # path to ds config json


@dataclass
class PathsConfig:
    graphwiz_repo_path: Optional[str] = None
    sns_repo_path: Optional[str] = None
    gita_repo_path: Optional[str] = None


@dataclass
class HybridConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sns: SNSConfig = field(default_factory=SNSConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


