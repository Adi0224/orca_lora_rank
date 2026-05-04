from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class ExperimentConfig:
    seed: int = 0
    method: str = "lora_only"  # lora_only | orca_otdd
    model_name: str = "EleutherAI/pythia-70m"
    max_length: int = 256
    lora_r: int = 4
    lora_alpha: int = 8  # 2 * lora_r

    # Data — same splits for both methods (deterministic via data_seed)
    hard_train_samples: int = 200
    val_samples: int = 50
    easy_pool_samples: int = 100
    easy_answer_threshold: int = 80  # |gold| < this → easy
    test_samples: int = 100  # cap on GSM8K test split (None-like: 0 = all 1319)
    data_seed: int = 42  # controls train/val/easy split deterministically

    # Stage A (OTDD alignment) — only used for orca_otdd
    embedder_epochs: int = 5
    alignment_lr: float = 1e-4
    adapter_bottleneck: int = 128
    otdd_maxsamples_per_class: int = 64
    otdd_use_exact: bool = False  # gaussian_approx default (faster)

    # Stage B (LoRA) — used for both methods
    lora_epochs: int = 2
    lora_lr: float = 1e-4
    batch_size: int = 2
    grad_accum: int = 4
    gradient_clip: float = 1.0

    # Runtime
    output_dir: str = "runs/default"
    bf16: bool = False  # off for CPU/MPS; auto-enable on capable CUDA
    device: str = "auto"  # auto | cuda | cpu | mps
    verbose: bool = True  # print console output (disable for CHTC batch)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
