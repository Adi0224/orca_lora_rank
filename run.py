#!/usr/bin/env python3
"""CLI entry point: train + test in one command."""

import sys
import os
import json
import argparse
import random

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

# Inject vendored OTDD into path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party", "orca_otdd"))

from config import ExperimentConfig
from data import make_splits
from model import load_model, attach_lora
from stage_a import run_stage_a
from stage_b import run_stage_b
from evaluate import evaluate_em, evaluate_ce_loss


def resolve_device(cfg):
    """Resolve 'auto' device to best available."""
    if cfg.device != "auto":
        return cfg.device
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Use MPS for lora_only, CPU for orca_otdd (OTDD has issues on MPS)
        if cfg.method == "orca_otdd":
            return "cpu"
        return "mps"
    return "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="ORCA × LoRA on GSM8K")
    # Add all config fields as CLI args
    for field_name, field_obj in ExperimentConfig.__dataclass_fields__.items():
        default = field_obj.default
        ftype = field_obj.type
        if ftype is bool:
            parser.add_argument(f"--{field_name}", action="store_true", default=default)
            parser.add_argument(f"--no-{field_name}", dest=field_name, action="store_false")
        elif ftype is int:
            parser.add_argument(f"--{field_name}", type=int, default=default)
        elif ftype is float:
            parser.add_argument(f"--{field_name}", type=float, default=default)
        else:
            parser.add_argument(f"--{field_name}", type=str, default=default)

    # --no-verbose already added by the bool handler above

    args = parser.parse_args()
    cfg = ExperimentConfig.from_dict(vars(args))

    # Resolve device
    cfg.device = resolve_device(cfg)

    # Auto-enable bf16 on capable CUDA
    if cfg.device == "cuda" and torch.cuda.is_bf16_supported():
        cfg.bf16 = True

    # Setup
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.output_dir, "config.json"))

    if cfg.verbose:
        print(f"Method: {cfg.method}")
        print(f"Device: {cfg.device}")
        print(f"Output: {cfg.output_dir}")
        print()

    # Load & split data
    if cfg.verbose:
        print("Loading GSM8K...")
    easy_pool, hard_train, hard_val, test_data = make_splits(cfg)
    if cfg.verbose:
        print(f"  Easy pool: {len(easy_pool)}, Hard train: {len(hard_train)}, "
              f"Hard val: {len(hard_val)}, Test: {len(test_data)}")
        print()

    # Initialize model
    if cfg.verbose:
        print("Loading model...")
    model = load_model(cfg)
    device = torch.device(cfg.device)
    model = model.to(device)
    tokenizer = model.tokenizer

    # Metrics collector
    metrics = {"config": vars(args)}

    # Stage A (OTDD alignment) — only for orca_otdd
    if cfg.method == "orca_otdd":
        if cfg.verbose:
            print("\n=== Stage A: OTDD Adapter Alignment ===")
        stage_a_result = run_stage_a(model, easy_pool, hard_train, cfg)
        metrics["stage_a_loss_curve"] = stage_a_result["loss_curve"]
        metrics["stage_a_wall_seconds"] = stage_a_result["wall_seconds"]
        if cfg.verbose:
            print(f"  Stage A done in {stage_a_result['wall_seconds']:.1f}s")
            print()

    # Attach LoRA for Stage B
    if cfg.verbose:
        print("=== Stage B: LoRA Training ===")
    model = attach_lora(model, cfg)

    # Run Stage B
    stage_b_result = run_stage_b(model, hard_train, hard_val, cfg)
    metrics["stage_b_train_loss_per_epoch"] = stage_b_result["train_loss_per_epoch"]
    metrics["stage_b_val_em_per_epoch"] = stage_b_result["val_em_per_epoch"]
    metrics["stage_b_wall_seconds"] = stage_b_result["wall_seconds"]
    if cfg.verbose:
        print(f"  Stage B done in {stage_b_result['wall_seconds']:.1f}s")
        print()

    # Evaluate on val
    if cfg.verbose:
        print("=== Validation Evaluation ===")
    val_result = evaluate_em(model, hard_val, tokenizer, device)
    if cfg.verbose:
        print(f"  Val EM: {val_result['correct']}/{val_result['total']} = {val_result['accuracy']:.4f}")

    # Save val predictions
    with open(os.path.join(cfg.output_dir, "val_predictions.json"), "w") as f:
        json.dump(val_result["predictions"], f, indent=2)

    # Evaluate on test
    if cfg.verbose:
        print("\n=== Test Evaluation ===")
    test_result = evaluate_em(model, test_data, tokenizer, device)
    test_ce = evaluate_ce_loss(model, test_data, tokenizer, device, cfg.max_length)
    metrics["test_em"] = test_result["accuracy"]
    metrics["test_correct"] = test_result["correct"]
    metrics["test_total"] = test_result["total"]
    metrics["test_ce_loss"] = test_ce["ce_loss"]
    metrics["test_predictions"] = test_result["predictions"]

    if cfg.verbose:
        print(f"  Test EM: {test_result['correct']}/{test_result['total']} = {test_result['accuracy']:.4f}")
        print(f"  Test CE Loss: {test_ce['ce_loss']:.4f}")
        # Print a few sample generations
        print("\n  Sample predictions:")
        for pred in test_result["predictions"][:3]:
            print(f"    Gold: {pred['gold']} | Predicted: {pred['predicted_num']} | "
                  f"{'✓' if pred['correct'] else '✗'}")
            print(f"    Generation: {pred['generation'][:100]}")
            print()

    # Save test predictions separately too
    with open(os.path.join(cfg.output_dir, "test_predictions.json"), "w") as f:
        json.dump(test_result["predictions"], f, indent=2)

    # Save metrics
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model artifacts
    # LoRA adapter
    lora_dir = os.path.join(cfg.output_dir, "lora_adapter")
    model.model.save_pretrained(lora_dir)
    if cfg.verbose:
        print(f"\n  LoRA adapter saved to {lora_dir}")

    # InputAdapter weights (if exists)
    if model.adapter is not None:
        adapter_path = os.path.join(cfg.output_dir, "adapter.pt")
        torch.save(model.adapter.state_dict(), adapter_path)
        if cfg.verbose:
            print(f"  InputAdapter saved to {adapter_path}")

    if cfg.verbose:
        print(f"\nDone! Results in {cfg.output_dir}/")


if __name__ == "__main__":
    main()
