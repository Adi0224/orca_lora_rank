#!/usr/bin/env python3
"""Rank sweep: runs both methods at ranks 4, 8, 16."""

import subprocess
import sys
import json
import os

RANKS = [4, 8, 16]
METHODS = ["lora_only", "orca_otdd"]

# Base config — override as needed
BASE_ARGS = {
    "lora_lr": "1e-4",
    "lora_epochs": "2",
    "hard_train_samples": "200",
    "val_samples": "50",
    "test_samples": "100",
    "embedder_epochs": "5",
    "data_seed": "42",
    "seed": "0",
}


def run_experiment(method, rank, output_dir):
    """Run a single experiment."""
    cmd = [
        sys.executable, "run.py",
        "--method", method,
        "--lora_r", str(rank),
        "--lora_alpha", str(rank * 2),
        "--output_dir", output_dir,
    ]
    for k, v in BASE_ARGS.items():
        cmd.extend([f"--{k}", v])

    print(f"\n{'='*60}")
    print(f"Running: {method} | rank={rank} | alpha={rank*2}")
    print(f"Output:  {output_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    results = {}

    for method in METHODS:
        for rank in RANKS:
            output_dir = f"runs/sweep_{method}_r{rank}"
            rc = run_experiment(method, rank, output_dir)

            if rc != 0:
                print(f"FAILED: {method} rank={rank} (exit code {rc})")
                continue

            # Load metrics
            metrics_path = os.path.join(output_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    m = json.load(f)
                results[f"{method}_r{rank}"] = {
                    "test_em": m.get("test_em", None),
                    "train_loss_final": m["stage_b_train_loss_per_epoch"][-1],
                    "val_em_final": m["stage_b_val_em_per_epoch"][-1],
                }
                if "stage_a_loss_curve" in m:
                    results[f"{method}_r{rank}"]["stage_a_final_loss"] = m["stage_a_loss_curve"][-1]

    # Print summary
    print(f"\n{'='*60}")
    print("SWEEP RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Rank':<6} {'Train Loss':<12} {'Val EM':<10} {'Test EM':<10}")
    print("-" * 60)
    for key, vals in results.items():
        method, rank = key.rsplit("_r", 1)
        print(f"{method:<20} {rank:<6} {vals['train_loss_final']:<12.4f} "
              f"{vals['val_em_final']:<10.4f} {vals['test_em']:<10.4f}")

    # Save summary
    summary_path = "runs/sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
