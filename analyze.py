#!/usr/bin/env python3
"""
Analyze sweep results: generate tables, figures, and statistics.
Run after unpacking all result tarballs into results/ directory.

Usage:
    python analyze.py --results_dir results
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("WARNING: matplotlib not installed. Skipping figures. pip install matplotlib")


def load_all_metrics(results_dir):
    """Load all metrics.json files from results/lora/ and results/orca/."""
    all_results = []

    for method_dir in ["lora", "orca"]:
        base = os.path.join(results_dir, method_dir)
        if not os.path.exists(base):
            continue
        for job_dir in sorted(os.listdir(base)):
            metrics_path = os.path.join(base, job_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    m = json.load(f)
                cfg = m.get("config", {})
                all_results.append({
                    "method": cfg.get("method", "unknown"),
                    "rank": cfg.get("lora_r", 0),
                    "seed": cfg.get("seed", 0),
                    "test_em": m.get("test_em", 0),
                    "test_ce_loss": m.get("test_ce_loss", None),
                    "train_loss_final": m["stage_b_train_loss_per_epoch"][-1] if m.get("stage_b_train_loss_per_epoch") else None,
                    "train_loss_curve": m.get("stage_b_train_loss_per_epoch", []),
                    "val_em_curve": m.get("stage_b_val_em_per_epoch", []),
                    "stage_a_loss_curve": m.get("stage_a_loss_curve", []),
                    "stage_a_wall": m.get("stage_a_wall_seconds", 0),
                    "stage_b_wall": m.get("stage_b_wall_seconds", 0),
                    "job_dir": job_dir,
                })

    return all_results


def compute_summary(results):
    """Group by (method, rank) and compute mean ± std."""
    grouped = defaultdict(list)
    for r in results:
        key = (r["method"], r["rank"])
        grouped[key].append(r)

    summary = {}
    for key, runs in sorted(grouped.items()):
        method, rank = key
        ems = [r["test_em"] for r in runs]
        ces = [r["test_ce_loss"] for r in runs if r["test_ce_loss"] is not None]
        train_losses = [r["train_loss_final"] for r in runs if r["train_loss_final"] is not None]

        summary[key] = {
            "method": method,
            "rank": rank,
            "n_seeds": len(runs),
            "test_em_mean": np.mean(ems),
            "test_em_std": np.std(ems),
            "test_ce_mean": np.mean(ces) if ces else None,
            "test_ce_std": np.std(ces) if ces else None,
            "train_loss_mean": np.mean(train_losses) if train_losses else None,
            "train_loss_std": np.std(train_losses) if train_losses else None,
            "runs": runs,
        }

    return summary


def print_table(summary):
    """Print results table."""
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print(f"{'Method':<15} {'Rank':<6} {'Seeds':<7} {'Test EM (↑)':<18} {'Test CE (↓)':<18} {'Train Loss':<18}")
    print("-" * 80)

    for key in sorted(summary.keys()):
        s = summary[key]
        em_str = f"{s['test_em_mean']:.4f} ± {s['test_em_std']:.4f}"
        ce_str = f"{s['test_ce_mean']:.4f} ± {s['test_ce_std']:.4f}" if s['test_ce_mean'] is not None else "N/A"
        tl_str = f"{s['train_loss_mean']:.4f} ± {s['train_loss_std']:.4f}" if s['train_loss_mean'] is not None else "N/A"
        print(f"{s['method']:<15} {s['rank']:<6} {s['n_seeds']:<7} {em_str:<18} {ce_str:<18} {tl_str:<18}")

    print("=" * 80)


def plot_em_by_rank(summary, output_dir):
    """Figure 1: Bar chart of Test EM by rank, grouped by method."""
    if not HAS_PLT:
        return

    ranks = [4, 8, 16]
    methods = ["lora_only", "orca_otdd"]
    labels = ["LoRA Only", "ORCA + OTDD"]
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ranks))
    width = 0.35

    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means = []
        stds = []
        for rank in ranks:
            key = (method, rank)
            if key in summary:
                means.append(summary[key]["test_em_mean"])
                stds.append(summary[key]["test_em_std"])
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + i * width, means, width, yerr=stds, label=label,
               color=color, capsize=5, alpha=0.85)

    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Test Exact Match Accuracy")
    ax.set_title("Test EM by LoRA Rank")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(ranks)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_test_em_by_rank.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/fig1_test_em_by_rank.png")


def plot_ce_by_rank(summary, output_dir):
    """Figure 2: Bar chart of Test CE loss by rank."""
    if not HAS_PLT:
        return

    ranks = [4, 8, 16]
    methods = ["lora_only", "orca_otdd"]
    labels = ["LoRA Only", "ORCA + OTDD"]
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ranks))
    width = 0.35

    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means = []
        stds = []
        for rank in ranks:
            key = (method, rank)
            if key in summary and summary[key]["test_ce_mean"] is not None:
                means.append(summary[key]["test_ce_mean"])
                stds.append(summary[key]["test_ce_std"])
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + i * width, means, width, yerr=stds, label=label,
               color=color, capsize=5, alpha=0.85)

    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Test Cross-Entropy Loss")
    ax.set_title("Test CE Loss by LoRA Rank (lower is better)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(ranks)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_test_ce_by_rank.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/fig2_test_ce_by_rank.png")


def plot_training_curves(summary, output_dir):
    """Figure 3: Training loss curves over epochs."""
    if not HAS_PLT:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"lora_only": "#4C72B0", "orca_otdd": "#DD8452"}
    linestyles = {4: "-", 8: "--", 16: ":"}

    for key, s in sorted(summary.items()):
        method, rank = key
        # Average training curves across seeds
        curves = [r["train_loss_curve"] for r in s["runs"] if r["train_loss_curve"]]
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)

        epochs = range(1, len(mean_curve) + 1)
        label_method = "LoRA" if method == "lora_only" else "ORCA"
        ax.plot(epochs, mean_curve, color=colors[method], linestyle=linestyles[rank],
                label=f"{label_method} r={rank}", linewidth=2)
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                        color=colors[method], alpha=0.1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Cross-Entropy Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_training_curves.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/fig3_training_curves.png")


def plot_stage_a_curve(summary, output_dir):
    """Figure 4: Stage A OTDD loss curve (orca_otdd only)."""
    if not HAS_PLT:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {4: "#DD8452", 8: "#55A868", 16: "#C44E52"}

    for key, s in sorted(summary.items()):
        method, rank = key
        if method != "orca_otdd":
            continue
        curves = [r["stage_a_loss_curve"] for r in s["runs"] if r["stage_a_loss_curve"]]
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        mean_curve = np.mean(curves, axis=0)

        epochs = range(1, len(mean_curve) + 1)
        ax.plot(epochs, mean_curve, color=colors[rank],
                label=f"Rank {rank}", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("OTDD Loss")
    ax.set_title("Stage A: OTDD Adapter Alignment Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_stage_a_otdd.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/fig4_stage_a_otdd.png")


def plot_val_em_curves(summary, output_dir):
    """Figure 5: Val EM over training epochs."""
    if not HAS_PLT:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"lora_only": "#4C72B0", "orca_otdd": "#DD8452"}
    linestyles = {4: "-", 8: "--", 16: ":"}

    for key, s in sorted(summary.items()):
        method, rank = key
        curves = [r["val_em_curve"] for r in s["runs"] if r["val_em_curve"]]
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        mean_curve = np.mean(curves, axis=0)

        epochs = range(1, len(mean_curve) + 1)
        label_method = "LoRA" if method == "lora_only" else "ORCA"
        ax.plot(epochs, mean_curve, color=colors[method], linestyle=linestyles[rank],
                label=f"{label_method} r={rank}", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Exact Match")
    ax.set_title("Validation EM Over Training")
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_val_em_curves.png"), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/fig5_val_em_curves.png")


def run_significance_tests(summary):
    """Statistical significance between methods at each rank."""
    from scipy import stats

    print("\n" + "=" * 80)
    print("STATISTICAL TESTS (paired by seed, LoRA vs ORCA at each rank)")
    print("=" * 80)

    ranks = [4, 8, 16]
    for rank in ranks:
        lora_key = ("lora_only", rank)
        orca_key = ("orca_otdd", rank)

        if lora_key not in summary or orca_key not in summary:
            continue

        lora_ems = sorted(summary[lora_key]["runs"], key=lambda r: r["seed"])
        orca_ems = sorted(summary[orca_key]["runs"], key=lambda r: r["seed"])

        lora_vals = [r["test_em"] for r in lora_ems]
        orca_vals = [r["test_em"] for r in orca_ems]

        if len(lora_vals) >= 2 and len(orca_vals) >= 2:
            # Paired t-test (paired by seed)
            t_stat, p_val = stats.ttest_rel(orca_vals, lora_vals)
            diff = np.mean(orca_vals) - np.mean(lora_vals)
            print(f"  Rank {rank}: ORCA - LoRA = {diff:+.4f} | t={t_stat:.3f}, p={p_val:.4f} "
                  f"{'*' if p_val < 0.05 else ''}")
        else:
            print(f"  Rank {rank}: Not enough seeds for significance test")

    print()


def save_summary_json(summary, output_dir):
    """Save summary as JSON for further analysis."""
    out = {}
    for key, s in summary.items():
        method, rank = key
        out[f"{method}_r{rank}"] = {
            "test_em_mean": s["test_em_mean"],
            "test_em_std": s["test_em_std"],
            "test_ce_mean": s["test_ce_mean"],
            "test_ce_std": s["test_ce_std"],
            "train_loss_mean": s["train_loss_mean"],
            "train_loss_std": s["train_loss_std"],
            "n_seeds": s["n_seeds"],
        }

    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing lora/ and orca/ subdirectories")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory to save figures and summary")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all results
    results = load_all_metrics(args.results_dir)
    if not results:
        print(f"No metrics.json found in {args.results_dir}/lora/ or {args.results_dir}/orca/")
        return

    print(f"Loaded {len(results)} runs")

    # Compute summary statistics
    summary = compute_summary(results)

    # Print table
    print_table(summary)

    # Generate figures
    plot_em_by_rank(summary, args.output_dir)
    plot_ce_by_rank(summary, args.output_dir)
    plot_training_curves(summary, args.output_dir)
    plot_stage_a_curve(summary, args.output_dir)
    plot_val_em_curves(summary, args.output_dir)

    # Statistical tests
    run_significance_tests(summary)

    # Save summary JSON
    save_summary_json(summary, args.output_dir)

    print("\nDone! Check figures/ for plots and summary.json")


if __name__ == "__main__":
    main()
