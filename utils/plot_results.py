
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys
import glob
import argparse

# Color scheme: one color per experiment type
COLORS = {
    "fl_baseline":    "#2196F3",   # blue
    "dp_fl_eps5000":  "#4CAF50",   # green
    "dp_fl_eps500":   "#FF9800",   # orange
    "dp_fl_eps100":   "#F44336",   # red
    "single_earth":   "#9C27B0",   # purple
    "single_mars":    "#795548",   # brown
    "single_moon":    "#607D8B",   # grey
}

LABELS = {
    "fl_baseline":    "FL (no DP)",
    "dp_fl_eps5000":  "DP-FL ε=5000",
    "dp_fl_eps500":   "DP-FL ε=500",
    "dp_fl_eps100":   "DP-FL ε=100",
    "single_earth":   "Single: Earth",
    "single_mars":    "Single: Mars",
    "single_moon":    "Single: Moon",
}


def load_results(results_dir):
    """
    Load all CSV files from a results directory.
    
    Returns:
        dict: {experiment_name: pandas.DataFrame}
    """
    dfs = {}
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return dfs
    
    for csv_path in csv_files:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)
        dfs[name] = df
        print(f"Loaded: {name} ({len(df)} rows)")
    
    return dfs


def plot_learning_curves(dfs, save_path="results/learning_curves.png"):
    """
    Plot reward vs. training round for all FL experiments.
    
    This is Figure 1 from the paper: shows how all methods improve over time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Only plot FL experiments (not single-agent, which uses steps not rounds)
    fl_experiments = [k for k in dfs.keys() if not k.startswith("single")]
    
    for exp_name in fl_experiments:
        df = dfs[exp_name]
        
        # Only rows where we actually evaluated
        df_eval = df.dropna(subset=["global_reward"])
        
        if df_eval.empty:
            continue
        
        color = COLORS.get(exp_name, "black")
        label = LABELS.get(exp_name, exp_name)
        
        ax.plot(
            df_eval["round"],
            df_eval["global_reward"],
            color=color,
            label=label,
            linewidth=2,
            marker="o",
            markersize=3,
        )
    
    # Add "solved" line at 200
    ax.axhline(y=200, color="black", linestyle="--", alpha=0.4, label="Solved (200)")
    
    ax.set_xlabel("Federated Learning Round", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("FedRL Learning Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-100, 350)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_final_comparison(dfs, save_path="results/final_comparison.png"):
    """
    Bar chart: final reward for each experiment.
    
    This is the key comparison: FedRL vs single-agent baselines.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    experiments_ordered = [
        "fl_baseline", "dp_fl_eps5000", "dp_fl_eps500", "dp_fl_eps100",
        "single_earth", "single_mars", "single_moon"
    ]
    
    rewards = []
    labels = []
    colors = []
    
    for exp_name in experiments_ordered:
        if exp_name not in dfs:
            continue
        
        df = dfs[exp_name]
        df_eval = df.dropna(subset=["global_reward"])
        
        if df_eval.empty:
            continue
        
        # Use last 3 evaluations as "final" to smooth out noise
        final_reward = df_eval["global_reward"].tail(3).mean()
        
        rewards.append(final_reward)
        labels.append(LABELS.get(exp_name, exp_name))
        colors.append(COLORS.get(exp_name, "grey"))
    
    x = range(len(labels))
    bars = ax.bar(x, rewards, color=colors, edgecolor="white", linewidth=1.5)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, rewards):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{val:.0f}",
            ha="center", va="bottom", fontsize=10
        )
    
    ax.axhline(y=200, color="black", linestyle="--", alpha=0.4)
    ax.text(len(labels) - 0.5, 205, "Solved", fontsize=9, alpha=0.6)
    
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Final Mean Reward", fontsize=12)
    ax.set_title("Final Performance Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(min(0, min(rewards) - 50), max(rewards) + 60)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_privacy_utility_tradeoff(dfs, save_path="results/dp_tradeoff.png"):
    """
    Shows the privacy-utility tradeoff: as epsilon decreases (more privacy),
    performance degrades.
    
    Key insight from the paper:
    - ε=5000: near-baseline performance (almost no noise)
    - ε=500:  slightly degraded (acceptable tradeoff)
    - ε=100:  severely degraded (too much noise)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Epsilon values and corresponding experiment names
    dp_experiments = [
        (5000, "dp_fl_eps5000"),
        (500,  "dp_fl_eps500"),
        (100,  "dp_fl_eps100"),
    ]
    
    epsilons = []
    final_rewards = []
    
    for eps, exp_name in dp_experiments:
        if exp_name not in dfs:
            continue
        df = dfs[exp_name].dropna(subset=["global_reward"])
        if df.empty:
            continue
        final_rewards.append(df["global_reward"].tail(3).mean())
        epsilons.append(eps)
    
    # Also add no-DP baseline as reference
    if "fl_baseline" in dfs:
        df_base = dfs["fl_baseline"].dropna(subset=["global_reward"])
        if not df_base.empty:
            baseline_reward = df_base["global_reward"].tail(3).mean()
            ax.axhline(y=baseline_reward, color=COLORS["fl_baseline"],
                      linestyle="--", linewidth=2, label="FL (no DP)")
    
    if epsilons:
        ax.plot(epsilons, final_rewards, "o-",
                color="#FF5722", linewidth=2.5, markersize=8,
                label="DP-FL", zorder=5)
        
        for eps, rew in zip(epsilons, final_rewards):
            ax.annotate(f"ε={eps}\n({rew:.0f})",
                       xy=(eps, rew), xytext=(eps, rew + 15),
                       ha="center", fontsize=9)
    
    ax.set_xscale("log")
    ax.set_xlabel("Privacy Budget (ε) — lower = more private", fontsize=12)
    ax.set_ylabel("Final Mean Reward", fontsize=12)
    ax.set_title("Privacy-Utility Tradeoff", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotate privacy levels
    ax.text(0.02, 0.98, "← More Private", transform=ax.transAxes,
            fontsize=9, va="top", color="grey")
    ax.text(0.98, 0.98, "Less Private →", transform=ax.transAxes,
            fontsize=9, va="top", ha="right", color="grey")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def generate_all_plots(results_dir="results"):
    """Generate all plots from saved CSVs."""
    print(f"\nLoading results from {results_dir}/...\n")
    dfs = load_results(results_dir)
    
    if not dfs:
        print("No results to plot. Run experiments first!")
        return
    
    print("\nGenerating plots...")
    plot_learning_curves(dfs, os.path.join(results_dir, "learning_curves.png"))
    plot_final_comparison(dfs, os.path.join(results_dir, "final_comparison.png"))
    plot_privacy_utility_tradeoff(dfs, os.path.join(results_dir, "dp_tradeoff.png"))
    
    print("\nDone! Plots saved to:", results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FedRL experiment results")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir)
