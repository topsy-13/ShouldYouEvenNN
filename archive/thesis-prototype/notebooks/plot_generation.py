import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_generation_dynamics(csv_path, save_path=None, show=True):
    """
    Plot key dynamics of ShouldYouEvenNN generations:
    - Validation accuracy mean and forecast mean
    - Probability and expected utility over generations
    - Computational effort accumulation

    Args:
        csv_path (str): Path to generation_logs.csv
        save_path (str): Optional path to save the resulting figure.
        show (bool): Whether to display the plot.
    """
    df = pd.read_csv(csv_path)
    
    # --- Guard clause ---
    if len(df) <= 1:
        print(f"[plot_generation_dynamics] Only {len(df)} row(s) in log — skipping plot.")
        return
    
    # --- setup ---
    sns.set(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # --- 1. Accuracy dynamics ---
    axes[0].plot(df["gen"], df["val_acc_mean"], label="Mean Validation Accuracy", color="C0", linewidth=2)
    axes[0].fill_between(df["gen"],
                         df["val_acc_mean"] - df["val_acc_std"],
                         df["val_acc_mean"] + df["val_acc_std"],
                         color="C0", alpha=0.2)
    if "fcst_mean" in df.columns:
        axes[0].plot(df["gen"], df["fcst_mean"], label="Mean Forecasted Accuracy", color="C1", linestyle="--", linewidth=2)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_title("Forecast and Validation Dynamics")

    # --- 2. Probability & Utility ---
    if "p_above_mean" in df.columns:
        axes[1].plot(df["gen"], df["p_above_mean"], label="Mean p(Above Baseline)", color="C2", linewidth=2)
    if "expected_utility_global" in df.columns:
        axes[1].plot(df["gen"], df["expected_utility_global"], label="Expected Utility (U)", color="C3", linestyle="--", linewidth=2)
    axes[1].axhline(0, color="gray", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Probability / Utility")
    axes[1].set_title("Decision Metrics Evolution")
    axes[1].legend()

    # --- 3. Compute usage ---
    if "cumulative_compute" in df.columns:
        axes[2].plot(df["gen"], df["cumulative_compute"], label="Cumulative Compute Time", color="C4", linewidth=2)
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Cumulative Time (s)")
    axes[2].set_title("Compute Utilization Across Generations")
    axes[2].legend()
    exp_id = csv_path.split('_generation_logs.csv')[0].split('/')[-1].split('_')[0]
    plt.suptitle(f'Dataset: {exp_id}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

import os
initial_files = os.listdir("experiments/testing/final_results")
exp_ids = [exp_id.split('.')[0] for exp_id in initial_files if exp_id.endswith('.json')]

for exp_id in exp_ids:
    plot_generation_dynamics(f"experiments/testing/final_results/csvs/{exp_id}_generation_logs.csv", save_path=f"experiments/testing/final_results/figures/{exp_id}_gen_dynamics.png")
