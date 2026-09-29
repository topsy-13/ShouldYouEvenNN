"""
EBE-NAS Results Analysis (research-paper version)
Compatible with all_experiments.csv columns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------------------
# 1. Load and summarize
# ---------------------------------------------------------------------
def load_experiments(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def summarize_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ebe_decision",
        "ebe_expected_utility",
        "ebe_p_above_goal",
        "ebe_benefit_est",
        "ebe_cost_est",
        "ebe_top_forecast_val_acc",
        "ebe_val_acc",
        "ebe_test_acc",
        "ebe_surpassed_val_baseline",
        "ebe_surpassed_test_baseline",
    ]
    summary = (
        df[cols]
        .agg(["mean", "std", "count"])
        .transpose()
        .rename_axis("metric")
        .reset_index()
    )
    return summary


def compute_success_rates(df: pd.DataFrame) -> dict:
    """Return success fractions for val/test surpassing baselines."""
    return {
        "val_success_rate": df["ebe_surpassed_val_baseline"].mean(),
        "test_success_rate": df["ebe_surpassed_test_baseline"].mean(),
    }


# ---------------------------------------------------------------------
# 2. Correlation and relationships
# ---------------------------------------------------------------------
def correlation_block(df: pd.DataFrame) -> pd.DataFrame:
    corr_cols = [
        "ebe_p_above_goal",
        "ebe_expected_utility",
        "ebe_benefit_est",
        "ebe_cost_est",
        "ebe_top_forecast_val_acc",
        "ebe_val_acc",
        "ebe_test_acc",
    ]
    return df[corr_cols].corr(method="pearson")


# ---------------------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------------------
def plot_expected_utility(df: pd.DataFrame, out_dir: str | Path = None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="ebe_p_above_goal",
        y="ebe_expected_utility",
        hue="ebe_surpassed_val_baseline",
        palette="coolwarm",
        s=60,
    )
    plt.axhline(0, color="black", linestyle="--", lw=1)
    plt.xlabel("Probability of beating baseline ($p$)")
    plt.ylabel("Expected Utility ($U$)")
    plt.title("Expected Utility Landscape across Datasets")
    plt.tight_layout()
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / "expected_utility_vs_p.png", dpi=300)
    return plt.gcf()


def plot_forecast_vs_val(df: pd.DataFrame, out_dir: str | Path = None):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=df,
        x="ebe_top_forecast_val_acc",
        y="ebe_val_acc",
        color="steelblue",
        s=60,
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Forecasted Validation Accuracy")
    plt.ylabel("Observed Validation Accuracy")
    plt.title("Forecast vs Actual Validation Accuracy")
    plt.tight_layout()
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / "forecast_vs_val.png", dpi=300)
    return plt.gcf()


def plot_time_efficiency(df: pd.DataFrame, out_dir: str | Path = None):
    plt.figure(figsize=(7, 5))
    sns.histplot(
        df["ebe_time_vs_hist_ratio"],
        bins=30,
        color="gray",
        edgecolor="black",
        alpha=0.7,
    )
    plt.xlabel("EBE / HGB Training Time Ratio")
    plt.ylabel("Frequency")
    plt.title("Time Efficiency of EBE-NAS relative to HistGradientBoosting")
    plt.tight_layout()
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / "time_efficiency_hist.png", dpi=300)
    return plt.gcf()


# ---------------------------------------------------------------------
# 4. LaTeX summary table (simple + safe)
# ---------------------------------------------------------------------
def latex_summary_table(df: pd.DataFrame) -> str:
    """
    Export a concise LaTeX summary table with safe escaping.
    Includes mean and std for the main EBE-NAS metrics.
    """
    tbl = (
        df[
            [
                "ebe_p_above_goal",
                "ebe_expected_utility",
                "ebe_benefit_est",
                "ebe_cost_est",
                "ebe_val_acc",
                "ebe_test_acc",
            ]
        ]
        .describe()
        .T[["mean", "std"]]
    )

    # Export to LaTeX with proper escaping of underscores and special chars
    latex_str = tbl.to_latex(
        float_format="%.3f",
        caption="Aggregate EBE-NAS Performance Summary",
        label="tab:ebe_summary",
        escape=True  # <-- key fix
    )

    return latex_str


# ---------------------------------------------------------------------
# 5. Full analysis routine
# ---------------------------------------------------------------------
def run_analysis(csv_path="all_experiments.csv", out_dir="analysis_outputs"):
    df = load_experiments(csv_path)
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    print("\n=== Core Summary ===")
    print(summarize_core_metrics(df))

    print("\n=== Success Rates ===")
    print(compute_success_rates(df))

    print("\n=== Correlations ===")
    print(correlation_block(df))

    plot_expected_utility(df, out)
    plot_forecast_vs_val(df, out)
    plot_time_efficiency(df, out)

    tex = latex_summary_table(df)
    with open(out / "ebe_summary_table.tex", "w") as f:
        f.write(tex)
    print(f"\nLaTeX table saved to {out/'ebe_summary_table.tex'}")


if __name__ == "__main__":
    run_analysis("./experiments/testing/final_results/summary/all_experiments.csv")
