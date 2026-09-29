"""Plotting helpers used to inspect population dynamics."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

__all__ = [
    "plot_generation_dynamics",
    "plot_forecast_vs_fidelity",
    "plot_es_learning_curve_from_ledger",
]


def plot_generation_dynamics(generation_logs: Sequence[Mapping], *, title: str = "Population dynamics") -> None:
    """Plot how the population evolves across generations."""

    if not generation_logs:
        print("No logs to plot.")
        return

    df = pd.DataFrame(generation_logs)
    generations = df["gen"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(generations, df["survivors"], marker="o", label="Survivors")
    ax1.bar(generations, df["convex_lb_discarded"], alpha=0.4, label="Convex LB discarded")
    ax1.bar(
        generations,
        df["hybrid_dropped"],
        bottom=df["convex_lb_discarded"],
        alpha=0.4,
        label="Hybrid dropped",
    )
    ax1.plot(generations, df["spawned"], marker="x", linestyle="--", color="green", label="Spawned")
    ax1.plot(generations, df["final_population"], marker="s", color="red", label="Final population")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Population counts")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(generations, df["cumulative_compute"], color="purple", marker="d", label="Cumulative compute (s)")
    ax2.set_ylabel("Cumulative compute (seconds)")
    ax2.legend(loc="upper right")

    plt.title(title)
    plt.tight_layout()
    # plt.show()


def plot_forecast_vs_fidelity(fidelity_report: Mapping[str, pd.DataFrame], *, title: str = "Forecast vs Fidelity") -> None:
    """Visualise the accuracy forecast compared to the ES evaluation."""

    details = fidelity_report.get("details")
    if details is None or not isinstance(details, pd.DataFrame):
        raise ValueError("fidelity_report must contain a 'details' DataFrame.")

    fcst = details["forecasted_val_acc"].astype(float)
    actual = details["fidelity_val_acc"].astype(float)
    deltas = details["delta"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(fcst, actual, c="blue", alpha=0.7)
    min_val = min(fcst.min(), actual.min())
    max_val = max(fcst.max(), actual.max())
    axes[0].plot([0, 1], [0, 1], "r--", label="Perfect forecast")
    axes[0].set_xlabel("Forecasted val_acc")
    axes[0].set_ylabel("Actual ES val_acc")
    axes[0].set_title("Forecast vs Actual")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0.2, 1.0)
    axes[0].set_ylim(0.2, 1.0)

    axes[1].hist(deltas, bins=15, color="purple", alpha=0.7)
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Delta (Actual - Forecast)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Forecast Error Distribution")

    plt.suptitle(title)
    plt.tight_layout()
    # plt.show()


def plot_es_learning_curve_from_ledger(row, title=None):
    """
    Plot ES learning curve from a fidelity_ledger row, using cumulative effort (mini-batches)
    as the horizontal axis. Includes the forecasted accuracy, horizon, and confidence interval.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # --- Extract learning curve ---
    curve = row.get("learning_curve")
    if curve is None:
        raise ValueError("No learning curve stored for this candidate row.")

    val_accs = curve.get("es_val_accs", [])
    if not val_accs:
        raise ValueError("Learning curve has no validation accuracy points.")

    n_points = len(val_accs)
    times = np.arange(1, n_points + 1, dtype=float)  # 1, 2, 3, ...

    # --- Forecast info ---
    fcst_acc = row.get("forecasted_val_acc")
    fcst_effort = row.get("forecast_horizon_effort", row.get("forecast_horizon_time"))
    ci_lower, ci_upper = row.get("forecasted_CI_low"), row.get("forecasted_CI_high")
    fcst_at = row.get("forecasted_at", times[-1])  # where forecast was made, fallback

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(times, val_accs, "o-", color="blue", label="ES val_acc")
    plt.axvline(fcst_at, color="black", linestyle="--", ymax=1, label="Forecast made at")

    # --- Forecast overlay ---
    if fcst_acc is not None and fcst_effort is not None:
        plt.axvline(fcst_effort, color="red", linestyle="--", label="Forecast horizon")
        plt.axhline(fcst_acc, color="red", linestyle="--",
                    label=f"Forecasted acc = {fcst_acc:.3f}")

        if ci_lower is not None and ci_upper is not None:
            # vertical whisker
            plt.vlines(fcst_effort, ci_lower, ci_upper, color="black", lw=1.5)
            # small horizontal ticks
            plt.hlines([ci_lower, ci_upper],
                       fcst_effort * 0.97, fcst_effort * 1.03,
                       color="black", lw=1.5)
            # subtle shaded box
            plt.fill_betweenx([ci_lower, ci_upper],
                              fcst_effort * 0.97, fcst_effort * 1.03,
                              color="gray", alpha=0.2)

    plt.xlabel("Cumulative Effort (mini-batches)")
    plt.ylabel("Validation Accuracy")
    plt.title(title or f"Candidate {row.get('id', '?')} — ES vs Forecast")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.0)
    sns.despine()
    plt.tight_layout()

