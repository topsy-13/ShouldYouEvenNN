"""Plotting helpers used to inspect population dynamics."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
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
    plt.show()


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
    bounds = [min(fcst.min(), actual.min()), max(fcst.max(), actual.max())]
    axes[0].plot(bounds, bounds, "r--", label="Perfect forecast")
    axes[0].set_xlabel("Forecasted val_acc")
    axes[0].set_ylabel("Actual ES val_acc")
    axes[0].set_title("Forecast vs Actual")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(deltas, bins=15, color="purple", alpha=0.7)
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Delta (Actual - Forecast)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Forecast Error Distribution")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_es_learning_curve_from_ledger(row: Mapping, *, title: str | None = None) -> None:
    """Plot the ES learning curve stored in a ledger row."""

    curve = row.get("learning_curve")
    if curve is None:
        raise ValueError("No learning curve stored for this candidate row.")

    val_accs = curve.get("es_val_accs")
    if not val_accs:
        raise ValueError("Learning curve has no validation accuracy points.")

    n_points = len(val_accs)
    forecast_acc = row.get("forecasted_val_acc")
    forecast_time = row.get("forecast_horizon_time")

    if forecast_time is not None:
        times = np.linspace(0, forecast_time, n_points)
    else:
        times = np.arange(n_points)

    plt.figure(figsize=(8, 5))
    plt.plot(times, val_accs, "o-", color="blue", label="ES val_acc")

    if forecast_acc is not None and forecast_time is not None:
        plt.axvline(forecast_time, color="red", linestyle="--", label="Forecast horizon")
        plt.axhline(forecast_acc, color="red", linestyle="--", label=f"Forecasted acc={forecast_acc:.3f}")

    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Validation Accuracy")
    plt.title(title or f"Candidate {row.get('id', '?')} ES vs Forecast")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

