import matplotlib.pyplot as plt
import pandas as pd

def plot_generation_dynamics(generation_logs, title="Population dynamics"):
    if not generation_logs:
        print("No logs to plot.")
        return

    df = pd.DataFrame(generation_logs)
    gens = df["gen"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Population dynamics
    ax1.plot(gens, df["survivors"], marker="o", label="Survivors (post hybrid drop)")
    ax1.bar(gens, df["convex_lb_discarded"], alpha=0.4, label="Convex LB discarded")
    ax1.bar(gens, df["hybrid_dropped"], bottom=df["convex_lb_discarded"],
            alpha=0.4, label="Hybrid dropped")
    ax1.plot(gens, df["spawned"], marker="x", linestyle="--", color="green", label="Spawned")
    ax1.plot(gens, df["final_population"], marker="s", color="red", label="Final population")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Population counts")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Compute dynamics on right axis
    ax2 = ax1.twinx()
    ax2.plot(gens, df["cumulative_compute"], color="purple", marker="d",
             label="Cumulative compute (s)")
    ax2.set_ylabel("Cumulative compute (seconds)")
    ax2.legend(loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.show()



def plot_forecast_vs_fidelity(fidelity_report, title="Forecast vs Fidelity"):
    """
    Visualize forecasted vs. ES-validated accuracies.

    fidelity_report: dict returned by Population.compare_forecast_vs_fidelity()
    """
    details = fidelity_report["details"]

    fcst = details["forecasted_val_acc"].astype(float)
    actual = details["fidelity_val_acc"].astype(float)
    deltas = details["delta"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Scatter: forecast vs actual ---
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

    # --- Histogram: forecast error ---
    axes[1].hist(deltas, bins=15, color="purple", alpha=0.7)
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Delta (Actual - Forecast)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Forecast Error Distribution")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_es_learning_curve_from_ledger(row, title=None):
    """
    Plot ES learning curve from a fidelity_ledger row with time axis scaled to the forecast horizon.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    lc = row["learning_curve"]
    if lc is None:
        raise ValueError("No learning curve stored for this candidate row.")

    val_accs = lc["es_val_accs"]
    n_points = len(val_accs)

    if n_points == 0:
        raise ValueError("Learning curve has no validation accuracy points.")

    # forecast info
    fcst_acc = row.get("forecasted_val_acc", None)
    fcst_time = row.get("forecast_horizon_time", None)

    # build x-axis: evenly spread validation checks over [0, forecast_horizon_time]
    if fcst_time is not None:
        times = np.linspace(0, fcst_time, n_points)
    else:
        times = np.arange(n_points)

    plt.figure(figsize=(8, 5))

    # ES curve
    plt.plot(times, val_accs, "o-", color="blue", label="ES val_acc")

    # forecast overlay
    if fcst_acc is not None and fcst_time is not None:
        plt.axvline(fcst_time, color="red", linestyle="--", label="Forecast horizon")
        plt.axhline(fcst_acc, color="red", linestyle="--", label=f"Forecasted acc={fcst_acc:.3f}")

    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Validation Accuracy")
    plt.title(title or f"Candidate {row['id']} ES vs Forecast")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1.0)
    plt.show()
