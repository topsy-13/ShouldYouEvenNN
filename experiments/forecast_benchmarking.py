# experiments/forecast_only_benchmark.py
import os, json, time, sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath("./src"))

import data_preprocessing as dp
from search_space import SearchSpace
from ebe import Population   # we’ll use its spawn/train/ledger/fidelity helpers only
from utils import init_global_seed, make_repro_context
from utils_viz import plot_forecast_vs_fidelity
from forecaster import forecast_generation
from scipy.stats import pearsonr, spearmanr

def run_forecast_only_benchmark(
    dataset_id: int,
    n_models: int = 25,
    seed: int = 13,
    task_type: str = "classification",
    min_val_points: int = 5,
    extra_full_passes: int = 50, # 50 epochs then
    save_dir: str = "./experiments/forecast_only"
):
    os.makedirs(save_dir, exist_ok=True)
    print('Testing dataset', dataset_id)
    # 0) Determinism
    init_global_seed(seed)
    repro = make_repro_context(seed)

    # 1) Data
    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=dataset_id, random_seed=seed, task_type=task_type, return_as="tensor"
    )

    in_size, out_size = dp.get_tensor_sizes(X_train, y_train, task_type=task_type)

    # 2) Search space + tiny “population” only for infra (no EBE loop)
    space = SearchSpace(input_size=in_size, output_size=out_size)
    starting_instances_proportion = 0.1
    starting_instances = int(starting_instances_proportion * len(X_train))
    pop = Population(search_space=space, size=n_models, starting_instances=starting_instances, seed=seed, task_type=task_type)
    
    # 3) One pass to log early curves (no pruning, no scoring)
    print('Training all models')
    n_repeats = 3  # as a start #minimal val points
    for i in range(n_repeats):
        print(f"[DEBUG] Re-training generation pass {i+1}/{n_repeats}")
        pop.train_generation(
            X_train=X_train, y_train=y_train,
            training_mode="oe",
            X_val=X_val, y_val=y_val,
            time_budget=None
        )
    # 4) Forecast: point + CI at projected horizon
    print('Forecasting all models')
    forecast_generation(
        candidates=pop.candidates,
        dataset_size=len(X_train),
        min_val_points=min_val_points,
        extra_full_passes=extra_full_passes
    )

    # 5) Freeze a ledger snapshot with forecasts for all models
    ledger = pop.build_ledger(export_as="pandas")
    ledger_path = os.path.join(save_dir, f"ledger_{dataset_id}.csv")
    ledger.to_csv(ledger_path, index=False)

    # 6) Fidelity training for ALL models (top_fraction=1). This rebuilds and trains
    #    each candidate until the horizon logic completes and returns actual best val accs.
    #    We need a baseline value to fill the API; pick a benign one (0.0) since we’re not
    #    doing decisions here.
    pop.current_snapshot = ledger.copy(deep=True)
    print('Fidelity of all models')

    fidelity = pop.fidelity_from_ledger(
        ledger_df=ledger,
        X_train=X_train, y_train=y_train,
        X_val=X_val,   y_val=y_val,
        top_fraction=1.0
    )
    fidelity_path = os.path.join(save_dir, f"fidelity_{dataset_id}.csv")
    fidelity.to_csv(fidelity_path, index=False)

    # 7) Core errors (MAE, Bias) using your helper
    from ebe import compare_forecast_vs_fidelity
    report = compare_forecast_vs_fidelity(fidelity)
    details = report["details"].copy()  # id, forecasted_val_acc, fidelity_val_acc, delta

    # 8) Extra metrics
    fcst = details["forecasted_val_acc"].astype(float).to_numpy()
    act  = details["fidelity_val_acc"].astype(float).to_numpy()
    # print('Forecasted acc:', fcst)
    # print('Real acc:', act)
    delta = act - fcst

    rmse = float(np.sqrt(np.mean((delta)**2)))
    # CI coverage if we have them in the ledger
    if "forecasted_CI_low" in fidelity and "forecasted_CI_high" in fidelity:
        lo = fidelity["forecasted_CI_low"].astype(float).to_numpy()
        hi = fidelity["forecasted_CI_high"].astype(float).to_numpy()
        # align order with details by id
        joined = details.merge(
            fidelity[["id", "forecasted_CI_low", "forecasted_CI_high"]],
            on="id", how="left"
        )
        lo = joined["forecasted_CI_low"].astype(float).to_numpy()
        hi = joined["forecasted_CI_high"].astype(float).to_numpy()
        covered = float(np.mean((act >= lo) & (act <= hi)))
        avg_ci_width = float(np.nanmean(hi - lo))
    else:
        covered = np.nan
        avg_ci_width = np.nan

    # correlations
    try:
        pear, _ = pearsonr(fcst, act)
    except Exception:
        pear = np.nan
    try:
        spear, _ = spearmanr(fcst, act)
    except Exception:
        spear = np.nan

    summary = {
        "dataset_id": dataset_id,
        "n_models": int(len(details)),
        "MAE": float(report["MAE"]),
        "Bias": float(report["Bias"]),
        "RMSE": rmse,
        "CI_coverage": covered,      # fraction of actuals inside [CI_low, CI_high]
        "Avg_CI_width": avg_ci_width,
        "Pearson_r": float(pear),
        "Spearman_rho": float(spear),
    }
    with open(os.path.join(save_dir, f"summary_{dataset_id}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 9) Visualization (scatter + error hist)
    try:
        plot_forecast_vs_fidelity(
            {"details": details.rename(columns={
                "forecasted_val_acc": "forecasted_val_acc",
                "fidelity_val_acc": "fidelity_val_acc",
                "delta": "delta"
            })},
            title=f"Forecast vs Fidelity (OpenML {dataset_id})"
        )
    except Exception:
        pass

    return summary, details, ledger, fidelity


if __name__ == "__main__":
    dataset_ids_path = 'experiments/datasets/openml_datasets.json'
    with open(dataset_ids_path) as f:
        dataset_ids = json.load(f)
    # omit_ids = [
    #             1111, 
    #             ] # 1111 got NAns


    crashed = {}

    omit_ids = [1169, 1590, 41156, 41147]
    for dataset_name, data_id in dataset_ids.items():
        if data_id not in omit_ids:
            print('Testing dataset:', dataset_name)
            save_dir = "./experiments/forecast_only"
            s, d, L, F = run_forecast_only_benchmark(dataset_id=data_id, n_models=25, seed=13)
            # print(s)
                # === 10) PLOTTING SECTION ===
            import matplotlib.pyplot as plt
            import ast

            print("\n[INFO] Generating visualizations...")

            # --- 1. Learning curves vs forecasts ---
            print("[Plot] Learning Curves vs Forecasts")
            fig, axes = plt.subplots(3, 3, figsize=(14, 10))
            axes = axes.ravel()

            sample_models = F.head(min(9, len(F)))
            for i, (_, row) in enumerate(sample_models.iterrows()):
                ax = axes[i]
                # parse the stored learning_curve (string -> dict)
                if isinstance(row["learning_curve"], str):
                    try:
                        lc = ast.literal_eval(row["learning_curve"])
                    except Exception:
                        lc = {}
                else:
                    lc = row["learning_curve"] or {}

                val_accs = lc.get("es_val_accs", [])
                epochs = np.arange(1, len(val_accs) + 1)

                if len(val_accs) > 0:
                    ax.plot(epochs, val_accs, lw=2, label="Validation Acc.")
                if not pd.isna(row.get("forecasted_val_acc")):
                    ax.axhline(row["forecasted_val_acc"], color="red", ls="--", label="Forecast")

                if not pd.isna(row.get("forecast_CI_low")) and not pd.isna(row.get("forecast_CI_high")):
                    ax.fill_between(
                        [0, max(1, len(epochs))],
                        row["forecast_CI_low"],
                        row["forecast_CI_high"],
                        color="red", alpha=0.15, label="Forecast CI"
                    )

                ax.set_title(f"Model {row['id']} | Final={row['fidelity_val_acc']:.2f}")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Validation Accuracy")
                ax.legend(loc="lower right", fontsize=8)
                ax.set_ylim(0.2, 1.0)


            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"learning_curves_{data_id}.png"), dpi=200)
            plt.close()

            # --- 2. Forecast vs Actual scatter ---
            print("[Plot] Forecast vs Actual Scatter")
            plt.figure(figsize=(6,6))
            plt.scatter(F["forecasted_val_acc"], F["fidelity_val_acc"], alpha=0.7)
            plt.plot([0,1],[0,1],'k--',label="Perfect prediction")
            plt.xlabel("Forecasted Validation Accuracy")
            plt.ylabel("Actual Validation Accuracy")
            plt.title(f"Forecast vs Actual (OpenML {data_id})")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"scatter_{data_id}.png"), dpi=200)
            plt.close()

            # --- 3. Forecast Error Histogram ---
            print("[Plot] Forecast Error Distribution")
            errors = F["fidelity_val_acc"] - F["forecasted_val_acc"]
            plt.figure(figsize=(6,4))
            plt.hist(errors, bins=15, alpha=0.7, color="gray", edgecolor="black")
            plt.axvline(errors.mean(), color="red", lw=2, label=f"Mean bias = {errors.mean():.3f}")
            plt.xlabel("Actual - Forecasted Accuracy")
            plt.ylabel("Frequency")
            plt.title(f"Forecast Error Distribution (OpenML {data_id})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"error_hist_{data_id}.png"), dpi=200)
            plt.close()

            print(f"[INFO] Plots saved to {save_dir}")

