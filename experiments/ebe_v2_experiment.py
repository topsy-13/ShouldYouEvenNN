# experiment.py
import os, json, time, traceback
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("./src"))
from utils import init_global_seed, make_repro_context
from data_preprocessing import get_preprocessed_data, get_tensor_sizes
from baseline_models import get_models_and_baseline_metric
from search_space import SearchSpace
from ebe import Population
from utils_viz import plot_forecast_vs_fidelity, plot_generation_dynamics

# ------------ knobs you actually care about ------------
POP_SIZE = 100
STARTING_INSTANCES = 0.1                # as fraction or absolute, you’re using absolute elsewhere; be consistent
MAX_GENS = 20
BASE_DROP = 0.1
MAX_DROP = 0.5

EXTRA_FULL_PASSES = 5                   # used by forecaster
TOP_FIDELITY_FRAC = 0.5
EXT_TRAIN_CAP_S = 60                    # hard cap for post-EBE extension
# -------------------------------------------------------

def secs_for_dataset(hist_time):
    # heuristic: at least 30s, or 3x the time taken by the baseline
    return max(30, hist_time * 3)

def run_one_dataset(data_id: int, seed: int = 13, outdir="runs"):
    t0 = time.time()
    out = os.path.join(outdir, str(data_id))
    os.makedirs(out, exist_ok=True)

    # --- reproducibility
    init_global_seed(seed)
    repro = make_repro_context(seed)

    # --- data
    X_train, y_train, X_val, y_val, X_test, y_test = get_preprocessed_data(
        dataset_id=data_id,
        scaling=True,
        categorical_strategy="label",
        return_as="tensor",
        random_seed=seed,
        task_type="classification",
        verbose=False
    )
    input_size, output_size = get_tensor_sizes(X_train, y_train, task_type="classification")
    n_rows = int(X_train.shape[0] + X_val.shape[0] + X_test.shape[0])

    ebe_time_budget = secs_for_dataset()

    # --- baseline
    baseline_metric, baseline_time, scoreboard, best_model = get_models_and_baseline_metric(
        X=np.vstack([X_train.numpy(), X_val.numpy()]),
        y=np.concatenate([y_train.numpy(), y_val.numpy()]),
        top_models=None,
        scoring_metric=DECISION_METRIC,
        random_state=seed,
        strategy="best"
    )
    scoreboard.to_csv(os.path.join(out, "baseline_scoreboard.csv"), index=False)

    # --- search space & population
    ss = SearchSpace(input_size=input_size, output_size=output_size)
    pop = Population(ss, size=POP_SIZE, starting_instances=STARTING_INSTANCES, seed=seed, task_type='classification')

    # --- EBE loop
    pop.run_ebe(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        baseline_metric=float(baseline_metric),
        max_generations=MAX_GENS,
        time_budget=ebe_time_budget,
        base_drop=BASE_DROP,
        max_drop=MAX_DROP,
        track_all_models=False
    )

    # save ledgers/logs
    ledger = pop.current_snapshot.copy()
    ledger.to_csv(os.path.join(out, "ledger.csv"), index=False)
    logs_df = pop.export_generation_logs(os.path.join(out, "generation_logs.csv"))

    # --- fidelity check and plots
    fid = pop.fidelity_ledger.copy()
    fid.to_csv(os.path.join(out, "fidelity_ledger.csv"), index=False)

    # Optional plots (guarded to avoid headless issues)
    try:
        plot_forecast_vs_fidelity(pop.compare_forecast_vs_fidelity(), title=f"Forecast vs Fidelity — {data_id}")
        plot_generation_dynamics(pop.generation_logs, title=f"Population dynamics — {data_id}")
    except Exception:
        pass

    # --- post-EBE extended training (already triggered inside run_ebe in your code if flip happens)
    # If you want to assert outcome explicitly:
    surpassed = False
    if hasattr(pop, "fidelity_ledger") and not fid.empty:
        best_row = fid.iloc[0]
        surpassed = bool(best_row["fidelity_val_acc"] >= float(baseline_metric))

    # --- summary
    decision = getattr(pop, "decision", False)
    EU = float(getattr(pop, "eu", 0.0))
    p = float(getattr(pop, "p", 0.0))
    top_fcst = float(ledger["forecasted_val_acc"].max()) if not ledger.empty else 0.0
    top_fcst_ci_h = float(ledger["forecast_CI_high"].max()) if ("forecast_CI_high" in ledger) and not ledger.empty else None
    fid_best = float(fid["fidelity_val_acc"].max()) if not fid.empty else None

    summary = {
        "data_id": data_id,
        "n_rows": n_rows,
        "baseline_metric": float(baseline_metric),
        "ebe_time_budget_s": ebe_time_budget,
        "ebe_decision": bool(decision),
        "ebe_EU": EU,
        "ebe_p": p,
        "top_fcst": top_fcst,
        "top_fcst_CI_high": top_fcst_ci_h,
        "fidelity_best": fid_best,
        "surpassed_baseline": bool(surpassed),
        "elapsed_baseline_s": baseline_time,
        "ebe_time_s": time.time() - t0
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary

def run_sweep(dataset_ids, seed=13, outdir="runs"):
    os.makedirs(outdir, exist_ok=True)
    results, errors = [], {}
    for i, data_id in enumerate(dataset_ids):
        try:
            print(f"\n=== Dataset {data_id} ({i+1}/{len(dataset_ids)}) ===")
            res = run_one_dataset(int(data_id), seed=seed + i, outdir=outdir)
            results.append(res)
        except Exception as e:
            errors[str(data_id)] = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"[ERROR] data_id={data_id}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(outdir, "aggregate_results.csv"), index=False)
    with open(os.path.join(outdir, "errors.json"), "w") as f:
        json.dump(errors, f, indent=2)
    return df, errors

if __name__ == "__main__":
    # drop your list here
    DATASET_IDS = [54, 1590, 41138]  # example
    run_sweep(DATASET_IDS, seed=13, outdir="runs_openml")
