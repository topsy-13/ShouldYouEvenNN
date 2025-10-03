# region Imports
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt


import time

sys.path.append(os.path.abspath("./src"))

import search_space
import data_preprocessing as dp
import baseline_models as bm
from utils import init_global_seed, make_repro_context

from ebe import Population
from utils_viz import plot_forecast_vs_fidelity, plot_generation_dynamics, plot_es_learning_curve_from_ledger

# endregion

def get_ml_metrics(data_id, seed):
    starting_path = './experiments/testing'

    with open(f'{starting_path}_results.json') as f:
        exp_results = json.load(f)
        naml_time_taken = exp_results["naml_time_taken"]
        naml_best_val_acc = exp_results["naml_best_validation_metric"]
        naml_max_test_acc = exp_results["naml_test_accuracy"]
        hist_time_taken = exp_results["hgb_training_time_sec"]
        hist_max_test_acc = exp_results["hgb_accuracy"]
        mlp_time_taken = exp_results['mlp_training_time']
    
    return naml_time_taken, naml_max_test_acc,  hist_time_taken, hist_max_test_acc, mlp_time_taken, naml_best_val_acc

def main(
         data_id, seed,
         X_train, y_train, X_val, y_val, 
         X_test, y_test,
         pop_size=25, 
         starting_instances_proportion=0.3,
         time_budget_factor=3):
    
    # region Set the scenario
    DATA_ID = data_id
    SEED = seed
    POP_SIZE = pop_size  # Individuals per generation
    BUDGET_FACTOR = time_budget_factor  # EBE gets MULTIPLIED the time of the baseline
    exp_id = f'{data_id}_{seed}'
    MAX_GENS = 200
    BASE_DROP = 0.2
    MAX_DROP = 0.3
    # --- reproducibility
    init_global_seed(seed)
    repro = make_repro_context(seed)

    
    # Get data from the json results of the Naive Experiment
    out = './experiments/ebe_vs/v3/experiments'
    
    naml_time_taken, naml_max_test_acc, hist_time_taken, hist_max_test_acc, mlp_time_taken, naml_best_val_acc = get_ml_metrics(data_id=DATA_ID, seed=SEED)
    # enregion

    # region EBE
    input_size, output_size = dp.get_tensor_sizes(X_train, y_train)
    s_space = search_space.SearchSpace(input_size=input_size, 
                                       output_size=output_size)

    ''' By default and to test the true potential of the new instance budget method, a starting number of instances is defined for the experiment'''
    starting_instances = int(starting_instances_proportion * len(X_train))
    ebe_start_time = time.time()
    pop = Population(s_space, 
                     size=POP_SIZE, 
                     starting_instances=starting_instances, 
                     seed=seed, 
                     task_type='classification')
    time_budget_ebe = max(hist_time_taken * BUDGET_FACTOR, mlp_time_taken, 60)
    # Store what budget was assigned
    if max(hist_time_taken * BUDGET_FACTOR, mlp_time_taken, 60) == hist_time_taken:
        time_used = 'HistGradientBoosting'
    elif max(hist_time_taken * BUDGET_FACTOR, mlp_time_taken, 60) == mlp_time_taken:
        time_used = 'MLP' 
    else:
        time_used = 'Minimum60s'

    print('Hist time', hist_time_taken)
    print('Assigned budget', time_budget_ebe)
    pop.run_ebe(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            baseline_metric=float(naml_best_val_acc),
            max_generations=MAX_GENS,
            time_budget=time_budget_ebe,
            base_drop=BASE_DROP,
            max_drop=MAX_DROP,
            track_all_models=False
        )
    
    ebe_end_time = time.time()
    ebe_time_taken = ebe_end_time - ebe_start_time
    final_decision = pop.final_decision()
    print('Final Decision:', final_decision)

    ledger = pop.current_snapshot.copy()
    ledger.to_csv(os.path.join(out, f"{exp_id}-ledger.csv"), index=False)
    logs_df = pop.export_generation_logs(os.path.join(out, f"{exp_id}_generation_logs.csv"))
    
    fid = None
    surpassed = False
    test_best = None               # <<< NEW
    test_beats_baseline = None     # <<< NEW
    if final_decision['ShouldYouEvenNN?']:

        # --- fidelity check and plots
        print('Evaluating fidelity...')
        fid = pop.fidelity_from_ledger(
            ledger_df=ledger,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,                 # <<< NEW
            baseline_metric=float(naml_best_val_acc),     # <<< NEW
            top_fraction=0.2
        )
        fid.to_csv(os.path.join(out, "fidelity_ledger.csv"), index=False)
        # Reporting section
        try:
            plot_forecast_vs_fidelity(pop.compare_forecast_vs_fidelity(), title=f"Forecast vs Fidelity — {data_id}")
            plt.savefig(os.path.join(out, f"{exp_id}_forecast_vs_fidelity.png"))
            plot_generation_dynamics(pop.generation_logs, title=f"Population dynamics — {data_id}")
            plt.savefig(os.path.join(out, f"{exp_id}_generation_dynamics.png"))
            # plot the lc of the best model 
            row = ledger.iloc[1]
            ebe_id = row['id']
            cand_id = ebe_id
            row = fid[fid["id"] == cand_id].iloc[0]
            plot_es_learning_curve_from_ledger(row)
            plt.savefig(os.path.join(out, f"{exp_id}_best_model_lc.png"))

        except Exception as e:
            print('Exception plotting', e)
            pass

        if not fid.empty:
            best_row = fid.iloc[0]
            test_best = best_row.get("test_acc")   # <<< NEW
            if test_best is not None:
                test_beats_baseline = bool(test_best >= float(naml_max_test_acc))  # <<< NEW
                surpassed = test_beats_baseline
            else:
                surpassed = bool(best_row["fidelity_val_acc"] >= float(naml_max_test_acc))

    # --- summary
    decision = getattr(pop, "decision", False)
    EU = float(getattr(pop, "eu", 0.0))
    p = float(getattr(pop, "p", 0.0))
    top_fcst = float(ledger["forecasted_val_acc"].max()) if not ledger.empty else 0.0
    top_fcst_ci_h = float(ledger["forecast_CI_high"].max()) if ("forecast_CI_high" in ledger) and not ledger.empty else None
    fid_best = float(fid["fidelity_val_acc"].max()) if fid is not None and not fid.empty else None

    summary = {
        # "data_id": data_id,
        # 'seed': seed,
        "ebe_baseline_metric": float(naml_max_test_acc),
        "ebe_time_budget": time_budget_ebe,
        "ebe_time_budget_used": time_used,
        "ebe_decision": bool(decision),
        "ebe_EU": EU,
        "ebe_p": p,
        "ebe_top_fcst": top_fcst,
        "ebe_top_fcst_CI_high": top_fcst_ci_h,
        "ebe_fidelity_best": fid_best,
        "ebe_surpassed_baseline": bool(surpassed),
        "ebe_test_best": test_best,                     # <<< NEW
        "ebe_test_beats_baseline": test_beats_baseline, # <<< NEW
        "ebe_elapsed_baseline_hist": hist_time_taken,
        "ebe_time_s": ebe_time_taken
    }
    # with open(os.path.join(out, f"{exp_id}-summary.json"), "w") as f:
    #     json.dump(summary, f, indent=2)

    # <<< NEW: print summary of test performance
    if test_best is not None:
        print(f'Baseline test_acc ={naml_max_test_acc:.4f}')
        print(f"Best test_acc={test_best:.4f}, beats baseline? {test_beats_baseline}")
    else:
        print("No test set evaluation performed.")
    
    return summary

if __name__ == "__main__":
    # Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
            dataset_id=54,
            scaling=True,
            random_seed=14125,
            return_as='tensor',
            task_type='classification',
            categorical_strategy='label', 
            verbose=True
        )
    main(data_id=54, seed=14125, 
         X_train=X_train, y_train=y_train, 
         X_val=X_val, y_val=y_val, 
         X_test=X_test, y_test=y_test,
         pop_size=25, 
         starting_instances_proportion=0.3,
         time_budget_factor=3)

