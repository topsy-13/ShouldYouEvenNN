# region Imports
import os
import sys
import json


import time

sys.path.append(os.path.abspath("./src"))

import search_space
import data_preprocessing as dp
from utils import init_global_seed, make_repro_context

from ebe import Population


# endregion

def get_ml_metrics(data_id, seed):
    starting_path = './experiments/testing/benchmark'

    with open(os.path.join(starting_path, f'{data_id}_{seed}_results.json'), 'r') as f:
        exp_results = json.load(f)
        naml_time_taken = exp_results["naml_time_taken"]
        naml_best_val_acc = exp_results["naml_best_validation_metric"]
        naml_max_test_acc = exp_results["naml_test_accuracy"]
        hist_time_taken = exp_results["hgb_training_time_sec"]
        hist_max_test_acc = exp_results["hgb_accuracy"]
        mlp_time_taken = exp_results['mlp_training_time']
    
    return naml_time_taken, naml_max_test_acc,  hist_time_taken, hist_max_test_acc, mlp_time_taken, naml_best_val_acc, exp_results

def main( 
         data_id, seed,
         X_train, y_train, X_val, y_val, 
         X_test, y_test,
         pop_size=25, 
         starting_instances_proportion=0.1,
         time_budget_factor=3):
    
    # === Scenario setup ===
    DATA_ID = data_id
    SEED = seed
    exp_id = f'{data_id}_{seed}'
    POP_SIZE = pop_size
    BUDGET_FACTOR = time_budget_factor
    MAX_GENS = 200
    BASE_DROP = 0.2
    MAX_DROP = 0.3

    init_global_seed(seed)
    repro = make_repro_context(seed)

    out = './experiments/testing/final_results'
    naml_time_taken, naml_max_test_acc, hist_time_taken, hist_max_test_acc, mlp_time_taken, naml_best_val_acc, exp_results = get_ml_metrics(data_id=DATA_ID, seed=SEED)

    # === Build search space and population ===
    input_size, output_size = dp.get_tensor_sizes(X_train, y_train)
    s_space = search_space.SearchSpace(input_size=input_size, output_size=output_size)

    starting_instances = int(starting_instances_proportion * len(X_train))
    ebe_start_time = time.time()
    pop = Population(s_space, 
                     size=POP_SIZE, 
                     starting_instances=starting_instances, 
                     seed=seed, 
                     task_type='classification')

    time_budget_ebe = max(hist_time_taken * BUDGET_FACTOR, mlp_time_taken, 60)
    if time_budget_ebe == hist_time_taken:
        time_used = 'HistGradientBoosting'
    elif time_budget_ebe == mlp_time_taken:
        time_used = 'MLP'
    else:
        time_used = 'Minimum60s'

    print('Hist time', hist_time_taken)
    print('Assigned budget', time_budget_ebe)

    # === Run EBE process ===
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
    ledger.to_csv(os.path.join(out, f"csvs/{exp_id}-ledger.csv"), index=False)
    logs_df = pop.export_generation_logs(os.path.join(out, f"csvs/{exp_id}_generation_logs.csv"))

    # === Evaluate final candidate after EBE ===
    ebe_val = None
    ebe_test = None
    ebe_beats_val = False
    ebe_beats_test = False

    if final_decision.get("ShouldYouEvenNN?", False):
        # get best candidate after EBE
        best_cand = max(pop.candidates.values(), key=lambda c: c.metrics.get("forecasted_val_acc", 0.0))
        model = best_cand.model

        # evaluate on validation set
        val_loader = dp.create_dataloader(
            X=X_val, y=y_val,
            batch_size=best_cand.batch_size,
            generator=pop.repro.torch_gen,
            seed_worker=pop.repro.seed_worker,
            shuffle=False
        )
        val_loss, val_acc = model.evaluate(val_loader)
        ebe_val = val_acc
        ebe_beats_val = bool(val_acc >= float(naml_best_val_acc))

        # evaluate on test set
        test_loader = dp.create_dataloader(
            X=X_test, y=y_test,
            batch_size=best_cand.batch_size,
            generator=pop.repro.torch_gen,
            seed_worker=pop.repro.seed_worker,
            shuffle=False
        )
        test_loss, test_acc = model.evaluate(test_loader)
        ebe_test = test_acc
        ebe_beats_test = bool(test_acc >= float(naml_max_test_acc))

        print(f"[Evaluation] val_acc={val_acc:.4f} | test_acc={test_acc:.4f}")
        print(f"[Evaluation] Beats val baseline? {ebe_beats_val} | Beats test baseline? {ebe_beats_test}")
    else:
        print("[EBE] No NN selected for evaluation.")

    # === Build summary ===
    decision = getattr(pop, "decision", False)
    EU = float(getattr(pop, "eu", 0.0))
    p = float(getattr(pop, "p", 0.0))
    top_fcst = float(ledger["forecasted_val_acc"].max()) if not ledger.empty else 0.0
    top_fcst_ci_h = float(ledger["forecast_CI_high"].max()) if ("forecast_CI_high" in ledger) and not ledger.empty else None

    ebe_summary = {
        # --- Experiment Context ---
        "data_id": data_id,
        "seed": seed,

        # --- Baseline References ---
        "baseline_val_acc": float(naml_best_val_acc),
        "baseline_test_acc": float(naml_max_test_acc),
        "baseline_hist_time_s": float(hist_time_taken),
        "baseline_mlp_time_s": float(mlp_time_taken),

        # --- Budget Configuration ---
        "ebe_time_budget": float(time_budget_ebe),
        "ebe_time_budget_source": time_used,
        "ebe_time_used_s": float(ebe_time_taken),
        "ebe_budget_efficiency": float(ebe_time_taken / time_budget_ebe) if time_budget_ebe else None,

        # --- Evolutionary Decision Summary ---
        "ebe_generations_completed": int(getattr(pop, "generations_completed", 0)),
        "ebe_decision": bool(decision),
        "ebe_expected_utility": float(EU),
        "ebe_p_above_goal": float(p),
        "ebe_benefit_est": float(getattr(pop, "benefit", 0.0)),
        "ebe_cost_est": float(getattr(pop, "cost", 0.0)),

        # --- Forecast Results ---
        "ebe_top_forecast_val_acc": float(top_fcst),
        "ebe_top_forecast_CI_high": float(top_fcst_ci_h) if top_fcst_ci_h is not None else None,

        # --- Final Evaluation ---
        "ebe_val_acc": float(ebe_val or 0.0),
        "ebe_test_acc": float(ebe_test or 0.0),
        "ebe_surpassed_val_baseline": bool(ebe_beats_val),
        "ebe_surpassed_test_baseline": bool(ebe_beats_test),

        # --- Efficiency Metrics ---
        "ebe_time_vs_hist_ratio": float(ebe_time_taken / hist_time_taken) if hist_time_taken else None,
        "ebe_time_vs_mlp_ratio": float(ebe_time_taken / mlp_time_taken) if mlp_time_taken else None,
    }

    # === Output results ===
    exp_results.update(ebe_summary)
    with open(f'{out}/{exp_id}.json', 'w') as f:
        json.dump(exp_results, f, indent=4)

    print("\n[SUMMARY]")
    print(f"Baseline val_acc={naml_best_val_acc:.4f} | test_acc={naml_max_test_acc:.4f}")
    print(f"EBE val_acc={ebe_val:.4f} | test_acc={ebe_test:.4f}")
    print(f"Beats val baseline? {ebe_beats_val} | Beats test baseline? {ebe_beats_test}")

    return ebe_summary


if __name__ == "__main__":
    # Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
            dataset_id=41147,
            scaling=True,
            random_seed=13,
            return_as='tensor',
            task_type='classification',
            categorical_strategy='label', 
            verbose=True
        )
    main(data_id=41147, seed=13, 
         X_train=X_train, y_train=y_train, 
         X_val=X_val, y_val=y_val, 
         X_test=X_test, y_test=y_test,
         pop_size=25, 
         starting_instances_proportion=0.3,
         time_budget_factor=3)

