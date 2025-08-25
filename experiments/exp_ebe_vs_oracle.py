import os
import sys
import json

import time

# region Imports
sys.path.append(os.path.abspath("./src"))

import generations
import search_space
import data_preprocessing as dp
import baseline_models as bm
from instance_sampling import create_dataloaders
# endregion


def main():

    # region Set the scenario
    DATA_ID = 54
    SEED = 13
    N_MODELS = 50  # for naiveautoml

    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=DATA_ID,
        scaling=True,
        random_seed=SEED,
        return_as='tensor',
        task_type='classification'
    )



    # NaiveAutoML baseline
    # ! This is showing reproducibility issues, it appears NaiveAutoML is not fully deterministic even with fixed seed
    print('Getting NaiveAutoML baseline...') 
    naml_results = {}
    baseline_metric, time_budget, naml_scoreboard = bm.get_models_and_baseline_metric(
        X_train, y_train,
        n_models=N_MODELS,
        random_state=SEED,
        strategy='mean'
    )  # Mean as baseline

    max_metric = bm.get_baseline_metric(naml_scoreboard, strategy='best')
    median_metric = bm.get_baseline_metric(naml_scoreboard, strategy='median')
    worst_metric = bm.get_baseline_metric(naml_scoreboard, strategy='worst')

    naml_scoreboard['pipeline'] = naml_scoreboard['pipeline'].astype(str)
    naml_results['seed'] = SEED
    naml_results['data_id'] = DATA_ID
    naml_results['time_taken'] = time_budget
    naml_results['baseline_metric'] = baseline_metric
    naml_results['max_metric'] = max_metric
    naml_results['median_metric'] = median_metric
    naml_results['worst_metric'] = worst_metric
    naml_results['scoreboard'] = naml_scoreboard.to_dict(orient='records')

    print("Baseline metric:", baseline_metric)
    print("Time for baseline models:", time_budget)

    BUDGET_FACTOR = 3  # EBE gets MULTIPLIED the time of the baseline
    nn_time_budget = time_budget * BUDGET_FACTOR
    print("Time budget for EBE:", nn_time_budget)

    # Export scoreboard for later analysis
    directory = f'experiments/ebe_vs_oracle/naml_baseline-models'
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{DATA_ID}.json"), 'w') as json_file:
        json.dump(naml_results, json_file, indent=4)

    # endregion

    # region EBE
    print('Starting EBE...')
    ebe_results = {}
    N_INDIVIDUALS = 200
    N_MAX_EPOCHS = 100
    percentile_drop = 15

    # Same search space for the whole experiment
    input_size, output_size = dp.get_tensor_sizes(X_train, y_train)
    s_space = search_space.SearchSpace(input_size, output_size)
    # The generation is created given the search space and the number of individuals.
    nn_start_time = time.time()
    generation = generations.Generation(s_space, N_INDIVIDUALS, starting_instances=500)
    generation.run_ebe(epochs=N_MAX_EPOCHS,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val, 
                    percentile_drop=percentile_drop,
                    baseline_metric=baseline_metric, 
                    time_budget=nn_time_budget,
                    epoch_threshold=3 # for forecasting
                    )
    nn_end_time = time.time()
    nn_taken_time = nn_end_time - nn_start_time
    
    ebe_results_df = generation.return_df()
    ebe_results_df[['activation_fn', 'optimizer_type']] = ebe_results_df[['activation_fn', 'optimizer_type']].astype(str)
    ebe_results_df = ebe_results_df.fillna('None')

    ebe_results['seed'] = SEED
    ebe_results['data_id'] = DATA_ID
    ebe_results['time_taken'] = nn_taken_time
    ebe_results['scoreboard'] = ebe_results_df.to_dict(orient='records')
    
    # Export ebe_results for later analysis
    directory = f'experiments/ebe_vs_oracle/ebe-models'
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{DATA_ID}.json"), 'w') as json_file:
        json.dump(ebe_results, json_file, indent=4)

    # endregion


if __name__ == "__main__":
    main()
