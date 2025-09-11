# region Imports
import os
import sys

import torch
import pandas as pd
import json

sys.path.append(os.path.abspath("./src"))

import data_preprocessing as dp
import baseline_models as bm

# endregion

# region NaiveTraining

def main(data_id, seed):
    exp_id = f'{data_id}_{seed}'

    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
            dataset_id=data_id,
            scaling=True,
            random_seed=seed,
            return_as='tensor',
            task_type='classification'
            )
    X_analysis = torch.cat([X_train, X_val], dim=0)
    y_analysis = torch.cat([y_train, y_val], dim=0)

    # print('Testing NaiveAutoML experiment...') 
    baseline_metric, time_taken, naml_scoreboard, best_naml_model = bm.get_models_and_baseline_metric(
        X_analysis, y_analysis,
        # n_models=NAIVE_MODELS, # commented because outputing all of the models
        random_state=seed,
    )  

    # Export scoreboard as csv
    export_path = f'./experiments/ebe_vs/v2/naml/{exp_id}_NAML.csv'
    naml_scoreboard.to_csv(export_path)

    # Test the best non neural pipeline + model on the test set
    accuracy = best_naml_model.score(X_test, y_test)

    # * Get baseline metrics (for later analysis I guess)
    # strategies = ["best", "median", "mean", "worst"]
    # naml_metrics = {
    #                 strategy: bm.get_baseline_metric(naml_scoreboard, strategy=strategy)
    #                 for strategy in strategies
    #                 }

    # Store NAML results
    naml_results = {
    "seed": seed,
    "data_id": data_id,
    "time_taken": time_taken,
    "max_training_metric": baseline_metric,
    "test_accuracy_best_non_nn_model": accuracy,
    }
    print("Baseline metric:", baseline_metric)
    print("Time for baseline models:", time_taken)
    print('Test performance acc:', accuracy)

    directory = f'experiments/ebe_vs/v2/naml'
    with open(os.path.join(directory, f"{exp_id}_NAML.json"), 'w') as json_file:
        json.dump(naml_results, json_file, indent=4)
    print('  -Naive results exported')    

if __name__ == "__main__":
    main(data_id=54, seed=13)




#endregion