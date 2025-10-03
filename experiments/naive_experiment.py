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

def main(data_id, seed, 
        X_analysis, y_analysis, X_test, y_test):
    exp_id = f'{data_id}_{seed}'
    
    # export_path = f'./experiments/ebe_vs/v4/experiments/{exp_id}.csv'


    # print('Testing NaiveAutoML experiment...') 
    validation_metric, time_taken, naml_scoreboard, best_naml_model = bm.get_models_and_baseline_metric(
        X_analysis, y_analysis,
        # n_models=NAIVE_MODELS, # commented because outputing all of the models
        random_state=seed,
    )  

    # Export scoreboard as csv
    # naml_scoreboard.to_csv(export_path)

    # Test the best non neural pipeline + model on the test set
    test_accuracy = best_naml_model.score(X_test, y_test)

    # * Get baseline metrics (for later analysis I guess)
    # strategies = ["best", "median", "mean", "worst"]
    # naml_metrics = {
    #                 strategy: bm.get_baseline_metric(naml_scoreboard, strategy=strategy)
    #                 for strategy in strategies
    #                 }

    # Store NAML results
    naml_results = {
    # "seed": seed,
    # "data_id": data_id,
    "naml_time_taken": time_taken,
    "naml_best_validation_metric": validation_metric,
    "naml_test_accuracy": test_accuracy,
    }
    print("Max Naive val metric:", validation_metric)
    print("Time taken from Naive:", time_taken)
    print('Naive best model Test acc:', test_accuracy)

    # directory = f'experiments/ebe_vs/v3'
    # with open(os.path.join(directory, f"{exp_id}_NAML.json"), 'w') as json_file:
    #     json.dump(naml_results, json_file, indent=4)
    # print('  -Naive results exported')
    return naml_results

if __name__ == "__main__":
    # Datasets to test
    SEED = 14125
    dataset_ids_path = 'experiments/datasets/openml_datasets.json'
    with open(dataset_ids_path) as f:
        dataset_ids = json.load(f)
    for dataset_name, data_id in dataset_ids.items():
        print(f"Starting dataset {data_id} - {dataset_name}")
        main(data_id=data_id, seed=SEED)
    print('All done for Naive!')    




#endregion