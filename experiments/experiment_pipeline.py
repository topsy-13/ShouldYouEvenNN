# region Imports
import os
import json
import pandas as pd

import basic_models_experiment
import naive_experiment
import mlp_basic_experiment
import ebe_experiment
import es_eval_from_ledger

# endregion


def main(dataset_name, data_id, budget_factor=3):

    print(f'Starting Experiment for {dataset_name} | {data_id}')
    # ML Testing
    print(' -Testing Standard Models')
    basic_models_experiment.main(data_id=data_id, seed=SEED)
    # Standard MLP
    print(' -Testing Standard MLP')
    mlp_basic_experiment.main(data_id=data_id, seed=SEED)
    # NAML Testing
    print(' -Testing NaiveAutoML')
    naive_experiment.main(data_id=data_id, seed=SEED)
    # EBE
    print(' -Testing EBE')
    while True:
            try:
                ebe_experiment.main(data_id=data_id, seed=SEED,
                                    time_budget_factor=budget_factor)
                break  # success, escape the loop
            except TimeoutError as e:
                print(f"{dataset_name} ({data_id}) failed with budget {budget_factor}: {e}")
                budget_factor += 1
                print(f"Retrying {dataset_name} with budget {budget_factor}...")

    # Training By Es
    print(' -Testing EBE-ES')
    ebe_results = pd.read_csv(f'./experiments/ebe_vs/v2/ebe/{data_id}_{SEED}_EBE.csv') 
    es_eval_from_ledger.evaluate_from_ledger(ebe_results,
                                        data_id=data_id, seed=SEED,
                                        top_fraction=0.2)
    # Oracle EBE
    # print(' -Testing EBE-Oracle')
    # oracle_experiment.main(data_id=data_id, seed=SEED,
    #                        n_max_epochs=1000, es_patience=150)

    print('Experiment concluded')


if __name__ == "__main__":
    
    # Datasets to test
    SEED = 14125
    dataset_ids_path = 'experiments/datasets/openml_datasets.json'
    with open(dataset_ids_path) as f:
        dataset_ids = json.load(f)
    omit_ids = [
                1111, 
                ] # 1111 got NAns

    crashed = {}

    for dataset_name, data_id in dataset_ids.items():
        if data_id not in omit_ids:
            try:
                main(dataset_name=dataset_name, data_id=data_id, budget_factor=3)
            except Exception as e:
                print(f"{dataset_name} ({data_id}) crashed: {e}")
                crashed[data_id] = str(e)
        else:
            pass

    # export all crashes into a single JSON
    with open("crashed_datasets_pipeline.json", "w") as f:
        json.dump(crashed, f, indent=4)

    print('There is hope')
    os.system("shutdown /s /t 1")