# region Imports
import os
import json

import basic_models_experiment
import naive_experiment
import mlp_basic_experiment
import ebe_experiment
import oracle_experiment
# endregion


def main(dataset_name, data_id):

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
    ebe_experiment.ebe_main(data_id=data_id, seed=SEED)
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
    omit_ids = [1590, 1111, 54,
                3, 12, 31, 1067
                ] # 1111 got NAns
    for dataset_name, data_id in dataset_ids.items():
       if data_id not in omit_ids:
        main(dataset_name=dataset_name, data_id=data_id)
       else: pass

    print('There is more hope')
    os.system("shutdown /s /t 1")