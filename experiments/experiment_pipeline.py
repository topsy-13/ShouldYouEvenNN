# region Imports
import os
import torch
import json
import pandas as pd

# import basic_models_experiment 
# import naive_experiment
# import mlp_basic_experiment
import ebe_experiment

import data_preprocessing as dp

# endregion


def main(dataset_name, data_id, seed):
    # Load Dataset
    exp_id = f'{data_id}_{seed}'
    exp_path = f'./experiments/testing/{exp_id}'
    # os.makedirs(exp_path, exist_ok=True)
    results_dict = {
        'exp_id': exp_id,
        'data_id': data_id,
        'seed': seed,
    }
    # with open(f'{exp_path}_results.json', 'w') as f:
    #     json.dump(results_dict, f, indent=4)        

    print(f'\n ** Starting Experiment for {dataset_name} | {data_id}')
    # print(f'Experiment ID: {exp_id}')
    print(f'- Loading Dataset {dataset_name} | {data_id}')

    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=data_id,
        scaling=True,
        random_seed=seed,
        return_as='tensor',
        task_type='classification',
        categorical_strategy='label', 
        verbose=False
    )
    X_analysis = torch.cat([X_train, X_val], dim=0)
    y_analysis = torch.cat([y_train, y_val], dim=0)
    # n_rows = int(X_analysis.shape[0])
    # n_features = int(X_analysis.shape[1])

    # results_dict['n_rows'] = n_rows
    # results_dict['n_features'] = n_features
    # # ML Testing
    # print(' -Testing HistGradientBoosting')
    # hgb_results = basic_models_experiment.main(data_id=data_id, seed=seed,
    #                              X_analysis=X_analysis, y_analysis=y_analysis, 
    #                              X_test=X_test, y_test=y_test)
    # results_dict.update(hgb_results)
    # with open(f'{exp_path}_results.json', 'w') as f:
    #     json.dump(results_dict, f, indent=4)  

    # # # Standard MLP
    # print(' -Testing Standard MLP') # MLPClassifier(random_state=seed, max_iter=1000, n_iter_no_change=100)
    # mlp_results = mlp_basic_experiment.main(data_id, SEED, 
    #                           X_analysis, y_analysis, X_test, y_test) 
    # results_dict.update(mlp_results)
    # with open(f'{exp_path}_results.json', 'w') as f:
    #     json.dump(results_dict, f, indent=4)  

    # # NAML Testing
    # print(' -Testing NaiveAutoML')
    # naive_results = naive_experiment.main(data_id, SEED, 
    #                       X_analysis, y_analysis, X_test, y_test)
    # results_dict.update(naive_results)
    
    # with open(f'{exp_path}_results.json', 'w') as f:
    #     json.dump(results_dict,f, indent=4)  
    
    # EBE
    print(' -Testing EBE')
    ebe_results = ebe_experiment.main(data_id, SEED, 
                        X_train,y_train, 
                        X_val, y_val,
                        X_test, y_test,
                        pop_size=30, 
                        starting_instances_proportion=0.2,
                        time_budget_factor=3)
    # results_dict.update(ebe_results)
    # with open(f'{exp_path}_results.json', 'w') as f:
    #     json.dump(results_dict, f, indent=4)  
    print('Experiment concluded')


if __name__ == "__main__":
    
    # Datasets to test
    SEED = 13
    dataset_ids_path = 'experiments/datasets/openml_datasets2.json'
    with open(dataset_ids_path) as f:
        dataset_ids = json.load(f)
    # omit_ids = [
    #             1111, 
    #             ] # 1111 got NAns

    import json
    import traceback
    import datetime

    crashed = {}

    for dataset_name, data_id in dataset_ids.items():
        try:
            main(dataset_name=dataset_name, data_id=data_id, seed=SEED)
        except Exception as e:
            tb_str = traceback.format_exc()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"[{timestamp}] {dataset_name} ({data_id}) crashed: {type(e).__name__} - {e}")
            
            crashed[data_id] = {
                "dataset_name": dataset_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": timestamp,
                "traceback": tb_str
            }

    # export all crashes into a single JSON
    with open("./experiments/testing/final_results/crashed_datasets_pipeline.json", "w") as f:
        json.dump(crashed, f, indent=4)


    # main(dataset_name='Vehicle', data_id=54, seed=SEED)
    print('There is hope')
    os.system("shutdown /s /t 60")