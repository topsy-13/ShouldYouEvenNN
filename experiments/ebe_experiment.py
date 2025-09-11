# region Imports
import os
import sys
import json
import pandas as pd

import time

sys.path.append(os.path.abspath("./src"))

import search_space
import data_preprocessing as dp
import baseline_models as bm

from ebe import Population
# endregion


def main(data_id=54, seed=12,
         n_individuals=100, 
         starting_instances_proportion=0.1,
         percentile_drop=10,
         time_budget_factor=3):
    
    # region Set the scenario
    DATA_ID = data_id
    SEED = seed
    N_INDIVIDUALS = n_individuals  # Individuals per generation
    BUDGET_FACTOR = time_budget_factor  # EBE gets MULTIPLIED the time of the baseline
    exp_id = f'{data_id}_{seed}'


    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=DATA_ID,
        scaling=True,
        random_seed=SEED,
        return_as='tensor',
        task_type='classification'
        )
    
    # Get data from the json results of the Naive Experiment
    starting_path = './experiments/ebe_vs/v2'
    naml_path = os.path.join(starting_path,'naml')

    with open(os.path.join(naml_path, f'{DATA_ID}_{SEED}_NAML.json')) as f:
        naml_results = json.load(f)
        naml_time_taken = naml_results["time_taken"]
        naml_max_test_acc = naml_results["test_accuracy_best_non_nn_model"]
    
    with open(os.path.join(starting_path, f'standard/{DATA_ID}_{SEED}_ML.json')) as f:
        standard_results = json.load(f)
        ml_time_taken = standard_results['total_training_time_sec']
        ml_max_test_acc = standard_results["best_accuracy"]
    # enregion
    # region EBE
    input_size, output_size = dp.get_tensor_sizes(X_train, y_train)
    s_space = search_space.SearchSpace(input_size=input_size, 
                                       output_size=output_size)

    ''' By default and to test the true potential of the new instance budget method, a starting number of instances is defined for the experiment'''
    starting_instances = int(len(X_train) 
                             * starting_instances_proportion)
    ebe_start_time = time.time()
    population = Population(s_space, 
                            size=N_INDIVIDUALS,
                            starting_instances=starting_instances,
                            seed=SEED)
    max_metric = max(naml_max_test_acc, ml_max_test_acc)
    time_budget_ebe = ml_time_taken * BUDGET_FACTOR
    ebe_results_final = population.run_ebe(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val, 
                        percentile_drop=percentile_drop,
                        baseline_metric=max_metric, 
                        time_budget=time_budget_ebe,
                        epoch_threshold=5,
                        track_all_models=True,
                        forecast_method='rational'
                    
                        )
    ebe_end_time = time.time()
    ebe_time_taken = ebe_end_time - ebe_start_time
    # Reporting section
    ebe_results_all = population.cumulative_ledger

    n_candidates_higher_fcsted = len(ebe_results_all
                                   [ebe_results_all['fcst_greater_than_baseline'] 
                                    == True])
    
    max_ebe_train_acc = ebe_results_final['train_acc'].apply(max).max()
    max_ebe_val_acc = ebe_results_final['val_acc'].apply(max).max()

    ebe_performance = {
        'seed': SEED,
        'data_id': DATA_ID,
        'time_budget': time_budget_ebe,
        'time_taken': ebe_time_taken,
        'naml_baseline_metric': naml_max_test_acc,
        'candidates_created': population.individuals_created,
        'n_above_baseline_fcst': n_candidates_higher_fcsted,
        'max_ebe_train_acc': max_ebe_train_acc,
        'max_ebe_val_acc': max_ebe_val_acc
        
    }

    # Export EBE results
    export_path = './experiments/ebe_vs/v2/ebe'
    # Export as json and dataframe

    ebe_results_final.to_csv(f'{export_path}/{exp_id}_EBE.csv', index=False)
    ebe_results_all.to_csv(f'{export_path}/{exp_id}_EBE-history.csv', index=False)
    
    with open(f"{export_path}/{exp_id}_EBE-summary.json", 'w') as json_file:
        json.dump(ebe_performance, json_file, indent=4)


    return ebe_performance, ebe_results_all, ebe_results_final

def train_by_es(n_individuals, ebe_results, seed, data_id):
    from architecture_generator import create_model_from_row
    from utils import set_seed
    exp_id = f'{data_id}_{seed}'

    N_INDIVIDUALS = n_individuals
    
    # Load data (yes, again)
    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=data_id,
        scaling=True,
        random_seed=seed,
        return_as='tensor',
        task_type='classification'
        )
    input_size, output_size = dp.get_tensor_sizes(X_train, y_train)
    print(' -Testing EBE final models by ES')
    ebe_results = ebe_results.copy()
    # Train the best models identified by EBE (and pray)
    N_TOP_MODELS_TO_TRAIN = int(N_INDIVIDUALS * 0.2) # * 20 percentile of models to return 
    # Extract their IDs
    top_model_ids = ebe_results['id'].head(N_TOP_MODELS_TO_TRAIN).unique().tolist()
    ebe_performance = {
        'seed': seed,
        'data_id': data_id,
    }
    

    es_results = []
    es_total_time = 0
    for n_model, model_id in enumerate(top_model_ids):
        # print('Training model by ES', n_model + 1, 
        #       'of', N_TOP_MODELS_TO_TRAIN)
        # Build from 0 the model
        model_row = ebe_results[ebe_results['id'] == model_id].iloc[0].copy()

        model_seed = int(model_row.get('arch_seed', None))
        model_batch_size = int(model_row.get('batch_size', None))
        g, seed_worker = set_seed(model_seed)
        # print('Batch_Size:',model_batch_size)
        
        train_loader = dp.create_dataloader(X=X_train, y=y_train, 
                       batch_size=model_batch_size, generator=g, seed_worker=seed_worker)
        val_loader = dp.create_dataloader(X=X_val, y=y_val, 
                            batch_size=model_batch_size, generator=g, seed_worker=seed_worker)
        test_loader = dp.create_dataloader(X=X_test, y=y_test, 
                            batch_size=model_batch_size, generator=g, seed_worker=seed_worker)

        model = create_model_from_row(model_row, input_size, output_size)

        es_start_time = time.time()
        # Train the model
        es_metrics = model.es_train(train_loader=train_loader, val_loader=val_loader,
                    es_patience=150, # epochs without improvement
                    max_epochs=1000, # cap for epochs
                    verbose=False, # print training progress,
                    return_lc=True
        )
        es_end_time = time.time()
        es_model_time = es_end_time - es_start_time 

        es_train_loss, es_train_acc, es_val_loss, es_val_acc, es_lc  = es_metrics

        model_row['es_train_loss'] = es_train_loss        
        model_row['es_val_loss'] = es_val_loss        
        model_row['es_train_acc'] = es_train_acc        
        model_row['es_val_acc'] = es_val_acc      
        for key, value in es_lc.items():
            model_row[key] = value  

        # Evaluate on test set
        es_test_loss, es_test_acc = model.evaluate(test_loader) 
        model_row['es_test_loss'] = es_test_loss
        model_row['es_test_acc'] = es_test_acc

        # Append the results to new_df
        es_results.append(model_row)
        es_total_time += es_model_time

    # Record the max performance found after ES
    es_results = pd.DataFrame(es_results)
    ebe_performance['training_time'] = es_total_time
    ebe_performance['max_train_acc'] = es_results['es_train_acc'].max()
    ebe_performance['min_train_loss'] = es_results['es_train_loss'].min()
    ebe_performance['max_val_acc'] = es_results['es_val_acc'].max()
    ebe_performance['min_val_loss'] = es_results['es_val_loss'].min()
    ebe_performance['max_test_acc'] = es_results['es_test_acc'].max()
    ebe_performance['min_test_loss'] = es_results['es_test_loss'].min()

    # Export as json and dataframe
    export_path = './experiments/ebe_vs/v2/ebe'

    es_results.to_csv(f'{export_path}/{exp_id}_ES_EBE.csv', index=False)
    with open(f"{export_path}/{exp_id}_ES_EBE-summary.json", 'w') as json_file:
        json.dump(ebe_performance, json_file, indent=4)


    return ebe_performance


def ebe_main(data_id, seed):
    ebe_performance, ebe_results_all, ebe_results_final = main(data_id=data_id, seed=seed, 
    n_individuals=130, starting_instances_proportion=0.1, time_budget_factor=3, percentile_drop=15)
    print('  -EBE Results Exported')
    
    train_by_es(data_id=data_id, seed=seed,
        n_individuals=130, 
        ebe_results=ebe_results_final)
    print('  -EBE-ES Results Exported')
    
    # endregion

if __name__ == "__main__":
    ebe_main(data_id=54, seed=13)

