# region Imports
import os
import sys
import json

import time
import torch

sys.path.append(os.path.abspath("./src"))

import generations
import search_space
import data_preprocessing as dp
import baseline_models as bm

from instance_sampling import create_dataloaders
from architecture_generator import create_model_from_row
# endregion

# region Oracle Training
def main(search_space, 
         data_id=54, seed=13, 
         n_individuals=3, 
         n_max_epochs=100,
         es_patience=100):

    # region Set the scenario
    DATA_ID = data_id
    SEARCH_SPACE = search_space
    SEED = seed
    N_INDIVIDUALS = n_individuals  # Individuals per generation
    N_MAX_EPOCHS = n_max_epochs
    ES_PATIENCE = 100


    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=DATA_ID,
        scaling=True,
        random_seed=SEED,
        return_as='tensor',
        task_type='classification'
        )
    
    print('Starting Oracle Training...')
    oracle_start_time = time.time()
    generation = generations.Generation(SEARCH_SPACE, N_INDIVIDUALS,
                                         starting_instances=len(X_train))

    input_size, output_size = dp.get_tensor_sizes(X_train, y_train)
    # Train Generation
    generation.train_generation(X_train, y_train,
                                training_mode='es', 
                                X_val=X_val, y_val=y_val,
                                es_patience=50, max_epochs=N_MAX_EPOCHS,
                                return_lc=True)
    oracle_training_time = time.time() - oracle_start_time
    print(f'Oracle Training time: {oracle_training_time:.2f} seconds')
    # Test Generation
    print('Testing Oracle trained generation...')
    generation.validate_generation(X_test, y_test, metric='test')
    oracle_full_time = time.time() - oracle_start_time
    print(f'Oracle Full time (training + testing): {oracle_full_time:.2f} seconds')
    _ = generation.return_df()
    oracle_results_df = generation.history
    print(oracle_results_df.head())

    # Export results
    oracle_results_df[['activation_fn', 'optimizer_type']] = oracle_results_df[['activation_fn', 'optimizer_type']].astype(str)
    oracle_results_df = oracle_results_df.fillna('None')
    results_path = f'./experiments/ebe_vs_oracle/oracle_training/{DATA_ID}_seed{SEED}'
    # As a json
    oracle_results_df.to_json(f'{results_path}_results.json', orient='records')



# endregion
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=54,
        scaling=True,
        random_seed=13,
        return_as='tensor',
        task_type='classification'
        )
    input_size, output_size = dp.get_tensor_sizes(X_train, y_train)
    s_space = search_space.SearchSpace(input_size, output_size)
    main(search_space=s_space)