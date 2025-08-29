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
         n_individuals=100, 
         n_max_epochs=1000,
         es_patience=150):

    # region Set the scenario
    DATA_ID = data_id
    SEARCH_SPACE = search_space
    SEED = seed
    N_INDIVIDUALS = n_individuals  # Individuals per generation
    N_MAX_EPOCHS = n_max_epochs
    ES_PATIENCE = es_patience


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
                                         starting_instances=len(X_train),
                                         seed=SEED)

    # Train Generation
    generation.train_generation(X_train, y_train,
                                training_mode='es', 
                                X_val=X_val, y_val=y_val,
                                es_patience=ES_PATIENCE, 
                                max_epochs=N_MAX_EPOCHS,
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
    oracle_results_df[['activation_fn', 'optimizer_type']] = oracle_results_df[['activation_fn', 'optimizer_type']].astype(str)
    oracle_results_df = oracle_results_df.fillna('None')

    # Drop unused columns for this experiment
    oracle_results_df = oracle_results_df.drop(columns=[
                                            "n_instances",
                                            "efforts",
                                            "train_loss",
                                            "train_acc",
                                            "val_loss",
                                            "val_acc",
                                        ])

    oracle_results = {
        'data_id': DATA_ID,
        'seed': SEED,
        'n_individuals': N_INDIVIDUALS,
        'n_max_epochs': N_MAX_EPOCHS,
        'es_patience': ES_PATIENCE,
        'oracle_training_time': oracle_training_time,
        'oracle_full_time': oracle_full_time,
        # 'best_train_accuracy': oracle_results_df['train_acc'].max(),
        # 'best_val_accuracy': oracle_results_df['val_acc'].max(),
        'best_test_accuracy': oracle_results_df['test_acc'].max(),
        'oracle_results_df': oracle_results_df.to_dict(orient='records')
    }

    # Export results
    results_path = f'./experiments/ebe_vs_oracle/oracle_training/{DATA_ID}'
    # As a json
    # oracle_results_df.to_json(f'{results_path}_results.json', orient='records')
    with open(f'{results_path}_oracle.json', 'w') as f:
        json.dump(oracle_results, f, indent=4)


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


    # TODO: Check replicability
