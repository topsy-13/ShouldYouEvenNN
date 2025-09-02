# region Imports
import os
import sys
import json

import time
import torch
import pandas as pd

sys.path.append(os.path.abspath("./src"))

import data_preprocessing as dp

from instance_sampling import create_dataloaders
from architecture_generator import create_model_from_row
from utils import set_seed
# endregion

# region Oracle Training
def main(
         data_id=54, seed=13, 
         n_individuals=100, 
         n_max_epochs=1000,
         es_patience=150):

    # region Set the scenario
    DATA_ID = data_id
    SEED = seed
    N_MAX_EPOCHS = n_max_epochs
    ES_PATIENCE = es_patience

    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=DATA_ID,
        scaling=True,
        random_seed=SEED,
        return_as='tensor',
        task_type='classification'
        )
    exp_id = f'{data_id}_{seed}'
    
    input_size, output_size = dp.get_tensor_sizes(X_train, y_train)

    
    print('Starting Oracle Training...')
    oracle_start_time = time.time()

    # Load the ebe models explored
    ebe_history = pd.read_csv(f'./experiments/ebe_vs/ebe-models/{exp_id}_EBE-history.csv')

    oracle_performance = {
        'seed': seed,
        'data_id': data_id
    }
    oracle_df = []
    model_ids = ebe_history['id'].unique().tolist()
    for n_model, model_id in enumerate(model_ids):
        print(f'Training model {n_model + 1} by ES',
              'of', len(model_ids))
        # Build from 0 the model
        model_row = ebe_history[ebe_history['id'] == model_id].iloc[0].copy()
        model_seed = int(model_row.get('arch_seed', None))
        model_batch_size = int(model_row.get('batch_size', None))
        
        train_loader = create_dataloaders(X=X_train, y=y_train, 
                            batch_size=model_batch_size)
        val_loader = create_dataloaders(X=X_val, y=y_val, 
                            batch_size=model_batch_size)
        test_loader = create_dataloaders(X=X_test, y=y_test, 
                            batch_size=model_batch_size)
        
        model = create_model_from_row(model_row, input_size, output_size)

        # Set seed for reproducibility
        set_seed(model_seed)

        # Train the model
        train_start = time.time()
        es_metrics = model.es_train(train_loader=train_loader, val_loader=val_loader,
                    es_patience=ES_PATIENCE, # epochs without improvement
                    max_epochs=N_MAX_EPOCHS, # cap for epochs
                    verbose=False, # print training progress
        )
        train_time = time.time() - train_start
        train_loss, train_acc, val_loss, val_acc  = es_metrics
        # Record the results
        model_row['es_train_loss'] = train_loss        
        model_row['es_val_loss'] = val_loss        
        model_row['es_train_acc'] = train_acc        
        model_row['es_val_acc'] = val_acc  
        model_row['es_time'] = train_time

        # Evaluate on test set
        es_test_loss, es_test_acc = model.evaluate(test_loader) 
        model_row['es_test_loss'] = es_test_loss
        model_row['es_test_acc'] = es_test_acc

        oracle_df.append(model_row)
    
    oracle_df = pd.DataFrame(oracle_df)
    oracle_performance['total_time'] = time.time() - oracle_start_time
    oracle_performance['max_train_acc'] = oracle_df['es_train_acc'].max()
    oracle_performance['min_train_loss'] = oracle_df['es_train_loss'].min()
    oracle_performance['max_val_acc'] = oracle_df['es_val_acc'].max()
    oracle_performance['min_val_loss'] = oracle_df['es_val_loss'].min()
    oracle_performance['max_test_acc'] = oracle_df['es_test_acc'].max()
    oracle_performance['min_test_loss'] = oracle_df['es_test_loss'].min()
    
    # Export as json and dataframe
    export_path = './experiments/ebe_vs/oracle'
    oracle_df.to_csv(f'{export_path}/{exp_id}_Oracle_all.csv', index=False)

    with open(f"{export_path}/{exp_id}_Oracle-summary.json", 'w') as json_file:
        json.dump(oracle_performance, json_file, indent=4)



# endregion
if __name__ == "__main__":
    main(data_id=54, seed=13)
    


    # TODO: Check replicability
