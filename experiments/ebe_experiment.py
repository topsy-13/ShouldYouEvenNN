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
# endregion

# region EBE 
def main(data_id=54, seed=13, naive_models=50, 
         strategy='mean', n_individuals=100, 
         n_max_epochs=100, n_top_models_to_train=10, 
         budget_factor=3, percentile_drop=15):
    
    # region Set the scenario
    DATA_ID = data_id
    SEED = seed
    NAIVE_MODELS = naive_models  #! for naiveautoml
    STRATEGY = strategy # Mean as baseline
    BUDGET_FACTOR = budget_factor  # EBE gets MULTIPLIED the time of the baseline
    N_INDIVIDUALS = n_individuals  # Individuals per generation
    N_MAX_EPOCHS = n_max_epochs  # Max epochs for EBE, honestly not that important
    N_TOP_MODELS_TO_TRAIN = n_top_models_to_train # For full training after EBE
    PERCENTILE_DROP = percentile_drop # Starting point to drop 

    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=DATA_ID,
        scaling=True,
        random_seed=SEED,
        return_as='tensor',
        task_type='classification'
        )
    


# endregion

