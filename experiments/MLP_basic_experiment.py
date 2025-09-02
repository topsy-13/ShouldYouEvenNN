# region Imports
import os
import sys
import json
import pandas as pd

import torch
import time

sys.path.append(os.path.abspath("./src"))

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import data_preprocessing as dp
from utils import set_seed

# endregion


def main(data_id, seed):
    exp_id = f'{data_id}_{seed}'

    mlp_results = {
        'seed': seed,
        'data_id': data_id
    }

    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=data_id,
        scaling=True,
        random_seed=seed,
        return_as='tensor',
        task_type='classification'
        )

    X_analysis = torch.cat([X_train, X_val], dim=0)
    y_analysis = torch.cat([y_train, y_val], dim=0)
    
    mlp_start_time = time.time()
    mlp = MLPClassifier(random_state=seed, max_iter=1000, n_iter_no_change=100)
    # defaults: hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200

    mlp.fit(X_train, y_train) # ? Analysis or training data
    
    # Predict
    y_pred_train = mlp.predict(X_train)
    y_pred_val = mlp.predict(X_val)
    y_pred_test = mlp.predict(X_test)

    # Results
    mlp_results['time_taken'] = time.time() - mlp_start_time
    mlp_results['train_acc'] = accuracy_score(y_train, y_pred_train)
    mlp_results['val_acc'] = accuracy_score(y_val, y_pred_val)
    mlp_results['test_acc'] = accuracy_score(y_test, y_pred_test)
    mlp_results['train_loss'] = mlp.loss_

    mlp_path = f'./experiments/ebe_vs/standard_MLP'
    with open(f"{mlp_path}/{exp_id}_MLP.json", 'w') as json_file:
        json.dump(mlp_results, json_file, indent=4)
    
    print('MLP results exported')


if __name__ == "__main__":
    main(data_id=54, seed=13)    
