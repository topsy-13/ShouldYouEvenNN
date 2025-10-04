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

# endregion


def main(data_id, seed, 
        X_analysis, y_analysis, 
        X_test, y_test):
    
    exp_id = f'{data_id}_{seed}'
    # print('Testing Standard MLP experiment...')
    mlp_results = {
        # 'seed': seed,
        # 'data_id': data_id
    }
    
    mlp_start_time = time.time()
    mlp = MLPClassifier(random_state=seed)
    # defaults: hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200

    mlp.fit(X_analysis, y_analysis) # ? Analysis or training data
    mlp_results['mlp_training_time'] = time.time() - mlp_start_time
    
    # Predict
    y_pred_train = mlp.predict(X_analysis)
    y_pred_test = mlp.predict(X_test)
    # y_pred_test = mlp.predict(X_test)

    # Results
    mlp_results['mlp_train_acc'] = accuracy_score(y_analysis, y_pred_train)
    mlp_results['mlp_train_loss'] = mlp.loss_
    # mlp_results['val_acc'] = accuracy_score(y_val, y_pred_val)
    mlp_results['mlp_test_acc'] = accuracy_score(y_test, y_pred_test)

    # mlp_path = f'./experiments/ebe_vs/v3'
    # with open(f"{mlp_path}/{exp_id}_MLP.json", 'w') as json_file:
    #     json.dump(mlp_results, json_file, indent=4)
    
    # print('  -MLP results exported')
    return mlp_results

# if __name__ == "__main__":
#     main(data_id=54, seed=13)    
