import sys
import os
sys.path.append(os.path.abspath("./src"))
import data_preprocessing as dp

import time
import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis


def main(data_id=54, seed=13):
    # Load dataset
    exp_id = f'{data_id}_{seed}'

    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
            dataset_id=data_id,
            scaling=True,
            random_seed=seed,
            return_as='tensor',
            task_type='classification'
            )
    X_analysis = torch.cat([X_train, X_val], dim=0)
    y_analysis = torch.cat([y_train, y_val], dim=0)

    # Convert X
    if isinstance(X_analysis, torch.Tensor):
        X_analysis = X_analysis.detach().cpu().numpy()
        X_test = X_test.detach().cpu().numpy()

    # Convert y
    if isinstance(y_analysis, torch.Tensor):
        y_analysis = y_analysis.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()



    def run_standard_models(X_analysis, y_analysis, X_test, y_test):
        # Define candidate models
        models = {
                "LogisticRegression": LogisticRegression(max_iter=500),
                "SVC": SVC(),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "HistGradientBoosting": HistGradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(algorithm='SAMME'),
                "KNN": KNeighborsClassifier(),
                "NaiveBayes": GaussianNB(),
                "QDA": QuadraticDiscriminantAnalysis(),
                "LDA": LinearDiscriminantAnalysis(),
                }
        
        results = {}
        best_model = None
        best_acc = -np.inf
        hist_gradient_training_time = None
        # Timer for all models
        total_start_time = time.time()
        for name, model in models.items():
            start_time = time.time()
            model.fit(X_analysis, y_analysis)
            train_time = time.time() - start_time
            
            # Capture HistGradientBoosting training time specifically
            if name == "HistGradientBoosting":
                hist_gradient_training_time = train_time
                
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = {
                "test_acc": acc,
                "training_time_sec": train_time
            }
            if acc > best_acc:
                best_acc = acc
                best_model = name

        # Total time for all models
        total_time = time.time() - total_start_time
        
        # Build JSON summary
        summary = {
        "best_model": best_model,
        "best_accuracy": best_acc,
        "total_training_time_sec": total_time,
        "hist_gradient_boosting_training_time_sec": hist_gradient_training_time,
        "all_results": results
        }
        return summary

    summary = run_standard_models(X_analysis, y_analysis, 
                                  X_test, y_test)
    
    export_path = f'./experiments/ebe_vs/v2/standard'
    with open(f"{export_path}/{exp_id}_ML.json", "w") as f:
        json.dump(summary, f, indent=4)

    # print(json.dumps(summary, indent=4))
    print('  -Standard ML Models results exported')

if __name__ == "__main__":
    main(data_id=54, seed=13)