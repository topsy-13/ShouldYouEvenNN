import naiveautoml
import pandas as pd
import torch
import numpy as np
import time

def get_best_models(X, y, 
                    top_models=None, 
                    scoring_metric='accuracy',
                    random_state=13, 
                    **kwargs):
    
    # Convert X
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    elif not isinstance(X, np.ndarray):
        raise TypeError(f"Unsupported type for X: {type(X)}")

    # Convert y
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    elif isinstance(y, pd.Series):
        y = y.to_numpy()
    elif not isinstance(y, np.ndarray):
        raise TypeError(f"Unsupported type for y: {type(y)}")

    naml = naiveautoml.NaiveAutoML(
        scoring=scoring_metric, 
        random_state=random_state,
        **kwargs
    )
    naml.fit(X, y)

    scoreboard = naml.history
    # print(scoreboard)

    if top_models is None:
        scoreboard = scoreboard.sort_values(by=scoring_metric, ascending=False)
    else:
        scoreboard = scoreboard.sort_values(by=scoring_metric, ascending=False).head(top_models)

    scoreboard = scoreboard[['pipeline', scoring_metric]]
    best_model = naml.chosen_model

    return scoreboard, best_model


def get_baseline_metric(scoreboard: pd.DataFrame, strategy: str='best'):
    if strategy == 'best':
        baseline_metrics = scoreboard.iloc[:,1].max()
    elif strategy == 'worst':
        baseline_metrics = scoreboard.iloc[:,1].min()
    elif strategy == 'mean':
        baseline_metrics = scoreboard.iloc[:,1].mean()
    elif strategy == 'median':
        baseline_metrics = scoreboard.iloc[:,1].median()

    return baseline_metrics


def get_models_and_baseline_metric(X, y, top_models=None, 
                                    scoring_metric='accuracy', random_state=13,
                                    strategy='best',
                                    **kwargs
                                    ):
    # Start time
    start_time = time.time()

    # Get the best models
    scoreboard, best_model = get_best_models(X, y, top_models=top_models, scoring_metric=scoring_metric, random_state=random_state, **kwargs)
    
    # Get the baseline metric
    baseline_metrics = get_baseline_metric(scoreboard, 
                                           strategy=strategy)
    
    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Filter out neural models
    scoreboard['pipeline'] = scoreboard['pipeline'].astype(str)
    scoreboard = scoreboard[~scoreboard['pipeline'].str.contains('MLP')]

    return baseline_metrics, elapsed_time, scoreboard, best_model



from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

def run_standard_models(X_analysis, y_analysis, X_test, y_test):

    # Define candidate models
    models = {
            "LogisticRegression": LogisticRegression(max_iter=500),
            "SVC": SVC(),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "KNN": KNeighborsClassifier(),
            "NaiveBayes": GaussianNB(),
            "QDA": QuadraticDiscriminantAnalysis(),
            "LDA": LinearDiscriminantAnalysis(),
            }

    results = {}
    best_model = None
    best_acc = -np.inf

    # Timer for all models
    total_start_time = time.time()

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_analysis, y_analysis)
        train_time = time.time() - start_time

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
    "all_results": results
    }
    return summary