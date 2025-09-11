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
        max_hpo_iterations=0, # ! No Optimization
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

    start_time = time.time()
    # Get the best models
    scoreboard, best_model = get_best_models(X, y, top_models=top_models, scoring_metric=scoring_metric, random_state=random_state, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Get the baseline metric
    baseline_metrics = get_baseline_metric(scoreboard, 
                                           strategy=strategy)
    
    # Filter out neural models
    scoreboard['pipeline'] = scoreboard['pipeline'].astype(str)
    scoreboard = scoreboard[~scoreboard['pipeline'].str.contains('MLP')]

    return baseline_metrics, elapsed_time, scoreboard, best_model


