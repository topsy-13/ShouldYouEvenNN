import naiveautoml
import pandas as pd
import torch
import numpy as np
import time

def get_best_models(X, y, 
                    n_models=30, 
                    max_hpo_iterations=100, 
                    timeout=60, 
                    scoring_metric='accuracy'):
    
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
        timeout_candidate=timeout, 
        max_hpo_iterations=max_hpo_iterations,
        scoring=scoring_metric
    )
    naml.fit(X, y)

    scoreboard = naml.history
    # print(scoreboard)

    scoreboard = scoreboard.sort_values(by=scoring_metric, ascending=False).head(n_models)
    scoreboard = scoreboard[['pipeline', scoring_metric]]

    return scoreboard


def get_baseline_metric(scoreboard: pd.DataFrame):
    baseline_metrics = scoreboard.iloc[:,1].to_list()
    return baseline_metrics


def get_models_and_baseline_metric(X, y, n_models=10, 
                                    max_hpo_iterations=100, timeout=60, 
                                    scoring_metric='accuracy'):
    # Start time
    start_time = time.time()

    # Get the best models
    scoreboard = get_best_models(X, y, n_models, max_hpo_iterations, timeout, scoring_metric)
    
    # Get the baseline metric
    baseline_metrics = get_baseline_metric(scoreboard)
    
    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time

    return baseline_metrics, elapsed_time, scoreboard


def hypothesis_testing(baseline_metrics, 
                       nn_model_metrics, 
                       alpha=0.05):
    """
    Perform hypothesis testing to compare baseline metrics with nn model metrics.
    
    Args:
        baseline_metrics (list): List of baseline metrics.
        nn_model_metrics (list): List of nn model metrics.
        alpha (float): Significance level for the test.
        
    Returns:
        bool: True if the new model is significantly better than the baseline, False otherwise.
    """
    from scipy import stats
    
    t_statistic, p_value = stats.ttest_ind(baseline_metrics, nn_model_metrics)
    
    return p_value < alpha