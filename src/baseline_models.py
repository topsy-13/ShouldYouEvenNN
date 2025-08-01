import naiveautoml
import pandas as pd


def get_best_models(X, y, n_models=10, 
                    max_hpo_iterations=100, timeout=60, 
                    scoring_metric='neg_log_loss'):
    
    # Make X and y numpy arrays since they are tensors
    X = X.to_numpy()
    y = y.to_numpy()

    naml = naiveautoml.NaiveAutoML(timeout_candidate=timeout, max_hpo_iterations=max_hpo_iterations)
    naml.fit(X, y)
    scoreboard = naml.history
    scoreboard = scoreboard.sort_values(by=scoring_metric, ascending=False).head(n_models)
    scoreboard = scoreboard[['pipeline', scoring_metric]]

    return scoreboard


def get_baseline_metric(scoreboard: pd.DataFrame):
    baseline_metric = abs(scoreboard.iloc[:,1].mean())
    return baseline_metric


def get_models_and_baseline_metric(X, y, n_models=10, 
                                    max_hpo_iterations=100, timeout=60, 
                                    scoring_metric='neg_log_loss'):
    # Get the best models
    scoreboard = get_best_models(X, y, n_models, max_hpo_iterations, timeout, scoring_metric)
    
    # Get the baseline metric
    baseline_metric = get_baseline_metric(scoreboard)
    
    return baseline_metric