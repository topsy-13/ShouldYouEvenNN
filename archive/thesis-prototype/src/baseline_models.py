"""Lightweight wrappers around the ``naiveautoml`` baselines."""

from __future__ import annotations

import time
from typing import Literal, Optional, Tuple

import naiveautoml
import numpy as np
import pandas as pd
import torch

__all__ = [
    "get_best_models",
    "summarise_baseline",
    "get_models_and_baseline_metric",
]


def _to_numpy(array_like) -> np.ndarray:
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    if isinstance(array_like, (pd.DataFrame, pd.Series)):
        return array_like.to_numpy()
    if isinstance(array_like, np.ndarray):
        return array_like
    raise TypeError(f"Unsupported type: {type(array_like)!r}")


def get_best_models(
    X,
    y,
    *,
    top_models: Optional[int] = None,
    scoring_metric: str = "accuracy",
    random_state: int = 13,
    **kwargs,
) -> Tuple[pd.DataFrame, object]:
    """Fit ``NaiveAutoML`` and return its history alongside the chosen model."""

    X_np = _to_numpy(X)
    y_np = _to_numpy(y)

    automl = naiveautoml.NaiveAutoML(
        scoring=scoring_metric,
        random_state=random_state,
        max_hpo_iterations=0,
        **kwargs,
    )

    automl.fit(X_np, y_np)

    history = automl.history.sort_values(by=scoring_metric, ascending=False)
    if top_models is not None:
        history = history.head(top_models)

    return history[["pipeline", scoring_metric]], automl.chosen_model


def summarise_baseline(
    X,
    y,
    *,
    strategy: Literal["best", "worst", "mean", "median"] = "best",
    top_models: Optional[int] = None,
    scoring_metric: str = "accuracy",
    random_state: int = 13,
    **kwargs,
) -> Tuple[float, float, pd.DataFrame, object]:
    """Train baselines and return a summary suitable for reporting."""

    start = time.time()
    scoreboard, model = get_best_models(
        X,
        y,
        top_models=top_models,
        scoring_metric=scoring_metric,
        random_state=random_state,
        **kwargs,
    )
    elapsed = time.time() - start

    metric_column = scoreboard.columns[-1]
    series = scoreboard[metric_column]

    if strategy == "best":
        baseline = float(series.max())
    elif strategy == "worst":
        baseline = float(series.min())
    elif strategy == "mean":
        baseline = float(series.mean())
    elif strategy == "median":
        baseline = float(series.median())
    else:
        raise ValueError("strategy must be one of 'best', 'worst', 'mean', 'median'.")

    filtered = scoreboard.copy()
    filtered["pipeline"] = filtered["pipeline"].astype(str)
    filtered = filtered[~filtered["pipeline"].str.contains("MLP", na=False)]

    return baseline, elapsed, filtered, model


def get_models_and_baseline_metric(
    X,
    y,
    *,
    top_models: Optional[int] = None,
    scoring_metric: str = "accuracy",
    random_state: int = 13,
    strategy: Literal["best", "worst", "mean", "median"] = "best",
    **kwargs,
) -> Tuple[float, float, pd.DataFrame, object]:
    """Backward compatible alias used by experiments and notebooks."""

    return summarise_baseline(
        X,
        y,
        strategy=strategy,
        top_models=top_models,
        scoring_metric=scoring_metric,
        random_state=random_state,
        **kwargs,
    )

