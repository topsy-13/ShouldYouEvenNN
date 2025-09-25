"""Scoring utilities to prioritise promising candidates."""

from __future__ import annotations

from typing import Mapping

import numpy as np

EPSILON = 1e-8

__all__ = [
    "check_higher_than_baseline",
    "sigmoid_prob",
    "mc_prob",
    "compute_p_above_goal",
    "compute_and_log_p_above_goal",
    "score_individuals",
    "convex_lb_discard",
]


def check_higher_than_baseline(candidates: Mapping, baseline_metric: float) -> None:
    """Annotate whether each candidate beats the provided baseline."""

    for candidate in candidates.values():
        forecast = candidate.get_metric("forecasted_val_acc") or 0.0
        candidate.log_metric("fcst_greater_than_baseline", value=forecast >= baseline_metric)


def sigmoid_prob(
    forecast: float,
    slope: float,
    variance: float,
    goal: float,
    *,
    temp: float = 0.05,
    slope_penalty_scale: float = 5.0,
    variance_penalty_scale: float = 1.0,
) -> float:
    """Convert the margin to a probability using a penalised sigmoid."""

    margin = forecast - goal
    slope_factor = np.exp(-max(0.0, slope) * slope_penalty_scale)
    penalty = slope_penalty_scale * 0.1 * slope_factor + variance_penalty_scale * variance
    adjusted = margin - penalty
    prob = 1.0 / (1.0 + np.exp(-adjusted / (temp + EPSILON)))
    return float(np.clip(prob, 0.0, 1.0))


def mc_prob(
    forecast: float,
    variance: float,
    goal: float,
    *,
    n_samples: int = 500,
    min_std: float = 1e-3,
) -> float:
    """Estimate the probability via Monte Carlo sampling."""

    std = max(min_std, np.sqrt(max(variance, 0.0)))
    samples = np.random.normal(loc=forecast, scale=std, size=n_samples)
    return float(np.mean(samples > goal))


def compute_p_above_goal(
    candidate,
    goal: float,
    *,
    alpha: float = 0.7,
    temp: float = 0.05,
    mc_samples: int = 500,
) -> float:
    """Blend sigmoid and Monte Carlo probabilities for robustness."""

    forecast = candidate.metrics.get("forecasted_val_acc", 0.0)
    slope = candidate.metrics.get("slope_val_acc", 0.0)
    variance = candidate.metrics.get("var_val_acc", 0.02)

    sigmoid_component = sigmoid_prob(forecast, slope, variance, goal, temp=temp)
    monte_carlo_component = mc_prob(forecast, variance, goal, n_samples=mc_samples)

    combined = alpha * monte_carlo_component + (1.0 - alpha) * sigmoid_component
    return float(np.clip(combined, 0.0, 1.0))


def compute_and_log_p_above_goal(
    candidates: Mapping,
    goal_metric: float,
    *,
    alpha: float = 0.7,
    temp: float = 0.05,
    mc_samples: int = 300,
) -> None:
    """Populate ``p_above_goal`` for each candidate."""

    for candidate in candidates.values():
        probability = compute_p_above_goal(
            candidate,
            goal_metric,
            alpha=alpha,
            temp=temp,
            mc_samples=mc_samples,
        )
        candidate.metrics["p_above_goal"] = probability


def score_individuals(candidates: Mapping) -> None:
    """Normalise probabilities into a score distribution."""

    keys = list(candidates.keys())
    probs = np.array([candidates[key].metrics.get("p_above_goal", 0.0) for key in keys], dtype=float)

    if probs.sum() <= 0:
        scores = np.ones_like(probs) / len(probs)
    else:
        scores = probs / probs.sum()

    for key, score in zip(keys, scores):
        candidates[key].metrics["score"] = float(score)


def convex_lb_discard(candidate, goal: float, ref_budget: float) -> bool:
    """Decide whether to discard a candidate using a convex lower-bound rule."""

    efforts = candidate.efforts
    val_accs = candidate.get_metric("val", "acc")

    if not val_accs or len(val_accs) < 2 or len(efforts) < 2:
        return False

    x1, x2 = efforts[-2], efforts[-1]
    y1, y2 = val_accs[-2], val_accs[-1]

    slope = (y2 - y1) / (x2 - x1 + EPSILON)
    best_case = y2 + slope * (ref_budget - x2)
    return best_case < goal

