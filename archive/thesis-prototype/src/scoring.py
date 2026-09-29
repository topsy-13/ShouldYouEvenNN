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

    fcst = candidate.metrics.get("forecasted_val_acc", 0.0)
    slope = candidate.metrics.get("slope_val_acc", 0.0)
    var = candidate.metrics.get("var_val_acc", 0.02)

    s_prob = sigmoid_prob(fcst, slope, var, goal, temp=temp)
    m_prob = mc_prob(fcst, var, goal, n_samples=mc_samples)
    p = alpha * m_prob + (1.0 - alpha) * s_prob
    return float(np.clip(p, 0.0, 1.0))


def compute_and_log_p_above_goal(candidates,
                                 goal_metric,
                                 alpha=0.7, temp=0.05,
                                 mc_samples=300):
    """
    For each candidate, compute probability of beating the goal
    and log it into candidate.metrics["p_above_goal"].
    """
    for cand in candidates.values():
        # --- use CI high instead of mean forecast ---
        fc_high = cand.metrics.get("forecast_CI_high",
                                cand.metrics.get("forecasted_val_acc", 0.0))
        cand.metrics["fcst_used_for_prob"] = fc_high  # log for debugging

        # run probability computation but override forecast
        last_val = cand.get_metric("val", "acc", last_only=True) or 0.0

        # optimistic Monte Carlo + sigmoid, anchored at CI_high
        slope = cand.metrics.get("slope_val_acc", 0.0)
        var   = cand.metrics.get("var_val_acc", 0.02)

        s_prob = sigmoid_prob(fc_high, slope, var, goal_metric, temp=temp)
        m_prob = mc_prob(fc_high, var, goal_metric, n_samples=mc_samples)
        p_base = alpha * m_prob + (1.0 - alpha) * s_prob

        # --- blend with observed accuracy ---
        p_blend = 0.6 * p_base + 0.4 * last_val

        # --- apply floor to avoid starving decent candidates ---
        p_final = float(np.clip(max(p_blend, 0.05), 0.0, 1.0))

        cand.metrics["p_above_goal"] = p_final


def score_individuals(candidates):
    """
    Score individuals directly from their probability of beating the baseline.
    Probabilities are normalized to sum = 1.
    """
    keys = list(candidates.keys())
    probs = np.array([candidates[key].metrics.get("p_above_goal", 0.0) for key in keys], dtype=float)

    if probs.sum() <= 0:
        scores = np.ones_like(probs) / len(probs)
    else:
        scores = probs / probs.sum()

    # Assign back
    for i, key in enumerate(keys):
        candidates[key].metrics["score"] = float(scores[i])



def convex_lb_discard(candidate, goal, b_ref,
                      min_points=5, tail_points=3, margin=0.02):
    """
    Simplified convex lower-bound discard:
    - Uses recent trend (last few validation points) to estimate slope.
    - Discards if even an optimistic linear extrapolation to full budget
      stays below the baseline goal by a margin.
    """
    import numpy as np

    val_accs = candidate.get_metric("val", "acc")
    efforts = candidate.efforts

    # Too few observations → can't decide yet
    if len(val_accs) < min_points or len(efforts) < min_points:
        return False

    # Tail slope from last few points
    tail_y = np.array(val_accs[-tail_points:], float)
    tail_x = np.array(efforts[-tail_points:], float)
    slope = np.mean(np.diff(tail_y) / (np.diff(tail_x) + 1e-8))

    # Extrapolate optimistically to the full reference budget
    y_last, x_last = val_accs[-1], efforts[-1]
    best_case = y_last + slope * (b_ref - x_last)

    # Decide: if best-case is still below goal (with margin), discard
    hopeless = best_case < (goal - margin)
    return bool(hopeless)
