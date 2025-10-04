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



# def get_worst_individuals(population, baseline_metric,
#                           percentile_drop=15):
#     """
#     Drop the worst individuals based on p_above_goal.
#     Always preserve elites (top 10% by last_val_acc).
#     """
#     n_worst = max(1, int(population.size * percentile_drop / 100))
#     elite_count = max(1, int(0.1 * population.size))

#     # Gather candidates
#     candidates = []
#     for key, cand in population.candidates.items():
#         val_acc = cand.get_metric('val', 'acc', last_only=True) or 0.0
#         prob = cand.metrics.get("p_above_goal", 0.0)
#         candidates.append((key, val_acc, prob))

#     # Sort by probability ascending (lowest chance of beating baseline = worst)
#     sorted_by_prob = sorted(candidates, key=lambda x: x[2])

#     # Identify elites: top 10% by validation accuracy
#     elites = {
#         k for k, v, p in sorted(candidates, key=lambda x: x[1], reverse=True)[:elite_count]
#     }

#     # Collect worst individuals, skipping elites
#     worst = []
#     for k, v, p in sorted_by_prob:
#         if k not in elites and len(worst) < n_worst:
#             worst.append(k)

#     population.worst_individuals = worst

# def get_worst_individuals_hybrid(population,
#                                  base_drop=0.2,
#                                  max_drop=0.5):
#     """
#     Hybrid pruning for limited-time NAS with convex LB discard already active.

#     Steps:
#     1. Sort candidates by probability of beating baseline (descending).
#     2. Drop a percentile of the worst (percent grows with generations).
#     3. Always keep a small elite buffer by val_acc.
#     4. Let population shrink naturally — do not auto-respawn full size.
#     """

#     n = len(population.candidates)
#     if n <= 1:
#         population.worst_individuals = []
#         return

#     # --- adaptive drop fraction ---
#     # early gens: gentler, later gens: harsher
#     frac = min(max_drop, base_drop + 0.02 * population.generations_completed)
#     n_drop = max(1, int(n * frac))

#     # --- rank by prob ---
#     ranked = sorted(
#         population.candidates.items(),
#         key=lambda kv: kv[1].metrics.get("p_above_goal", 0.0),
#         reverse=True
#     )

#     # --- elite buffer (top 10% by val_acc) ---
#     elite_count = max(1, int(0.1 * n))
#     elites = {
#         k for k, c in sorted(
#             population.candidates.items(),
#             key=lambda kv: kv[1].get_metric("val", "acc", last_only=True) or 0.0,
#             reverse=True
#         )[:elite_count]
#     }

#     # survivors = top (n - n_drop) + elites
#     survivors = {k for k, _ in ranked[:n - n_drop]} | elites
#     worst = [k for k, _ in ranked if k not in survivors]

#     population.worst_individuals = worst

#     print(f"[Hybrid] Candidates={n}, drop={len(worst)}, keep={len(survivors)}")


def convex_lb_discard(candidate, goal, b_ref,
                      min_points=5, margin=0.02):
    """
    Softer convex LB discard using first and last val_acc points.
    - Needs at least min_points anchors.
    - Slope from very first to most recent accuracy, not just last 2.
    - Requires CI_high also below goal before discarding.
    """

    val_accs = candidate.get_metric("val", "acc")
    efforts = candidate.efforts

    if len(val_accs) < min_points or len(efforts) < min_points:
        return False  # too early

    # first and last points
    x1, x2 = efforts[0], efforts[-1]
    y1, y2 = val_accs[0], val_accs[-1]

    slope = (y2 - y1) / (x2 - x1 + 1e-8)
    best_case = y2 + slope * (b_ref - x2)

    ci_high = candidate.metrics.get("forecast_CI_high",
                                    candidate.metrics.get("forecasted_val_acc", y2))

    hopeless = (best_case < (goal - margin)) and (ci_high < goal)
    return hopeless
