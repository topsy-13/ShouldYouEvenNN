# import numpy as np

def check_higher_than_baseline(candidates, baseline_metric):
    active_individuals = candidates.keys()
    for i in active_individuals:
        candidate = candidates[i]
        last_fcst_acc = candidate.get_metric("forecasted_val_acc") or 0.0
        
        candidate.log_metric("fcst_greater_than_baseline", value=last_fcst_acc >= baseline_metric)

import numpy as np

EPS = 1e-8

def sigmoid_prob(fcst, slope, var, goal, temp=0.05,
                 slope_penalty_scale=5.0, var_penalty_scale=1.0):
    """
    Convert margin into probability with slope/variance penalties.
    """
    margin = fcst - goal
    slope_factor = np.exp(-max(0.0, slope) * slope_penalty_scale)
    penalty = slope_penalty_scale * 0.1 * slope_factor + var_penalty_scale * var
    adjusted_margin = margin - penalty
    prob = 1.0 / (1.0 + np.exp(-adjusted_margin / (temp + EPS)))
    return float(np.clip(prob, 0.0, 1.0))


def mc_prob(fcst, var, goal, n_samples=500, min_std=1e-3):
    """
    Monte Carlo probability: sample from Normal(fcst, var).
    """
    std = max(min_std, np.sqrt(max(var, 0.0)))
    samples = np.random.normal(loc=fcst, scale=std, size=n_samples)
    return float(np.mean(samples > goal))


def compute_p_above_goal(candidate, goal,
                         alpha=0.7, temp=0.05,
                         mc_samples=500):
    """
    Compute probability that candidate beats the goal.
    - alpha: weight for Monte Carlo vs sigmoid.
    """
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
        p = compute_p_above_goal(cand, goal_metric,
                                 alpha=alpha, temp=temp,
                                 mc_samples=mc_samples)
        cand.metrics["p_above_goal"] = p


def score_individuals(candidates):
    """
    Score individuals directly from their probability of beating the baseline.
    Probabilities are normalized to sum = 1.
    """
    keys = list(candidates.keys())
    probs = np.array(
        [candidates[k].metrics.get("p_above_goal", 0.0) for k in keys],
        dtype=float
    )

    # Avoid all-zero case
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


def convex_lb_discard(candidate, goal, b_ref):
    """
    Convex Lower-Bound Discard Rule.
    
    Idea:
    - Learning curves are usually |monotone and convex (improve quickly, then flatten).
    - Use the last two observed anchors (effort, val_acc) to draw a straight line (secant).
    - That line represents the *best-case extension* of current progress.
    - If even this optimistic extrapolation at reference budget (b_ref) 
      cannot beat the goal, then discard the candidate early.

    Args:
        candidate (Candidate): Individual model with effort and val_acc history.
        goal (float): Target accuracy to beat (e.g. baseline or incumbent).
        b_ref (int): Reference effort/budget to project to.

    Returns:
        bool: True if candidate should be discarded, False otherwise.
    """
    efforts = candidate.efforts
    val_accs = candidate.get_metric("val", "acc")

    # need at least 2 anchors to compute slope
    if not val_accs or len(val_accs) < 2 or len(efforts) < 2:
        return False

    x1, x2 = efforts[-2], efforts[-1]
    y1, y2 = val_accs[-2], val_accs[-1]

    slope = (y2 - y1) / (x2 - x1 + 1e-8)
    best_case = y2 + slope * (b_ref - x2)
    return best_case < goal
