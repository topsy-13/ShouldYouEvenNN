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
    margin = fcst - goal
    slope_factor = np.exp(-max(0.0, slope) * slope_penalty_scale)
    penalty = slope_penalty_scale * 0.1 * slope_factor + var_penalty_scale * var
    adjusted_margin = margin - penalty
    prob = 1.0 / (1.0 + np.exp(-adjusted_margin / (temp + EPS)))
    return float(np.clip(prob, 0.0, 1.0))


def mc_prob(fcst, var, goal, n_samples=500, min_std=1e-3):
    std = max(min_std, np.sqrt(max(var, 0.0)))
    samples = np.random.normal(loc=fcst, scale=std, size=n_samples)
    return float(np.mean(samples > goal))


def compute_p_above_goal(candidate, goal,
                         alpha=0.7, temp=0.05,
                         mc_samples=500):
    fcst = candidate.metrics.get("forecasted_val_acc", 0.0)
    slope = candidate.metrics.get("slope_val_acc", 0.0)
    var = candidate.metrics.get("var_val_acc", 0.0)

    s_prob = sigmoid_prob(fcst, slope, var, goal, temp=temp)
    m_prob = mc_prob(fcst, var, goal, n_samples=mc_samples)
    p = alpha * m_prob + (1.0 - alpha) * s_prob
    return float(np.clip(p, 0.0, 1.0))


def score_individuals(candidates, baseline_metric,
                      rescale_method="minmax", temperature=1.0):
    """
    Score individuals using their pre-computed probability (p_above_goal).
    Rescales across the population so evolution has selection pressure.
    """
    # Gather probabilities
    keys = list(candidates.keys())
    probs = np.array([candidates[k].metrics.get("p_above_goal", 0.0) for k in keys], dtype=float)

    # Rescale
    if rescale_method == "softmax":
        probs_clipped = np.clip(probs, 1e-9, 1 - 1e-9)
        logits = np.log(probs_clipped / (1 - probs_clipped))
        scaled = logits / max(1e-6, temperature)
        exps = np.exp(scaled - np.max(scaled))
        scores = exps / (exps.sum() + EPS)

    elif rescale_method == "minmax":
        lo, hi = probs.min(), probs.max()
        if hi - lo < 1e-12:
            scores = np.ones_like(probs) / len(probs)
        else:
            scaled = (probs - lo) / (hi - lo)
            scores = scaled / (scaled.sum() + EPS)

    elif rescale_method == "rank":
        ranks = np.argsort(np.argsort(-probs)).astype(float)
        scores = 1.0 - (ranks / max(1.0, (len(probs)-1)))
        scores = scores / (scores.sum() + EPS)

    else:
        raise ValueError("Unknown rescale method")

    # Assign back
    for i, key in enumerate(keys):
        candidates[key].metrics["score"] = float(scores[i])


def get_worst_individuals(population, baseline_metric,
                          percentile_drop=15):
    """
    Drop the worst individuals based on p_above_goal.
    Always preserve elites (top 10% by last_val_acc).
    """
    n_worst = max(1, int(population.size * percentile_drop / 100))
    elite_count = max(1, int(0.1 * population.size))

    # Gather candidates
    candidates = []
    for key, cand in population.candidates.items():
        val_acc = cand.get_metric('val', 'acc', last_only=True) or 0.0
        prob = cand.metrics.get("p_above_goal", 0.0)
        candidates.append((key, val_acc, prob))

    # Sort by probability ascending (lowest chance of beating baseline = worst)
    sorted_by_prob = sorted(candidates, key=lambda x: x[2])

    # Identify elites: top 10% by validation accuracy
    elites = {
        k for k, v, p in sorted(candidates, key=lambda x: x[1], reverse=True)[:elite_count]
    }

    # Collect worst individuals, skipping elites
    worst = []
    for k, v, p in sorted_by_prob:
        if k not in elites and len(worst) < n_worst:
            worst.append(k)

    population.worst_individuals = worst
