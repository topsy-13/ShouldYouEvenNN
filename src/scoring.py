
def check_higher_than_baseline(candidates, baseline_metric):
    active_individuals = candidates.keys()
    for i in active_individuals:
        candidate = candidates[i]
        last_fcst_acc = candidate.get_metric("forecasted_val_acc") or 0.0
        
        candidate.log_metric("fcst_greater_than_baseline", value=last_fcst_acc >= baseline_metric)


def score_individuals(candidates, baseline_metric):
        check_higher_than_baseline(candidates, baseline_metric)
        active_individuals = candidates.keys()
        for i in active_individuals:
            candidate = candidates[i]
            last_val_acc = candidate.get_metric('val', 'acc', last_only=True) or 0.0
            last_fcst_acc = candidate.get_metric("forecasted_val_acc") or 0.0
            slope = candidate.metrics.get("slope_val_acc", 0.0)
            variance = candidate.metrics.get("var_val_acc", 0.0)
            gap = candidate.metrics.get("gap_val_acc", 0.0)

            if last_fcst_acc < baseline_metric:
                # Below baseline → check momentum
                if slope > 0.01 and gap > 0:  # improving fast enough
                    score = 0.3 * last_val_acc + 0.5 * last_fcst_acc + 0.2 * slope
                else:
                    score = max(last_val_acc, 0.0)
            else:
                # Beating baseline forecast
                fcst_gain = last_fcst_acc - baseline_metric
                score = 0.6 * last_fcst_acc + 0.3 * fcst_gain + 0.1 * slope

            candidate.log_metric('score', value=score)


def get_worst_individuals(population, baseline_metric, 
                          percentile_drop=15):
    """
    Identify worst individuals to drop.

    Rules:
    - If forecasts exist, drop those below baseline first.
    - If no forecasts, rank by raw val_acc.
    - Always preserve top 10% (elites).
    """
    n_worst = max(1, int(population.size * percentile_drop / 100))
    elite_count = max(1, int(0.1 * population.size))  # preserve 10%

    # Build list of candidates
    candidates = []
    for key, cand in population.candidates.items():
        val_acc = cand.get_metric('val', 'acc', last_only=True)
        fcst_acc = cand.metrics.get("forecasted_val_acc", None)
        score = cand.metrics.get("score", None)
        candidates.append((key, val_acc, fcst_acc, score))

    # Separate forecasted vs non-forecasted
    with_fcst = [c for c in candidates if c[2] is not None]
    without_fcst = [c for c in candidates if c[2] is None]

    # Case 1: forecasts exist
    if with_fcst:
        below_baseline = [k for k, v, f, s in with_fcst if f < (baseline_metric or 0)]
        sorted_all = sorted(candidates, key=lambda x: (x[3] if x[3] is not None else 0))
    # Case 2: no forecasts → fallback to val_acc
    else:
        below_baseline = []
        sorted_all = sorted(candidates, key=lambda x: (x[1] if x[1] is not None else 0))

    # Identify elites (top 10% by val_acc)
    elites = {k for k, v, f, s in sorted(candidates, key=lambda x: (x[1] or 0), reverse=True)[:elite_count]}

    # Build worst list
    worst = []
    # Drop below-baseline first
    for k in below_baseline:
        if k not in elites and len(worst) < n_worst:
            worst.append(k)
    # Fill rest with lowest scorers
    for k, v, f, s in sorted_all:
        if k not in elites and k not in worst and len(worst) < n_worst:
            worst.append(k)

    population.worst_individuals = worst
