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
    # Start time
    start_time = time.time()

    # Get the best models
    scoreboard, best_model = get_best_models(X, y, top_models=top_models, scoring_metric=scoring_metric, random_state=random_state, **kwargs)
    
    # Get the baseline metric
    baseline_metrics = get_baseline_metric(scoreboard, 
                                           strategy=strategy)
    
    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Filter out neural models
    scoreboard['pipeline'] = scoreboard['pipeline'].astype(str)
    scoreboard = scoreboard[~scoreboard['pipeline'].str.contains('MLP')]

    return baseline_metrics, elapsed_time, scoreboard, best_model


def hypothesis_testing(
    non_nn_scoreboard,
    nn_model_scoreboard,
    alpha=0.05,
    metric='final_val_acc',
    mode='superiority',
    equivalence_margin=0.01
):
    """
    Hypothesis testing comparing baseline vs neural network models.

    Modes:
        - 'superiority': NN significantly > baseline
        - 'non_inferiority': NN not worse than baseline
        - 'equivalence': NN and baseline are statistically equivalent 
                         within equivalence_margin (TOST)

    Args:
        scoreboard (pd.DataFrame): Baseline metrics (metric assumed in 2nd col).
        nn_model_scoreboard (pd.DataFrame): Neural net metrics.
        alpha (float): Significance level.
        metric (str): Column name with NN metrics.
        mode (str): 'superiority', 'non_inferiority', or 'equivalence'.
        equivalence_margin (float): Margin for equivalence testing (TOST).
    
    Returns:
        dict with results
    """
    import numpy as np
    from scipy import stats

    # Extract metrics
    baseline = non_nn_scoreboard.iloc[:, 1].to_numpy(dtype=float)
    nn = nn_model_scoreboard[metric].to_numpy(dtype=float)

    # Means & sizes
    mean_diff = np.mean(nn) - np.mean(baseline)
    nx, ny = len(baseline), len(nn)

    # Normality check (Shapiro breaks if n > 5000, so skip then)
    def is_normal(x):
        if len(x) < 5000:
            _, p = stats.shapiro(x)
            return p > 0.05
        return True  # large n -> CLT safety blanket

    normal = is_normal(baseline) and is_normal(nn)

    # Defaults
    t_stat, p_val = None, None
    significant = False
    test_used = None

    # --- Superiority ---
    if mode == 'superiority':
        if normal:
            test_used = "Welch's t-test"
            t_stat, p_val = stats.ttest_ind(nn, baseline, equal_var=False, alternative='greater')
        else:
            test_used = "Mann-Whitney U"
            t_stat, p_val = stats.mannwhitneyu(nn, baseline, alternative='greater')
        significant = p_val < alpha

    # --- Non-inferiority ---
    elif mode == 'non_inferiority':
        if normal:
            test_used = "Welch's t-test (non-inferiority)"
            # Test if NN < baseline (bad). If not significant, and mean_diff >= 0 → non-inferior
            t_stat, p_val = stats.ttest_ind(nn, baseline, equal_var=False, alternative='less')
        else:
            test_used = "Mann-Whitney U (non-inferiority)"
            t_stat, p_val = stats.mannwhitneyu(nn, baseline, alternative='less')
        significant = (mean_diff >= 0) and (p_val > alpha)

    # --- Equivalence (TOST) ---
    elif mode == 'equivalence':
        test_used = "TOST equivalence"
        if normal:
            se = np.sqrt(np.var(nn, ddof=1)/ny + np.var(baseline, ddof=1)/nx)
            df = min(nx, ny) - 1  # conservative
            t1 = (mean_diff - (-equivalence_margin)) / se
            p1 = 1 - stats.t.cdf(t1, df)
            t2 = (mean_diff - equivalence_margin) / se
            p2 = stats.t.cdf(t2, df)
            p_val = max(p1, p2)
            significant = (p1 < alpha) and (p2 < alpha)
        else:
            # Non-parametric "equivalence" is trickier. Simplify: CI overlap method.
            ci_low = np.percentile(nn - baseline.mean(), 2.5)
            ci_high = np.percentile(nn - baseline.mean(), 97.5)
            p_val = None
            significant = (ci_low > -equivalence_margin) and (ci_high < equivalence_margin)
            test_used += " (approx, bootstrap/percentile CI)"

    # Effect size
    pooled_sd = np.sqrt(((nx-1)*np.var(baseline, ddof=1) +
                         (ny-1)*np.var(nn, ddof=1)) / (nx+ny-2))
    cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

    return {
        "test": test_used,
        "mode": mode,
        "mean_diff": float(mean_diff),
        "p_value": None if p_val is None else float(p_val),
        "significant": significant,
        "baseline_mean": float(np.mean(baseline)),
        "nn_mean": float(np.mean(nn)),
        "effect_size_cohen_d": float(cohen_d),
        "normality_assumption": normal
    }


def hypothesis_report(results: dict, detail: str = "medium") -> str:
    """
    Build a human-readable report from hypothesis_testing results.
    Stepwise levels:
        short ⊂ medium ⊂ long

    Args:
        results (dict): Output from hypothesis_testing.
        detail (str): 'short', 'medium', or 'long'.

    Returns:
        str: Formatted report.
    """
    mode = results.get("mode", "unknown").replace("_", " ")
    verdict = "YES" if results["significant"] else "NO"
    baseline = results["baseline_mean"]
    nn = results["nn_mean"]
    diff = results["mean_diff"]
    pval = results.get("p_value")

    # --- Short ---
    if mode == "superiority":
        conclusion = (
            "NN models are statistically superior to the baseline."
            if results["significant"] else
            "NN models are not significantly better than the baseline."
        )
    elif mode == "non inferiority":
        conclusion = (
            "NN models are competitive (not worse than baseline)."
            if results["significant"] else
            "NN models may be worse than baseline."
        )
    elif mode == "equivalence":
        conclusion = (
            "NN models are statistically equivalent to baseline within margin."
            if results["significant"] else
            "NN models are not equivalent to baseline."
        )
    else:
        conclusion = "Test mode not recognized."

    short_str = f"[{mode.upper()}] Significant: {verdict} | {conclusion}"

    # --- Medium ---
    medium_str = (
        f"{short_str}\n"
        f"Hypothesis test ({mode}): {results['test']}\n"
        f"Baseline mean = {baseline:.4f}, NN mean = {nn:.4f}, "
        f"Mean diff = {diff:.4f}\n"
        f"p-value = {pval:.4g} | Significant: {verdict}"
    )

    # --- Long ---
    long_str = (
        f"{medium_str}\n"
        f"--- Detailed Stats ---\n"
        f"Effect Size (d)   : {results['effect_size_cohen_d']:.3f}\n"
        f"Normality Assumed : {results['normality_assumption']}"
    )

    if detail == "short":
        return short_str
    elif detail == "medium":
        return medium_str
    elif detail == "long":
        return long_str
    else:
        raise ValueError("detail must be 'short', 'medium', or 'long'")

def run_hypothesis_and_report(
    baseline_scores: list,
    nn_scores: list,
    mode: str = "non_inferiority",
    alpha: float = 0.05,
    margin: float = 0.01,
    detail: str = "medium",
    metric='final_val_acc'
) -> str:
    """
    Runs hypothesis_testing and builds a report in one step.

    Args:
        baseline_scores (list): Scores from baseline model.
        nn_scores (list): Scores from neural models.
        mode (str): "superiority", "non_inferiority", or "equivalence".
        alpha (float): Significance level.
        margin (float): Margin for equivalence / non-inferiority.
        detail (str): 'short', 'medium', or 'long' (report detail).

    Returns:
        str: Hypothesis test report.
    """
    # Run the statistical test
    results = hypothesis_testing(
        non_nn_scoreboard=baseline_scores,
        nn_model_scoreboard=nn_scores,
        alpha=alpha,
        mode=mode,
        equivalence_margin=margin,
        metric=metric,
    )

    # Build the report
    report = hypothesis_report(results, detail=detail)

    return report
