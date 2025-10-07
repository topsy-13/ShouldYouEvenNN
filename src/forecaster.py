import math
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


def rational_model(x, a, b):
    return (a * x) / (b + x)

def forecast_accuracy(x_values, accuracies, max_x=None, model_type='sigmoid', degree=2):
    """
    Forecast accuracy given progress data with pessimism-aware adjustments.
    Uses sigmoid fit plus conservative blending to avoid runaway optimism.
    """

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from scipy.optimize import curve_fit

    def sigmoid(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def rational_model(x, a, b):
        return (a * x) / (b + x)

    X = np.array(x_values).reshape(-1, 1)
    y = np.array(accuracies)

    if max_x is None:
        max_x = np.max(X)

    if len(y) < 2:
        return float(y[-1]) if len(y) else 0.0

    # --- base forecast ---
    try:
        if model_type == 'sigmoid':
            p0 = [1.0, 1.0, np.median(x_values)]
            popt, _ = curve_fit(sigmoid, X.flatten(), y, p0=p0,
                                bounds=([0, 0, 0], [1.0, 10, np.inf]))
            raw_fcst = sigmoid(max_x, *popt)
        elif model_type == 'rational':
            popt, _ = curve_fit(
                rational_model,
                X.flatten(), y,
                bounds=([0.0, 0.01], [1.0, np.inf]),
                maxfev=10000
            )
            raw_fcst = rational_model(max_x, *popt)
        elif model_type == 'linear':
            model = LinearRegression().fit(X, y)
            raw_fcst = model.predict([[max_x]])[0]
        elif model_type == 'polynomial':
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(X, y)
            raw_fcst = model.predict([[max_x]])[0]
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    except Exception:
        raw_fcst = y[-1]

    # --- pessimism-aware adjustments ---
    last_val = y[-1]

    # 1. Ensemble with last observed
    alpha = 0.7
    blended = alpha * raw_fcst + (1 - alpha) * last_val

    # 2. Slope-aware penalty (if curve flattening, downscale optimism)
    dx = X[-1] - X[-2] + 1e-8
    slope = (y[-1] - y[-2]) / dx
    penalty = np.exp(-5 * max(0, slope))  # flat slope → heavier discount
    forecast = blended * penalty + last_val * (1 - penalty)

    return float(np.clip(forecast, 0.0, 1.0))

def forecast_with_ci(x_values, accuracies, max_x=None,
                     model_type='rational', degree=2, alpha=0.05):
    """
    Forecast validation accuracy with a more stable confidence interval.
    Uses residual-based uncertainty and falls back gracefully if fitting fails.
    Returns (forecast_mean, lower, upper).
    """
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.stats import t
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    # --- sanity checks ---
    X = np.array(x_values, dtype=float).reshape(-1, 1)
    y = np.array(accuracies, dtype=float)
    if len(y) < 2:
        last = float(y[-1]) if len(y) else 0.0
        return last, last * 0.95, min(1.0, last * 1.05)

    if max_x is None:
        max_x = float(np.max(X))

    # --- smoothing for noisy curves ---
    if len(y) > 3:
        y = np.convolve(y, np.ones(3) / 3, mode='same')

    # --- define models ---
    def sigmoid(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def rational_model(x, a, b):
        return (a * x) / (b + x)

    # --- fit and forecast ---
    forecast = None
    std_err = 0.05
    try:
        if model_type == 'rational':
            popt, _ = curve_fit(rational_model, X.flatten(), y,
                                bounds=([0.0, 0.001], [1.0, np.inf]),
                                maxfev=20000)
            forecast = rational_model(max_x, *popt)
            residuals = y - rational_model(X.flatten(), *popt)
        elif model_type == 'sigmoid':
            p0 = [1.0, 1.0, np.median(x_values)]
            popt, _ = curve_fit(sigmoid, X.flatten(), y, p0=p0,
                                bounds=([0, 0, 0], [1.0, 10, np.inf]),
                                maxfev=20000)
            forecast = sigmoid(max_x, *popt)
            residuals = y - sigmoid(X.flatten(), *popt)
        elif model_type == 'linear':
            model = LinearRegression().fit(X, y)
            forecast = model.predict([[max_x]])[0]
            residuals = y - model.predict(X)
        elif model_type == 'polynomial':
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(X, y)
            forecast = model.predict([[max_x]])[0]
            residuals = y - model.predict(X)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        std_err = np.std(residuals)
    except Exception:
        # fallback: simple slope extrapolation
        slope = (y[-1] - y[-2]) / (X[-1] - X[-2] + 1e-8)
        forecast = y[-1] + slope * (max_x - X[-1])
        std_err = abs(slope) * 0.1

    # --- CI from residual variance ---
    dof = max(1, len(y) - 1)
    tval = t.ppf(1 - alpha / 2, dof)
    ci_range = tval * std_err
    lower = float(np.clip(forecast - ci_range, 0.0, 1.0))
    upper = float(np.clip(forecast + ci_range, 0.0, 1.0))
    forecast = float(np.clip(forecast, 0.0, 1.0))

    return forecast, lower, upper



def forecast_generation(candidates, dataset_size, 
                        min_val_points=5, extra_full_passes=10):

    for cand in candidates.values():
        val_times, val_accs = get_val_acc_vs_time(cand)

        if len(val_times) == 0 or len(val_accs) == 0:
            cand.metrics["forecasted_val_acc"] = 0.0
            # print(f"[DEBUG] Candidate {cand.id}: empty validation curve.")
            continue

        if len(val_accs) < min_val_points:
            last_val = val_accs[-1]
            fc = float(min(1.0, last_val + 0.05))
            cand.metrics["forecasted_val_acc"] = fc
            # print(f"[DEBUG] Candidate {cand.id}: too few points ({len(val_accs)}), "
                #   f"default forecast={fc:.4f}")
            continue

        T_future = project_future_time(cand, dataset_size, extra_full_passes=extra_full_passes)
        cand.metrics["forecast_horizon_time"] = T_future         # legacy (seconds)
        cand.metrics["forecast_horizon_effort"] = T_future       # new deterministic axis

        
        if T_future is None:
            fc = float(val_accs[-1])
            cand.metrics["forecasted_val_acc"] = fc
            # print(f"[DEBUG] Candidate {cand.id}: missing T_future, using last val={fc:.4f}")
            continue

        try:
            fc, lo, hi = forecast_with_ci(val_times, val_accs, max_x=T_future, model_type="rational")
            cand.metrics["forecasted_val_acc"] = fc
            cand.metrics["forecast_CI_low"]  = lo
            cand.metrics["forecast_CI_high"] = hi
        except Exception as e:
            print(f"[DEBUG][ERROR] Candidate {cand.id}: forecast_with_ci failed ({e})")
            fc = float(val_accs[-1])
            cand.metrics["forecasted_val_acc"] = float(np.clip(fc, 0.0, 1.0))
            cand.metrics["forecast_CI_low"]  = None
            cand.metrics["forecast_CI_high"] = None

        # --- DEBUG: track forecast CI creation ---
        fcst_val = cand.metrics.get("forecasted_val_acc")
        ci_low   = cand.metrics.get("forecast_CI_low")
        ci_high  = cand.metrics.get("forecast_CI_high")
        if ci_low is None or ci_high is None:
            pass
            # print(f"[DEBUG][forecast_generation] Candidate {cand.id}: "
            #       f"forecast={fcst_val:.4f}, CI missing ({ci_low}, {ci_high})")
        else:
            # print(f"[DEBUG][forecast_generation] Candidate {cand.id}: "
            #       f"forecast={fcst_val:.4f}, CI=({ci_low:.4f}, {ci_high:.4f})")
            pass


def get_val_acc_vs_time(candidate):
    # prefer deterministic effort
    if getattr(candidate, "cumulative_effort", None):
        xs = np.asarray(candidate.cumulative_effort, dtype=float)
    # elif getattr(candidate, "cumulative_times", None):
    #     xs = np.asarray(candidate.cumulative_times, dtype=float)
    else:
        effs = np.asarray(candidate.efforts or [], dtype=float)
        xs = np.cumsum(effs) if effs.size else np.array([])

    if xs.size == 0:
        return [], []

    val_accs = candidate.get_metric("val", "acc") or []
    if not val_accs:
        return [], []

    k = min(len(val_accs), xs.size)
    return xs[:k], val_accs[:k]




def project_future_time(candidate, dataset_size, extra_full_passes=10):
    """
    Returns effort horizon (in mini-batches), not seconds:
      now + N passes of max(current_passed, one_full_dataset_pass)
    """
    import math
    bs = int(candidate.batch_size)
    if bs <= 0:
        return None

    # batches seen so far (deterministic)
    e_now = int(candidate.cumulative_effort[-1]) if getattr(candidate, "cumulative_effort", None) else 0

    # batches per full pass over dataset
    batches_per_dataset = math.ceil(dataset_size / bs)

    # conservative: future = N * max(already_seen, one_full_pass)
    e_future = extra_full_passes * max(e_now, batches_per_dataset)

    return e_now + e_future
