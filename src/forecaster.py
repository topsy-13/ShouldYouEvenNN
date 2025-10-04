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
    Forecast accuracy with confidence interval.
    Returns (forecast_mean, lower, upper).
    """
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.stats import t

    X = np.array(x_values).reshape(-1, 1)
    y = np.array(accuracies)

    if max_x is None:
        max_x = np.max(X)

    # pick model
    if model_type == 'linear':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        forecast = model.predict([[max_x]])[0]
        # crude std
        residuals = y - model.predict(X)
        std_err = np.std(residuals)
    elif model_type == 'rational':
        popt, pcov = curve_fit(rational_model, X.flatten(), y,
                               bounds=([0.0, 0.01], [1.0, np.inf]),
                               maxfev=10000)
        forecast = rational_model(max_x, *popt)
        perr = np.sqrt(np.diag(pcov))
        std_err = np.max(perr)
    elif model_type == 'sigmoid':
        p0 = [1.0, 1.0, np.median(x_values)]
        popt, pcov = curve_fit(sigmoid, X.flatten(), y, p0=p0,
                               bounds=([0, 0, 0], [1.0, 10, np.inf]))
        forecast = sigmoid(max_x, *popt)
        perr = np.sqrt(np.diag(pcov))
        std_err = np.max(perr)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # CI from t-distribution
    dof = max(1, len(y) - 1)
    tval = t.ppf(1 - alpha/2, dof)
    lower = forecast - tval * std_err
    upper = forecast + tval * std_err

    return float(np.clip(forecast, 0, 1)), float(np.clip(lower, 0, 1)), float(np.clip(upper, 0, 1))


def forecast_generation(candidates, dataset_size, 
                        min_val_points=5, extra_full_passes=10):
    
    for cand in candidates.values():
        val_times, val_accs = get_val_acc_vs_time(cand)

        if len(val_times) == 0 or len(val_accs) == 0:
            cand.metrics["forecasted_val_acc"] = 0.0
            continue

        if len(val_accs) < min_val_points:
            last_val = val_accs[-1]
            cand.metrics["forecasted_val_acc"] = float(min(1.0, last_val + 0.05))

            continue

        T_future = project_future_time(cand, dataset_size, extra_full_passes=extra_full_passes)
        cand.metrics["forecast_horizon_time"] = T_future
        if T_future is None:
            cand.metrics["forecasted_val_acc"] = float(val_accs[-1])
            continue

        try:
            fc, lo, hi = forecast_with_ci(val_times, val_accs, max_x=T_future, model_type="rational")
            cand.metrics["forecasted_val_acc"] = fc 
            cand.metrics["forecast_CI_low"] = lo
            cand.metrics["forecast_CI_high"] = hi

        except Exception:
            fc = float(val_accs[-1])
            cand.metrics["forecasted_val_acc"] = float(np.clip(fc, 0.0, 1.0))


def get_val_acc_vs_time(candidate) -> tuple[np.ndarray, Sequence[float]]:
    """Return validation accuracies aligned with their timestamps."""

    cumulative_times = getattr(candidate, "cumulative_times", None)
    if cumulative_times:
        times = np.asarray(cumulative_times, dtype=float)
    else:
        efforts = np.asarray(candidate.efforts or [], dtype=float)
        times = np.cumsum(efforts) if efforts.size else np.array([])

    if times.size == 0:
        return [], []

    val_accs = candidate.get_metric("val", "acc") or []
    if not val_accs:
        return [], []

    k = min(len(val_accs), times.size)
    return times[:k], val_accs[:k]


def project_future_time(
    candidate,
    dataset_size: int,
    *,
    growth: float = 1.4,
    extra_full_passes: int = 3,
) -> Optional[float]:
    """Estimate the absolute time horizon used for forecasting."""

def project_future_time(candidate, dataset_size, extra_full_passes=10):
    """
    Returns absolute time horizon (seconds) to forecast at:
    now + N passes of the cumulative anchor history
    OR N passes of the full dataset — whichever is larger.
    """
    bs = int(candidate.batch_size)
    batch_times = candidate.efforts or []
    if not batch_times or bs <= 0:
        return None

    bt = float(np.median(batch_times))
    t_now = candidate.cumulative_times[-1] if candidate.cumulative_times else float(np.sum(batch_times))

    # total seen over *all anchors*
    n_total_seen = sum(candidate.n_instances)
    t_passed = math.ceil(n_total_seen / bs) * bt

    t_dataset = math.ceil(dataset_size / bs) * bt

    t_future = extra_full_passes * max(t_passed, t_dataset)

    # print(f"Candidate {candidate.id}: n_total_seen={n_total_seen}, last_seen={candidate.n_instances[-1]}, bs={bs}, bt={bt:.2f}, "
    #       f"t_now={t_now:.2f}, t_passed={t_passed:.2f}, t_dataset={t_dataset:.2f}, t_future={t_future:.2f}")
    return t_now + t_future
