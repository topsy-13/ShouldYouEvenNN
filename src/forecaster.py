import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def rational_model(x, a, b):
    return (a * x) / (b + x)

def forecast_accuracy(x_values, accuracies, max_x=None, model_type='rational', degree=2):
    """
    Forecast accuracy given progress data.

    Parameters
    ----------
    x_values : list or array
        Monotonic increasing measure of effort (batches, instances, or cumulative wall time).
    accuracies : list or array
        Validation accuracies corresponding to x_values.
    max_x : float or int, optional
        Extrapolation point. Default = max observed x.
    model_type : str
        'linear', 'polynomial', 'sigmoid', 'rational'.
    degree : int
        Degree for polynomial fitting.
    """
    X = np.array(x_values).reshape(-1, 1)
    y = np.array(accuracies)

    if max_x is None:
        max_x = np.max(X)

    try:
        if (
            np.isnan(X.astype(float)).any() or 
            np.isinf(X.astype(float)).any() or 
            np.isnan(y.astype(float)).any() or 
            np.isinf(y.astype(float)).any()
        ):
            return None
    except (TypeError, ValueError):
        return None

    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        forecast = model.predict([[max_x]])[0]

    elif model_type == 'polynomial':
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        forecast = model.predict([[max_x]])[0]

    elif model_type == 'sigmoid':
        p0 = [1.0, 1.0, np.median(x_values)]
        try:
            popt, _ = curve_fit(sigmoid, X.flatten(), y, p0=p0, 
                                bounds=([0, 0, 0], [1.0, 10, np.inf]))
            forecast = sigmoid(max_x, *popt)
        except RuntimeError:
            forecast = y[-1]

    elif model_type == 'rational':
        try:
            popt, _ = curve_fit(
                rational_model,
                X.flatten(),
                y,
                bounds=([0.0, 0.01], [1.0, np.inf]),
                maxfev=10000
            )
            forecast = rational_model(max_x, *popt)
        except RuntimeError:
            forecast = y[-1]

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return float(np.clip(forecast, 0.0, 1.0))


# forecaster.py
def forecast_generation(candidates, dataset_size, 
                        min_val_points=3, 
                        growth=1.4, extra_full_passes=3):
    
    for cand in candidates.values():
        val_times, val_accs = get_val_acc_vs_time(cand)

        if len(val_times) == 0 or len(val_accs) == 0:
            cand.metrics["forecasted_val_acc"] = 0.0
            continue

        if len(val_accs) < min_val_points:
            last_val = val_accs[-1]
            cand.metrics["forecasted_val_acc"] = float(min(1.0, last_val + 0.15))
            continue

        T_future = project_future_time(cand, dataset_size, growth=growth, extra_full_passes=extra_full_passes)
        if T_future is None:
            cand.metrics["forecasted_val_acc"] = float(val_accs[-1])
            continue

        try:
            fc = forecast_accuracy(val_times, val_accs, max_x=T_future, model_type="rational")
        except Exception:
            fc = float(val_accs[-1])

        cand.metrics["forecast_horizon_time"] = T_future
        cand.metrics["forecasted_val_acc"] = float(np.clip(fc, 0.0, 1.0))



# EPS = 1e-8

# def sigmoid_prob(fcst, slope, var, goal, temp=0.05,
#                  slope_penalty_scale=5.0, var_penalty_scale=1.0):
#     margin = fcst - goal
#     slope_factor = np.exp(-max(0.0, slope) * slope_penalty_scale)
#     penalty = slope_penalty_scale * 0.1 * slope_factor + var_penalty_scale * var
#     adjusted_margin = margin - penalty
#     prob = 1.0 / (1.0 + np.exp(-adjusted_margin / (temp + EPS)))
#     return float(np.clip(prob, 0.0, 1.0))


# def mc_prob(fcst, var, goal, n_samples=500, min_std=1e-3):
#     std = max(min_std, np.sqrt(max(var, 0.0)))
#     samples = np.random.normal(loc=fcst, scale=std, size=n_samples)
#     return float(np.mean(samples > goal))

def annotate_probabilities(candidates, goal_metric, temp=0.05):
    """
    Assign probability of surpassing the goal using only the rational forecast.
    """
    for cand in candidates.values():
        fcst = cand.metrics.get("forecasted_val_acc", 0.0)

        # Margin over the baseline
        margin = fcst - goal_metric

        # Convert margin into probability
        prob = 1.0 / (1.0 + np.exp(-margin / (temp + 1e-8)))
        prob = float(np.clip(prob, 0.0, 1.0))

        cand.metrics["p_above_goal"] = prob


import numpy as np

def get_val_acc_vs_time(candidate):
    # Prefer the explicit cumulative list you added
    times = np.array(getattr(candidate, "cumulative_times", []), dtype=float)
    if times.size == 0:
        # Fallback to cumulative sum of efforts
        efforts = np.array(candidate.efforts or [], dtype=float)
        times = np.cumsum(efforts) if efforts.size else np.array([])

    if times.size == 0:
        return [], []

    val_accs = candidate.get_metric("val", "acc")
    if not val_accs:
        return [], []

    k = min(len(val_accs), times.size)
    return times[:k], val_accs[:k]

import math


def project_future_time(candidate, dataset_size, growth=1.4, extra_full_passes=3):
    """
    Returns an *absolute* time horizon (seconds) to forecast at:
    now + time_to_ramp_to_full + extra_full_passes * time_full_pass
    """
    bs = int(candidate.batch_size)
    batch_times = candidate.efforts or []
    if not batch_times or bs <= 0:
        return 'F'  # not enough info to project

    # robust per-batch estimate
    bt = float(np.median(batch_times))

    # current absolute time on the clock
    t_now = (candidate.cumulative_times[-1] 
             if getattr(candidate, "cumulative_times", None) 
             else float(np.sum(batch_times)))

    # ramp-to-full (one pass at each anchor)
    n = int(candidate.n_instances[-1])
    t_add = 0.0
    while n < dataset_size:
        batches = math.ceil(n / bs)
        t_add += batches * bt
        n = min(int(n * growth), dataset_size)

    # full pass at cap
    t_full = math.ceil(dataset_size / bs) * bt
    t_add += extra_full_passes * t_full

    return t_now + t_add
