
import numpy as np


def forecast_with_ci(
    x_values,
    accuracies,
    max_x=None,
    model_type="rational",
    degree=2,
    alpha=0.05,
    debug=False,
):
    """
    Conservative yet adaptive forecast of validation accuracy with CI.
    - Supports rational, sigmoid, linear, polynomial fits.
    - Penalises overconfident upward slopes when data are scarce.
    - Returns (forecast_mean, lower_CI, upper_CI).
    """
    from scipy.optimize import curve_fit
    from scipy.stats import t
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    X = np.array(x_values, dtype=float).reshape(-1, 1)
    y = np.array(accuracies, dtype=float)
    n = len(y)

    if n < 2:
        last = float(y[-1]) if n else 0.0
        return last, last * 0.95, min(1.0, last * 1.05)

    if max_x is None:
        max_x = float(np.max(X))

    # --- models ---
    def sigmoid(x, L, k, x0): return L / (1 + np.exp(-k * (x - x0)))
    def rational(x, a, b):    return (a * x) / (b + x)

    # --- fit & forecast ---
    try:
        if model_type == "rational":
            popt, _ = curve_fit(
                rational, X.flatten(), y,
                bounds=([0, 1e-4], [1, np.inf]), maxfev=20000
            )
            y_pred = rational(X.flatten(), *popt)
            forecast = rational(max_x, *popt)

        elif model_type == "sigmoid":
            p0 = [1.0, 1.0, np.median(x_values)]
            popt, _ = curve_fit(
                sigmoid, X.flatten(), y, p0=p0,
                bounds=([0, 0, 0], [1.0, 10, np.inf]), maxfev=20000
            )
            y_pred = sigmoid(X.flatten(), *popt)
            forecast = sigmoid(max_x, *popt)

        elif model_type == "linear":
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            forecast = model.predict([[max_x]])[0]

        elif model_type == "polynomial":
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(X, y)
            y_pred = model.predict(X)
            forecast = model.predict([[max_x]])[0]

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        residuals = y - y_pred
        std_err = np.std(residuals)

    except Exception as e:
        # fallback: simple linear extrapolation
        slope = (y[-1] - y[-2]) / (X[-1] - X[-2] + 1e-8)
        forecast = y[-1] + slope * (max_x - X[-1])
        std_err = abs(slope) * 0.1
        if debug:
            print(f"[forecast_with_ci] Fallback used: {e}")

    # --- adaptive slope moderation ---
    slope = (y[-1] - y[-2]) / (X[-1] - X[-2] + 1e-8)
    if slope > 0:
        # damp optimism depending on history length
        optimism = np.exp(-0.4 * n)
        forecast -= optimism * 0.2 * (forecast - y[-1])
    else:
        # tiny uplift if recovering from dip
        forecast += 0.1 * abs(slope) * np.exp(-n / 5)

    # --- CI calculation ---
    dof = max(1, n - 1)
    tval = t.ppf(1 - alpha / 2, dof)
    ci = tval * (std_err + 1e-8)
    lower = float(np.clip(forecast - ci, 0, 1))
    upper = float(np.clip(forecast + ci, 0, 1))
    forecast = float(np.clip(forecast, 0, 1))

    if debug:
        print(f"[forecast_with_ci] model={model_type}, n={n}, slope={slope:.4f}, "
              f"forecast={forecast:.3f}, CI=({lower:.3f}, {upper:.3f})")

    return forecast, lower, upper



def forecast_generation(candidates, dataset_size, 
                        min_val_points=5, extra_full_passes=5):

    for cand in candidates.values():
        val_times, val_accs = get_val_acc_vs_effort(cand)

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
            fc, lo, hi = forecast_with_ci_morphing(val_times, val_accs, max_x=T_future)

            cand.metrics["forecasted_val_acc"] = fc
            cand.metrics["forecast_CI_low"]  = lo
            cand.metrics["forecast_CI_high"] = hi
        except Exception as e:
            print(f"[DEBUG][ERROR] Candidate {cand.id}: forecast_with_ci failed ({e})")
            fc = float(val_accs[-1])
            cand.metrics["forecasted_val_acc"] = float(np.clip(fc, 0.0, 1.0))
            cand.metrics["forecast_CI_low"]  = None
            cand.metrics["forecast_CI_high"] = None


def get_val_acc_vs_effort(candidate):
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




def project_future_time(candidate, dataset_size, extra_full_passes=50):
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

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t

def forecast_with_ci_morphing(
    x_values,
    accuracies,
    max_x=None,
    alpha=0.05,
    debug=False,
):
    """
    Forecasts validation accuracy using an Inverse Power Law:
        f(x) = a - b / (x + c)^d

    Returns:
        forecast_mean, lower_CI, upper_CI

    Behavior:
        - Handles short curves robustly.
        - Provides asymptotic (limit) forecast 'a'.
        - Computes confidence interval using residual variance.
    """

    x = np.array(x_values, dtype=float)
    y = np.array(accuracies, dtype=float)
    n = len(y)

    # Fallback for very short curves
    if n < 2:
        last = float(y[-1]) if n else 0.0
        return last, max(0.0, last * 0.95), min(1.0, last * 1.05)

    if max_x is None:
        max_x = float(np.max(x))

    # Define inverse power law
    def inv_power(x, a, b, c, d):
        return a - b / np.power(x + c, d)

    # Parameter bounds to keep things stable
    bounds = (
        [0.0, 0.0, 0.0, 0.1],   # lower bounds
        [1.0, 1.0, 1e3, 10.0],  # upper bounds
    )

    # Initial guess: asymptote near last value, small b, gentle slope
    p0 = [min(1.0, y[-1] + 0.05), 0.5, 1.0, 1.0]

    try:
        popt, pcov = curve_fit(inv_power, x, y, p0=p0, bounds=bounds, maxfev=20000)
        forecast = float(np.clip(inv_power(max_x, *popt), 0.0, 1.0))
        y_pred = inv_power(x, *popt)
        residuals = y - y_pred
        std_err = np.std(residuals)
    except Exception as e:
        if debug:
            print(f"[InversePowerForecast] Fallback linear used: {e}")
        # fallback: naive extrapolation
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2] + 1e-8)
        forecast = y[-1] + slope * (max_x - x[-1])
        std_err = abs(slope) * 0.1

    # Confidence interval
    dof = max(1, n - 1)
    tval = t.ppf(1 - alpha / 2, dof)
    ci = tval * (std_err + 1e-8)

    lower = float(np.clip(forecast - ci, 0, 1))
    upper = float(np.clip(forecast + ci, 0, 1))

    if debug:
        print(f"[InversePowerForecast] n={n}, forecast={forecast:.3f}, "
              f"CI=({lower:.3f}, {upper:.3f})")

    return forecast, lower, upper

