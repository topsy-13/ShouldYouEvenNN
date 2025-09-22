import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def rational_model(x, a, b):
    return (a * x) / (b + x)

def forecast_accuracy(efforts, accuracies, max_effort=1000, model_type='rational', degree=2):
    X = np.array(efforts).reshape(-1, 1)
    y = np.array(accuracies)

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
        forecast = model.predict([[max_effort]])[0]

    elif model_type == 'polynomial':
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        forecast = model.predict([[max_effort]])[0]

    elif model_type == 'sigmoid':
        p0 = [1.0, 1.0, np.median(efforts)]
        try:
            popt, _ = curve_fit(sigmoid, X.flatten(), y, p0=p0, bounds=([0, 0, 0], [1.0, 10, np.inf]))
            forecast = sigmoid(max_effort, *popt)
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
            forecast = rational_model(max_effort, *popt)
        except RuntimeError:
            forecast = y[-1]

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return float(np.clip(forecast, 0.0, 1.0))


def forecast_generation(candidates, effort_threshold=3,
                        method='rational', min_epochs_for_rational=5,
                        total_batches=1000):
    """
    Generate optimistic forecasts for all candidates.
    - Very forgiving with minimal data.
    - Boosts forecasts aggressively in early batches.
    - Always favors upward trajectories unless model is flatlining badly.
    """

    for i, candidate in candidates.items():
        efforts = candidate.efforts
        val_accs = candidate.get_metric('val', 'acc')

        if not efforts or not val_accs:
            continue

        # Early skip threshold
        if candidate.epochs_trained < effort_threshold or len(val_accs) < effort_threshold:
            # With ultra minimal data, enforce an optimistic prior
            last_val = val_accs[-1]
            forecasted_accuracy = min(1.0, last_val + 0.15)  # "hopeful bump"
            candidate.metrics["forecasted_val_acc"] = forecasted_accuracy
            candidate.metrics["slope_val_acc"] = 0.0
            candidate.metrics["var_val_acc"] = 0.0
            candidate.metrics["gap_val_acc"] = 0.0
            continue

        # --- Slope forecast ---
        slope = (val_accs[-1] - val_accs[0]) / max(1e-6, efforts[-1] - efforts[0])
        slope_boost = np.log1p((total_batches / max(1, efforts[-1])))  # scale slope optimism
        slope_fcst = float(np.clip(val_accs[-1] + slope * (max(efforts) * slope_boost), 0.0, 1.0))

        variance = float(np.var(val_accs))
        last_gap = val_accs[-1] - np.mean(val_accs[:-1]) if len(val_accs) > 1 else 0.0

        # --- Rational forecast ---
        rational_fcst = None
        if len(val_accs) >= min_epochs_for_rational:
            try:
                rational_fcst = forecast_accuracy(efforts, val_accs, model_type='rational')
            except Exception:
                rational_fcst = None

        # --- Linear forecast ---
        linear_fcst = None
        try:
            linear_fcst = forecast_accuracy(efforts, val_accs, model_type='linear')
        except Exception:
            linear_fcst = None

        # --- Ensemble optimistic rule ---
        forecasts = [fc for fc in [slope_fcst, rational_fcst, linear_fcst] if fc is not None]
        base_fcst = max(forecasts) if forecasts else val_accs[-1]

        # --- Optimism boost ---
        batches_seen = efforts[-1]
        optimism_boost = (1 - batches_seen / total_batches) * 0.25  # bigger boost early
        optimistic_fcst = min(1.0, base_fcst + optimism_boost)

        # Guarantee at least +0.15 over last val unless clearly collapsing
        final_fcst = max(optimistic_fcst, val_accs[-1] + 0.15)
        final_fcst = float(np.clip(final_fcst, 0.0, 1.0))

        candidate.metrics["forecasted_val_acc"] = final_fcst
        candidate.metrics["slope_val_acc"] = slope
        candidate.metrics["var_val_acc"] = variance
        candidate.metrics["gap_val_acc"] = last_gap
