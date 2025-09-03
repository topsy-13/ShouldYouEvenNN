import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def rational_model(x, a, b):
    return (a * x) / (b + x)

def forecast_accuracy(efforts, accuracies, max_effort=300, model_type='rational', degree=2):
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


def forecast_generation(candidates, effort_threshold=3):
    
    active_individuals = candidates.keys()
    for i in active_individuals:
        candidate = candidates[i]
        efforts = candidate.efforts
        val_accs = candidate.get_metric('val', 'acc')

        if candidate.epochs_trained >= effort_threshold and len(val_accs) >= effort_threshold:
            # Existing rational forecast
            forecasted_accuracy = forecast_accuracy(efforts, val_accs, model_type='rational')
            
            # New lightweight extras
            slope = (val_accs[-1] - val_accs[0]) / max(1e-6, efforts[-1] - efforts[0])
            variance = float(np.var(val_accs))
            last_gap = val_accs[-1] - np.mean(val_accs[:-1]) if len(val_accs) > 1 else 0.0

            candidate.metrics["forecasted_val_acc"] = forecasted_accuracy
            candidate.metrics["slope_val_acc"] = slope
            candidate.metrics["var_val_acc"] = variance
            candidate.metrics["gap_val_acc"] = last_gap
        else: 
            pass  # Not enough data to forecast
