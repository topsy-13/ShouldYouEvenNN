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


def _prepare_series(x_values: Sequence[float], accuracies: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert input sequences to float arrays and validate them."""
    try:
        X = np.asarray(x_values, dtype=float).reshape(-1, 1)
        y = np.asarray(accuracies, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Input sequences must be numeric") from None

    if X.size == 0 or y.size == 0:
        raise ValueError("Input sequences cannot be empty")

    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("Input sequences must contain finite values")

    return X, y


def _forecast_linear(X: np.ndarray, y: np.ndarray, target: float, **_: object) -> float:
    model = LinearRegression()
    model.fit(X, y)
    return float(model.predict([[target]])[0])


def _forecast_polynomial(
    X: np.ndarray,
    y: np.ndarray,
    target: float,
    *,
    degree: int = 2,
    **_: object,
) -> float:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    return float(model.predict([[target]])[0])


def _forecast_sigmoid(X: np.ndarray, y: np.ndarray, target: float, **_: object) -> float:
    p0 = [1.0, 1.0, float(np.median(X))]
    popt, _ = curve_fit(sigmoid, X.flatten(), y, p0=p0, bounds=([0, 0, 0], [1.0, 10, np.inf]))
    return float(sigmoid(target, *popt))


def _forecast_rational(X: np.ndarray, y: np.ndarray, target: float, **_: object) -> float:
    popt, _ = curve_fit(
        rational_model,
        X.flatten(),
        y,
        bounds=([0.0, 0.01], [1.0, np.inf]),
        maxfev=10000,
    )
    return float(rational_model(target, *popt))


FORECASTERS: Dict[str, Callable[..., float]] = {
    "linear": _forecast_linear,
    "polynomial": _forecast_polynomial,
    "sigmoid": _forecast_sigmoid,
    "rational": _forecast_rational,
}


def forecast_accuracy(
    x_values: Sequence[float],
    accuracies: Sequence[float],
    max_x: Optional[float] = None,
    *,
    model_type: str = "rational",
    degree: int = 2,
) -> Optional[float]:
    """Forecast the final accuracy given early learning curve samples."""

    try:
        X, y = _prepare_series(x_values, accuracies)
    except ValueError:
        return None

    target = float(max_x) if max_x is not None else float(np.max(X))

    forecaster = FORECASTERS.get(model_type)
    if forecaster is None:
        raise ValueError(f"Unsupported model_type: {model_type}")

    try:
        forecast = forecaster(X, y, target, degree=degree)
    except RuntimeError:
        forecast = float(y[-1])

    return float(np.clip(forecast, 0.0, 1.0))


def _early_exit_forecast(val_accs: Sequence[float]) -> float:
    return float(np.clip(val_accs[-1], 0.0, 1.0))


def forecast_generation(
    candidates: Mapping[str, Any],
    dataset_size: int,
    *,
    min_val_points: int = 3,
    growth: float = 1.4,
    extra_full_passes: int = 3,
) -> None:
    """Annotate each candidate with a forecasted validation accuracy."""

    for cand in candidates.values():
        val_times, val_accs = get_val_acc_vs_time(cand)

        if len(val_times) == 0 or len(val_accs) == 0:
            cand.metrics["forecasted_val_acc"] = 0.0
            continue

        if len(val_accs) < min_val_points:
            cand.metrics["forecasted_val_acc"] = float(min(1.0, val_accs[-1] + 0.15))
            continue

        horizon = project_future_time(
            cand,
            dataset_size,
            growth=growth,
            extra_full_passes=extra_full_passes,
        )

        if horizon is None:
            cand.metrics["forecasted_val_acc"] = _early_exit_forecast(val_accs)
            continue

        forecast = forecast_accuracy(
            val_times,
            val_accs,
            max_x=horizon,
            model_type="rational",
        )

        if forecast is None:
            forecast = _early_exit_forecast(val_accs)

        cand.metrics["forecast_horizon_time"] = horizon
        cand.metrics["forecasted_val_acc"] = forecast


def annotate_probabilities(candidates, goal_metric, temp=0.05):
    """Assign probability of surpassing the goal using the rational forecast."""

    for cand in candidates.values():
        fcst = cand.metrics.get("forecasted_val_acc", 0.0)

        margin = fcst - goal_metric
        prob = 1.0 / (1.0 + np.exp(-margin / (temp + 1e-8)))
        cand.metrics["p_above_goal"] = float(np.clip(prob, 0.0, 1.0))


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

    batch_size = int(getattr(candidate, "batch_size", 0) or 0)
    batch_times = np.asarray(candidate.efforts or [], dtype=float)
    if batch_times.size == 0 or batch_size <= 0:
        return None

    bt = float(np.median(batch_times))

    cumulative_times = getattr(candidate, "cumulative_times", None)
    if cumulative_times:
        t_now = float(cumulative_times[-1])
    else:
        t_now = float(batch_times.sum())

    instance_history = getattr(candidate, "n_instances", None) or []
    if not instance_history:
        return None

    n = int(instance_history[-1])
    t_add = 0.0
    while n < dataset_size:
        batches = math.ceil(n / batch_size)
        t_add += batches * bt
        n = min(int(n * growth), dataset_size)

    t_full = math.ceil(dataset_size / batch_size) * bt
    t_add += extra_full_passes * t_full

    return t_now + t_add
