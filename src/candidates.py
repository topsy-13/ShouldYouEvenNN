"""Candidate abstraction used by the evolutionary search."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

__all__ = ["Candidate"]


def _default_metrics() -> Dict[str, Any]:
    return {
        "train": {"loss": [], "acc": []},
        "val": {"loss": [], "acc": []},
        "test": {"acc": []},
        "forecasted_val_acc": 0.0,
        "score": 0.0,
        "fcst_greater_than_baseline": False,
    }


@dataclass
class Candidate:
    """Represent an individual model and its learning trajectory."""

    model: Any
    architecture: Mapping[str, Any]
    starting_instances: int = 100
    identifier: Optional[int] = None

    batch_size: Optional[int] = field(init=False)
    n_instances: List[int] = field(default_factory=list)
    epochs_trained: int = 0
    batches_trained: int = 0
    cumulative_times: List[float] = field(default_factory=list)
    efforts: List[float] = field(default_factory=list)
    metrics: MutableMapping[str, Any] = field(default_factory=_default_metrics)

    def __post_init__(self) -> None:
        self.id = self.identifier
        self.batch_size = int(self.architecture.get("batch_size", 0) or 0)
        self.n_instances.append(self.starting_instances)

    def log_effort(self, batch_time: float) -> None:
        """Record the time spent on a training batch."""

        self.efforts.append(batch_time)
        self.batches_trained += 1

        if self.cumulative_times:
            self.cumulative_times.append(self.cumulative_times[-1] + batch_time)
        else:
            self.cumulative_times.append(batch_time)

    def update_n_instances(self, n_instances: int) -> None:
        self.n_instances.append(n_instances)

    def add_metric(self, split: str, name: str, initial_value: Optional[Any] = None) -> None:
        """Create a new metric entry for the candidate."""

        if split in {"train", "val", "test"}:
            split_metrics = self.metrics.setdefault(split, {})
            if isinstance(split_metrics, dict):
                split_metrics[name] = [] if initial_value is None else initial_value
            else:
                self.metrics[split] = {name: [] if initial_value is None else initial_value}
        else:
            self.metrics[split] = initial_value if initial_value is not None else 0.0

    def log_metric(self, split: str, metric: Optional[str] = None, value: Any = None) -> None:
        """Append a value to a metric or set a scalar metric directly."""

        if metric is None:
            self.metrics[split] = value
            return

        container = self.metrics.setdefault(split, {})
        if not isinstance(container, dict):
            container = {}
            self.metrics[split] = container

        series = container.setdefault(metric, [])
        if isinstance(series, list):
            series.append(value)
        else:
            container[metric] = value

    def get_metric(self, split: str, metric: Optional[str] = None, *, last_only: bool = False):
        """Retrieve a metric value with optional ``last_only`` behaviour."""

        value = self.metrics.get(split)
        if isinstance(value, dict):
            if metric is None or metric not in value:
                raise KeyError(f"Metric '{metric}' not found in split '{split}'.")
            metric_values = value[metric]
            if isinstance(metric_values, list):
                return metric_values[-1] if last_only and metric_values else metric_values
            return metric_values
        if metric is not None:
            raise KeyError(f"Split '{split}' does not contain nested metrics.")
        return value

    def build_dict(self) -> Dict[str, Any]:
        """Return a flattened representation suitable for logging."""

        flat_arch = {f"arch_{key}": val for key, val in self.architecture.items()}
        flat_arch["arch_rng_state"] = str(self.architecture.get("rng_state"))

        flat_metrics: Dict[str, Any] = {}
        for key, value in self.metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_metrics[f"{key}_{sub_key}"] = sub_value
                    if isinstance(sub_value, list) and sub_value:
                        flat_metrics[f"last_{key}_{sub_key}"] = sub_value[-1]
            else:
                flat_metrics[key] = value

        if self.efforts:
            flat_metrics["total_batch_time"] = float(sum(self.efforts))
            flat_metrics["avg_batch_time"] = float(fmean(self.efforts))
        else:
            flat_metrics["total_batch_time"] = 0.0
            flat_metrics["avg_batch_time"] = 0.0

        return {
            "id": self.id,
            "batch_size": self.batch_size,
            "n_instances": self.n_instances,
            "epochs_trained": self.epochs_trained,
            "batches_trained": self.batches_trained,
            "efforts": self.efforts,
            "cumulative_times": self.cumulative_times,
            **flat_arch,
            **flat_metrics,
        }

    def next_anchor(self, *, growth: float = 1.4, max_cap: Optional[int] = None) -> None:
        """Expand the number of instances processed in the next round."""

        previous = self.n_instances[-1]
        next_value = int(previous * growth)
        if max_cap is not None:
            next_value = min(next_value, max_cap)
        self.update_n_instances(next_value)

    def __str__(self) -> str:
        train_loss = self.get_metric("train", "loss", last_only=True)
        train_acc = self.get_metric("train", "acc", last_only=True)
        val_loss = self.get_metric("val", "loss", last_only=True)
        val_acc = self.get_metric("val", "acc", last_only=True)
        test_acc = None
        try:
            test_acc = self.get_metric("test", "acc", last_only=True)
        except KeyError:
            pass
        score = self.metrics.get("score")

        arch_summary = ", ".join(
            f"{key}={value}" for key, value in self.architecture.items() if key != "layers"
        )

        return (
            "Candidate(\n"
            f"  Arch: {arch_summary}\n"
            f"  Train: loss={train_loss}, acc={train_acc}\n"
            f"  Val:   loss={val_loss}, acc={val_acc}\n"
            f"  Test:  acc={test_acc}\n"
            f"  Score: {score}\n"
            ")"
        )

