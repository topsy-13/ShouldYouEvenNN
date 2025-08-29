import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# region Individuals
class Candidate:
    def __init__(self, model, architecture, 
                 starting_instances=100, id_counter=None):
        self.id = id_counter
        self.model = model
        self.architecture = architecture
        self.batch_size = architecture.get("batch_size")
        self.n_instances = [starting_instances]
        self.epochs_trained = 0
        # self.proportion_instances = []
        self.efforts = []
        self.metrics = {
            "train": {"loss": [], "acc": []},
            "val": {"loss": [], "acc": []},
            "test": {"loss": [], "acc": []},
            "forecasted_val_acc": 0.0,
            "score": 0.0,
            "forecast_gain": 0.0,
            "fcst_greater_than_baseline": False
        }
        


    def update_n_instances(self, n_instances):
        self.n_instances.append(n_instances)
  

    def add_metric(self, split, name, initial_value=None):
        """
        Add a new metric dynamically.
        split: "train", "val", "test" or None (for global metrics).
        name: metric name.
        initial_value: starting value (list or scalar).
        """
        if split in ["train", "val", "test"]:
            self.metrics[split][name] = initial_value if initial_value is not None else []
        else:
            self.metrics[name] = initial_value if initial_value is not None else 0.0
            

    def log_metric(self, split, metric=None, value=None):
            if split in self.metrics and isinstance(self.metrics[split], dict):
                if metric not in self.metrics[split]:
                    self.metrics[split][metric] = []
                if isinstance(self.metrics[split][metric], list):
                    self.metrics[split][metric].append(value)
                else:
                    self.metrics[split][metric] = value
            else:
                self.metrics[split] = value


    def get_metric(self, split, metric=None, last_only=False):
        """
        Retrieve a metric value from the nested metrics dict.
        If last_only=True and metric stores a list, return only the last value.
        """
        if split in self.metrics and isinstance(self.metrics[split], dict):
            if metric not in self.metrics[split]:
                raise KeyError(f"Metric '{metric}' not found in split '{split}'.")
            values = self.metrics[split][metric]
            if isinstance(values, list):
                return values[-1] if last_only and values else values
            return values
        elif split in self.metrics:
            # Scalars like score, forecasted_val_acc
            return self.metrics[split]
        else:
            raise KeyError(f"Split '{split}' not found in metrics.")
        

    def __str__(self):
        # Pull latest values for readability
        train_loss = self.get_metric("train", "loss", last_only=True)
        train_acc = self.get_metric("train", "acc", last_only=True)
        val_loss = self.get_metric("val", "loss", last_only=True)
        val_acc = self.get_metric("val", "acc", last_only=True)
        test_acc = self.get_metric("test", "acc", last_only=True)
        score = self.get_metric("score")

        arch_summary = ", ".join(f"{k}={v}" for k, v in self.architecture.items() if k != "layers")

        return (
            f"Candidate(\n"
            f"  Arch: {arch_summary}\n"
            f"  Train: loss={train_loss}, acc={train_acc}\n"
            f"  Val:   loss={val_loss}, acc={val_acc}\n"
            f"  Test:  acc={test_acc}\n"
            f"  Score: {score}\n"
            f")"
        )
    
    def build_dict(self):
        # Flatten architecture
        flat_arch = {f"arch_{k}": v for k, v in self.architecture.items()}

        # Flatten metrics, but keep lists intact
        flat_metrics = {}
        for k, v in self.metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    # keep lists as full lists
                    flat_metrics[f"{k}_{sub_k}"] = sub_v
            else:
                flat_metrics[k] = v
        
        for key in ["train_acc", "train_loss", "val_acc", "val_loss"]:
            values = flat_metrics.get(key, [])
            flat_metrics[f"last_{key}"] = values[-1] if values else None

        # Combine everything
        candidate_dict = {
            "id": self.id,
            "batch_size": self.batch_size,
            "n_instances": self.n_instances,
            "epochs_trained": self.epochs_trained,
            "efforts": self.efforts,
            **flat_arch,
            **flat_metrics
        }
        return candidate_dict

# endregion