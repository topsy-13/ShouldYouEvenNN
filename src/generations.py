import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import gc
import data_preprocessing as dp
from instance_sampling import resolve_instance_budget, sample_data, create_dataloaders

import time

# from baseline_models import get_models_and_baseline_metric
from forecaster import forecast_accuracy
from utils import set_seed

# region Individuals
class Candidate:
    def __init__(self, model, architecture):
        self.model = model
        self.architecture = architecture
        self.batch_size = architecture.get("batch_size")
        self.n_instances = []
        self.proportion_instances = []
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
# endregion

# region Generations
class Generation():
    def __init__(self, search_space, n_individuals,
                 starting_instances=100, seed=None,
                 task_type='classification'):
        
        if seed is None:
            seed = random.randint(0, 100000)

        self.seed = seed
        self.task_type = task_type
        self.search_space = search_space
        self.max_individuals = n_individuals
        self.n_individuals = n_individuals
        self.generation = self.build_generation()
        self.n_instances = starting_instances
        

    def build_generation(self):
        generation = {}
        for i in range(self.n_individuals):
            architecture = self.search_space.sample_architecture(seed=i * self.seed)
            model = self.search_space.create_model(architecture, task_type=self.task_type)
            generation[i] = Candidate(model, architecture)

        return generation
    
    def train_generation(self, X_train, y_train, 
                         training_mode='oe', X_val=None, y_val=None, 
                         **kwargs):

        for i in range(self.n_individuals):
            candidate = self.generation[i]
            seed = candidate.architecture.get("seed", None)
            model = candidate.model
            batch_size = candidate.batch_size
            # Cap it to the max instances of training data
            self.n_instances = min(self.n_instances, len(X_train))
            dataset_fraction = self.n_instances / len(X_train)
            
            # Set seed for reproducibility
            set_seed(seed)
            # Sample data based on the instance budget
            X_sampled, y_sampled = sample_data(X_train, y_train, self.n_instances, mode="absolute")

            # Create a DataLoader with the architecture-specific batch size
            train_loader = create_dataloaders(X=X_sampled, y=y_sampled, batch_size=batch_size)
            # Train the model
            if training_mode == 'oe':
                train_loss, train_acc = model.oe_train(train_loader)
                candidate.log_metric('train', 'loss', train_loss)
                candidate.log_metric('train', 'acc', train_acc)
                
                current_epoch = len(candidate.metrics["train"]["loss"])
                self.effort = dataset_fraction * current_epoch
                candidate.efforts.append(self.effort)
                candidate.log_metric('n_instances', value=self.n_instances)

            elif training_mode == 'es':
                print(f'Training individual {i+1}/{self.n_individuals} with Early Stopping...')
                assert X_val is not None and y_val is not None, "X_val and y_val must be provided for early stopping training."

                val_loader = create_dataloaders(X=X_val, y=y_val, batch_size=batch_size)
                es_results = model.es_train(train_loader, val_loader, **kwargs)
                # Unpack results
                best_train_loss, best_train_acc, best_val_loss, best_val_acc, learning_curve = es_results
                results = {
                    'final_train_loss': best_train_loss,
                    'final_train_acc': best_train_acc,
                    'final_val_loss': best_val_loss,
                    'final_val_acc': best_val_acc,
                    'learning_curve_es': learning_curve
                }
                candidate.metrics["final"] = results


    def validate_generation(self, X_val, y_val, metric='val'):

        for i in range(self.n_individuals):
            candidate = self.generation[i]
            model = candidate.model
            batch_size = candidate.batch_size
            seed = candidate.architecture.get("seed", None)
            set_seed(seed)
            
            # Create a DataLoader with the architecture-specific batch size
            val_loader = create_dataloaders(X=X_val, y=y_val, batch_size=batch_size)
            val_loss, val_acc = model.evaluate(val_loader)

            candidate.log_metric(metric, 'loss', val_loss)
            candidate.log_metric(metric, 'acc', val_acc)


    def build_new_models(self, search_space):
        self.n_individuals = len(self.generation)  # Update the count

        # Build new models based on the amount of dropped individuals
        n_new_models = self.max_individuals - self.n_individuals
        n_basic_models = int(n_new_models * 0.5)  # 50% of the new models will be basic
        n_advanced_models = n_new_models - n_basic_models  # The rest will be evolutions

        new_generation = {}
        start_idx = self.n_individuals  # so to not overwrite existing ones
        for i in range(n_basic_models):
            architecture = search_space.sample_architecture(seed=i*5 + self.seed)
            model = search_space.create_model(architecture, task_type=self.task_type)
            # Create a new model entry but that starts i as the len of the generation
            # so that it does not overwrite the existing ones
            new_generation[start_idx + i] = Candidate(model=model, 
                                                      architecture=architecture)
        
        # Create mutations of the existing models
        for i in range(n_advanced_models):
            # Select sample of individuals to evolve
            parents = self.weighted_random_selection(k=2,
                                                     seed=i+self.seed*3)
            parent1 = parents[0].architecture
            parent2 = parents[1].architecture

            # Crossover
            child_architecture = self.crossover(parent1, parent2, 
                                                seed=i*7 + self.seed)
            
            # Create a new model with the child architecture
            child_model = search_space.create_model(child_architecture, task_type=self.task_type)

            # Add the child model to the generation
            new_generation[i 
                           + n_basic_models + self.n_individuals] = Candidate(model=child_model, architecture=child_architecture)

        # Merge the new models into the existing generation
        self.generation.update(new_generation)
        self.n_individuals = len(self.generation)  # Update the count
        
        return
    

    def weighted_random_selection(self, k=2, seed=None):
        """
        Select k candidates based on weighted probabilities from fitness scores.
        Higher fitness = higher probability of being chosen.
        Falls back to uniform selection if not enough non-zero probabilities.
        """
        if seed is None:
            seed = random.randint(0, 100000)
        
        set_seed(seed)
        candidates = list(self.generation.keys())
        
        # Extract scores
        scores = np.array([self.generation[i].metrics['score'] for i in candidates], dtype=float)
        
        # Normalize scores to get probabilities
        total = scores.sum()
        if total > 0:
            probabilities = scores / total
        else:
            probabilities = np.zeros_like(scores)
        
        # Handle edge case: fewer non-zero probabilities than k
        nonzero_candidates = [c for c, prob in zip(candidates, probabilities) if prob > 0]
        if len(nonzero_candidates) < k:
            # fallback to uniform random selection
            selected_indices = np.random.choice(candidates, size=k, replace=False)
        else:
            selected_indices = np.random.choice(candidates, size=k, replace=False, p=probabilities)
        
        return [self.generation[i] for i in selected_indices]


    def crossover(self, parent1, parent2, seed=None):
        """Create a child model configuration from two parents."""
        seed = seed if seed is not None else random.randint(0, 100000)
        set_seed(seed)
        child = {}

        for key in parent1.keys():
            if key == 'hidden_layers':
                # one-point crossover on list
                cut = random.randint(1, min(len(parent1[key]), len(parent2[key])) - 1)
                child[key] = parent1[key][:cut] + parent2[key][cut:]
            elif key == 'scheduler_params' and isinstance(parent1[key], dict) and isinstance(parent2[key], dict):
                all_keys = set(parent1[key].keys()) | set(parent2[key].keys())  # union of keys
                child[key] = {}
                for k in all_keys:
                    if k in parent1[key] and k in parent2[key]:
                        child[key][k] = random.choice([parent1[key][k], parent2[key][k]])
                    elif k in parent1[key]:
                        child[key][k] = parent1[key][k]
                    else:
                        child[key][k] = parent2[key][k]
            else:
                # Simple gene pick
                child[key] = random.choice([parent1[key], parent2[key]])

        return child
    

    def get_worst_individuals(self, percentile_drop=15):
        n_worst_individuals = max(1, int(self.n_individuals * percentile_drop / 100))  # Ensure at least 1

        # Sort individuals by score (lower is worse)
        sorted_generation = sorted(
            self.generation.items(),
            key=lambda x: x[1].metrics["score"],  # access via Candidate.metrics
            reverse=False
        )

        # Extract keys of the worst individuals
        self.worst_individuals = [key for key, _ in sorted_generation[:n_worst_individuals]]


    def drop_worst_individuals(self):
        # Move all worst models to CPU first
        for idx in self.worst_individuals:
            candidate = self.generation[idx]
            if hasattr(candidate.model, "cpu"):
                candidate.model = candidate.model.cpu()
        
        # Remove worst individuals
        for idx in self.worst_individuals:
            del self.generation[idx]
        
        # Re-index the remaining individuals to maintain continuous keys
        self.generation = {new_idx: val for new_idx, (_, val) in enumerate(self.generation.items())}
        self.n_individuals = len(self.generation)  # Update the count

        # Clean up GPU memory once
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    def forecast_generation(self, effort_threshold=3):
        for i in range(self.n_individuals):
            candidate = self.generation[i]
            efforts = candidate.efforts
            val_accs = candidate.get_metric('val', 'acc')

            if len(efforts) >= effort_threshold and len(val_accs) >= effort_threshold:
                forecasted_accuracy = forecast_accuracy(efforts, val_accs, model_type='rational')
                candidate.metrics["forecasted_val_acc"] = forecasted_accuracy
            else: 
                pass  # Not enough data to forecast
    

    def score_individuals(self, baseline_metric):
        self.check_higher_than_baseline(baseline_metric)

        for i in range(self.n_individuals):
            candidate = self.generation[i]
            # ? TODO: use the baseline metric to drop models 
            # get the last validation accuracy and the forecasted
            last_val_acc = candidate.get_metric('val', 'acc', last_only=True)
            last_fcst_acc = candidate.get_metric("forecasted_val_acc")
            if last_fcst_acc is None:
                last_fcst_acc = 0.0
            fcst_gain = last_fcst_acc - baseline_metric
            candidate.log_metric('forecast_gain', value=fcst_gain)

            # * Playground for later
            # score models by how much higher their forecast is compared to the baseline:
            score = max(0.6 * last_val_acc + 0.4 * fcst_gain, 0)
            candidate.log_metric('score', value=score)


    def check_higher_than_baseline(self, baseline_metric):
        for i in range(self.n_individuals):
            candidate = self.generation[i]
            last_fcst_acc = candidate.get_metric("forecasted_val_acc") or 0.0
            
            candidate.log_metric("fcst_greater_than_baseline", value=last_fcst_acc >= baseline_metric)


    def get_best_model(self, n_candidate=0):
        # Sort individuals by score in descending order
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Get the n-th best model
        nth_best_individual = sorted_generation[n_candidate][0]
        return self.generation[nth_best_individual]["model"]


    def return_df(self):
        records = []
        
        for i, candidate in self.generation.items():
            # Start from architecture dict
            base = dict(candidate.architecture)
            base.update({
                "id": i,
                "batch_size": candidate.batch_size,
                "n_instances": candidate.n_instances,
                "efforts": candidate.efforts,
                "train_loss": candidate.metrics["train"]["loss"],
                "train_acc": candidate.metrics["train"]["acc"],
                "val_loss": candidate.metrics["val"]["loss"],
                "val_acc": candidate.metrics["val"]["acc"],
                "test_loss": candidate.metrics["test"]["loss"],
                "test_acc": candidate.metrics["test"]["acc"],
                "forecasted_val_acc": candidate.metrics["forecasted_val_acc"],
                "fcst_greater_than_baseline": candidate.metrics["fcst_greater_than_baseline"],
            })
            records.append(base)

        df = pd.DataFrame.from_records(records)

        # Add last epoch metrics
        df['last_epoch_val_acc'] = df['val_acc'].apply(
            lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else np.nan)
        df['last_epoch_val_loss'] = df['val_loss'].apply(
            lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else np.nan
        )
        df['forecasted_val_acc'] = df['forecasted_val_acc'].fillna(0.0)
        # Sort by forecasted accuracy descending
        self.history = df.sort_values("forecasted_val_acc", ascending=False).reset_index(drop=True).copy()

        return self.history


    def run_generation(self,
                       X_train, y_train, X_val, y_val,
                       percentile_drop=25, goal_metric=None,
                       epoch_threshold=3):
    
        # Generation is trained, and dropped
        self.train_generation(X_train, y_train)
        self.validate_generation(X_val, y_val)
        self.forecast_generation(effort_threshold=epoch_threshold)
        self.score_individuals(baseline_metric=goal_metric)
        self.n_instances *= 2  # Increase instance budget for next generation
        self.n_instances = min(self.n_instances, len(X_train))

        self.get_worst_individuals(percentile_drop)
        self.drop_worst_individuals()
        self.build_new_models(self.search_space)

        return self.generation
    
    def run_ebe(self,
            X_train, y_train, X_val, y_val,
            percentile_drop=25, epochs=50, baseline_metric=None,
            time_budget=60, epoch_threshold=3):
    
        self.epoch_threshold = epoch_threshold
        start_time = time.time()
        
        for epoch in range(epochs):
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time >= time_budget:
                print(f"Time budget exceeded at epoch {epoch + 1}: {elapsed_time:.2f} seconds")
                if epoch <= self.epoch_threshold - 1:
                    print("Not enough epochs completed for forecasting, stopping EBE.")
                    print('Try to reduce the number of candidates or increase the time budget')
                    break
                break

            print(f"Epoch {epoch + 1}/{epochs}")

            self.generation = self.run_generation(X_train, y_train,
                                                X_val, y_val,
                                                percentile_drop=percentile_drop,
                                                goal_metric=baseline_metric,
                                                epoch_threshold=epoch_threshold)

            self.num_models = len(self.generation)
            if self.num_models <= 10:
                print(f"Only {self.num_models} models left, stopping EBE.")
                break

            # Increase drop but limit to 50%
            percentile_drop = min(percentile_drop + 10, 50)

        results = self.return_df()
        print("EBE process completed.")
