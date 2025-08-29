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
import json


# from baseline_models import get_models_and_baseline_metric
from forecaster import forecast_accuracy
from utils import set_seed
from candidates import Candidate

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
        self.starting_instances = starting_instances
        self.individuals_created = 0
        self.generation = self.build_generation()
        self.starting_snapshot = self.build_snapshot().copy(deep=True)
        

    def build_generation(self):
        generation = {}
        for i in range(self.n_individuals):
            architecture = self.search_space.sample_architecture(seed=i * self.seed)
            model = self.search_space.create_model(architecture, task_type=self.task_type)
            generation[i] = Candidate(model, architecture, 
                                      starting_instances=self.starting_instances,
                                      id_counter=i)
            self.individuals_created += 1

        return generation
    
    def train_generation(self, X_train, y_train, 
                         training_mode='oe', X_val=None, y_val=None, 
                         **kwargs):
        active_individuals = self.generation.keys()
        for i in active_individuals:
            candidate = self.generation[i]
            seed = candidate.architecture.get("seed", None)
            model = candidate.model
            batch_size = candidate.batch_size
            # Get the last instance budget for the candidate
            # Cap it to the max instances of training data
            n_instances = min(candidate.n_instances[-1], len(X_train))
            dataset_fraction = n_instances / len(X_train)
            
            # Set seed for reproducibility
            set_seed(seed)
            # Sample data based on the instance budget
            X_sampled, y_sampled = sample_data(X_train, y_train, 
                                               n_instances, mode="absolute")

            # Create a DataLoader with the architecture-specific batch size
            train_loader = create_dataloaders(X=X_sampled, y=y_sampled, 
                                              batch_size=batch_size)

            # Train the model
            if training_mode == 'oe':
                train_loss, train_acc = model.oe_train(train_loader)
                candidate.log_metric('train', 'loss', train_loss)
                candidate.log_metric('train', 'acc', train_acc)
                
                current_epoch = candidate.epochs_trained + 1
                candidate.efforts.append(dataset_fraction * current_epoch)
                candidate.update_n_instances(n_instances * 2)  # Double for next time
                candidate.epochs_trained += 1

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
                candidate.metrics["es_results"] = results


    def validate_generation(self, X_val, y_val, metric='val'):

        active_individuals = self.generation.keys()
        for i in active_individuals:
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
        start_idx = self.individuals_created  # so to not overwrite existing ones
        for i in range(n_basic_models):
            architecture = search_space.sample_architecture(seed=i*5 + self.seed)
            model = search_space.create_model(architecture, task_type=self.task_type)
            # Create a new model entry but that starts i as the len of the generation
            # so that it does not overwrite the existing ones
            new_generation[start_idx + i] = Candidate(model=model, 
                                                      architecture=architecture, starting_instances=self.starting_instances, id_counter=self.individuals_created + 1)
            self.individuals_created += 1
        
        # Create mutations of the existing models
        start_idx = self.individuals_created  # so to not overwrite existing ones
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
            new_generation[start_idx + i] = Candidate(model=child_model, architecture=child_architecture, id_counter=self.individuals_created + 1, starting_instances=self.starting_instances)
            self.individuals_created += 1


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
        
        self.n_individuals = len(self.generation)  # Update the count

        # Clean up GPU memory once
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    def forecast_generation(self, effort_threshold=3):
        active_individuals = self.generation.keys()

        for i in active_individuals:
            candidate = self.generation[i]
            efforts = candidate.efforts
            val_accs = candidate.get_metric('val', 'acc')

            if candidate.epochs_trained >= effort_threshold and len(val_accs) >= effort_threshold:
                forecasted_accuracy = forecast_accuracy(efforts, val_accs, model_type='rational')
                candidate.metrics["forecasted_val_acc"] = forecasted_accuracy
            else: 
                pass  # Not enough data to forecast
    

    def check_higher_than_baseline(self, baseline_metric):
        active_individuals = self.generation.keys()
        for i in active_individuals:
            candidate = self.generation[i]
            last_fcst_acc = candidate.get_metric("forecasted_val_acc") or 0.0
            
            candidate.log_metric("fcst_greater_than_baseline", value=last_fcst_acc >= baseline_metric)


    def score_individuals(self, baseline_metric):
        self.check_higher_than_baseline(baseline_metric)
        active_individuals = self.generation.keys()
        for i in active_individuals:
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


    def get_best_model(self, n_candidate=0):
        # Sort individuals by score in descending order
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Get the n-th best model
        nth_best_individual = sorted_generation[n_candidate][0]
        return self.generation[nth_best_individual].model


    # def build_df(self):
        
    #     records = []    
    #     for i, candidate in self.generation.items():
    #         # Start from architecture dict
    #         base = dict(candidate.architecture)
    #         metrics = candidate.metrics

    #         base.update({
    #             "id": candidate.id,
    #             "batch_size": candidate.batch_size,
    #             "n_instances": candidate.n_instances,
    #             "efforts": candidate.efforts,
    #             "train_loss": metrics["train"]["loss"],
    #             "train_acc": metrics["train"]["acc"],
    #             "val_loss": metrics["val"]["loss"],
    #             "val_acc": metrics["val"]["acc"],
    #             "test_loss": metrics["test"]["loss"],
    #             "test_acc": metrics["test"]["acc"],
    #             "forecasted_val_acc": metrics.get("forecasted_val_acc", 0.0),
    #             "fcst_greater_than_baseline": metrics.get("fcst_greater_than_baseline", False),
    #             # Include es_results if present, else None
    #             "es_results": metrics.get("es_results", None)
    #         })
    #         records.append(base)

    #     df = pd.DataFrame.from_records(records)

    #     # Add last epoch metrics
    #     df['last_epoch_val_acc'] = df['val_acc'].apply(
    #         lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else np.nan)
    #     df['last_epoch_val_loss'] = df['val_loss'].apply(
    #         lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else np.nan
    #     )
    #     df['forecasted_val_acc'] = df['forecasted_val_acc'].fillna(0.0)
    #     # Sort by forecasted accuracy descending
    #     self.history = df.sort_values("forecasted_val_acc", ascending=False).reset_index(drop=True).copy()

    #     return self.history


    def run_generation(self,
                       X_train, y_train, X_val, y_val,
                       percentile_drop=25, goal_metric=None,
                       epoch_threshold=3, track_all_models=False):


        self.build_new_models(self.search_space)
        # Generation is trained, and dropped
        self.train_generation(X_train, y_train)
        self.validate_generation(X_val, y_val)
        self.forecast_generation(effort_threshold=epoch_threshold)
        self.score_individuals(baseline_metric=goal_metric)
        self.current_snapshot = self.build_snapshot()

        # Optionally maintain cumulative history (merge without duplicates)
        if track_all_models:
            self.cumulative_snapshot = (
                pd.concat([self.cumulative_snapshot, self.current_snapshot])
                .drop_duplicates(subset="id", keep="last")
                # .reset_index(drop=True)
            )
            
        self.get_worst_individuals(percentile_drop)
        self.drop_worst_individuals()

        return self.generation
    

    def run_ebe(self,
            X_train, y_train, X_val, y_val,
            percentile_drop=25, epochs=50, baseline_metric=None,
            time_budget=60, epoch_threshold=3, 
            track_all_models=False):

        # Snapshots: baseline vs evolving
        self.current_snapshot = self.starting_snapshot
        self.cumulative_snapshot = self.starting_snapshot if track_all_models else None

        self.epoch_threshold = epoch_threshold
        start_time = time.time()
        
        for epoch in range(epochs):
            print('Built models:',self.individuals_created)
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time >= time_budget:
                print(f"Time budget exceeded at epoch {epoch + 1}: {elapsed_time:.2f} seconds")
                if epoch <= self.epoch_threshold - 1:
                    print("Not enough epochs completed for forecasting, stopping EBE.")
                    print("Try to reduce the number of candidates or increase the time budget.")
                    break
                break

            print(f"Epoch {epoch + 1}/{epochs}")

            # Run one evolutionary generation
            self.generation = self.run_generation(
                X_train, y_train,
                X_val, y_val,
                percentile_drop=percentile_drop,
                goal_metric=baseline_metric,
                epoch_threshold=epoch_threshold,
                track_all_models=track_all_models
            )

            # Increase drop but cap at 50%
            percentile_drop = min(percentile_drop + 10, 50)

        print("EBE process completed.")

        return self.current_snapshot  # return the latest by default




    def build_snapshot(self, export_as='pandas'):
        current_candidates = []
        active_individuals = self.generation.keys()
        for i in active_individuals:
            candidate = self.generation[i]
            current_candidates.append(candidate.build_dict())           
        
        if export_as == 'pandas':
            df = pd.DataFrame(current_candidates).sort_values(by='forecasted_val_acc', ascending=False)
            return df.copy(deep=True)
        elif export_as == 'json':
            return json.dumps(current_candidates, indent=4)


