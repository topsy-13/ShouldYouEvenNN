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
            
            child_architecture = self.mutate_architecture(child_architecture, mutation_rate=0.3)

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
    
    def mutate_architecture(self, architecture, mutation_rate=0.3, seed=None):
        """
        Mutate a given architecture by tweaking hidden layers, batch size,
        learning rate, or dropout. mutation_rate is the probability any key mutates.
        """
        if seed is None:
            seed = random.randint(0, 100000)
        set_seed(seed)

        mutated = dict(architecture)  # copy

        # hidden_layers tweak
        if 'hidden_layers' in mutated and random.random() < mutation_rate:
            layers = mutated['hidden_layers'][:]
            if layers and random.random() < 0.5:
                # Add a new layer
                layers.append(random.choice([32, 64, 128, 256]))
            else:
                # Perturb an existing one
                idx = random.randrange(len(layers))
                layers[idx] = max(4, int(layers[idx] * random.choice([0.5, 1.5])))
            mutated['hidden_layers'] = layers

        # learning rate tweak
        if 'lr' in mutated and random.random() < mutation_rate:
            mutated['lr'] = mutated['lr'] * random.choice([0.5, 1.5])

        # dropout tweak
        if 'dropout' in mutated and random.random() < mutation_rate:
            mutated['dropout'] = min(max(0.0, mutated['dropout'] + random.uniform(-0.1, 0.1)), 0.7)

        # batch size tweak
        if 'batch_size' in mutated and random.random() < mutation_rate:
            mutated['batch_size'] = int(max(8, mutated['batch_size'] * random.choice([0.5, 2])))

        return mutated


    def get_worst_individuals(self, percentile_drop=15, baseline_metric=None):
        """
        Identify worst individuals to drop.

        Rules:
        - If forecasts exist, drop those below baseline first.
        - If no forecasts, rank by raw val_acc.
        - Always preserve top 10% (elites).
        """
        if baseline_metric is not None:
            self.score_individuals(baseline_metric)

        n_worst = max(1, int(self.n_individuals * percentile_drop / 100))
        elite_count = max(1, int(0.1 * self.n_individuals))  # preserve 10%

        # Build list of candidates
        candidates = []
        for key, cand in self.generation.items():
            val_acc = cand.get_metric('val', 'acc', last_only=True)
            fcst_acc = cand.metrics.get("forecasted_val_acc", None)
            score = cand.metrics.get("score", None)
            candidates.append((key, val_acc, fcst_acc, score))

        # Separate forecasted vs non-forecasted
        with_fcst = [c for c in candidates if c[2] is not None]
        without_fcst = [c for c in candidates if c[2] is None]

        # Case 1: forecasts exist
        if with_fcst:
            below_baseline = [k for k, v, f, s in with_fcst if f < (baseline_metric or 0)]
            sorted_all = sorted(candidates, key=lambda x: (x[3] if x[3] is not None else 0))
        # Case 2: no forecasts → fallback to val_acc
        else:
            below_baseline = []
            sorted_all = sorted(candidates, key=lambda x: (x[1] if x[1] is not None else 0))

        # Identify elites (top 10% by val_acc)
        elites = {k for k, v, f, s in sorted(candidates, key=lambda x: (x[1] or 0), reverse=True)[:elite_count]}

        # Build worst list
        worst = []
        # Drop below-baseline first
        for k in below_baseline:
            if k not in elites and len(worst) < n_worst:
                worst.append(k)
        # Fill rest with lowest scorers
        for k, v, f, s in sorted_all:
            if k not in elites and k not in worst and len(worst) < n_worst:
                worst.append(k)

        self.worst_individuals = worst


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
            last_val_acc = candidate.get_metric('val', 'acc', last_only=True) or 0.0
            last_fcst_acc = candidate.get_metric("forecasted_val_acc") or 0.0
            slope = candidate.metrics.get("slope_val_acc", 0.0)
            variance = candidate.metrics.get("var_val_acc", 0.0)
            gap = candidate.metrics.get("gap_val_acc", 0.0)

            if last_fcst_acc < baseline_metric:
                # Below baseline → check momentum
                if slope > 0.01 and gap > 0:  # improving fast enough
                    score = 0.3 * last_val_acc + 0.5 * last_fcst_acc + 0.2 * slope
                else:
                    score = max(last_val_acc, 0.0)
            else:
                # Beating baseline forecast
                fcst_gain = last_fcst_acc - baseline_metric
                score = 0.6 * last_fcst_acc + 0.3 * fcst_gain + 0.1 * slope

            candidate.log_metric('score', value=score)



    def get_best_model(self, n_candidate=0):
        # Sort individuals by score in descending order
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Get the n-th best model
        nth_best_individual = sorted_generation[n_candidate][0]
        return self.generation[nth_best_individual].model


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
            percentile_drop=25, max_epochs=200, 
            baseline_metric=None,
            time_budget=60, epoch_threshold=3, 
            track_all_models=False):

        # Snapshots: baseline vs evolving
        self.current_snapshot = self.starting_snapshot
        self.cumulative_snapshot = self.starting_snapshot if track_all_models else None

        self.epoch_threshold = epoch_threshold
        start_time = time.time()
        
        for epoch in range(max_epochs):
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time >= time_budget:
                print(f"Time budget exceeded at epoch {epoch + 1}: {elapsed_time:.2f} seconds")
                if epoch <= self.epoch_threshold - 1:
                    print("Not enough epochs completed for forecasting, stopping EBE.")
                    print("Try to reduce the number of candidates or increase the time budget.")
                    break
                break

            print(f"Epoch {epoch + 1}")

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
            df = pd.DataFrame(current_candidates).sort_values(by='score', ascending=False)
            return df.copy(deep=True)
        elif export_as == 'json':
            return json.dumps(current_candidates, indent=4)


