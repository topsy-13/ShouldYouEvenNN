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
            generation[i] = {
                "model": model,
                "architecture": architecture,
                "batch_size": architecture['batch_size'],
                "n_instances": [],
                "proportion_instances": [],
                'effort': [],
                'train_loss': [],
                "train_acc": [],
                'val_loss': [],
                "val_acc": [],
                "forecasted_val_acc": 0.0,
                "score": 0.0,
                "forecast_gain": 0.0,
                'higher_than_baseline': False
            }
        return generation
    
    def train_generation(self, X_train, y_train):
        for i in range(self.n_individuals):
            seed = self.generation[i]["architecture"].get("seed", None)
            model = self.generation[i]["model"]
            batch_size = self.generation[i]["batch_size"]
            dataset_fraction = self.n_instances / len(X_train)
            
            # Set seed for reproducibility
            set_seed(seed)
            # Sample data based on the instance budget
            X_sampled, y_sampled = sample_data(X_train, y_train, self.n_instances, mode="absolute")

            # Cap it to the max instances
            self.n_instances = min(self.n_instances, len(X_train))
            # Create a DataLoader with the architecture-specific batch size
            train_loader = create_dataloaders(X=X_sampled, y=y_sampled, batch_size=batch_size)
            # print(train_loader.dataset.tensors[0].shape)
            # print(train_loader.dataset.tensors[1].shape)

            train_loss, train_acc = model.oe_train(train_loader)
            self.generation[i]["train_loss"].append(train_loss)  
            self.generation[i]["train_acc"].append(train_acc)
            num_epochs = len(self.generation[i]['train_loss'])
            self.effort = dataset_fraction * num_epochs
            self.generation[i]["n_instances"].append(self.n_instances)
            self.generation[i]["effort"].append(self.effort)
    

    def validate_generation(self, X_val, y_val):
        for i in range(self.n_individuals):
            model = self.generation[i]["model"]
            batch_size = self.generation[i]["batch_size"]
            seed = self.generation[i]["architecture"].get("seed", None)
            set_seed(seed)
            # Sample data based on the instance budget
            X_sampled, y_sampled = sample_data(X_val, y_val, self.n_instances, mode="absolute", task_type=self.task_type)
            
            # Create a DataLoader with the architecture-specific batch size
            val_loader = create_dataloaders(X=X_sampled, y=y_sampled, batch_size=batch_size)
            
            val_loss, val_acc = model.evaluate(val_loader)
            self.generation[i]["val_loss"].append(val_loss)
            self.generation[i]["val_acc"].append(val_acc)


    def get_worst_individuals(self,
                              percentile_drop=15):
    
        n_worst_individuals = max(1, int(self.n_individuals * percentile_drop / 100))  # Ensure at least 1

        # Sort individuals by score ranking (higher score is better)
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["score"], reverse=False)

        # Extract the keys of the worst individuals
        self.worst_individuals = [key for key, _ in sorted_generation[:n_worst_individuals]]

    def build_new_models(self, search_space):
        self.n_individuals = len(self.generation)  # Update the count

        # Build new models based on the amount of dropped individuals
        n_new_models = self.max_individuals - self.n_individuals
        n_basic_models = int(n_new_models * 0.5)  # 50% of the new models will be basic
        n_advanced_models = n_new_models - n_basic_models  # The rest will be evolutions

        new_generation = {}
        for i in range(n_basic_models):
            architecture = search_space.sample_architecture(seed=i*5 + self.seed)
            model = search_space.create_model(architecture, task_type=self.task_type)
            # Create a new model entry but that starts i as the len of the generation
            # so that it does not overwrite the existing ones
            new_generation[i + self.n_individuals] = {
                "model": model,
                "architecture": architecture,
                "batch_size": architecture['batch_size'],
                "n_instances": [],
                "proportion_instances": [],
                'effort': [],
                'train_loss': [],
                "train_acc": [],
                'val_loss': [],
                "val_acc": [],
                "forecasted_val_acc": 0.0,
                "score": 0.0,
                "forecast_gain": 0.0,
                'higher_than_baseline': False
            }
        
        # Create mutations of the existing models
        for i in range(n_advanced_models):
            # Select sample of individuals to evolve
            parents = self.weighted_random_selection(k=2,
                                                     seed=i+self.seed*3)
            parent1 = parents[0]['architecture']
            parent2 = parents[1]['architecture']

            # Crossover
            child_architecture = self.crossover(parent1, parent2, 
                                                seed=i*7 + self.seed)
            # Create a new model with the child architecture
            child_model = search_space.create_model(child_architecture, task_type=self.task_type)
            # Add the child model to the generation
            new_generation[i + n_basic_models + self.n_individuals] = {
                "model": child_model,
                "architecture": child_architecture,
                "batch_size": child_architecture['batch_size'],
                "n_instances": [],
                "proportion_instances": [],
                'effort': [],
                'train_loss': [],
                "train_acc": [],
                'val_loss': [],
                "val_acc": [],
                "forecasted_val_acc": 0.0,
                "score": 0.0,
                "forecast_gain": 0.0,
                'higher_than_baseline': False
            }    
        # Merge the new models into the existing generation
        self.generation.update(new_generation)
        self.n_individuals = len(self.generation)  # Update the count
        
        return
    
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

    def drop_worst_individuals(self):
        # Clean up GPU memory before removing references
        for idx in self.worst_individuals:
            if hasattr(self.generation[idx]["model"], "cpu"):
                self.generation[idx]["model"] = self.generation[idx]["model"].cpu()
            # Force garbage collection for the model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Remove worst individuals
        for idx in self.worst_individuals:
            del self.generation[idx]
        
        # Re-index the remaining individuals to maintain continuous keys
        self.generation = {new_idx: val for new_idx, (_, val) in enumerate(self.generation.items())}
        self.n_individuals = len(self.generation)  # Update the count


    def weighted_random_selection(self, k=2, seed=None):
        """
        Select k candidates based on weighted probabilities from fitness scores.
        Higher fitness = higher probability of being chosen.
        """
        if seed is None:
            seed = random.randint(0, 100000)
        
        set_seed(seed)
        candidates = list(self.generation.keys())
        scores = np.array([self.generation[i]['score'] for i in candidates])
        probabilities = scores / scores.sum()  # Normalize to get probabilities
        selected_indices = np.random.choice(candidates, size=k, p=probabilities)
        return [self.generation[i] for i in selected_indices]


    def drop_all_except_best(self):
        # Sort individuals by score ranking (higher score is better)
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["score"], reverse=False)
        
        # Keep only the best individual
        best_individual = sorted_generation[0][0]
        best_model_data = self.generation[best_individual]
        
        # Clean up GPU memory for models that will be discarded
        for idx, data in self.generation.items():
            if idx != best_individual:
                if hasattr(data["model"], "cpu"):
                    data["model"] = data["model"].cpu()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.generation = {0: best_model_data}
        self.n_individuals = 1

    def train_best_individual(self, X_train, y_train, num_epochs=1):
        best_model = self.generation[0]["model"]
        batch_size = self.generation[0]["batch_size"]

        # Create a DataLoader with the architecture-specific batch size
        train_loader = create_dataloaders(X=X_train, y=y_train, batch_size=batch_size)
        best_model.oe_train(train_loader, num_epochs=num_epochs)
    
    def forecast_generation(self, epoch_threshold=3):
        for i in range(self.n_individuals):
            efforts = self.generation[i]["effort"]
            val_accs = self.generation[i]["val_acc"]
            if len(efforts) >= epoch_threshold and len(val_accs) >= epoch_threshold:
                forecasted_accuracy = forecast_accuracy(efforts, val_accs, model_type='rational')
                self.generation[i]["forecasted_val_acc"] = forecasted_accuracy
            else: 
                pass  # Not enough data to forecast
    
    def score_individuals(self, baseline_metric):
        self.check_higher_than_baseline(baseline_metric)

        for i in range(self.n_individuals):
            # TODO: use the baseline metric to drop models
            last_val_acc = self.generation[i]['val_acc'][-1]
            # get the last forecasted accuracy if empty then 0

            if not self.generation[i]["forecasted_val_acc"]:
                last_fcst_acc = 0.0
            else:
                last_fcst_acc = self.generation[i]["forecasted_val_acc"]

            # print('Last valacc:', last_val_acc)
            # print('Forecast valacc' ,self.generation[i]["forecasted_val_acc"])
            # Score is a combination of current accuracy and forecasted 
            score = 0.7 * last_val_acc + 0.3 * last_fcst_acc
            # * score models by how much higher their forecast is compared to the baseline:
            self.generation[i]['forecast_gain'] = last_fcst_acc - baseline_metric

            self.generation[i]['score'] = 0.5 * last_val_acc + 0.5 * self.generation[i]['forecast_gain']
            
            self.generation[i]["score"] = score

    def check_higher_than_baseline(self, baseline_metric):
        for i in range(self.n_individuals):
            if not self.generation[i]["forecasted_val_acc"]:
                last_fcst_acc = 0.0
            else:
                last_fcst_acc = self.generation[i]["forecasted_val_acc"]
            self.generation[i]['higher_than_baseline'] = last_fcst_acc >= baseline_metric

    def get_best_model(self, n_candidate=0):
        # Sort individuals by score in descending order
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Get the n-th best model
        nth_best_individual = sorted_generation[n_candidate][0]
        return self.generation[nth_best_individual]["model"]

    def return_df(self):
        # As a dataframe
        architectures = []
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        batch_sizes = []
        efforts = []
        instance_sizes = []
        forecasted_accuracies = []
        higher_than_baseline = []

        for i in range(self.n_individuals):
            architectures.append(self.generation[i]["architecture"])
            train_losses.append(self.generation[i]["train_loss"])
            train_accs.append(self.generation[i]["train_acc"])
            val_losses.append(self.generation[i]["val_loss"])
            val_accs.append(self.generation[i]["val_acc"])
            batch_sizes.append(self.generation[i]["batch_size"])
            efforts.append(self.generation[i]["effort"])   
            instance_sizes.append(self.generation[i]["n_instances"])   
            forecasted_accuracies.append(self.generation[i]["forecasted_val_acc"])   
            higher_than_baseline.append(self.generation[i]["higher_than_baseline"])

        # Create a DataFrame with the architectures and their corresponding metrics
        architectures_df = pd.DataFrame(architectures)
        architectures_df['train_loss'] = train_losses
        architectures_df['train_acc'] = train_accs
        architectures_df['val_loss'] = val_losses
        architectures_df['val_acc'] = val_accs
        architectures_df['batch_size'] = batch_sizes
        architectures_df['efforts'] = efforts
        architectures_df['n_instances'] = instance_sizes
        architectures_df['fcst_accuracy'] = forecasted_accuracies
        architectures_df['higher_than_baseline'] = higher_than_baseline

        df = pd.DataFrame(architectures_df)
        df['last_epoch_val_acc'] = df['val_acc'].apply(
            lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else np.nan)
        df['last_epoch_val_loss'] = df['val_loss'].apply(
            lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else np.nan
        )
        
        # Drop untrained models
        df = df[df['last_epoch_val_acc'].notna()]

        # Sort by forecasted accuracy in descending order
        self.history = df.sort_values('fcst_accuracy', ascending=False).reset_index(drop=True)

        return self.history

    def run_generation(self,
                       X_train, y_train, X_val, y_val,
                       percentile_drop=25, goal_metric=None,
                       epoch_threshold=3):
    
        # Generation is trained, and dropped
        self.train_generation(X_train, y_train)
        self.validate_generation(X_val, y_val)
        self.forecast_generation(epoch_threshold=epoch_threshold)
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


