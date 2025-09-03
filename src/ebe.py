import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import gc
import data_preprocessing as dp
from instance_sampling import sample_data, create_dataloaders

import time
import json


# from baseline_models import get_models_and_baseline_metric
from forecaster import forecast_generation
from utils import set_seed
from candidates import Candidate
from evolution import breed_and_mutate
from scoring import score_individuals, get_worst_individuals

# region Generations
class Population():
    
    def __init__(self, search_space, size,
                 starting_instances=100, 
                 seed=None,
                 task_type='classification'):
        
        self.seed = seed or random.randint(0, 100000)
        self.task_type = task_type
        self.search_space = search_space
        self.max_individuals = size
        self.size = size
        self.starting_instances = starting_instances
        self.individuals_created = 0
        self.candidates = self.spawn_candidates()
        self.initial_ledger = self.build_ledger().copy(deep=True)
        

    def spawn_candidates(self):
        """Initial population."""
        candidates_pool = {}
        for i in range(self.size):
            architecture = self.search_space.sample_architecture(seed=i * self.seed)
            model = self.search_space.create_model(architecture, task_type=self.task_type)
            candidates_pool[i] = Candidate(model, architecture, 
                                            starting_instances=self.starting_instances,
                                            id_counter=i)
            self.individuals_created += 1
        return candidates_pool
    

    def train_generation(self, X_train, y_train, 
                         training_mode='oe', 
                         X_val=None, y_val=None, 
                         **kwargs):
        
        """Train all candidates."""
        active_candidates = self.candidates.keys()
        for i in active_candidates:
            candidate = self.candidates[i]
            seed = candidate.architecture.get("seed", None)
            model = candidate.model
            batch_size = candidate.batch_size

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
                # Store epochs and efforts
                current_epoch = candidate.epochs_trained + 1
                candidate.efforts.append(dataset_fraction * current_epoch)
                candidate.update_n_instances(n_instances * 2)  # Double for next time
                candidate.epochs_trained += 1

            elif training_mode == 'es':
                print(f'Training individual {i+1}/{self.size} with Early Stopping...')
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

        active_individuals = self.candidates.keys()
        for i in active_individuals:
            candidate = self.candidates[i]
            model = candidate.model
            batch_size = candidate.batch_size
            seed = candidate.architecture.get("seed", None)
            set_seed(seed)
            
            # Create a DataLoader with the architecture-specific batch size
            val_loader = create_dataloaders(X=X_val, y=y_val, batch_size=batch_size)
            val_loss, val_acc = model.evaluate(val_loader)

            candidate.log_metric(metric, 'loss', val_loss)
            candidate.log_metric(metric, 'acc', val_acc)


    def spawn_new_candidates(self, search_space):
        self.size = len(self.candidates)  # Update the count

        # Build new models based on the amount of dropped individuals
        n_new_models = self.max_individuals - self.size
        n_basic_models = int(n_new_models * 0.5)  # 50% of the new models will be basic
        n_advanced_models = n_new_models - n_basic_models  # The rest will be evolutions

        new_generation = {}
        for i in range(n_basic_models):
            architecture = search_space.sample_architecture(seed=i*5 + self.seed)
            model = search_space.create_model(architecture, task_type=self.task_type)
            new_generation[self.individuals_created] = Candidate(model=model, 
                                                      architecture=architecture, 
                                                      starting_instances=self.starting_instances, 
                                                      id_counter=self.individuals_created + 1)
            self.individuals_created += 1
        
        # Crossover + Mutations
        for i in range(n_advanced_models):
            # Build the child model and architecture
            child_architecture = breed_and_mutate(candidates=self.candidates, seed=i)
            child_model = search_space.create_model(child_architecture, 
                                                    task_type=self.task_type)
            # Add the child model to the generation
            new_generation[self.individuals_created] = Candidate(model=child_model, architecture=child_architecture, id_counter=self.individuals_created + 1,  starting_instances=self.starting_instances)
            self.individuals_created += 1

        # Merge the new models into the existing generation
        self.candidates.update(new_generation)
        self.size = len(self.candidates)  # Update the count
        
        return
    

    def drop_worst_individuals(self):
        # Move all worst models to CPU first
        for idx in self.worst_individuals:
            candidate = self.candidates[idx]
            if hasattr(candidate.model, "cpu"):
                candidate.model = candidate.model.cpu()
        
        # Remove worst individuals
        for idx in self.worst_individuals:
            del self.candidates[idx]
        
        self.size = len(self.candidates)  # Update the count

        # Clean up GPU memory once
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def run_generation(self,
                       X_train, y_train, X_val, y_val,
                       percentile_drop=25, goal_metric=None,
                       epoch_threshold=3, track_all_models=False):


        self.spawn_new_candidates(self.search_space)
        # Generation is trained, and dropped
        self.train_generation(X_train, y_train)
        self.validate_generation(X_val, y_val)
        forecast_generation(self.candidates, effort_threshold=epoch_threshold)
        score_individuals(self.candidates, baseline_metric=goal_metric)
        self.current_snapshot = self.build_ledger()

        # Optionally maintain cumulative history (merge without duplicates)
        if track_all_models:
            self.cumulative_ledger = (
                pd.concat([self.cumulative_ledger, 
                           self.current_snapshot])
                .drop_duplicates(subset="id", keep="last")
                # .reset_index(drop=True)
            )
            
        get_worst_individuals(self, percentile_drop)
        self.drop_worst_individuals()

        return self.candidates
    

    def run_ebe(self,
            X_train, y_train, X_val, y_val,
            percentile_drop=25, max_epochs=200, 
            baseline_metric=None,
            time_budget=60, epoch_threshold=3, 
            track_all_models=False):

        # Snapshots: baseline vs evolving
        self.current_snapshot = self.initial_ledger
        self.cumulative_ledger = self.initial_ledger if track_all_models else None

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
            self.candidates = self.run_generation(
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

        return self.cumulative_ledger  # return the latest by default




    def build_ledger(self, export_as='pandas'):
        current_candidates = []
        active_individuals = self.candidates.keys()
        for i in active_individuals:
            candidate = self.candidates[i]
            current_candidates.append(candidate.build_dict())           
        
        if export_as == 'pandas':
            df = pd.DataFrame(current_candidates).sort_values(by='score', ascending=False)
            return df.copy(deep=True)
        elif export_as == 'json':
            return json.dumps(current_candidates, indent=4)


