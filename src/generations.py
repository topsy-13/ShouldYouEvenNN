import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import gc
import data_preprocessing as dp
from instance_sampling import resolve_instance_budget, sample_data, create_dataloaders

# from baseline_models import get_models_and_baseline_metric
from forecaster import forecast_accuracy

# region Generations
class Generation():
    def __init__(self, search_space, n_individuals, starting_instances=100, 
                 task_type='classification'):
        self.task_type = task_type
        self.search_space = search_space
        self.n_individuals = n_individuals
        self.generation = self.build_generation()
        self.n_instances = starting_instances

    def build_generation(self):
        generation = {}
        for i in range(self.n_individuals):
            architecture = self.search_space.sample_architecture()
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
                "forecasted_val_acc": []
            }
        return generation
    
    def train_generation(self, X_train, y_train):
        for i in range(self.n_individuals):
            model = self.generation[i]["model"]
            batch_size = self.generation[i]["batch_size"]
            dataset_fraction = self.n_instances / len(X_train)
            num_epochs = len(self.generation[i]['train_loss'])

            self.effort = dataset_fraction * num_epochs
            # Sample data based on the instance budget
            X_sampled, y_sampled = sample_data(X_train, y_train, self.n_instances, mode="absolute")
            
            # Create a DataLoader with the architecture-specific batch size
            train_loader = create_dataloaders(X=X_sampled, y=y_sampled, batch_size=batch_size)

            train_loss, train_acc = model.oe_train(train_loader)
            self.generation[i]["train_loss"].append(train_loss)  
            self.generation[i]["train_acc"].append(train_acc)
            self.generation[i]["n_instances"].append(self.n_instances)
            self.generation[i]["effort"].append(self.effort)
    

    def validate_generation(self, X_val, y_val):
        for i in range(self.n_individuals):
            model = self.generation[i]["model"]
            batch_size = self.generation[i]["batch_size"]
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

        # Sort individuals by validation loss in descending order (higher loss is worse)
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["val_loss"], reverse=True) #? Should the criterion be val_loss or val_acc?

        # Extract the keys of the worst individuals
        self.worst_individuals = [key for key, _ in sorted_generation[:n_worst_individuals]]

        

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

    def drop_all_except_best(self):
        # Sort individuals by validation loss in ascending order (lower loss is better)
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["val_loss"])
        
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
            if len(efforts) > epoch_threshold and len(val_accs) > epoch_threshold:
                forecasted_accuracy = forecast_accuracy(efforts, val_accs, model_type='rational')
                self.generation[i]["forecasted_val_acc"] = forecasted_accuracy
            else: 
                pass  # Not enough data to forecast
    
    def get_best_model(self):
        # Sort individuals by validation accuracy in descending order
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["val_acc"][-1], reverse=True)
        
        # Return the best model
        best_individual = sorted_generation[0][0]
        return self.generation[best_individual]["model"]

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

        df = pd.DataFrame(architectures_df)
        # Sort by validation accuracy in descending order
        df['final_val_acc'] = df['val_acc'].apply(lambda x: x[-1] if isinstance(x, list) else x)
        df['final_val_loss'] = df['val_loss'].apply(lambda x: x[-1] if isinstance(x, list) else x)
        
        self.history = df.sort_values('final_val_acc', ascending=False).reset_index(drop=True)
        return self.history

    def run_generation(self,
                       X_train, y_train, X_val, y_val,
                       percentile_drop=25, goal_metric=None):
    
        # Generation is trained, and dropped
        self.train_generation(X_train, y_train)
        self.validate_generation(X_val, y_val)
        self.forecast_generation()
        self.n_instances *= 2  # Increase instance budget for next generation
        self.n_instances = min(self.n_instances, len(X_train))


        self.get_worst_individuals(percentile_drop)
        self.drop_worst_individuals()

        return self.generation
    
    def run_ebe(self,
                X_train, y_train, X_val, y_val,
                percentile_drop=25, epochs=5):
        
    # Estimate best performance in other models 
        # goal_metric = get_models_and_baseline_metric(X_train, y_train)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs} - Running EBE")

            self.generation = self.run_generation(X_train, y_train, 
                                                  X_val, y_val,
                                                  percentile_drop=percentile_drop)

            self.num_models = len(self.generation)
            if self.num_models <= 10:
                print(f"Only {self.num_models} models left, stopping EBE.")
                break

            # Increase drop but limit to 50%
            percentile_drop = min(percentile_drop + 5, 50)
