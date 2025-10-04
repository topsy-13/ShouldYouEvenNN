import torch
import torch.nn as nn
import torch.optim as optim

import copy
import ast
import pandas as pd
import time

class DynamicNN(nn.Module):  # MLP
    def __init__(self, input_size, output_size, 
                 hidden_layers, 
                 activation_fn, dropout_rate,
                 lr, optimizer_type, 
                 weight_decay=0, momentum=None,
                 use_skip_connections=False,
                 initializer='xavier_uniform', lr_scheduler='none',
                 scheduler_params={},
                 device=None,
                 task_type='classification',
                 rng=None
                 ):
        
        # super(DynamicNN, self).__init__()
        super().__init__()
        self.rng = rng or torch.Generator().manual_seed(0)  # fallback

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_skip_connections = use_skip_connections
        self.task_type = task_type

        layers = []
        prev_size = input_size
        for size in hidden_layers:
            layer = nn.Linear(prev_size, size)

            # Apply initializer deterministically
            if initializer == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight, generator=self.rng)
            elif initializer == 'xavier_normal':
                nn.init.xavier_normal_(layer.weight, generator=self.rng)
            elif initializer == 'kaiming_uniform':
                nn.init.kaiming_uniform_(layer.weight, generator=self.rng)
            elif initializer == 'kaiming_normal':
                nn.init.kaiming_normal_(layer.weight, generator=self.rng)

            layers.append(layer)
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

        # Get the optimizer class from string if needed
        optimizer_class = self._get_optimizer_class(optimizer_type)

        # Configure optimizer
        optimizer_kwargs = {'lr': lr}
        if weight_decay > 0:
            optimizer_kwargs['weight_decay'] = weight_decay
        
        # Handle momentum parameter
        if momentum is not None:
            # For SGD optimizer
            if optimizer_class == optim.SGD:
                optimizer_kwargs['momentum'] = momentum
            # For RMSprop optimizer which also supports momentum
            elif optimizer_class == optim.RMSprop:
                optimizer_kwargs['momentum'] = momentum

        self.optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

        # Scheduler
        if lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 10),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        elif lr_scheduler == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_params.get('gamma', 0.9)
            )
        elif lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get('T_max', 10)
            )
        else:
            self.scheduler = None

        if self.task_type == 'classification':
           self.criterion = nn.CrossEntropyLoss()
        elif self.task_type == 'regression':
            self.criterion = nn.MSELoss()
    
    def _get_optimizer_class(self, optimizer_type):
        """
        Convert optimizer_type to an actual optimizer class.
        
        Args:
            optimizer_type: Can be:
                - A direct reference to an optimizer class (e.g., optim.Adam)
                - A string representation of a class (e.g., "<class 'torch.optim.rmsprop.RMSprop'>")
                - A string name of an optimizer (e.g., "Adam", "SGD", "RMSprop")
        
        Returns:
            The optimizer class
        """
        # If optimizer_type is already a class, return it
        if isinstance(optimizer_type, type):
            return optimizer_type
        
        # If optimizer_type is a string representation of a class like "<class 'torch.optim.rmsprop.RMSprop'>"
        if isinstance(optimizer_type, str) and optimizer_type.startswith("<class '") and "'" in optimizer_type:
            try:
                # Extract the class path (handle both with and without closing bracket)
                if optimizer_type.endswith("'>"):
                    class_path = optimizer_type[8:-2]  # Remove "<class '" and "'>"
                else:
                    # Handle case where the closing bracket is missing
                    class_path = optimizer_type[8:].split("'")[0]
                
                # Split the path into components
                components = class_path.split('.')
                
                # Import the module and get the class
                module_path = '.'.join(components[:-1])  # e.g., 'torch.optim.rmsprop'
                class_name = components[-1]  # e.g., 'RMSprop'
                
                # Dynamically import the module
                module = __import__(module_path, fromlist=[class_name])
                
                # Get the class from the module
                return getattr(module, class_name)
            
            except (ImportError, AttributeError, ValueError) as e:
                raise ValueError(f"Failed to parse optimizer class from '{optimizer_type}': {e}")
        
        # If optimizer_type is a simple string name of an optimizer
        if isinstance(optimizer_type, str):
            optimizer_map = {
                'sgd': optim.SGD,
                'adam': optim.Adam,
                'adamw': optim.AdamW,
                'rmsprop': optim.RMSprop,
                'adagrad': optim.Adagrad,
                'adadelta': optim.Adadelta
            }
            
            optimizer_key = optimizer_type.lower()
            if optimizer_key in optimizer_map:
                return optimizer_map[optimizer_key]
            
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        raise TypeError(f"optimizer_type must be a class, class string, or name string, got {type(optimizer_type)}")

    def forward(self, x):
        if not self.use_skip_connections:
            return self.network(x)

        result = x
        idx = 0

        for module in self.network:
            if isinstance(module, nn.Linear) and idx > 0:
                output = module(result)
                if result.shape == output.shape:
                    result = output + result
                else:
                    result = output
            else:
                result = module(result)
            idx += 1

        return result

        
    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()


    # def oe_train(self, train_loader, val_loader=None,
    #          num_epochs=1, val_frequency=1):
    #     """
    #     Train the model, logging per-batch metrics.
    #     - val_loader: optional validation loader
    #     - val_frequency: run validation every k batches (default=1 = every batch)
    #     Returns:
    #         batch_logs: list of dicts with train/val metrics
    #         total_time: total wall-clock time spent in training
    #     """
    #     self.train()
    #     batch_logs = []
    #     total_batches = 0
    #     total_time = 0.0

    #     for epoch in range(num_epochs):
    #         for features, labels in train_loader:
    #             start = time.time()
    #             total_batches += 1

    #             # --- forward + backward ---
    #             features, labels = features.to(self.device), labels.to(self.device)
    #             if features.dim() > 2:
    #                 features = features.view(features.size(0), -1)

    #             self.optimizer.zero_grad()
    #             outputs = self(features)
    #             loss = self.criterion(outputs, labels)
    #             loss.backward()
    #             self.optimizer.step()

    #             batch_time = time.time() - start
    #             total_time += batch_time

    #             with torch.no_grad():
    #                 _, predicted = torch.max(outputs, 1)
    #                 train_acc = (predicted == labels).float().mean().item()

    #             log_entry = {
    #                 "train_loss": loss.item(),
    #                 "train_acc": train_acc,
    #                 "val_loss": None,  # will be filled if validated
    #                 "val_acc": None,
    #                 "batch_effort": total_batches,
    #                 "batch_time": batch_time,
    #             }

    #             # --- validation (conditional) ---
    #             if val_loader is not None and (total_batches % val_frequency == 0):
    #                 self.eval()
    #                 correct, total, val_loss_sum = 0, 0, 0.0
    #                 with torch.no_grad():
    #                     for v_features, v_labels in val_loader:
    #                         v_features, v_labels = v_features.to(self.device), v_labels.to(self.device)
    #                         if v_features.dim() > 2:
    #                             v_features = v_features.view(v_features.size(0), -1)

    #                         v_outputs = self(v_features)
    #                         v_loss = self.criterion(v_outputs, v_labels)
    #                         val_loss_sum += v_loss.item() * v_features.size(0)

    #                         _, v_pred = torch.max(v_outputs, 1)
    #                         correct += (v_pred == v_labels).sum().item()
    #                         total += v_labels.size(0)

    #                 log_entry["val_acc"] = correct / total
    #                 log_entry["val_loss"] = val_loss_sum / total
    #                 self.train()

    #             batch_logs.append(log_entry)

    #     return batch_logs, total_time


    # TODO: recheck this behavior with new batch logging    
    def horizon_train(self, candidate, train_loader, val_loader,
                      task_type='classification', return_lc=False):
        """
        Train until the forecast horizon (seconds) is reached.
        Validation runs after every batch. No early stopping.
        """

        horizon = candidate.metrics.get("forecast_horizon_time", None)
        if horizon is None:
            raise ValueError("Forecast horizon not found in candidate.metrics")

        if return_lc:
            learning_curve = {
                'es_train_losses': [], 'es_val_losses': [],
                'es_train_accs': [], 'es_val_accs': []
            }

        total_time = 0.0
        keep_training = True
        epoch = 0

        while keep_training and total_time < horizon:
            epoch += 1
            self.train()

            for features, labels in train_loader:
                start = time.time()
                features, labels = features.to(self.device), labels.to(self.device)
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)

                self.optimizer.zero_grad()
                outputs = self(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                batch_time = time.time() - start
                total_time += batch_time
                candidate.log_effort(batch_time)

                with torch.no_grad():
                    if task_type == 'classification':
                        _, predicted = torch.max(outputs, 1)
                        train_acc = (predicted == labels).float().mean().item()
                    else:
                        train_acc = None

                candidate.log_metric("train", "loss", loss.item())
                candidate.log_metric("train", "acc", train_acc)

                # ---- validation immediately after each batch ----
                self.eval()
                correct, total, val_loss_sum = 0, 0, 0.0
                with torch.no_grad():
                    for v_features, v_labels in val_loader:
                        v_features, v_labels = v_features.to(self.device), v_labels.to(self.device)
                        if v_features.dim() > 2:
                            v_features = v_features.view(v_features.size(0), -1)
                        v_outputs = self(v_features)
                        v_loss = self.criterion(v_outputs, v_labels)
                        val_loss_sum += v_loss.item() * v_features.size(0)
                        if task_type == 'classification':
                            _, v_pred = torch.max(v_outputs, 1)
                            correct += (v_pred == v_labels).sum().item()
                        total += v_labels.size(0)

                val_loss = val_loss_sum / total
                val_acc = (correct / total) if task_type == 'classification' else None
                candidate.log_metric("val", "loss", val_loss)
                candidate.log_metric("val", "acc", val_acc)

                if return_lc:
                    learning_curve['es_train_losses'].append(loss.item())
                    learning_curve['es_val_losses'].append(val_loss)
                    if task_type == 'classification':
                        learning_curve['es_train_accs'].append(train_acc)
                        learning_curve['es_val_accs'].append(val_acc)

                # stop if horizon exceeded mid-epoch
                if total_time >= horizon:
                    keep_training = False
                    break

        if return_lc:
            candidate.metrics["learning_curve_es"] = learning_curve

        return (
            candidate.get_metric("train", "loss", last_only=True),
            candidate.get_metric("train", "acc", last_only=True),
            candidate.get_metric("val", "loss", last_only=True),
            candidate.get_metric("val", "acc", last_only=True),
            learning_curve if return_lc else None
        )



    def evaluate(self, val_loader):
        task_type = self.task_type
        self.eval()
        
        total = 0
        running_loss = 0.0
        correct = 0  # Only used for classification

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                if features.dim() > 2:
                    features = features.view(features.size(0), -1)

                outputs = self(features)
                loss = self.criterion(outputs, labels)
                batch_size = features.size(0)
                running_loss += loss.item() * batch_size
                total += batch_size

                if task_type == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()

        val_loss = running_loss / total
        val_accuracy = (correct / total) if task_type == 'classification' else None

        return val_loss, val_accuracy


import numpy as np
def create_model_from_row(row, input_size, output_size, task_type='classification'):

    # Hidden layers
    hidden_layers = row.get('arch_hidden_layers', [128, 64])
    if isinstance(hidden_layers, str):
        hidden_layers = ast.literal_eval(hidden_layers)

    # Activation function
    activation_raw = row.get('arch_activation_fn', nn.ReLU)
    if isinstance(activation_raw, str):
        activation_name = activation_raw
    elif hasattr(activation_raw, '__name__'):
        activation_name = activation_raw.__name__
    else:
        activation_name = 'ReLU'

    activation_map = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
    }
    activation_fn = activation_map.get(activation_name, nn.ReLU)

    # Dropout
    dropout_rate = row.get('arch_dropout_rate', 0.0)

    # Optimizer and learning rate
    lr = row.get('arch_lr', 0.001)
    optimizer_type = row.get('arch_optimizer_type', 'adam')
    weight_decay = row.get('arch_weight_decay', 0.0)
    momentum = None if pd.isna(row.get('arch_momentum', None)) else row['arch_momentum']
   

    # Skip connections
    use_skip = row.get('arch_use_skip_connections', False)

    # Initializer
    initializer = row.get('arch_initializer', 'xavier_uniform')

    # LR Scheduler
    lr_scheduler = row.get('arch_lr_scheduler', 'none')
    scheduler_params = row.get('arch_scheduler_params', {})
    if isinstance(scheduler_params, str):
        scheduler_params = ast.literal_eval(scheduler_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = DynamicNN(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        activation_fn=activation_fn,
        dropout_rate=dropout_rate,
        lr=lr,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        momentum=momentum,
        use_skip_connections=use_skip,
        initializer=initializer,
        lr_scheduler=lr_scheduler,
        scheduler_params=scheduler_params,
        device=device,
        task_type=task_type
    ).to(device)

    return model