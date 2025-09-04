import torch
import torch.nn as nn
import torch.optim as optim

import math
import random

from architecture_generator import DynamicNN
from utils import set_seed

# region Search Space
class SearchSpace():
    
    def __init__(self, input_size, output_size,
                 min_layers=2, max_layers=50, 
                 min_neurons=3, max_neurons=500,
                 activation_fns=[nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.GELU],
                 dropout_rates=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                 min_learning_rate=0.0001, max_learning_rate=0.1,
                 min_batch_size=32, max_batch_size=1024,
                 weight_decays=[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                 momentum_values=[0.8, 0.9, 0.95, 0.99],
                 layer_norm_options=[True, False],
                 skip_connection_options=[True, False],
                 initializers=['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'],
                 lr_schedulers=['step', 'exponential', 'cosine', 'none'],
                 arch_shapes=["constant", "pyramid", "inv_pyramid", "hourglass", "triangular", "irregular"]):

        # Store parameters    
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [min_layers, max_layers]
        self.neurons = [min_neurons, max_neurons]
        self.arch_shapes = arch_shapes
        self.activation_fns = activation_fns
        
        # Store log bounds for learning rate
        self.log_min_lr = math.log10(min_learning_rate)
        self.log_max_lr = math.log10(max_learning_rate)
        
        self.dropout_rates = dropout_rates
        self.optimizers = [optim.Adam, optim.SGD, optim.RMSprop, optim.AdamW]
        self.weight_decays = weight_decays
        self.momentum_values = momentum_values
        self.layer_norm_options = layer_norm_options
        self.skip_connection_options = skip_connection_options
        self.initializers = initializers
        self.lr_schedulers = lr_schedulers

        # Build batch sizes considering powers of 2
        power = 1
        self.batch_sizes = []
        while power <= max_batch_size:
            if power >= min_batch_size:
                self.batch_sizes.append(power)
            power *= 2


    def _generate_hidden_layers(self, shape, depth):
        min_w, max_w = self.neurons
        if shape == "constant":
            width = random.randint(min_w, max_w)
            return [width] * depth
        elif shape == "pyramid":
            return sorted([random.randint(min_w, max_w) for _ in range(depth)], reverse=True)
        elif shape == "inv_pyramid":
            return sorted([random.randint(min_w, max_w) for _ in range(depth)])
        elif shape == "hourglass":
            half = depth // 2
            down = sorted([random.randint(min_w, max_w) for _ in range(half+1)], reverse=True)
            up = down[:-1][::-1] if depth % 2 == 0 else down[::-1]
            return down + up
        elif shape == "triangular":
            if random.random() < 0.5:  # grow
                return sorted([random.randint(min_w, max_w) for _ in range(depth)])
            else:  # shrink
                return sorted([random.randint(min_w, max_w) for _ in range(depth)], reverse=True)
        elif shape == "irregular":
            return [random.randint(min_w, max_w) for _ in range(depth)]
        else:
            raise ValueError(f"Unknown shape: {shape}")


    def sample_architecture(self, seed=None):
        # Set seed for reproducibility
        if seed is None:
            seed = random.randint(0, 100000)

        set_seed(seed)

        depth = random.randint(self.layers[0], self.layers[1])
        shape = random.choice(self.arch_shapes)
        hidden_layers = self._generate_hidden_layers(shape, depth)

        activation_fn = random.choice(self.activation_fns)
        dropout_rate = random.choice(self.dropout_rates)
        optimizer_type = random.choice(self.optimizers)
        
        # Sample learning rate on a logarithmic scale
        log_lr = random.uniform(self.log_min_lr, self.log_max_lr)
        learning_rate = 10 ** log_lr
        
        # Sample weight decay on logarithmic scale if it's not zero
        weight_decay = random.choice(self.weight_decays)
        momentum = random.choice(self.momentum_values) if optimizer_type in [optim.SGD] else None
        
        # Sample other parameters
        batch_size = random.choice(self.batch_sizes)
        # use_layer_norm = random.choice(self.layer_norm_options)
        use_skip_connections = random.choice(self.skip_connection_options)
        initializer = random.choice(self.initializers)
        lr_scheduler = random.choice(self.lr_schedulers)
        
        # Sample hyperparameters specific to schedulers
        scheduler_params = {}
        if lr_scheduler == 'step':
            scheduler_params['step_size'] = random.choice([5, 10, 20, 30])
            scheduler_params['gamma'] = random.choice([0.1, 0.5, 0.9])
        elif lr_scheduler == 'exponential':
            scheduler_params['gamma'] = random.choice([0.9, 0.95, 0.99])
        elif lr_scheduler == 'cosine':
            scheduler_params['T_max'] = random.choice([10, 50, 100])
        
        return {
            'hidden_layers': hidden_layers,
            'shape': shape,
            'depth': depth,
            'activation_fn': activation_fn,
            'dropout_rate': dropout_rate,
            'optimizer_type': optimizer_type,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'batch_size': batch_size,
            'use_skip_connections': use_skip_connections,
            'initializer': initializer,
            'lr_scheduler': lr_scheduler,
            'scheduler_params': scheduler_params,
            'seed':seed
        }


    def create_model(self, architecture, task_type='classification'):
        hidden_layers = architecture["hidden_layers"]
        activation_fn = architecture["activation_fn"]
        dropout_rate = architecture["dropout_rate"]
        optimizer_type = architecture["optimizer_type"]
        learning_rate = architecture["learning_rate"]
        self.batch_size = architecture["batch_size"]  #* extract the batch size for dataloader


        # Extract new parameters with defaults if not present (for backward compatibility)
        weight_decay = architecture.get("weight_decay", 0)
        momentum = architecture.get("momentum", None)
        use_skip_connections = architecture.get("use_skip_connections", False)
        initializer = architecture.get("initializer", "xavier_uniform")
        lr_scheduler = architecture.get("lr_scheduler", "none")
        scheduler_params = architecture.get("scheduler_params", {})
        
        # Create model with all parameters
        model = DynamicNN(
            self.input_size, self.output_size, 
            hidden_layers, activation_fn, 
            dropout_rate, learning_rate, optimizer_type,
            weight_decay=weight_decay,
            momentum=momentum,
            use_skip_connections=use_skip_connections,
            initializer=initializer,
            lr_scheduler=lr_scheduler,
            scheduler_params=scheduler_params,
            device=self.device,
            task_type=task_type
        ).to(self.device)
        
        return model
    
# endregion