import torch
import torch.nn as nn
import torch.optim as optim

import math

from architecture_generator import DynamicNN
import numpy as np

# region Search Space
class SearchSpace():
    
    def __init__(self, input_size, output_size,
                 min_layers=2, max_layers=25, 
                 min_neurons=3, max_neurons=100,
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


    def _pick(self, rng, seq):
        idx = rng.integers(0, len(seq))
        return seq[idx]

    def _randint(self, rng, a, b_inclusive):
        return int(rng.integers(a, b_inclusive + 1))

    def _uniform(self, rng, lo, hi):
        return float(rng.uniform(lo, hi))

    def _generate_hidden_layers(self, rng, shape, depth):
        min_w, max_w = self.neurons
        if shape == "constant":
            width = self._randint(rng, min_w, max_w)
            return [width] * depth
        elif shape == "pyramid":
            return sorted([self._randint(rng, min_w, max_w) for _ in range(depth)], reverse=True)
        elif shape == "inv_pyramid":
            return sorted([self._randint(rng, min_w, max_w) for _ in range(depth)])
        elif shape == "hourglass":
            half = depth // 2
            down = sorted([self._randint(rng, min_w, max_w) for _ in range(half+1)], reverse=True)
            up = down[:-1][::-1] if depth % 2 == 0 else down[::-1]
            return down + up
        elif shape == "triangular":
            grow = rng.random() < 0.5
            vals = [self._randint(rng, min_w, max_w) for _ in range(depth)]
            return sorted(vals) if grow else sorted(vals, reverse=True)
        elif shape == "irregular":
            return [self._randint(rng, min_w, max_w) for _ in range(depth)]
        else:
            raise ValueError(f"Unknown shape: {shape}")

    def sample_architecture(self, seed=None, rng: np.random.Generator = None):
        # Prefer passed-in RNG; fall back to per-call local RNG (still reproducible)
        if rng is None:
            rng = np.random.default_rng(seed if seed is not None else None)

        depth = self._randint(rng, self.layers[0], self.layers[1])
        shape = self._pick(rng, self.arch_shapes)
        hidden_layers = self._generate_hidden_layers(rng, shape, depth)

        activation_fn = self._pick(rng, self.activation_fns)
        dropout_rate  = self._pick(rng, self.dropout_rates)
        optimizer_type = self._pick(rng, self.optimizers)

        # log-uniform learning rate
        log_lr = rng.uniform(self.log_min_lr, self.log_max_lr)
        learning_rate = 10 ** log_lr

        weight_decay = self._pick(rng, self.weight_decays)
        momentum = self._pick(rng, self.momentum_values) if optimizer_type.__name__ == "SGD" else None

        batch_size = self._pick(rng, self.batch_sizes)
        use_skip_connections = self._pick(rng, self.skip_connection_options)
        initializer = self._pick(rng, self.initializers)
        lr_scheduler = self._pick(rng, self.lr_schedulers)

        scheduler_params = {}
        if lr_scheduler == 'step':
            scheduler_params['step_size'] = self._pick(rng, [5, 10, 20, 30])
            scheduler_params['gamma']     = self._pick(rng, [0.1, 0.5, 0.9])
        elif lr_scheduler == 'exponential':
            scheduler_params['gamma']     = self._pick(rng, [0.9, 0.95, 0.99])
        elif lr_scheduler == 'cosine':
            scheduler_params['T_max']     = self._pick(rng, [10, 50, 100])

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
            # 'seed': draw_seed  # optional legacy field; now meaningless
        }


    def create_model(self, architecture, task_type='classification',
                     rng=None):
        # hidden layers etc.
        hidden_layers = architecture["hidden_layers"]
        activation_fn = architecture["activation_fn"]
        dropout_rate  = architecture["dropout_rate"]
        optimizer_type = architecture["optimizer_type"]
        learning_rate = architecture["learning_rate"]

        weight_decay = architecture.get("weight_decay", 0)
        momentum = architecture.get("momentum", None)
        use_skip_connections = architecture.get("use_skip_connections", False)
        initializer = architecture.get("initializer", "xavier_uniform")
        lr_scheduler = architecture.get("lr_scheduler", "none")
        scheduler_params = architecture.get("scheduler_params", {})

        # force deterministic device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            device=device,
            task_type=task_type,
            rng=rng  # NEW: give the model its reproducibility stream
        ).to(device)

        return model

    
# endregion