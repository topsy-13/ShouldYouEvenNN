import random 
import numpy as np
from utils import set_seed


def weighted_random_selection(candidates, k=2, seed=None):
        """
        Select k candidates based on weighted probabilities from fitness scores.
        Higher fitness = higher probability of being chosen.
        Falls back to uniform selection if not enough non-zero probabilities.
        """
        if seed is None:
            seed = random.randint(0, 100000)
        
        set_seed(seed)
        candidates_pool = list(candidates.keys())
        
        # Extract scores
        scores = np.array([candidates[i].metrics['score'] for i in candidates_pool], dtype=float)
        
        # Normalize scores to get probabilities
        total = scores.sum()
        if total > 0:
            probabilities = scores / total
        else:
            probabilities = np.zeros_like(scores)
        
        # Handle edge case: fewer non-zero probabilities than k
        nonzero_candidates = [c for c, prob in zip(candidates_pool, probabilities) if prob > 0]
        if len(nonzero_candidates) < k:
            # fallback to uniform random selection
            selected_indices = np.random.choice(candidates_pool, size=k, replace=False)
        else:
            selected_indices = np.random.choice(candidates_pool, size=k, replace=False, p=probabilities)
        
        return [candidates[i] for i in selected_indices]


def crossover(parent1, parent2, seed=None):
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


def mutate_architecture(architecture, mutation_rate=0.3, seed=None):
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


def breed_and_mutate(candidates, seed):
    # Select parents at random weighting their score
    parents = weighted_random_selection(candidates, seed=seed)
    parent1 = parents[0].architecture
    parent2 = parents[1].architecture

    # Cross them
    child_architecture = crossover(parent1, parent2, seed=seed)
    # Mutate the child
    child_architecture = mutate_architecture(child_architecture, mutation_rate=0.3)

    return child_architecture

