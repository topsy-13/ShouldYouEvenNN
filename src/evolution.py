import random 
import numpy as np
from utils import set_seed
from numpy.random import choice


def weighted_random_selection(candidates, n_parents=2, seed=None):
    """
    Select parents based on their probability-derived scores.
    - Uses candidate.metrics["score"], already normalized in scoring.py
    """
    EPS = 1e-8

    seed = seed if seed is not None else random.randint(0, 100000)
    _, _ = set_seed(seed)
    
    keys = list(candidates.keys())
    scores = np.array([candidates[k].metrics.get("score", 0.0) for k in keys], dtype=float)

    # Normalize to sum=1 (just in case scoring step didn't)
    if scores.sum() <= 0:
        scores = np.ones_like(scores) / len(scores)
    else:
        scores = scores / (scores.sum() + EPS)

    selected_keys = choice(keys, size=n_parents, replace=False, p=scores)
    return [candidates[k] for k in selected_keys]


def select_elites(candidates, elite_fraction=0.1):
    """
    Select elites purely by validation accuracy (last epoch).
    """
    n_elites = max(1, int(elite_fraction * candidates.size))
    sorted_cands = sorted(
        candidates.values(),
        key=lambda c: c.get_metric('val', 'acc', last_only=True) or 0.0,
        reverse=True
    )
    return sorted_cands[:n_elites]


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

    # Rename the architecture shape to mutated
    mutated['shape'] = 'mutated'
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

