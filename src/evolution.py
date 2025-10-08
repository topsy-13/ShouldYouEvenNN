import numpy as np

def weighted_random_selection(candidates, n_parents=2, 
                              rng: np.random.Generator = None):
    rng = rng or np.random.default_rng()
    keys = list(candidates.keys())
    scores = np.array([c.metrics.get("score", 0.0) for c in candidates.values()], dtype=float)

    # if all zero or negative → uniform
    if scores.sum() <= 0:
        probs = np.ones_like(scores) / len(scores)
    else:
        probs = scores / scores.sum()

    # ensure exact normalization
    probs = np.clip(probs, 0, 1)
    probs = probs / probs.sum()

    idx = rng.choice(len(keys), size=n_parents, replace=False, p=probs)
    return [candidates[keys[i]] for i in idx]



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


def crossover(parent1, parent2, rng: np.random.Generator = None):
    rng = rng or np.random.default_rng()
    child = {}
    for key in parent1.keys():
        if key == 'hidden_layers':
            cut = int(rng.integers(1, min(len(parent1[key]), len(parent2[key]))))
            child[key] = parent1[key][:cut] + parent2[key][cut:]
        elif key == 'scheduler_params' and isinstance(parent1[key], dict) and isinstance(parent2[key], dict):
            all_keys = set(parent1[key]) | set(parent2[key])
            child[key] = {k: (parent1[key].get(k) if rng.random() < 0.5 else parent2[key].get(k)) for k in all_keys}
        else:
            child[key] = parent1[key] if rng.random() < 0.5 else parent2[key]
    return child


def mutate_architecture(architecture, mutation_rate=0.2, 
                        rng: np.random.Generator = None):
    rng = rng or np.random.default_rng()
    mutated = dict(architecture)
    if 'hidden_layers' in mutated and rng.random() < mutation_rate:
        layers = mutated['hidden_layers'][:]
        if layers and rng.random() < 0.5:
            layers.append(int(rng.choice([32, 64, 128, 256])))
        else:
            idx = int(rng.integers(0, len(layers)))
            layers[idx] = max(4, int(layers[idx] * (0.5 if rng.random() < 0.5 else 1.5)))
        mutated['hidden_layers'] = layers
    if 'lr' in mutated and rng.random() < mutation_rate:
        mutated['lr'] = mutated['lr'] * (0.5 if rng.random() < 0.5 else 1.5)
    if 'dropout' in mutated and rng.random() < mutation_rate:
        mutated['dropout'] = float(np.clip(mutated['dropout'] + rng.uniform(-0.1, 0.1), 0.0, 0.7))
    if 'batch_size' in mutated and rng.random() < mutation_rate:
        mutated['batch_size'] = int(max(8, mutated['batch_size'] * (0.5 if rng.random() < 0.5 else 2.0)))
    mutated['shape'] = 'mutated'
    return mutated

def breed_and_mutate(candidates, rng: np.random.Generator = None):
    rng = rng or np.random.default_rng()
    p1, p2 = weighted_random_selection(candidates, rng=rng)
    child = crossover(p1.architecture, p2.architecture, rng=rng)
    return mutate_architecture(child, mutation_rate=0.3, rng=rng)