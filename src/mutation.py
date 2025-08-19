import random
import copy

def crossover_candidates(parent1, parent2, method='uniform'):
    child = {}
    keys = list(parent1.keys())

    if method == 'uniform':
        # 50% chance to take from parent1
        for k in keys:
            child[k] = parent1[k] if random.random() < 0.5 else parent2[k]

    elif method == 'single_point':
        # Pick a split point
        point = random.randint(1, len(keys)-1)
        split_keys = keys[:point]
        rest_keys = keys[point:]
        for k in split_keys:
            child[k] = parent1[k]
        for k in rest_keys:
            child[k] = parent2[k]

    return child