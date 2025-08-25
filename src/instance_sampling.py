import numpy as np
import data_preprocessing as dp
from utils import set_seed

def resolve_instance_budget(X, budget, mode="absolute", max_cap=None):
    """
    Resolves the number of instances to use from X, based on budget.

    Parameters:
        X (np.ndarray or pd.DataFrame): Input data.
        budget (float or int): Budget value (percentage or absolute).
        mode (str): 'percent' or 'absolute'.
        max_cap (int, optional): Max instances to allow regardless of budget.

    Returns:
        int: Number of instances to use.
    """
    total = len(X)

    if mode == "percent":
        n_instances = int(total * budget)
    elif mode == "absolute":
        n_instances = int(min(budget, total))
    else:
        raise ValueError("Unknown budget mode. Use 'percent' or 'absolute'.")

    if max_cap:
        n_instances = min(n_instances, max_cap)

    return n_instances

from sklearn.model_selection import StratifiedShuffleSplit

def sample_data(X, y, budget, mode="absolute", max_cap=None, task_type='classification', seed=None):

    n_samples = resolve_instance_budget(X, budget, mode=mode, max_cap=max_cap)
    total_samples = len(X)

    if n_samples >= total_samples:
        # Just return the full dataset
        return X, y

    if task_type == 'classification':
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
        train_index, _ = next(sss.split(X, y))
        selected_indices = train_index

    elif task_type == 'regression':
        set_seed(seed)
        selected_indices = np.random.choice(total_samples, size=n_samples, replace=False)

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return X[selected_indices], y[selected_indices]


def create_dataloaders(X, y, 
                       batch_size,
                       return_as='loaders'):

    # Create DataLoaders
    dataset, dataloader = dp.create_dataset_and_loader(X, y,
                                                       batch_size=batch_size)
    if return_as == 'loaders':
        return dataloader
    else: 
        return dataset
    
