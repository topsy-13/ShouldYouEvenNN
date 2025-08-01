import numpy as np
import data_preprocessing as dp
# TODO: Incorporate this into the generations py module

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


def sample_data(X, y, budget, mode="absolute", max_cap=None):
    n_samples = resolve_instance_budget(X, budget, mode=mode, max_cap=max_cap)
    indices = np.random.choice(len(X), size=n_samples, replace=False)

    return X[indices], y[indices]


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
    
