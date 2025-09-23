import openml
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# region Loading data
def load_openml_dataset(dataset_id=334, verbose=False):
    """Loads dataset from OpenML and returns it as a Pandas DataFrame."""
    dataset = openml.datasets.get_dataset(dataset_id)
    if verbose:
        print(f"Loading Dataset: {dataset.name}")
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    return X, y


def preprocess_features(X_train, X_val, X_test,
                        categorical_strategy="label", verbose=False):
    """Fit encoders on train only, apply to val/test."""
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_columns:
        if verbose:
            print(f"Categorical features detected: {categorical_columns}")

        if categorical_strategy == "onehot":
            # Fit one-hot encoder on train only
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            X_train_enc = pd.DataFrame(encoder.fit_transform(X_train[categorical_columns]))
            X_val_enc   = pd.DataFrame(encoder.transform(X_val[categorical_columns]))
            X_test_enc  = pd.DataFrame(encoder.transform(X_test[categorical_columns]))

            # Drop original cat columns + concat encoded
            X_train = X_train.drop(columns=categorical_columns).reset_index(drop=True)
            X_val   = X_val.drop(columns=categorical_columns).reset_index(drop=True)
            X_test  = X_test.drop(columns=categorical_columns).reset_index(drop=True)

            X_train = pd.concat([X_train.reset_index(drop=True), X_train_enc], axis=1)
            X_val   = pd.concat([X_val.reset_index(drop=True), X_val_enc], axis=1)
            X_test  = pd.concat([X_test.reset_index(drop=True), X_test_enc], axis=1)

        elif categorical_strategy == "label":
            # Label encode column by column (fit on train, apply to others)
            for col in categorical_columns:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_val[col]   = le.transform(X_val[col].astype(str))
                X_test[col]  = le.transform(X_test[col].astype(str))

        else:
            raise ValueError("categorical_strategy must be 'onehot' or 'label'.")

    return X_train, X_val, X_test


import numpy as np
import pandas as pd

def preprocess_target(y_train, y_val, y_test, encode_labels=True, min_class_count=2, verbose=False):
    """
    Fit label encoder on y_train, apply to val/test safely.
    Rare classes (fewer than min_class_count in the whole dataset) are mapped to 'other'.
    """
    if not encode_labels:
        return y_train, y_val, y_test

    # Combine all splits to detect rare classes
    y_all = np.concatenate([np.array(y_train), np.array(y_val), np.array(y_test)])
    counts = pd.Series(y_all).value_counts()
    rare_classes = counts[counts < min_class_count].index.tolist()

    if rare_classes and verbose:
        print(f"Collapsing rare classes {rare_classes} -> 'other'")

    def replace_rare(y):
        return np.array([lbl if lbl not in rare_classes else "__other__" for lbl in y])

    y_train = replace_rare(y_train)
    y_val   = replace_rare(y_val)
    y_test  = replace_rare(y_test)

    # Fit LabelEncoder on train
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # Map val/test safely
    y_val_enc  = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    return y_train_enc, y_val_enc, y_test_enc


def split_data(X, y, test_size=0.2, val_size=0.2, random_seed=None, stratify=True):
    """Splits data into train, validation, and test sets with optional stratification."""
    
    stratify_y = y if stratify else None

    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_y
    )
    
    # Then split train/val
    stratify_y_train_val = y_train_val if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        random_state=random_seed,
        stratify=stratify_y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test, scaler_type="standard"):
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }
    scaler = scalers.get(scaler_type, StandardScaler())
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    return X_train, X_val, X_test


def get_preprocessed_data(dataset_id=334, scaling=True, 
                          scaler_type="standard",
                          categorical_strategy="label", return_as="tensor",
                          random_seed=None, X=None, y=None,
                          task_type="classification", verbose=False):

    if dataset_id is not None:
        dataset = openml.datasets.get_dataset(dataset_id)
        if verbose:
            print(f"Loading Dataset: {dataset.name}")
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Split raw first
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_seed=random_seed)

    # Encode features
    X_train, X_val, X_test = preprocess_features(X_train, X_val, X_test,
                                                 categorical_strategy, verbose=verbose)

    # Encode target
    encode_labels = True if task_type == "classification" else False
    y_train, y_val, y_test = preprocess_target(y_train, y_val, y_test,
                                               encode_labels, verbose=verbose)

    # Scale
    if scaling:
        X_train, X_val, X_test = scale_features(X_train, X_val, X_test, scaler_type=scaler_type)

    # Convert to torch if asked
    if return_as == "tensor":
        X_train, X_val, X_test = map(lambda arr: torch.tensor(arr, dtype=torch.float32),
                                     [X_train, X_val, X_test])
        if task_type == "classification":
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_val   = torch.tensor(y_val, dtype=torch.long)
            y_test  = torch.tensor(y_test, dtype=torch.long)
        else:
            y_train = torch.tensor(y_train, dtype=torch.float32)
            y_val   = torch.tensor(y_val, dtype=torch.float32)
            y_test  = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test

def convert_to_tensor(X_train, X_val, X_test, y_train, y_val, y_test, return_as='tensor', task_type='classification'):
    """Converts data to PyTorch tensors, handling regression vs classification."""
    # Ensure the targets are NumPy arrays
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.to_numpy()
    if isinstance(y_val, pd.DataFrame):
        y_val = y_val.to_numpy()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.to_numpy()

    if return_as == 'tensor':
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        
        if task_type == 'classification':
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_val = torch.tensor(y_val, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
        elif task_type == 'regression':
            y_train = torch.tensor(y_train, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    return X_train, X_val, X_test, y_train, y_val, y_test



def get_tensor_sizes(X_train, y_train, task_type='classification'):
    """
    Determine input and output sizes for PyTorch tensors

    Parameters:
    -----------
    X_train : torch.Tensor
        Input features tensor
    y_train : torch.Tensor
        Labels/target tensor
    task_type : str
        'classification' or 'regression'

    Returns:
    --------
    tuple: (input_size, output_size)
    """
    
    # Input size
    if len(X_train.shape) == 2:
        input_size = X_train.shape[1]
    elif len(X_train.shape) == 1:
        input_size = 1
    else:
        print('Images detected')
        input_size = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]

    # Output size
    if task_type == 'classification':
        output_size = len(torch.unique(y_train))
    elif task_type == 'regression':
        output_size = 1 if y_train.dim() == 1 else y_train.shape[1]
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    return input_size, output_size


from torch.utils.data import TensorDataset, DataLoader

def create_dataloader(X, y, batch_size, generator, seed_worker, 
                      shuffle=True):
    """
    Wrap (X, y) tensors into a reproducible DataLoader.
    
    Args:
        X (Tensor): Features.
        y (Tensor): Labels.
        batch_size (int): Mini-batch size.
        generator (torch.Generator): From set_seed().
        seed_worker (callable): From set_seed().
        shuffle (bool): Whether to shuffle the dataset.
    
    Returns:
        DataLoader
    """
    dataset = TensorDataset(X, y)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    
    return loader

    

# endregion
