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
def load_openml_dataset(dataset_id=334):
    """Loads dataset from OpenML and returns it as a Pandas DataFrame."""
    dataset = openml.datasets.get_dataset(dataset_id)
    print(f"Loading Dataset: {dataset.name}")
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    return X, y

def preprocess_features(X, categorical_strategy='label'):
    """Encodes categorical features if present."""
    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_columns:
        print(f"Categorical features detected: {categorical_columns}")
        
        if categorical_strategy == 'onehot':
            X = pd.get_dummies(X, columns=categorical_columns)  # One-hot encoding
        elif categorical_strategy == 'label':
            for col in categorical_columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))  # Label encoding
        else:
            raise ValueError("categorical_strategy must be 'onehot' or 'label'.")

    return X.values  # Convert DataFrame to NumPy array

def preprocess_target(y, encode_labels=True):
    """Encodes target labels if they are categorical."""
    if isinstance(y, pd.Series):
        y = y.values  # Keep it raw

    if encode_labels and not is_numeric_dtype(y):
        print("Class column is not numeric. Applying LabelEncoder.")
        y = LabelEncoder().fit_transform(y)

    return y
    

def split_data(X, y, test_size=0.2, val_size=0.2, random_seed=None):
    """Splits data into train, validation, and test sets."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test, scaler_type='standard'):
    """Scales features using StandardScaler or MinMaxScaler."""
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }
    scaler = scalers.get(scaler_type, StandardScaler())  # Default: StandardScaler

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test

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

def get_preprocessed_data(dataset_id=334, scaling=True, scaler_type='standard', 
                          categorical_strategy='label',
                          return_as='tensor', random_seed=None, X=None, y=None, task_type='classification'):
    """Full pipeline to load, preprocess, and return dataset."""
    
    if dataset_id is not None:
        # Load data
        X, y = load_openml_dataset(dataset_id)
    
    X = X.copy()
    y = y.copy()

    # Convert categorical features to numeric
    X = preprocess_features(X, categorical_strategy)

    # Convert target variable if needed
    if task_type == 'classification':
        encode_labels = True
    else: encode_labels = False

    y = preprocess_target(y, encode_labels)

    # Split the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_seed=random_seed)

    # Scale features if needed
    if scaling:
        X_train, X_val, X_test = scale_features(X_train, X_val, X_test, scaler_type=scaler_type)

    # Convert to tensors if required
    X_train, X_val, X_test, y_train, y_val, y_test = convert_to_tensor(
        X_train, X_val, X_test, y_train, y_val, y_test, return_as, task_type=task_type
    )

    print(f'Data loaded successfully! Format: {return_as}')
    print(f'Training data shape: {X_train.shape}')
    print(f'y_training data shape: {y_train.shape}')
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_dataset_and_loader(X, y, batch_size, shuffle=True):
    """
    Creates a TensorDataset from X and y tensors and wraps it in a DataLoader.
    """
    # Create the TensorDataset
    dataset = TensorDataset(X, y)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, loader


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


def create_dataloaders(X, y, 
                       batch_size,
                       return_as='loaders'):

    # Create DataLoaders
    dataset, dataloader = create_dataset_and_loader(X, y,
                                                       batch_size=batch_size)
    if return_as == 'loaders':
        return dataloader
    else: 
        return dataset
    

# endregion
