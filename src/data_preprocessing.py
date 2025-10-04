"""Data loading and preprocessing helpers used by experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence, Tuple

import numpy as np
import openml
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "DatasetSplits",
    "PreprocessingConfig",
    "load_openml_dataset",
    "split_data",
    "preprocess_features",
    "preprocess_target",
    "scale_features",
    "convert_to_tensors",
    "create_dataloader",
    "get_preprocessed_data",
]


@dataclass
class DatasetSplits:
    """Container bundling train/validation/test splits."""

    X_train: pd.DataFrame | np.ndarray
    X_val: pd.DataFrame | np.ndarray
    X_test: pd.DataFrame | np.ndarray
    y_train: Sequence
    y_val: Sequence
    y_test: Sequence


@dataclass
class PreprocessingConfig:
    """Configuration describing the preprocessing pipeline."""

    dataset_id: Optional[int] = 334
    scaling: bool = True
    scaler_type: Literal["standard", "minmax"] = "standard"
    categorical_strategy: Literal["label", "onehot"] = "label"
    return_as: Literal["tensor", "array"] = "tensor"
    random_seed: Optional[int] = None
    task_type: Literal["classification", "regression"] = "classification"
    test_size: float = 0.2
    val_size: float = 0.2
    min_class_count: int = 2


def load_openml_dataset(dataset_id: int, *, verbose: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Return an OpenML dataset as feature/target dataframes."""

    dataset = openml.datasets.get_dataset(dataset_id)
    if verbose:
        print(f"Loading Dataset: {dataset.name}")

    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X = pd.DataFrame(X)
    y = pd.Series(y, name=dataset.default_target_attribute)
    return X, y


def preprocess_features(X_train, X_val, X_test,
                        categorical_strategy="label", verbose=False):
    """Fit encoders on train only, apply to val/test safely."""

    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test)

        if categorical_strategy == "onehot":
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            X_train_enc = pd.DataFrame(encoder.fit_transform(X_train[categorical_columns]))
            X_val_enc   = pd.DataFrame(encoder.transform(X_val[categorical_columns]))
            X_test_enc  = pd.DataFrame(encoder.transform(X_test[categorical_columns]))

            # Drop originals and concat encoded
            X_train = X_train.drop(columns=categorical_columns).reset_index(drop=True)
            X_val   = X_val.drop(columns=categorical_columns).reset_index(drop=True)
            X_test  = X_test.drop(columns=categorical_columns).reset_index(drop=True)

            X_train = pd.concat([X_train, X_train_enc], axis=1)
            X_val   = pd.concat([X_val, X_val_enc], axis=1)
            X_test  = pd.concat([X_test, X_test_enc], axis=1)

        elif categorical_strategy == "label":
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_columns:
                le = LabelEncoder()
                le.fit(X_train[col].astype(str))

                # extend classes_ with "__other__"
                le_classes = list(le.classes_)
                if "__other__" not in le_classes:
                    le_classes.append("__other__")
                le.classes_ = np.array(le_classes)

                def safe_transform(series):
                    return series.astype(str).map(lambda x: x if x in le.classes_ else "__other__")

                X_train[col] = le.transform(safe_transform(X_train[col]))
                X_val[col]   = le.transform(safe_transform(X_val[col]))
                X_test[col]  = le.transform(safe_transform(X_test[col]))

    X_train, X_val, X_test = splits.X_train.copy(), splits.X_val.copy(), splits.X_test.copy()
    categorical_columns = [col for col in X_train.columns if _is_categorical(X_train[col])]

    if not categorical_columns:
        return splits

    if verbose:
        print(f"Categorical features detected: {categorical_columns}")

    if categorical_strategy == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        encoder.fit(X_train[categorical_columns])

        def encode(df: pd.DataFrame) -> pd.DataFrame:
            encoded = pd.DataFrame(
                encoder.transform(df[categorical_columns]),
                columns=encoder.get_feature_names_out(categorical_columns),
                index=df.index,
            )
            return pd.concat([df.drop(columns=categorical_columns), encoded], axis=1)

        X_train, X_val, X_test = map(encode, (X_train, X_val, X_test))
    elif categorical_strategy == "label":
        for column in categorical_columns:
            categories = {
                value: index
                for index, value in enumerate(sorted(map(str, X_train[column].unique())))
            }

            def encode(series: pd.Series) -> pd.Series:
                mapped = series.astype(str).map(categories)
                return mapped.fillna(-1).astype(int)

            X_train[column] = encode(X_train[column])
            X_val[column] = encode(X_val[column])
            X_test[column] = encode(X_test[column])
    else:
        raise ValueError("categorical_strategy must be either 'label' or 'onehot'.")


import numpy as np
import pandas as pd

def preprocess_target(y_train, y_val, y_test, encode_labels=True, min_class_count=2, verbose=False):
    """
    Fit label encoder on y_train, apply to val/test safely.
    Rare classes (fewer than min_class_count in the whole dataset) are mapped to '__other__'.
    Any unseen classes in val/test are also mapped to '__other__'.
    """
    if not encode_labels:
        return splits

    y_train = np.asarray(splits.y_train)
    y_val = np.asarray(splits.y_val)
    y_test = np.asarray(splits.y_test)

    combined = np.concatenate([y_train, y_val, y_test])
    value_counts = pd.Series(combined).value_counts()
    rare_classes = value_counts[value_counts < min_class_count].index.tolist()

    if rare_classes and verbose:
        print(f"Collapsing rare classes {rare_classes} -> '__other__'")

    def replace_rare(values: np.ndarray) -> np.ndarray:
        return np.array([label if label not in rare_classes else "__other__" for label in values])

    y_train = replace_rare(y_train)
    y_val = replace_rare(y_val)
    y_test = replace_rare(y_test)

    # Fit LabelEncoder only on train (plus the '__other__' bucket if needed)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    unique_train = np.unique(y_train).tolist()
    if "__other__" in y_train or "__other__" in y_val or "__other__" in y_test:
        if "__other__" not in unique_train:
            unique_train.append("__other__")
    le.fit(unique_train)

    def safe_encode(le, y):
        return np.array([lbl if lbl in le.classes_ else "__other__" for lbl in y])

    y_train_enc = le.transform(safe_encode(le, y_train))
    y_val_enc   = le.transform(safe_encode(le, y_val))
    y_test_enc  = le.transform(safe_encode(le, y_test))


def scale_features(
    splits: DatasetSplits,
    *,
    scaler_type: Literal["standard", "minmax"] = "standard",
) -> DatasetSplits:
    """Apply feature scaling using the provided scaler type."""


def split_data(X, y, test_size=0.2, val_size=0.2, random_seed=None, stratify=True):
    """Splits data into train, validation, and test sets with optional stratification."""
    
    stratify_y = y if stratify else None

    X_train = scaler.fit_transform(splits.X_train)
    X_val = scaler.transform(splits.X_val)
    X_test = scaler.transform(splits.X_test)

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

from sklearn.impute import SimpleImputer

def impute_features(X_train, X_val, X_test,
                    num_strategy="mean", cat_strategy="most_frequent",
                    verbose=False):
    """
    Impute missing values separately for numeric and categorical features.
    
    Parameters
    ----------
    num_strategy : str
        'mean', 'median', or 'constant' for numeric columns.
    cat_strategy : str
        'most_frequent' or 'constant' for categorical columns.
    """
    # Identify types
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

    # --- numeric ---
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy=num_strategy)
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_val[num_cols]   = num_imputer.transform(X_val[num_cols])
        X_test[num_cols]  = num_imputer.transform(X_test[num_cols])

    # --- categorical ---
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy=cat_strategy)
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_val[cat_cols]   = cat_imputer.transform(X_val[cat_cols])
        X_test[cat_cols]  = cat_imputer.transform(X_test[cat_cols])

    if verbose:
        n_missing = (
            X_train.isna().sum().sum() +
            X_val.isna().sum().sum() +
            X_test.isna().sum().sum()
        )
        print(f"Imputation applied. Remaining NaNs: {n_missing}")

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

    # --- NEW: handle missing values before encoding ---
    X_train, X_val, X_test = impute_features(
        X_train, X_val, X_test,
        num_strategy="mean",
        cat_strategy="most_frequent",
        verbose=verbose
    )

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
        raise ValueError("task_type must be 'classification' or 'regression'.")

    y_train = torch.as_tensor(splits.y_train, dtype=target_dtype)
    y_val = torch.as_tensor(splits.y_val, dtype=target_dtype)
    y_test = torch.as_tensor(splits.y_test, dtype=target_dtype)

    return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def create_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap tensors into a ``DataLoader`` with deterministic behaviour."""

    dataset = TensorDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def get_preprocessed_data(
    *,
    config: PreprocessingConfig,
    X: Optional[pd.DataFrame] = None,
    y: Optional[Sequence] = None,
    verbose: bool = False,
) -> DatasetSplits:
    """High-level helper that orchestrates the full preprocessing pipeline."""

    if config.dataset_id is not None and (X is None or y is None):
        X, y = load_openml_dataset(config.dataset_id, verbose=verbose)
    elif X is None or y is None:
        raise ValueError("Either provide dataset_id or explicit X/y data.")

    splits = split_data(
        X,
        y,
        test_size=config.test_size,
        val_size=config.val_size,
        random_seed=config.random_seed,
    )

    splits = preprocess_features(
        splits,
        categorical_strategy=config.categorical_strategy,
        verbose=verbose,
    )

    splits = preprocess_target(
        splits,
        encode_labels=config.task_type == "classification",
        min_class_count=config.min_class_count,
        verbose=verbose,
    )

    if config.scaling:
        splits = scale_features(splits, scaler_type=config.scaler_type)

    if config.return_as == "tensor":
        splits = convert_to_tensors(splits, task_type=config.task_type)
    elif config.return_as != "array":
        raise ValueError("return_as must be either 'tensor' or 'array'.")

    return splits

