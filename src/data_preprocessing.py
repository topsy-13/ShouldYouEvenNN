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


def split_data(
    X: pd.DataFrame,
    y: Sequence,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: Optional[int] = None,
    stratify: bool = True,
) -> DatasetSplits:
    """Split a dataset into train/validation/test partitions."""

    stratify_y = y if stratify else None

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_y,
    )

    stratify_train_val = y_train_val if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_seed,
        stratify=stratify_train_val,
    )

    return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def _is_categorical(series: pd.Series) -> bool:
    return series.dtype == "object" or series.dtype.name == "category" or not is_numeric_dtype(series)


def preprocess_features(
    splits: DatasetSplits,
    *,
    categorical_strategy: Literal["label", "onehot"] = "label",
    verbose: bool = False,
) -> DatasetSplits:
    """Encode categorical features consistently across splits."""

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

    return DatasetSplits(X_train, X_val, X_test, splits.y_train, splits.y_val, splits.y_test)


def preprocess_target(
    splits: DatasetSplits,
    *,
    encode_labels: bool = True,
    min_class_count: int = 2,
    verbose: bool = False,
) -> DatasetSplits:
    """Optionally encode labels and collapse rare classes into ``__other__``."""

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

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)

    return DatasetSplits(splits.X_train, splits.X_val, splits.X_test, y_train_enc, y_val_enc, y_test_enc)


def scale_features(
    splits: DatasetSplits,
    *,
    scaler_type: Literal["standard", "minmax"] = "standard",
) -> DatasetSplits:
    """Apply feature scaling using the provided scaler type."""

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be either 'standard' or 'minmax'.")

    X_train = scaler.fit_transform(splits.X_train)
    X_val = scaler.transform(splits.X_val)
    X_test = scaler.transform(splits.X_test)

    return DatasetSplits(X_train, X_val, X_test, splits.y_train, splits.y_val, splits.y_test)


def convert_to_tensors(
    splits: DatasetSplits,
    *,
    task_type: Literal["classification", "regression"] = "classification",
) -> DatasetSplits:
    """Convert arrays to ``torch.Tensor`` objects while respecting the task type."""

    X_train = torch.as_tensor(splits.X_train, dtype=torch.float32)
    X_val = torch.as_tensor(splits.X_val, dtype=torch.float32)
    X_test = torch.as_tensor(splits.X_test, dtype=torch.float32)

    if task_type == "classification":
        target_dtype = torch.long
    elif task_type == "regression":
        target_dtype = torch.float32
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

