"""Utility helpers for deterministic experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["init_global_seed", "ReproContext", "make_repro_context"]


def init_global_seed(base_seed: int = 13) -> None:
    """Seed all major random number generators for reproducibility."""

    random.seed(base_seed)
    os.environ["PYTHONHASHSEED"] = str(base_seed)

    np.random.seed(base_seed)

    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.__version__ >= "1.8":
        torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class ReproContext:
    """Container holding RNG objects that share a common seed."""

    numpy_generator: np.random.Generator
    torch_generator: torch.Generator
    python_rng: random.Random
    base_seed: int

    def worker_init_fn(self, worker_id: int) -> None:
        """Seed NumPy/Python RNGs for a data-loading worker."""

        worker_seed = self.base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def make_repro_context(seed: int) -> ReproContext:
    """Create seeded RNGs for NumPy, PyTorch and Python's ``random``."""

    numpy_generator = np.random.default_rng(seed)

    torch_generator = torch.Generator()
    torch_generator.manual_seed(seed)

    python_rng = random.Random(seed)

    return ReproContext(
        numpy_generator=numpy_generator,
        torch_generator=torch_generator,
        python_rng=python_rng,
        base_seed=seed,
    )
