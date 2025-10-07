"""Quick smoke test verifying deterministic behaviour."""

from __future__ import annotations

import pandas as pd

from ebe import Population
from search_space import SearchSpace
from utils import init_global_seed
import torch


def test_full_run_repro():
    init_global_seed(42)
    search_space = SearchSpace(input_size=10, output_size=2)
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    X_train, y_train = X[:80], y[:80]
    X_val, y_val = X[80:], y[80:]

    # First run
    pop1 = Population(search_space, size=3, seed=42, starting_instances=10)
    ledger1 = pop1.run_generation(X_train, y_train, X_val, y_val, baseline_metric=0.7)
    ledger1 = pop1.build_ledger()

    # Second independent run (new Population, same seed)
    pop2 = Population(search_space, size=3, seed=42, starting_instances=10)
    ledger2 = pop2.run_generation(X_train, y_train, X_val, y_val, baseline_metric=0.7)
    ledger2 = pop2.build_ledger()
    def _drop_observational(df):
        return df.drop(
            columns=[c for c in df.columns
                    if "wall_time" in c or "total_wall_time" in c or "avg_wall_time" in c],
            errors="ignore"
        )

    ledger1 = _drop_observational(ledger1)
    ledger2 = _drop_observational(ledger2)

    pd.testing.assert_frame_equal(
        ledger1.reset_index(drop=True),
        ledger2.reset_index(drop=True),
        check_dtype=False, check_exact=False, atol=1e-7
    )

    print("Full reproducibility test passed (ignoring timing jitter).")


if __name__ == "__main__":
    test_full_run_repro()
