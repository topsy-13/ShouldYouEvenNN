"""Quick smoke test verifying deterministic behaviour."""

from __future__ import annotations

import pandas as pd

from .ebe import Population
from .search_space import SearchSpace
from .utils import init_global_seed


def test_reproducibility() -> None:
    """Ensure that two independent runs produce identical ledgers."""

    init_global_seed(123)

    search_space = SearchSpace(input_size=10, output_size=2)

    pop1 = Population(search_space, size=4, starting_instances=50, seed=123)
    ledger1 = pop1.build_ledger(export_as="pandas")

    pop2 = Population(search_space, size=4, starting_instances=50, seed=123)
    ledger2 = pop2.build_ledger(export_as="pandas")

    pd.testing.assert_frame_equal(
        ledger1.reset_index(drop=True),
        ledger2.reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
    )

    print("Reproducibility test passed — identical ledgers with same seed")


if __name__ == "__main__":
    test_reproducibility()

