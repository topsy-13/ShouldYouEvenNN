# Tests

Tests apply to the active installed package. Install the development extra before
running `python -m pytest`. The initial checks verify imports outside the checkout
and ensure historical modules are not exposed by the installation.

Add tests for scientific invariants as implementations arrive. Default checks must
remain small and offline; archived notebooks and experiments are not test inputs.
