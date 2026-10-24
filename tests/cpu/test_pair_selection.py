"""Pair-selection helper characterization tests.

Scope: the four index-selection helpers in
``openrlhf/utils/convert_to_dataset.py`` only. The full file-conversion
pipeline (set iteration order, file IO, swapping) is intentionally out of
scope for this round.

Observed behavior pinned here:
- ``get_best_quartile_indices`` returns the best and the 4th-best entries of
  the descending sort (fixed index 3), NOT a general quartile computation.
- ``get_random_indices`` uses the Python ``random`` module; same seed and
  same call sequence give identical pairs.

RNG isolation: the conftest autouse fixture saves/restores the global
``random`` state around every test in this directory, so the tests below may
seed and consume ``random`` freely. The guard itself (normal and exceptional
exit) is verified in ``test_random_state_guard_*``.
"""

import random

import pytest

from tests.support.legacy_imports import (
    isolated_legacy_imports,
    load_convert_to_dataset,
)
from tests.support.random_state import preserve_python_random_state


@pytest.fixture()
def ctd():
    with isolated_legacy_imports():
        yield load_convert_to_dataset()


def test_best_worst_exact_indices(ctd):
    assert ctd.get_best_worst_indices([1.0, 3.0, 2.0]) == (1, 0)
    assert ctd.get_best_worst_indices([0.5]) == (0, 0)


def test_best_worst_ties_take_first_occurrence(ctd):
    rewards = [2.0, 2.0, 1.0]
    assert max(rewards) == rewards[0]
    assert ctd.get_best_worst_indices(rewards) == (0, 2)
    rewards = [1.0, 2.0, 2.0]
    assert min(rewards) == rewards[0]
    assert ctd.get_best_worst_indices(rewards) == (1, 0)


def test_best_second_exact_indices(ctd):
    assert ctd.get_best_second_indices([1.0, 3.0, 2.0]) == (1, 2)
    assert ctd.get_best_second_indices([5.0, 1.0]) == (0, 1)


def test_best_second_short_list_returns_none(ctd):
    assert ctd.get_best_second_indices([1.0]) == (None, None)
    assert ctd.get_best_second_indices([]) == (None, None)


def test_best_quartile_selects_fourth_best(ctd):
    """Current behavior: fixed 4th entry of the descending sort (index 3)."""
    rewards = [5.0, 1.0, 4.0, 2.0, 3.0]
    # descending sort: indices [0, 2, 4, 3, 1] -> best=0, 4th best=3
    assert ctd.get_best_quartile_indices(rewards) == (0, 3)
    assert ctd.get_best_quartile_indices([4.0, 3.0, 2.0, 1.0]) == (0, 3)


def test_best_quartile_short_list_returns_none(ctd):
    assert ctd.get_best_quartile_indices([3.0, 2.0, 1.0]) == (None, None)


def test_random_indices_are_seeded_and_distinct(ctd):
    """Same seed + same call sequence give identical pairs; isolation of the
    global random state is provided by the conftest autouse fixture."""
    rewards = [0.1 * i for i in range(10)]
    random.seed(1234)
    first = ctd.get_random_indices(rewards)
    random.seed(1234)
    second = ctd.get_random_indices(rewards)
    assert first == second
    idx1, idx2 = first
    assert 0 <= idx1 < len(rewards)
    assert 0 <= idx2 < len(rewards)
    assert idx1 != idx2


def test_random_state_guard_restores_on_normal_exit(ctd):
    random.seed(2024)
    random.random()  # advance deterministically
    before = random.getstate()
    with preserve_python_random_state():
        random.seed(7)
        ctd.get_random_indices([0.1 * i for i in range(10)])
        ctd.get_random_indices([0.1 * i for i in range(10)])
    assert random.getstate() == before


def test_random_state_guard_restores_on_exceptional_exit():
    random.seed(2024)
    before = random.getstate()
    with pytest.raises(RuntimeError, match="probe"):
        with preserve_python_random_state():
            random.seed(7)
            random.random()
            raise RuntimeError("probe")
    assert random.getstate() == before
