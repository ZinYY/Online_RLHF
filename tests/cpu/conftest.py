"""Shared fixtures for the legacy CPU tests.

Import boundary: ``tests/support`` is imported as a namespace package from the
repository root, so these never execute the production package aggregators.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.support.legacy_imports import (  # noqa: E402
    isolated_legacy_imports,
    load_convert_to_dataset,
    load_legacy_loss,
    load_rm_score_selection,
    load_trainer,
)
from tests.support.random_state import preserve_python_random_state  # noqa: E402


@pytest.fixture(autouse=True)
def _preserve_global_random_state():
    """Every test in tests/cpu must leave the global RNG state untouched:
    the Python ``random`` module and the torch CPU generator."""
    with preserve_python_random_state():
        torch_state = torch.get_rng_state()
        try:
            yield
        finally:
            torch.set_rng_state(torch_state)


@pytest.fixture(
    params=["rm", "rm_active", "rm_hvp"],
    ids=["rm_trainer_head_hvp", "rm_active_trainer_head_hvp", "rm_trainer_hvp"],
)
def legacy_trainer(request):
    """All legacy HVP reward-model trainer modules, loaded from this checkout.

    "rm_hvp" is the full-model trainer; its HVP wraps the autograd math in a
    deepspeed.zero.GatheredParameters gate, which tests bracket with the
    scoped pass-through (see tests/support/legacy_imports.py).
    """
    with isolated_legacy_imports():
        loss_mod = load_legacy_loss()
        trainer_mod = load_trainer(request.param)
        yield types.SimpleNamespace(
            loss=loss_mod, trainer=trainer_mod, kind=request.param
        )


@pytest.fixture()
def legacy_convert_to_dataset():
    """The real openrlhf/utils/convert_to_dataset.py module."""
    with isolated_legacy_imports():
        yield load_convert_to_dataset()


@pytest.fixture()
def legacy_rm_score_selection():
    """The real openrlhf/utils/rm_score_selection.py module (fresh per test,
    so its module-level _global_V accumulator cannot leak between tests)."""
    with isolated_legacy_imports():
        yield load_rm_score_selection()
