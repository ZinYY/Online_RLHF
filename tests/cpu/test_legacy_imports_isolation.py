"""Regression tests for the isolation adapter itself (tests/support/legacy_imports.py).

These tests protect the namespace-management contract so the adapter cannot
pollute later tests in the same process:
- original pre-existing objects are restored on exit;
- restoration also happens on exceptional exit;
- third-party modules imported inside the block are NOT cleaned up;
- pre-existing openrlhf.* entries cannot shadow this checkout's modules;
- DeepSpeed metadata reads succeed while real training API access fails.
"""

import sys
import types
from pathlib import Path

import pytest

from tests.support.legacy_imports import (
    REPO_ROOT,
    isolated_legacy_imports,
    load_trainer,
)


def test_preexisting_module_object_is_restored():
    fake_loss = types.ModuleType("openrlhf.models.loss")
    fake_loss.MARKER = object()
    sys.modules["openrlhf.models.loss"] = fake_loss
    try:
        with isolated_legacy_imports():
            loaded = load_trainer("rm")
            assert loaded.__name__ == "openrlhf.trainer.rm_trainer_head_hvp"
            assert sys.modules["openrlhf.models.loss"] is not fake_loss
        assert sys.modules["openrlhf.models.loss"] is fake_loss
        assert sys.modules["openrlhf.models.loss"].MARKER is not None
    finally:
        sys.modules.pop("openrlhf.models.loss", None)


def test_restoration_happens_on_exceptional_exit():
    with pytest.raises(RuntimeError, match="probe"):
        with isolated_legacy_imports():
            raise RuntimeError("probe")
    assert "openrlhf" not in sys.modules
    assert "deepspeed" not in sys.modules


def test_unrelated_modules_are_not_cleaned_up():
    probe = types.ModuleType("_third_party_probe")
    with isolated_legacy_imports():
        load_trainer("rm_active")
        sys.modules["_third_party_probe"] = probe
        import torch  # noqa: F401  (already imported by conftest elsewhere)

        assert "torch" in sys.modules
    assert sys.modules["_third_party_probe"] is probe
    assert "torch" in sys.modules
    sys.modules.pop("_third_party_probe", None)


def test_preexisting_openrlhf_submodule_is_isolated():
    """A stale openrlhf.utils from outside this checkout must not be reused."""
    stale_utils = types.ModuleType("openrlhf.utils")
    stale_utils.__path__ = ["/nonexistent/stale/path"]
    sys.modules["openrlhf.utils"] = stale_utils
    try:
        with isolated_legacy_imports():
            load_trainer("rm")
            loaded_path = sys.modules["openrlhf.utils.distributed_sampler"].__file__
            assert loaded_path is not None
            assert (
                Path(loaded_path)
                .resolve()
                .is_relative_to((REPO_ROOT / "openrlhf").resolve())
            )
    finally:
        sys.modules.pop("openrlhf.utils", None)


def test_deepspeed_metadata_reads_but_api_fails():
    with isolated_legacy_imports():
        import deepspeed

        assert deepspeed.__name__ == "deepspeed"
        # Metadata reads must succeed (returning None is fine) ...
        assert deepspeed.__file__ is None
        assert deepspeed.__spec__ is not None
        # ... while any real training API access fails fast.
        with pytest.raises(RuntimeError, match="deepspeed.zero"):
            deepspeed.zero.GatheredParameters
