"""Damping schedule characterization tests for the legacy head-HVP trainers.

The schedule lives in ``get_current_damping`` and is gated by normalized
time ``t = total_steps / damping_growth_rate`` whenever
``damping_growth_rate > 1.0`` (otherwise ``total_steps / total_T``).

Observed behaviors pinned here (from the prior audit, re-verified locally):
- ``constant`` returns ``base_damping`` until ``t >= 1.0``, then exactly 1.0.
- ``cosine`` can fall back below 1.0 once ``t`` exceeds 1.0
  (``min(base * (2 - cos(pi * t)), 1.0)`` oscillates).
These tests pin current behavior; they do not assert that the schedule is
monotone, and they do not prove the scheduled value is dynamically passed
into the HVP (see docs/maintenance/review.md).
"""

import numpy as np
import pytest

from tests.support.legacy_imports import make_trainer_shell

BASE = 0.8
GROWTH = 100
TOTAL_T = 400

COSINE_AUDIT_STEPS = (0, 50, 100, 150, 200)
COSINE_AUDIT_EXPECTED = (0.8, 1.0, 1.0, 1.0, 0.8)


def _damping_at(
    legacy_trainer, strategy, steps, base=BASE, growth=GROWTH, total_t=TOTAL_T
):
    trainer = make_trainer_shell(
        legacy_trainer.trainer,
        cg_damping=base,
        damping_strategy=strategy,
        damping_growth_rate=growth,
        total_steps=steps,
        total_T=total_t,
    )
    return trainer.get_current_damping()


def test_constant_switches_to_one_at_threshold(legacy_trainer):
    assert _damping_at(legacy_trainer, "constant", 0) == BASE
    assert _damping_at(legacy_trainer, "constant", 50) == BASE
    assert _damping_at(legacy_trainer, "constant", 100) == 1.0
    assert _damping_at(legacy_trainer, "constant", 150) == 1.0


def test_linear_grows_then_caps(legacy_trainer):
    assert _damping_at(legacy_trainer, "linear", 0) == BASE
    assert _damping_at(legacy_trainer, "linear", 50) == 0.9
    assert _damping_at(legacy_trainer, "linear", 100) == 1.0
    assert _damping_at(legacy_trainer, "linear", 200) == 1.0


def test_log_schedule_is_capped(legacy_trainer):
    t = 1.0
    expected = min(BASE * (1 + np.log(1 + t)), 1.0)
    assert _damping_at(legacy_trainer, "log", 100) == pytest.approx(expected)


def test_square_schedule_is_capped(legacy_trainer):
    t = 1.0
    expected = min(BASE * (1 + t**2), 1.0)
    assert _damping_at(legacy_trainer, "square", 100) == pytest.approx(expected)


def test_cosine_matches_audit_sequence(legacy_trainer):
    """Re-verifies the audit sequence base=0.8, growth=100, total_T=400:
    cosine at steps 0/50/100/150/200 is 0.8/1/1/1/0.8 (non-monotone)."""
    actual = tuple(
        _damping_at(legacy_trainer, "cosine", steps) for steps in COSINE_AUDIT_STEPS
    )
    assert actual == COSINE_AUDIT_EXPECTED


def test_cosine_falls_back_after_threshold(legacy_trainer):
    t = 2.0  # steps=200
    expected = min(BASE * (2 - np.cos(np.pi * t)), 1.0)
    assert expected == BASE  # cos(2*pi) == 1
    assert _damping_at(legacy_trainer, "cosine", 200) == pytest.approx(expected)


def test_unknown_strategy_returns_base(legacy_trainer):
    assert _damping_at(legacy_trainer, "", 0) == BASE
    assert _damping_at(legacy_trainer, "not_a_strategy", 500) == BASE


def test_default_growth_rate_uses_steps_over_growth(legacy_trainer):
    """With the constructor default growth=5.0 (>1), t = total_steps / 5 and
    total_T is ignored by the schedule."""
    trainer = make_trainer_shell(
        legacy_trainer.trainer,
        cg_damping=BASE,
        damping_strategy="linear",
        damping_growth_rate=5.0,
        total_steps=10,
        total_T=400,
    )
    # t = 10 / 5 = 2 -> capped at 1.0, regardless of total_T=400.
    assert trainer.get_current_damping() == 1.0
