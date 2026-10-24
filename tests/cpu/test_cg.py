"""Conjugate gradient characterization tests for the legacy HVP reward trainers.

Isolation notes:
- ``args.damping`` (the direction-mixing weight) is set to 0.0 so the solver
  output is the pure CG result; ``cg_damping > 0`` supplies the Tikhonov-style
  regularization. This is NOT the default CLI training configuration.
- ``residual_tol`` is compared against the SQUARED residual norm inside the
  production loop (``r_norm_sq < residual_tol``), not the residual norm.
  This is interface semantics (the name can mislead), pinned by
  ``test_cg_residual_tol_compares_squared_residual``.

Fixed defect (CG-1, fixed on branch ``fix/cg-fletcher-reeves-beta``, see
docs/maintenance/review.md): the Fletcher-Reeves beta — the ratio of the NEW
to the OLD squared residual — used to be computed as ``r_norm_sq /
r_norm_sq`` (both operands the freshly updated value), i.e. beta = 1.0
whenever its denominator was nonzero. The production solvers now save the old
value before updating the residual, and the two tests below pin the corrected
behavior so the one-line regression cannot return silently.
"""

import types

import pytest
import torch

from tests.support import logistic
from tests.support.legacy_imports import make_trainer_shell, trainer_hvp_gate


def _solve(
    legacy_trainer, max_iter, damping_mix=0.0, residual_tol=1e-10, damping=0.2, seed=7
):
    Z, theta, rejected = logistic.make_logistic_problem(seed=seed)
    trainer = make_trainer_shell(
        legacy_trainer.trainer,
        cg_damping=damping,
        args=types.SimpleNamespace(damping=damping_mix),
    )
    loss, flat_grad = logistic.pairwise_loss_and_flat_grad(
        legacy_trainer.loss.PairWiseLoss, Z, theta, rejected
    )
    with trainer_hvp_gate(legacy_trainer.kind):
        result = trainer.conjugate_gradient_solver(
            [theta], loss, flat_grad, max_iter=max_iter, residual_tol=residual_tol
        )
    return (
        result,
        flat_grad,
        Z,
        theta,
        rejected,
        damping,
    )


def _relative_residual(operator, x, rhs):
    return (torch.linalg.norm(operator @ x - rhs) / torch.linalg.norm(rhs)).item()


def test_cg_output_contract_at_dimension_budget(legacy_trainer):
    """Shape/dtype/device/finiteness at the regression fixture's 4-step budget."""
    x, flat_grad, *_ = _solve(legacy_trainer, max_iter=4)
    assert x.shape == flat_grad.shape
    assert x.dtype == torch.float64
    assert x.device.type == "cpu"
    assert torch.isfinite(x).all()


# CG-1 regression guard: the audit fixture's SPD system must be solved to
# rel. residual <= 1e-5 within dim=4 steps now that the Fletcher-Reeves beta
# uses the new/old squared-residual ratio. (Before the fix this needed an
# xfail: 4-step rel. residual was 0.231158.)
def test_cg_solves_spd_system_within_dimension_steps(legacy_trainer):
    """Standard linear CG on this SPD system reaches rel. residual <= 1e-5
    within dim=4 steps (float64). Before the CG-1 fix the 4-step relative
    residual here was 0.231158; after the fix it is ~4.5e-6 (recorded in
    docs/maintenance/review.md)."""
    x, flat_grad, Z, theta, rejected, damping = _solve(legacy_trainer, max_iter=4)
    operator = logistic.regularized_hessian(Z, theta, rejected, damping)
    reference = torch.linalg.solve(operator, flat_grad)

    assert torch.allclose(x, reference, rtol=1e-4, atol=1e-6)
    assert _relative_residual(operator, x, flat_grad) <= 1e-5


def test_cg_converges_on_conjugate_direction_system(legacy_trainer):
    """Anti-revert regression: a 3x3 SPD system whose eigenvectors are far
    from the residual directions — fast convergence REQUIRES genuinely
    conjugate search directions. With beta pinned to 1.0 this system stalls;
    with the Fletcher-Reeves ratio it solves within dim steps. The operator
    is exercised through the production solver on a real quadratic loss."""
    dim = 3
    angle = torch.tensor(0.9, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    eigenvalues = torch.diag(torch.tensor([1.0, 30.0, 200.0], dtype=torch.float64))
    operator = rotation @ eigenvalues @ rotation.T
    operator = 0.5 * (operator + operator.T)  # guard against asymmetry noise
    rhs = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

    theta = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float64))
    loss = 0.5 * theta @ operator @ theta - rhs @ theta
    flat_grad = torch.cat(
        [g.reshape(-1) for g in torch.autograd.grad(loss, [theta], create_graph=True)]
    )
    rhs = flat_grad.detach()  # the solver solves A x = flat_grad (here A@0 - rhs)

    trainer = make_trainer_shell(
        legacy_trainer.trainer,
        cg_damping=0.0,
        args=types.SimpleNamespace(damping=0.0),
    )
    with trainer_hvp_gate(legacy_trainer.kind):
        x = trainer.conjugate_gradient_solver(
            [theta], loss, flat_grad, max_iter=dim, residual_tol=1e-12
        )

    reference = torch.linalg.solve(operator, rhs)
    assert torch.allclose(x, reference, rtol=1e-6, atol=1e-8)
    assert _relative_residual(operator, x, rhs) <= 1e-6


def test_cg_full_mix_returns_gradient_direction(legacy_trainer):
    """With the mixing weight args.damping = 1.0 the output is flat_grad itself."""
    x, flat_grad, *_ = _solve(legacy_trainer, max_iter=2, damping_mix=1.0)
    assert torch.equal(x, flat_grad)


def test_cg_residual_tol_compares_squared_residual(legacy_trainer):
    """Discriminating check: pick tol strictly between ||r1||^2 and ||r1|| for
    the independently derived one-step residual r1. Stopping after the first
    update (returning exactly the one-step iterate) is only possible if the
    loop compares the SQUARED residual against tol."""
    Z, theta, rejected = logistic.make_logistic_problem()
    damping = 0.2
    loss, flat_grad = logistic.pairwise_loss_and_flat_grad(
        legacy_trainer.loss.PairWiseLoss, Z, theta, rejected
    )
    operator = logistic.regularized_hessian(Z, theta, rejected, damping)
    g = flat_grad.detach()
    # Independent one-step iterate, keeping the production denominator's 1e-8.
    alpha = (g @ g) / (g @ operator @ g + 1e-8)
    r1 = g - alpha * operator @ g
    squared = (r1 @ r1).item()
    norm = torch.linalg.norm(r1).item()
    # Reference values for this fixture (independent computation):
    #   ||r1|| ~= 0.09713945660306085, ||r1||^2 ~= 0.009436074029137943
    assert squared == pytest.approx(0.009436074029137943, rel=1e-12)
    assert norm == pytest.approx(0.09713945660306085, rel=1e-12)
    tol = (squared + norm) / 2
    assert squared < tol < norm

    trainer = make_trainer_shell(
        legacy_trainer.trainer,
        cg_damping=damping,
        args=types.SimpleNamespace(damping=0.0),
    )
    x = None
    with trainer_hvp_gate(legacy_trainer.kind):
        x = trainer.conjugate_gradient_solver(
            [theta], loss, flat_grad, max_iter=10, residual_tol=tol
        )
    assert torch.allclose(x, alpha * g, rtol=1e-7, atol=1e-9)


def test_cg_truncated_default_steps_stay_finite(legacy_trainer):
    """The production default is num_cg_steps=3; a truncated run must still
    return a finite, correctly shaped direction (accuracy is NOT asserted)."""
    x, flat_grad, *_ = _solve(legacy_trainer, max_iter=3)
    assert x.shape == flat_grad.shape
    assert x.dtype == torch.float64
    assert torch.isfinite(x).all()


def test_cg_is_deterministic(legacy_trainer):
    x1, _, *_ = _solve(legacy_trainer, max_iter=4)
    x2, _, *_ = _solve(legacy_trainer, max_iter=4)
    assert torch.equal(x1, x2)


def _solve_with_args(legacy_trainer, args_namespace, max_iter=3):
    Z, theta, rejected = logistic.make_logistic_problem()
    trainer = make_trainer_shell(
        legacy_trainer.trainer,
        cg_damping=0.2,
        args=args_namespace,
    )
    loss, flat_grad = logistic.pairwise_loss_and_flat_grad(
        legacy_trainer.loss.PairWiseLoss, Z, theta, rejected
    )
    with trainer_hvp_gate(legacy_trainer.kind):
        result = trainer.conjugate_gradient_solver(
            [theta], loss, flat_grad, max_iter=max_iter, residual_tol=1e-10
        )
    return result, flat_grad


def test_cg_mixing_weight_defaults_to_damping_bitwise(legacy_trainer):
    """Compare missing, None and explicit damping on the SAME current solver.

    Equality does not compare against the pre-beta-fix numerical output.
    """
    fallback, _ = _solve_with_args(legacy_trainer, types.SimpleNamespace(damping=0.8))
    explicit, _ = _solve_with_args(
        legacy_trainer, types.SimpleNamespace(damping=0.8, cg_mixing_weight=0.8)
    )
    explicit_none, _ = _solve_with_args(
        legacy_trainer, types.SimpleNamespace(damping=0.8, cg_mixing_weight=None)
    )
    assert torch.equal(fallback, explicit)
    assert torch.equal(fallback, explicit_none)


def test_cg_mixing_weight_overrides_damping(legacy_trainer):
    """An explicit cg_mixing_weight overrides the damping-based mixing:
    w=0.0 gives the pure CG direction regardless of args.damping, and an
    intermediate w blends linearly per the documented formula."""
    pure, g = _solve_with_args(
        legacy_trainer, types.SimpleNamespace(damping=0.8, cg_mixing_weight=0.0)
    )
    zero_damping_ref, _ = _solve_with_args(
        legacy_trainer, types.SimpleNamespace(damping=0.0)
    )
    assert torch.equal(pure, zero_damping_ref)

    blended, g = _solve_with_args(
        legacy_trainer, types.SimpleNamespace(damping=0.8, cg_mixing_weight=0.3)
    )
    expected = 0.3 * g + 0.7 * pure
    assert torch.allclose(blended, expected, rtol=1e-7, atol=1e-9)


def test_cg_single_iteration_does_not_mix(legacy_trainer):
    pure, _ = _solve_with_args(
        legacy_trainer,
        types.SimpleNamespace(damping=0.8, cg_mixing_weight=0.0),
        max_iter=1,
    )
    full_weight, grad = _solve_with_args(
        legacy_trainer,
        types.SimpleNamespace(damping=0.8, cg_mixing_weight=1.0),
        max_iter=1,
    )
    assert torch.equal(pure, full_weight)
    # Discriminates against accidentally returning the gradient for w=1.
    assert not torch.allclose(pure, grad)
