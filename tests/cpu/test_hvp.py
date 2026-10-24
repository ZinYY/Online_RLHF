"""HVP characterization tests for the legacy HVP reward trainers.

These tests call the real ``hessian_vector_product`` implementations with a
small float64 logistic reward model on CPU and compare against an
independently computed analytic Hessian (see ``tests/support/logistic.py``).
The production method adds ``cg_damping * v`` to the raw Hessian-vector
product, so the reference is always the regularized Hessian.

The full-model trainer ("rm_hvp") wraps its autograd math in deepspeed's
GatheredParameters gate; ``trainer_hvp_gate`` brackets those calls with the
scoped no-op pass-through documented in tests/support/legacy_imports.py.
"""

import torch

from tests.support import logistic
from tests.support.legacy_imports import make_trainer_shell, trainer_hvp_gate

RTOL = 1e-7
ATOL = 1e-9


def _hvp(legacy_trainer, Z, theta, rejected, vector, damping):
    trainer = make_trainer_shell(legacy_trainer.trainer, cg_damping=damping)
    loss, flat_grad = logistic.pairwise_loss_and_flat_grad(
        legacy_trainer.loss.PairWiseLoss, Z, theta, rejected
    )
    with trainer_hvp_gate(legacy_trainer.kind):
        return trainer.hessian_vector_product([theta], loss, vector, flat_grad)


def test_hvp_matches_regularized_analytic_hessian(legacy_trainer):
    Z, theta, rejected = logistic.make_logistic_problem()
    damping = 0.2
    vector = torch.tensor([0.3, -0.7, 1.1, 0.2], dtype=torch.float64)

    actual = _hvp(legacy_trainer, Z, theta, rejected, vector, damping)
    expected = logistic.regularized_hessian(Z, theta, rejected, damping) @ vector

    assert actual.shape == vector.shape
    assert actual.dtype == torch.float64
    assert actual.device.type == "cpu"
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_hvp_zero_vector(legacy_trainer):
    Z, theta, rejected = logistic.make_logistic_problem()
    zero = torch.zeros(4, dtype=torch.float64)

    actual = _hvp(legacy_trainer, Z, theta, rejected, zero, damping=0.2)

    assert actual.shape == zero.shape
    assert torch.isfinite(actual).all()
    assert torch.all(actual == 0)


def test_hvp_clears_parameter_grads(legacy_trainer):
    """Documented side effect: hessian_vector_product resets param.grad to None."""
    Z, theta, rejected = logistic.make_logistic_problem()
    theta.grad = torch.ones_like(theta)
    vector = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    _hvp(legacy_trainer, Z, theta, rejected, vector, damping=0.2)

    assert theta.grad is None


def test_hvp_reuses_single_loss_graph(legacy_trainer):
    """Two HVP calls with different vectors on the SAME loss/gradient graph:
    the graph is built once; only hessian_vector_product's retain_graph path
    makes the second call possible."""
    Z, theta, rejected = logistic.make_logistic_problem()
    damping = 0.2
    trainer = make_trainer_shell(legacy_trainer.trainer, cg_damping=damping)
    loss, flat_grad = logistic.pairwise_loss_and_flat_grad(
        legacy_trainer.loss.PairWiseLoss, Z, theta, rejected
    )
    hessian = logistic.regularized_hessian(Z, theta, rejected, damping)
    vectors = [
        torch.tensor([0.3, -0.7, 1.1, 0.2], dtype=torch.float64),
        torch.tensor([-0.4, 0.9, 0.05, -1.3], dtype=torch.float64),
    ]
    for vector in vectors:
        with trainer_hvp_gate(legacy_trainer.kind):
            actual = trainer.hessian_vector_product([theta], loss, vector, flat_grad)
        assert actual.shape == vector.shape
        assert torch.isfinite(actual).all()
        assert torch.allclose(actual, hessian @ vector, rtol=RTOL, atol=ATOL)
