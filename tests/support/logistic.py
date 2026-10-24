"""Small CPU float64 logistic reward problem for the legacy HVP/CG tests.

The analytic Hessian is computed independently of the production
``hessian_vector_product`` implementation, so the tests compare the real
autograd-based code against a closed-form reference.
"""

import torch

COLUMN_SCALE = (0.5, 1.0, 2.0, 3.0)
THETA_VALUES = (0.1, -0.2, 0.05, 0.15)


def make_logistic_problem(seed=7, n=16, d=4, dtype=torch.float64):
    """Fixed small logistic reward model: chosen = Z @ theta, rejected = 0.

    Fixture lineage: reproduces the matrix family used in the prior audit
    (seed=7, normal Z with scaled columns, fixed theta). Kept small and
    well-conditioned on purpose; float64 everywhere.
    """
    if d != len(COLUMN_SCALE):
        raise ValueError("fixture only defined for d=4")
    gen = torch.Generator().manual_seed(seed)
    Z = torch.randn((n, d), generator=gen, dtype=dtype)
    Z = Z * torch.tensor(COLUMN_SCALE, dtype=dtype)
    theta = torch.nn.Parameter(torch.tensor(THETA_VALUES, dtype=dtype))
    rejected = torch.zeros((n, 1), dtype=dtype)
    return Z, theta, rejected


def pairwise_loss_and_flat_grad(loss_cls, Z, theta, rejected):
    """Real PairWiseLoss/LogExpLoss value and its create-graph gradient."""
    chosen = (Z @ theta).unsqueeze(-1)
    loss = loss_cls()(chosen, rejected, None)
    grads = torch.autograd.grad(loss, [theta], create_graph=True, retain_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    return loss, flat_grad


def analytic_hessian(Z, theta, rejected):
    """H = Z.T @ diag(p * (1 - p)) @ Z / n for p = sigmoid(Z @ theta - rejected)."""
    logits = (Z @ theta).unsqueeze(-1) - rejected
    p = torch.sigmoid(logits)
    weights = (p * (1 - p)).squeeze(-1)
    return (Z.T * weights) @ Z / Z.shape[0]


def regularized_hessian(Z, theta, rejected, damping):
    """H + damping * I, matching the damping term added by hessian_vector_product."""
    n = Z.shape[1]
    return analytic_hessian(Z, theta, rejected) + damping * torch.eye(n, dtype=Z.dtype)
