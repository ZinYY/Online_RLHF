"""Characterization tests for ``openrlhf/utils/rm_score_selection.py``.

The module is loaded fresh per test (conftest fixture), so its module-level
``_global_V`` accumulator used by ``uncertainty_score`` cannot leak between
tests.

Naming notes (fixed on branch ``fix/cg-fletcher-reeves-beta``): "apo" is now
an accepted alias of "uncertainty_score" (the docstring already advertised
it), and ``one_step_fisher_score`` is a deprecated alias of the accurately
named ``one_step_embedding_l2_score``. See docs/maintenance/review.md.
"""

import pytest
import torch


def _batch():
    chosen_reward = torch.tensor([[2.0], [0.5], [1.5]])
    reject_reward = torch.tensor([[1.0], [0.5], [0.0]])
    chosen_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rejected_emb = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    return chosen_reward, reject_reward, chosen_emb, rejected_emb


def test_fisher_score_is_squared_embedding_distance(legacy_rm_score_selection):
    """Despite the "Fisher" name, the score is the L2-squared difference of
    the chosen/rejected embeddings."""
    _, _, chosen_emb, rejected_emb = _batch()
    scores = legacy_rm_score_selection.one_step_fisher_score(
        None, None, chosen_emb, rejected_emb
    )
    diffs = (chosen_emb - rejected_emb).reshape(3, -1)
    expected = torch.einsum("ij,ij->i", diffs, diffs)
    assert torch.allclose(scores, expected)
    assert scores.dtype == torch.float32


def test_reward_diff_score_is_absolute_difference(legacy_rm_score_selection):
    chosen_reward, reject_reward, _, _ = _batch()
    scores = legacy_rm_score_selection.reward_diff_score(
        chosen_reward, reject_reward, None, None
    )
    assert torch.allclose(scores, torch.tensor([1.0, 0.0, 1.5]))


def test_margin_score_defaults_to_ones(legacy_rm_score_selection):
    chosen_reward, reject_reward, _, _ = _batch()
    scores = legacy_rm_score_selection.margin_based_score(
        chosen_reward, reject_reward, None, None
    )
    diff = (chosen_reward - reject_reward).squeeze()
    assert torch.allclose(scores, -torch.abs(diff - 1.0))


def test_random_score_shape_and_seeding(legacy_rm_score_selection):
    chosen_reward, reject_reward, _, _ = _batch()
    torch.manual_seed(123)
    first = legacy_rm_score_selection.random_score(
        chosen_reward, reject_reward, None, None
    )
    torch.manual_seed(123)
    second = legacy_rm_score_selection.random_score(
        chosen_reward, reject_reward, None, None
    )
    assert first.shape == chosen_reward.squeeze().shape
    assert torch.equal(first, second)
    assert (first >= 0).all() and (first <= 1).all()


def test_uncertainty_score_uses_accumulated_v_matrix(legacy_rm_score_selection):
    mod = legacy_rm_score_selection
    chosen_reward, reject_reward, chosen_emb, rejected_emb = _batch()
    emb_dim = chosen_emb.size(-1)

    mod.reset_apo_v_matrix()
    first = mod.uncertainty_score(
        chosen_reward, reject_reward, chosen_emb, rejected_emb
    )
    assert first.shape == (3,)
    assert torch.isfinite(first).all()

    # After the first call, V = 1e-5 * I + sum of outer products; a second
    # call accumulates further, so scores change (documents the global state).
    v_after_first = mod._global_V.clone()
    assert v_after_first.shape == (emb_dim, emb_dim)
    assert torch.allclose(
        v_after_first,
        1e-5 * torch.eye(emb_dim)
        + sum(
            torch.ger(d, d) for d in (chosen_emb - rejected_emb).float().reshape(3, -1)
        ),
    )
    second = mod.uncertainty_score(
        chosen_reward, reject_reward, chosen_emb, rejected_emb
    )
    assert not torch.allclose(first, second)

    mod.reset_apo_v_matrix()
    assert mod._global_V is None


def test_get_score_fn_dispatches_known_types(legacy_rm_score_selection):
    mod = legacy_rm_score_selection
    assert mod.get_score_fn("fisher") is mod.one_step_embedding_l2_score
    assert mod.get_score_fn("reward_diff") is mod.reward_diff_score
    assert mod.get_score_fn("margin") is mod.margin_based_score
    assert mod.get_score_fn("random") is mod.random_score
    assert mod.get_score_fn("uncertainty_score") is mod.uncertainty_score


def test_get_score_fn_accepts_apo_alias(legacy_rm_score_selection):
    mod = legacy_rm_score_selection
    # "apo" is documented in the docstring and now aliases uncertainty_score.
    assert mod.get_score_fn("apo") is mod.uncertainty_score


def test_fisher_alias_is_deprecated_but_equivalent(legacy_rm_score_selection):
    mod = legacy_rm_score_selection
    assert mod.one_step_fisher_score is mod.one_step_embedding_l2_score
    _, _, chosen_emb, rejected_emb = _batch()
    old_name = mod.one_step_fisher_score(None, None, chosen_emb, rejected_emb)
    new_name = mod.one_step_embedding_l2_score(None, None, chosen_emb, rejected_emb)
    assert torch.equal(old_name, new_name)


def test_get_score_fn_rejects_unknown_types(legacy_rm_score_selection):
    mod = legacy_rm_score_selection
    with pytest.raises(ValueError, match="Unknown score type"):
        mod.get_score_fn("not_a_score")
