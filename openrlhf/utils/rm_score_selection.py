'''different score methods for the reward model'''

import torch

# Global V matrix for APO score calculation
_global_V = None

def one_step_fisher_score(chosen_reward, reject_reward, chosen_embeddings, rejected_embeddings, margin=None):
    """Compute the importance score for a sample based on Fisher Information Matrix.
    For the last layer (value head) with sigmoid activation, we can compute the Fisher
    Information Matrix directly without computing gradients.

    Args:
        chosen_reward: Reward values for the chosen responses in a batch
        reject_reward: Reward values for the rejected responses in a batch
        chosen_embeddings: Embeddings for the chosen responses in a batch
        rejected_embeddings: Embeddings for the rejected responses in a batch
        margin: Optional margin values for the loss function

    Returns:
        scores: The importance scores for the batch of samples
    """
    with torch.no_grad():
        # Compute scores for each sample in the batch
        scores = []
        for c_emb, r_emb in zip(chosen_embeddings, rejected_embeddings):
            # Compute embedding difference
            diff = (c_emb - r_emb).reshape(-1)
            
            # Compute score as embedding difference magnitude
            score = torch.dot(diff, diff)  # L2 norm squared of the difference
            scores.append(score.item())
        
        return torch.tensor(scores)


def reward_diff_score(chosen_reward, reject_reward, chosen_embeddings, rejected_embeddings, margin=None):
    """Compute the importance score based on the absolute difference between chosen and rejected rewards.

    Args:
        chosen_reward: Reward values for the chosen responses in a batch
        reject_reward: Reward values for the rejected responses in a batch
        chosen_embeddings: Embeddings for the chosen responses in a batch (not used)
        rejected_embeddings: Embeddings for the rejected responses in a batch (not used)
        margin: Optional margin values for the loss function (not used)

    Returns:
        scores: The importance scores for the batch of samples
    """
    with torch.no_grad():
        return torch.abs(chosen_reward - reject_reward).squeeze()


def margin_based_score(chosen_reward, reject_reward, chosen_embeddings, rejected_embeddings, margin=None):
    """Compute the importance score based on how close the sample is to the margin.
    Samples closer to the margin are considered more important.

    Args:
        chosen_reward: Reward values for the chosen responses in a batch
        reject_reward: Reward values for the rejected responses in a batch
        chosen_embeddings: Embeddings for the chosen responses in a batch (not used)
        rejected_embeddings: Embeddings for the rejected responses in a batch (not used)
        margin: Optional margin values for the loss function

    Returns:
        scores: The importance scores for the batch of samples
    """
    with torch.no_grad():
        if margin is None:
            margin = torch.ones_like(chosen_reward)
        diff = (chosen_reward - reject_reward).squeeze()
        return -torch.abs(diff - margin)  # Negative because we want samples closer to margin to have higher scores


def random_score(chosen_reward, reject_reward, chosen_embeddings, rejected_embeddings, margin=None):
    """Generate random importance scores for the batch.

    Args:
        chosen_reward: Reward values for the chosen responses in a batch (used only for shape)
        reject_reward: Reward values for the rejected responses in a batch (not used)
        chosen_embeddings: Embeddings for the chosen responses in a batch (not used)
        rejected_embeddings: Embeddings for the rejected responses in a batch (not used)
        margin: Optional margin values for the loss function (not used)

    Returns:
        scores: Random scores for the batch samples
    """
    with torch.no_grad():
        # Generate random scores with the same shape as chosen_reward
        return torch.rand_like(chosen_reward).squeeze()


def uncertainty_score(chosen_reward, reject_reward, chosen_embeddings, rejected_embeddings, margin=None):
    """Compute the importance score based on APO (Adaptive Preference Optimization) method.
    This method uses Fisher Information Matrix to select the most informative samples.
    The V matrix is accumulated across multiple calls to this function.

    Args:
        chosen_reward: Reward values for the chosen responses in a batch (not used)
        reject_reward: Reward values for the rejected responses in a batch (not used)
        chosen_embeddings: Embeddings for the chosen responses in a batch
        rejected_embeddings: Embeddings for the rejected responses in a batch
        margin: Optional margin values for the loss function (not used)

    Returns:
        scores: The importance scores for the batch samples based on APO selection
    """
    global _global_V
    
    with torch.no_grad():
        # Convert embeddings to float32 for computation
        chosen_embeddings_float = chosen_embeddings.float()
        rejected_embeddings_float = rejected_embeddings.float()
        
        # Compute embedding differences in float32
        diff_embeddings_float = chosen_embeddings_float - rejected_embeddings_float
        
        # Initialize V matrix if not exists (in float32)
        if _global_V is None or _global_V.device != chosen_embeddings.device:
            _global_V = 1e-5 * torch.eye(chosen_embeddings.size(-1), device=chosen_embeddings.device, dtype=torch.float32)
        
        # Update V matrix with outer products
        for diff in diff_embeddings_float:
            outer_product = torch.ger(diff, diff)
            _global_V += outer_product
        
        # Compute scores using matrix operations (all in float32)
        V_inv = torch.inverse(_global_V)
        norm_mat = torch.matmul(torch.matmul(diff_embeddings_float, V_inv), diff_embeddings_float.transpose(-2, -1))
        scores = torch.diag(norm_mat)
        
        # Convert scores back to original dtype
        scores = scores.to(chosen_embeddings.dtype)
        
        return scores


def reset_apo_v_matrix():
    """Reset the global V matrix used in APO score calculation."""
    global _global_V
    _global_V = None


def get_score_fn(score_type: str):
    """Get the scoring function based on the score type.

    Args:
        score_type: The type of scoring function to use.
            Options: "fisher", "reward_diff", "margin", "random", "apo"

    Returns:
        The corresponding scoring function
    """
    score_fns = {
        "fisher": one_step_fisher_score,
        "reward_diff": reward_diff_score,
        "margin": margin_based_score,
        "random": random_score,
        "uncertainty_score": uncertainty_score,
    }
    # print with red:
    print(f"\033[91mScore type: {score_type}. Available options: {list(score_fns.keys())}\033[0m")
    
    if score_type not in score_fns:
        raise ValueError(f"Unknown score type: {score_type}. Available options: {list(score_fns.keys())}")
    
    return score_fns[score_type]
