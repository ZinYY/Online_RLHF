import json
import argparse
from typing import List, Dict, Tuple, Set

import numpy as np
from tqdm import tqdm
from collections import defaultdict


def get_ngrams(sequence: List[float], n: int) -> Set[Tuple[int, ...]]:
    """Get n-grams from a sequence based on ranking order.
    
    Args:
        sequence: List of values
        n: Size of n-grams
        
    Returns:
        Set of n-gram tuples containing indices in sorted order
    """
    # Get indices sorted by values
    sorted_indices = sorted(range(len(sequence)), key=lambda k: sequence[k])
    # Convert to ranking (0 to len-1)
    rankings = [0] * len(sequence)
    for rank, idx in enumerate(sorted_indices):
        rankings[idx] = rank
    
    # Create n-grams from rankings
    ngrams = set()
    for i in range(len(rankings) - n + 1):
        ngram = tuple(rankings[i:i + n])
        ngrams.add(ngram)
    return ngrams


def calculate_order_similarity(seq1: List[float], seq2: List[float], n: int = 2) -> float:
    """Calculate order-sensitive Jaccard similarity between two sequences.
    
    Args:
        seq1: First sequence of values
        seq2: Second sequence of values
        n: Size of n-grams to use (default: 2)
        
    Returns:
        Similarity score between 0 and 1
    """
    if len(seq1) != len(seq2):
        return 0.0
    
    # Get n-grams for both sequences
    ngrams1 = get_ngrams(seq1, n)
    ngrams2 = get_ngrams(seq2, n)
    
    # Calculate Jaccard similarity
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return intersection / union if union > 0 else 1.0


def load_and_process_file(file_path: str) -> Dict[str, Dict[str, List]]:
    """Load json file and organize data by input text.
    
    Args:
        file_path: Path to json file
        
    Returns:
        Dict mapping input text to dict containing outputs and rewards
    """
    processed_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            input_text = item["input"]
            # Create mapping from output to its reward
            output_to_reward = {output: reward for output, reward in zip(item["outputs"], item["rewards"])}
            processed_data[input_text] = {
                "outputs"         : item["outputs"],
                "output_to_reward": output_to_reward
            }
    return processed_data


def compare_rewards(eval_file: str, true_file: str, output_file: str):
    """Compare evaluated rewards with ground truth rewards.
    
    Args:
        eval_file: Path to file containing evaluated rewards
        true_file: Path to file containing ground truth rewards
        output_file: Output file path for comparison results
    """
    # Load and process both files
    print("Processing evaluated rewards file... {}".format(eval_file))
    eval_data = load_and_process_file(eval_file)
    print("Processing ground truth rewards file... {}".format(true_file))
    gt_data = load_and_process_file(true_file)
    
    # Compare rewards for each input
    print("Comparing rewards...")
    comparison_results = []
    
    # Get intersection of inputs from both files
    common_inputs = set(eval_data.keys()) & set(gt_data.keys())
    
    for input_text in tqdm(common_inputs):
        eval_item = eval_data[input_text]
        gt_item = gt_data[input_text]
        
        # Get rewards for outputs in eval file's order
        eval_rewards = []
        gt_rewards = []
        
        # Use eval file's output order as reference
        for output in eval_item["outputs"]:
            eval_rewards.append(eval_item["output_to_reward"][output])
            # Find corresponding reward in ground truth
            if output in gt_item["output_to_reward"]:
                gt_rewards.append(gt_item["output_to_reward"][output])
            else:
                print(f"Warning: Output not found in ground truth file for input: {input_text[:100]}...")
                continue
        
        # sort the eval_rewards and gt_rewards (according to the eval_rewards)
        eval_rewards, gt_rewards = zip(*sorted(zip(eval_rewards, gt_rewards), key=lambda x: x[0]))
        
        # calculate mean of abs diff of the eval_rewards and gt_rewards:
        # Normalize rewards to [0,1] range before calculating difference
        eval_rewards_norm = (np.array(eval_rewards) - np.min(eval_rewards)) / (np.max(eval_rewards) - np.min(eval_rewards) + 1e-6)
        gt_rewards_norm = (np.array(gt_rewards) - np.min(gt_rewards)) / (np.max(gt_rewards) - np.min(gt_rewards) + 1e-6)
        mean_abs_diff = np.mean(np.abs(eval_rewards_norm - gt_rewards_norm))
        
        # Calculate order similarity if we have valid rewards
        if len(eval_rewards) == len(gt_rewards) and len(eval_rewards) > 0:
            # Calculate similarities with different n-gram sizes
            similarities = {
                f"order_similarity_n{n}": calculate_order_similarity(eval_rewards, gt_rewards, n)
                for n in [2] if n <= len(eval_rewards)
            }
        else:
            similarities = {
                "order_similarity_n2": 0.0
            }
        
        comparison_results.append({
            "input"        : input_text,
            "eval_rewards" : eval_rewards,
            "gt_rewards"   : gt_rewards,
            **similarities,
            "mean_abs_diff": mean_abs_diff
        })
    
    # Write results
    print(f"Writing comparison results for {len(comparison_results)} inputs...")
    with open(output_file, 'w') as f:
        for item in comparison_results:
            f.write(json.dumps(item) + '\n')
    
    # Print average similarities
    avg_similarities = defaultdict(list)
    for item in comparison_results:
        for k, v in item.items():
            if k.startswith("order_similarity"):
                avg_similarities[k].append(v)
            elif k == "mean_abs_diff":
                avg_similarities[k].append(v)
    
    print("\nAverage similarities across all inputs:")
    for k, v in avg_similarities.items():
        print(f"{k}: {sum(v) / len(v):.4f}")


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--eval_file', type=str, required=True,
    #                     help='File containing evaluated rewards')
    # parser.add_argument('--true_file', type=str, required=True,
    #                     help='File containing ground truth rewards')
    # parser.add_argument('--output_file', type=str, required=True,
    #                     help='Output file path for comparison results')
    
    parser.add_argument('--eval_file', type=str,
                        # required=True,
                        help='File containing evaluated rewards')
    parser.add_argument('--true_file', type=str,
                        # required=True,
                        help='File containing ground truth rewards')
    parser.add_argument('--output_file', type=str,
                        # required=True,
                        help='Output file path for comparison results')
    
    args = parser.parse_args()
    
    # args.eval_file = './checkpoint_online_ultraFB_gemma_2/gemma-2-2b-ppo_generated_dataset_current_t=5.json'
    # args.true_file = './checkpoint_online_ultraFB_gemma_2/gemma-2-2b-ppo_rm_eval_current_t=5.json'
    # args.output_file = './checkpoint_online_ultraFB_gemma_2/gemma-2-2b-ppo_rm_current_t=5_comparison.json'
    
    compare_rewards(args.eval_file, args.true_file, args.output_file)


if __name__ == '__main__':
    main()
