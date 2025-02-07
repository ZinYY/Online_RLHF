import json
import argparse
from typing import List, Dict, Tuple, Set

import numpy as np
from tqdm import tqdm
from collections import defaultdict


def calculate_mean_scores(file_path: str) -> float:
    """Calculate mean of sum of chosen_score and rejected_score for each item.

    Args:
        file_path: Path to json file containing scores

    Returns:
        Mean value of the sum of scores
    """
    total_sum = 0
    count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            score_sum = item["chosen_score"] + item["rejected_score"]
            total_sum += score_sum
            count += 1
    
    return total_sum / count if count > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_file', type=str,
                        help='File containing chosen and rejected scores')
    
    args = parser.parse_args()
    
    # args.score_file = './checkpoint_online_ultraFB_llama3_ppo_hvp_best_worst/llama3.2-1b-ppo_current_t=4_output_PPO_dataset.json'
    if args.score_file:
        mean_score = calculate_mean_scores(args.score_file)
        print(f"\n***Mean sum of chosen and rejected scores: {mean_score:.4f}")


if __name__ == '__main__':
    main()
