import json
import argparse
from typing import List, Dict
import subprocess
from pathlib import Path

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

def evaluate_iterations(eval_dir: str, start_t: int = 0, end_t: int = 4):
    """Evaluate multiple iterations of PPO dataset results.
    
    Args:
        eval_dir: Directory containing evaluation files
        start_t: Starting iteration number (inclusive)
        end_t: Ending iteration number (inclusive)
    """
    results_file = Path(eval_dir) / "mean_scores.txt"
    
    # Clear previous results
    with open(results_file, "w") as f:
        f.write("Evaluation Results:\n")
    
    for current_t in range(start_t, end_t + 1):
        dataset_file = Path(eval_dir) / f"llama3.2-1b-ppo_current_t={current_t}_output_PPO_dataset.json"
        
        if not dataset_file.exists():
            print(f"Warning: File not found for iteration {current_t}: {dataset_file}")
            continue
            
        try:
            mean_score = calculate_mean_scores(dataset_file)
            
            # Save to results file
            with open(results_file, "a") as f:
                f.write(f"Iteration {current_t}, mean of chosen and rejected scores: {mean_score:.4f}\n")
                
            print(f"Processed iteration {current_t}: mean score = {mean_score:.4f}")
            
        except Exception as e:
            print(f"Error processing iteration {current_t}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', type=str, required=True,
                        help='Directory containing evaluation files')
    parser.add_argument('--start_t', type=int, default=0,
                        help='Starting iteration number (inclusive)')
    parser.add_argument('--end_t', type=int, default=4,
                        help='Ending iteration number (inclusive)')
    
    args = parser.parse_args()
    evaluate_iterations(args.eval_dir, args.start_t, args.end_t)

if __name__ == '__main__':
    main() 