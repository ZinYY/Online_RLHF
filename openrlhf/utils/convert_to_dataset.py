import json
import argparse
import random
import re  # Add re import for regex support
from typing import List, Dict, Tuple
from tqdm import tqdm


def clean_content(text: str) -> str:
    """Clean special tokens from text content.
    
    Args:
        text: Input text with special tokens
        
    Returns:
        Cleaned text without special tokens
    """
    # Remove system message with date using regex
    date_pattern = r"<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n\nCutting Knowledge Date: December 2023\nToday Date: .*?\n\n<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>\n\n"
    text = re.sub(date_pattern, "", text)
    
    # Remove any extra whitespace
    text = text.strip()
    

    # List of patterns to remove
    patterns = [
        "<bos><start_of_turn>user\n",
        "\n\nA:<end_of_turn>\n<start_of_turn>model\n",
        "<eos>",
        "<start_of_turn>user\n",
        "<end_of_turn>\n<start_of_turn>model\n",
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
        '<|eot_id|><|start_header_id|>assistant<|end_header_id|>',
        '<|eot_id|>'
    ]
    
    # Remove each pattern
    for pattern in patterns:
        text = text.replace(pattern, "")
    
    
    return text

def format_chat_template(prompt: str, response: str) -> List[Dict[str, str]]:
    """Format prompt and response into chat template format.
    
    Args:
        prompt: The input prompt text
        response: The model's response text
        
    Returns:
        List of message dictionaries in chat format
    """
    return [
        {"role": "user", "content": clean_content(prompt)},
        {"role": "assistant", "content": clean_content(response)}
    ]

def get_best_worst_indices(rewards: List[float]) -> Tuple[int, int]:
    """Get indices of best and worst rewards."""
    max_idx = rewards.index(max(rewards))
    min_idx = rewards.index(min(rewards))
    return max_idx, min_idx

def get_best_second_indices(rewards: List[float]) -> Tuple[int, int]:
    """Get indices of best and second best rewards."""
    sorted_indices = sorted(range(len(rewards)), key=lambda k: rewards[k], reverse=True)
    if len(sorted_indices) < 2:
        return None, None
    return sorted_indices[0], sorted_indices[1]

def get_best_quartile_indices(rewards: List[float]) -> Tuple[int, int]:
    """Get indices of best and 75th percentile rewards."""
    sorted_indices = sorted(range(len(rewards)), key=lambda k: rewards[k], reverse=True)
    if len(sorted_indices) < 4:
        return None, None
    # quartile_idx = len(sorted_indices) // 4
    return sorted_indices[0], sorted_indices[3]

def get_random_indices(rewards: List[float]) -> Tuple[int, int]:
    """Get two random indices from the rewards list.
    
    Args:
        rewards: List of reward values
        
    Returns:
        Tuple of two different random indices, or (None, None) if list has less than 2 elements
    """
    if len(rewards) < 2:
        return None, None
    idx1 = random.randint(0, len(rewards) - 1)
    idx2 = random.randint(0, len(rewards) - 2)
    if idx2 >= idx1:
        idx2 += 1
    return idx1, idx2

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


def convert_to_preference_dataset(eval_file: str, true_file: str, output_file: str, strategy: str = "best_worst"):
    """Convert json files with multiple outputs and rewards to preference dataset format.
    
    Args:
        eval_file: Path to file containing evaluated rewards (used for selection)
        true_file: Path to file containing ground truth rewards (used for scores)
        output_file: Output file path
        strategy: Strategy for selecting response pairs:
            - "best_worst": Select best and worst responses
            - "best_second": Select best and second best responses
            - "best_quartile": Select best and 75th percentile responses
            - "random": Select two random responses
    """
    # Load and process both files
    print("Processing evaluated rewards file...")
    eval_data = load_and_process_file(eval_file)
    print("Processing ground truth rewards file...")
    true_data = load_and_process_file(true_file)
    
    # Select strategy function
    strategy_funcs = {
        "best_worst": get_best_worst_indices,
        "best_second": get_best_second_indices,
        "best_quartile": get_best_quartile_indices,
        "random": get_random_indices
    }
    if strategy not in strategy_funcs:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of {list(strategy_funcs.keys())}")
    
    get_indices = strategy_funcs[strategy]
    
    # Get intersection of inputs from both files
    common_inputs = set(eval_data.keys()) & set(true_data.keys())
    
    # Convert format
    preference_data = []
    swapped_count = 0
    for input_text in tqdm(common_inputs):
        eval_item = eval_data[input_text]
        true_item = true_data[input_text]
        
        # Get indices based on evaluated rewards
        eval_rewards = [eval_item["output_to_reward"][output] for output in eval_item["outputs"]]
        chosen_idx, rejected_idx = get_indices(eval_rewards)
        
        # Skip if indices are invalid
        if chosen_idx is None or rejected_idx is None or chosen_idx == rejected_idx:
            continue
        
        # Get the outputs and their true rewards
        chosen_output = eval_item["outputs"][chosen_idx]
        rejected_output = eval_item["outputs"][rejected_idx]
        
        # Get true rewards for these outputs
        if chosen_output in true_item["output_to_reward"] and rejected_output in true_item["output_to_reward"]:
            chosen_score = true_item["output_to_reward"][chosen_output]
            rejected_score = true_item["output_to_reward"][rejected_output]
            
            # Swap if chosen score is less than rejected score
            if chosen_score < rejected_score:
                chosen_output, rejected_output = rejected_output, chosen_output
                chosen_score, rejected_score = rejected_score, chosen_score
                swapped_count += 1
        else:
            print(f"Warning: Output not found in ground truth file for input: {input_text[:100]}...")
            continue
            
        preference_item = {
            "prompt": clean_content(input_text),
            "chosen"        : format_chat_template(input_text, chosen_output),
            "rejected"      : format_chat_template(input_text, rejected_output),
            "chosen_score"  : chosen_score,
            "rejected_score": rejected_score
        }
        preference_data.append(preference_item)
    
    # Write output file
    print(f"Writing {len(preference_data)} samples to {output_file}")
    print(f"Swapped {swapped_count} pairs to ensure chosen_score > rejected_score")
    with open(output_file, 'w') as f:
        for item in preference_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Converted {len(preference_data)} samples to {output_file} using strategy: {strategy}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str,
                        # required=True,
                        help='File containing evaluated rewards (used for selection)')
    parser.add_argument('--true_file', type=str,
                        # required=True,
                        help='File containing ground truth rewards (used for scores)')
    parser.add_argument('--output_file', type=str,
                        # required=True,
                        help='Output file path')
    parser.add_argument('--strategy', type=str, default="best_worst",
                        choices=["best_worst", "best_second", "best_quartile", "random"],
                        help='Strategy for selecting response pairs')
    
    args = parser.parse_args()
    
    # args.eval_file = './checkpoint_online_ultraFB_gemma_2/gemma-2-2b-ppo_generated_dataset_current_t=5.json'
    # args.true_file = './checkpoint_online_ultraFB_gemma_2/gemma-2-2b-ppo_rm_eval_current_t=5.json'
    # args.output_file = './checkpoint_online_ultraFB_gemma_2/gemma-2-2b-ppo_current_t=5_output_PPO_dataset.json'
    
    convert_to_preference_dataset(args.eval_file, args.true_file, args.output_file, args.strategy)

if __name__ == '__main__':
    main()
