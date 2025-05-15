#!/usr/bin/env python3
import os
import subprocess
import argparse
from datetime import datetime
import random


def get_random_port():
    """Generate a random port number between 20000 and 65535"""
    return random.randint(20000, 65535)


def run_command(cmd, env=None):
    """Run a command and return its output"""
    try:
        # Print command in red
        RED = "\033[91m"
        RESET = "\033[0m"
        print(f"\n{RED}Executing command:{RESET}")
        print(f"{RED}{cmd}{RESET}\n")
        
        result = subprocess.run(cmd, shell=True, check=True, text=True, env=env)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        return False


def get_gpu_config():
    """Get GPU configuration based on hardware"""
    try:
        gpu_model = subprocess.check_output("nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits", shell=True).decode().strip()
        if "A800" in gpu_model:
            return "0,2,3,4", 32  # gpu_nodes, train_batch_size for A800
        else:
            return "0,1,2,3", 32  # gpu_nodes, train_batch_size for others
    except:
        print("Error detecting GPU configuration. Using default values.")
        return "0,1,2,3", 32


def train_reward_model_active_SGD(gpu_nodes, train_batch_size, args):
    """Train the reward model using active learning"""
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    deepspeed --include=localhost:{gpu_nodes} --master_port {get_random_port()} --module openrlhf.cli.active_train_rm_head \\
     --save_path ./checkpoint_mixture2/llama3-8b-rm-active \\
     --save_steps -1 \\
     --logging_steps 1 \\
     --eval_steps 100 \\
     --train_batch_size {train_batch_size} \\
     --micro_train_batch_size 8 \\
     --max_len 1024 \\
     --max_samples 50000 \\
     --max_select_samples 6400 \\
     --retrain_step {args.retrain_step} \\
     --pretrain ./checkpoint_mixture2/llama3-8b-sft \\
     --bf16 \\
     --max_epochs 1 \\
     --zero_stage 0 \\
     --learning_rate 1e-3 \\
     --score_type {args.score_type} \\
     --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \\
     --train_split train_prefs \\
     --eval_split test_prefs \\
     --apply_chat_template \\
     --chosen_key chosen \\
     --rejected_key rejected \\
     --flash_attn \\
     --seed {args.seed} \\
     --packing_samples \\
     --use_wandb {args.wandb_key} \\
     --wandb_run_name llama3-8b-rm-head-active-SGD-{args.score_type} \\
     --wandb_project openrlhf_RM_mixture2_head_active
    """
    return run_command(cmd)


def train_reward_model_active_hvp(gpu_nodes, train_batch_size, args):
    """Train the reward model using active learning"""
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    deepspeed --include=localhost:{gpu_nodes} --master_port {get_random_port()} --module openrlhf.cli.active_train_rm_head_hvp \\
     --save_path ./checkpoint_mixture2/llama3-8b-rm-active \\
     --save_steps -1 \\
     --logging_steps 1 \\
     --eval_steps 100 \\
     --train_batch_size {train_batch_size} \\
     --micro_train_batch_size 8 \\
     --max_len 1024 \\
     --max_samples 50000 \\
     --max_select_samples 6400 \\
     --damping {args.damping} \\
     --damping_strategy linear \\
     --damping_growth_rate {args.damping_growth_rate} \\
     --use_hvp \
     --pretrain ./checkpoint_mixture2/llama3-8b-sft \\
     --bf16 \\
     --max_epochs 1 \\
     --zero_stage 0 \\
     --learning_rate 1e-3 \\
     --score_type {args.score_type} \\
     --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \\
     --apply_chat_template \\
     --chosen_key chosen \\
     --rejected_key rejected \\
     --flash_attn \\
     --seed {args.seed} \\
     --packing_samples \\
     --use_wandb {args.wandb_key} \\
     --wandb_run_name llama3-8b-rm-head-active-hvp-{args.score_type} \\
     --wandb_project openrlhf_RM_mixture2_head_active
    """
    return run_command(cmd)


def main():
    # Add color codes
    RED = "\033[91m***"
    RESET = "\033[0m"
    
    parser = argparse.ArgumentParser(description='Active Learning Pipeline for LLaMA 3 with Mixture2 Dataset')
    parser.add_argument('--score_type', type=str, default='random',
                        choices=['random', 'reward_diff', 'margin', 'fisher', 'uncertainty_score'],
                        help='Scoring strategy for active learning')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_key', type=str, default='48de847750dd8d971bd7cce0720e512a8e5cc067', help='Weights & Biases API key')
    parser.add_argument('--method', type=str, default='sgd', choices=['sgd', 'hvp'],
                        help='Training method: SGD or HVP')
    parser.add_argument('--damping', type=float, default=0.8, help='Damping factor for HVP')
    parser.add_argument('--damping_growth_rate', type=int, default=100, help='Damping growth rate for HVP')
    parser.add_argument('--retrain_step', type=int, default=200, help='Number of steps between retraining on buffer')
    args = parser.parse_args()
    
    # Get GPU configuration
    gpu_nodes, train_batch_size = get_gpu_config()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs("./checkpoint_mixture2", exist_ok=True)
    
    # Train Reward Model with Active Learning
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{current_time} {RED}Training Reward Model with Active Learning ({args.score_type}, {args.method.upper()}){RESET}")
    
    if args.method == 'sgd':
        success = train_reward_model_active_SGD(gpu_nodes, train_batch_size, args)
    else:  # hvp
        success = train_reward_model_active_hvp(gpu_nodes, train_batch_size, args)
    
    if not success:
        print("Failed to train reward model with active learning")
        return


if __name__ == "__main__":
    main()
