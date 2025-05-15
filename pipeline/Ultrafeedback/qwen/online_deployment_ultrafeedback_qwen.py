#!/usr/bin/env python3
import os
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Callable, Any, Tuple
import json


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
            return "0,2,3,4", 4, 4, 48  # gpu_nodes, train_batch_size, tp_size, max_num_seqs for A800
        else:
            return "0,1,2,3", 4, 4, 24  # gpu_nodes, train_batch_size, tp_size, max_num_seqs for others
    except:
        print("Error detecting GPU configuration. Using default values.")
        return "0,1,2,3", 4, 4, 24


def train_reward_model_dataset0(current_t, total_T, gpu_nodes, train_batch_size, **kwargs):
    """Train the reward model for the current iteration"""
    pretrain_path = "./checkpoint_ultraFB/qwen2.5-7b-sft" if current_t == 0 else f"./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/qwen2.5-7b-rm"
    
    # Choose dataset based on current iteration
    # if current_t == 0:
    dataset_args = """--dataset HuggingFaceH4/ultrafeedback_binarized \\
     --train_split train_prefs \\
     --eval_split test_prefs \\"""
    # else:
    #     dataset_args = f"--dataset ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_current_t={current_t - 1}_output_PPO_dataset.json \\"
    
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    deepspeed --include=localhost:{gpu_nodes} --master_port 27009 --module openrlhf.cli.online_train_rm_head \\
     --save_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/qwen2.5-7b-rm \\
     --save_steps -1 \\
     --logging_steps 1 \\
     --eval_steps -1 \\
     --train_batch_size 4 \\
     --micro_train_batch_size 1 \\
     --max_len 1024 \\
     --max_samples 4 \\
     --pretrain {pretrain_path} \\
     --bf16 \\
     --max_epochs 1 \\
     --zero_stage 0 \\
     --learning_rate 1e-3 \\
     {dataset_args}
     --apply_chat_template \\
     --chosen_key chosen \\
     --rejected_key rejected \\
     --flash_attn \\
     --packing_samples \\
     --wandb_project online_openrlhf_RM_ultraFB \\
     --total_T 1 \\
     --current_t 0 \\
     --t_length 1
    """
    # MLE: use all previous data to train the reward model
    return run_command(cmd)


def train_reward_model(current_t, total_T, gpu_nodes, train_batch_size, **kwargs):
    """Train the reward model for the current iteration"""
    
    for tt in range(1, current_t + 1):
        pretrain_path = "./checkpoint_ultraFB/qwen2.5-7b-sft" if current_t == 0 else f"./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/qwen2.5-7b-rm"
        
        # Choose dataset based on current iteration
        if current_t == 0:
            dataset_args = """--dataset HuggingFaceH4/ultrafeedback_binarized \\
         --train_split train_prefs \\
         --eval_split test_prefs \\"""
        else:
            dataset_args = f"--dataset ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_current_t={tt - 1}_output_PPO_dataset.json \\"
        
        cmd = f"""
        export CUDA_VISIBLE_DEVICES={gpu_nodes}
        deepspeed --include=localhost:{gpu_nodes} --master_port 27009 --module openrlhf.cli.online_train_rm_head \\
         --save_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/qwen2.5-7b-rm \\
         --save_steps -1 \\
         --logging_steps 1 \\
         --eval_steps -1 \\
         --train_batch_size {4 if tt > 0 else 4} \\
         --micro_train_batch_size {1 if tt > 0 else 1} \\
         --max_len 1024 \\
         --max_samples {50000 if tt > 0 else 4} \\
         --pretrain {pretrain_path} \\
         --bf16 \\
         --max_epochs 1 \\
         --zero_stage 0 \\
         --learning_rate 1e-3 \\
         {dataset_args}
         --apply_chat_template \\
         --chosen_key chosen \\
         --rejected_key rejected \\
         --flash_attn \\
         --packing_samples \\
         --wandb_project online_openrlhf_RM_ultraFB \\
         --total_T 1 \\
         --current_t 0 \\
         --t_length 1
        """
    # MLE: use all previous data to train the reward model
    return run_command(cmd)


def train_reward_model_hvp(current_t, total_T, gpu_nodes, train_batch_size, **kwargs):
    """Train the reward model using HVP for the current iteration"""
    pretrain_path = "./checkpoint_ultraFB/qwen2.5-7b-sft" if current_t == 0 else f"./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/qwen2.5-7b-rm"
    
    # Choose dataset based on current iteration
    if current_t == 0:
        dataset_args = """--dataset HuggingFaceH4/ultrafeedback_binarized \\
     --train_split train_prefs \\
     --eval_split test_prefs \\"""
    else:
        dataset_args = f"--dataset ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_current_t={current_t - 1}_output_PPO_dataset.json \\"
    
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    deepspeed --include=localhost:{gpu_nodes} --master_port 27009 --module openrlhf.cli.online_train_rm_head_hvp \\
     --save_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/qwen2.5-7b-rm \\
     --save_steps -1 \\
     --logging_steps 1 \\
     --eval_steps -1 \\
     --train_batch_size 4 \\
     --micro_train_batch_size 1 \\
     --max_len 1024 \\
     --max_samples {50000 if current_t > 0 else 4} \\
     --damping 0.8 \\
     --damping_strategy linear \\
     --damping_growth_rate 100 \\
     --num_cg_steps 3 \\
     --use_hvp \\
     --pretrain {pretrain_path} \\
     --bf16 \\
     --max_epochs 1 \\
     --zero_stage 0 \\
     --learning_rate 1e-3 \\
     {dataset_args}
     --apply_chat_template \\
     --chosen_key chosen \\
     --rejected_key rejected \\
     --packing_samples \\
     --wandb_project online_openrlhf_RM_ultraFB_hvp \\
     --total_T 1 \\
     --current_t 0
    """
    # HVP: use only current data to train the reward model
    return run_command(cmd)


def train_dpo(gpu_nodes, train_batch_size, current_t, **kwargs):
    """Train using DPO"""
    pretrain_path = "meta-llama/Llama-3.2-1B-Instruct" if current_t == 0 else f"./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo"
    
    # Choose dataset based on current iteration
    if current_t == 0:
        dataset_args = """--dataset HuggingFaceH4/ultrafeedback_binarized \\
     --train_split train_prefs \\
     --eval_split test_prefs \\"""
    else:
        dataset_args = f"--dataset ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_current_t={current_t - 1}_output_DPO_dataset.json \\"
    
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    deepspeed --include=localhost:{gpu_nodes} --master_port 27012 --module openrlhf.cli.train_dpo \\
       --save_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo \\
       --save_steps -1 \\
       --logging_steps 1 \\
       --eval_steps -1 \\
       --train_batch_size 4 \\
       --micro_train_batch_size 1 \\
       --pretrain {pretrain_path} \\
       --bf16 \\
       --max_epochs 1 \\
       --max_len 1024 \\
       --zero_stage 3 \\
       --learning_rate 5e-7 \\
       --beta 0.1 \\
       {dataset_args}
       --apply_chat_template \\
       --chosen_key chosen \\
       --rejected_key rejected \\
       --max_samples {args.dpo_initial_num} \\
       --flash_attn \\
       --load_checkpoint \\
       --wandb_project online_openrlhf_DPO_ultraFB
    """
    return run_command(cmd)


def train_ppo(gpu_nodes, train_batch_size, current_t, total_T, **kwargs):
    """Train using PPO with LoRA"""
    pretrain_path = "meta-llama/Llama-3.2-1B-Instruct" if current_t == 0 else f"./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-rlhf_lora16_merged"
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    deepspeed --include=localhost:{gpu_nodes} --master_port 27011 --module openrlhf.cli.train_ppo_online \
      --pretrain {pretrain_path} \
      --lora_rank 16 \
      --reward_pretrain ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/qwen2.5-7b-rm \
      --save_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-rlhf_lora16 \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --micro_train_batch_size 1 \
      --train_batch_size 4 \
      --micro_rollout_batch_size 1 \
      --rollout_batch_size 8 \
      --max_epochs 1 \
      --prompt_max_len 1024 \
      --generate_max_len 1024 \
      --zero_stage 2 \
      --bf16 \
      --actor_learning_rate 5e-7 \
      --critic_learning_rate 9e-6 \
      --init_kl_coef 0.01 \
      --prompt_data HuggingFaceH4/ultrafeedback_binarized \
      --prompt_split test_gen \
      --input_key messages \
      --apply_chat_template \
      --max_samples 10000 \
      --normalize_reward \
      --adam_offload \
      --flash_attn \
       --wandb_project online_openrlhf_PPO_ultraFB \
      --total_T {total_T} \
      --current_t {current_t}
    """
    if not run_command(cmd):
        return False
    
    # Merge PEFT PPO Model
    merge_cmd = f"python merge_peft.py --peft_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-rlhf_lora16"
    return run_command(merge_cmd)


def generate_dpo_responses(gpu_nodes, tp_size, current_t, max_num_seqs, total_T, **kwargs):
    """Generate responses using the trained model"""
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    python -m openrlhf.cli.online_batch_inference \
       --eval_task generate_vllm \
       --pretrain ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo \
       --max_new_tokens 1024 \
       --prompt_max_len 1024 \
       --max_samples 10000 \
       --dataset HuggingFaceH4/ultrafeedback_binarized \
       --dataset_split test_gen \
       --input_key messages \
       --apply_chat_template \
       --temperature 1.0 \
       --tp_size {tp_size} \
       --best_of_n 2 \
       --enable_prefix_caching \
       --max_num_seqs {max_num_seqs} \
       --iter 0 \
       --rollout_batch_size 10240 \
       --micro_batch_size 1 \
       --total_T {total_T} \
       --current_t {current_t} \
       --output_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_output_generate_vllm_current_t={current_t}.json
    """
    return run_command(cmd)


def generate_ppo_responses(gpu_nodes, tp_size, current_t, total_T, max_num_seqs, **kwargs):
    """Generate responses using the PPO trained model"""
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    python -m openrlhf.cli.online_batch_inference \
       --eval_task generate_vllm \
       --pretrain ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-rlhf_lora16_merged \
       --max_new_tokens 1024 \
       --prompt_max_len 1024 \
       --max_samples 10000 \
       --dataset HuggingFaceH4/ultrafeedback_binarized \
       --dataset_split test_gen \
       --input_key messages \
       --apply_chat_template \
       --temperature 1.0 \
       --tp_size {tp_size} \
       --best_of_n 64 \
       --enable_prefix_caching \
       --max_num_seqs {max_num_seqs} \
       --iter 0 \
       --rollout_batch_size 10240 \
       --micro_batch_size 1 \
       --total_T {total_T} \
       --current_t {current_t} \
       --output_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-rlhf_lora16_output_generate_vllm_current_t={current_t}.json
    """
    return run_command(cmd)


# use oracle RM to generate dataset
def evaluate_dpo_responses(gpu_nodes, current_t, total_T, **kwargs):
    """Evaluate the generated responses using the oracle reward model"""
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    deepspeed --include=localhost:{gpu_nodes} --master_port 27012 --module openrlhf.cli.online_batch_inference \
       --eval_task rm \
       --pretrain NCSOFT/Llama-3-OffsetBias-RM-8B \
       --bf16 \
       --max_len 4096 \
       --dataset ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_output_generate_vllm_current_t={current_t}.json  \
       --dataset_probs 1.0 \
       --zero_stage 0 \
       --post_processor make_dataset \
       --micro_batch_size 1 \
       --total_T 1 \
       --current_t 0 \
       --output_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_rm_oracle_current_t={current_t}.json
    """
    return run_command(cmd)


# use oracle RM to generate dataset
def evaluate_ppo_responses(gpu_nodes, current_t, total_T, **kwargs):
    """Evaluate the PPO generated responses using the oracle reward model"""
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    deepspeed --include=localhost:{gpu_nodes} --master_port 27012 --module openrlhf.cli.online_batch_inference \
       --eval_task rm \
       --pretrain NCSOFT/Llama-3-OffsetBias-RM-8B \
       --bf16 \
       --max_len 4096 \
       --dataset ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-rlhf_lora16_output_generate_vllm_current_t={current_t}.json  \
       --dataset_probs 1.0 \
       --zero_stage 0 \
       --post_processor make_dataset \
       --micro_batch_size 1 \
       --total_T 1 \
       --current_t 0 \
       --output_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_rm_oracle_current_t={current_t}.json
    """
    return run_command(cmd)


def reward_dataset_ppo(gpu_nodes, current_t, total_T, **kwargs):
    """Create the dataset using PPO's generation and current reward model for the current iteration"""
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    deepspeed --include=localhost:{gpu_nodes} --master_port 27012 --module openrlhf.cli.online_batch_inference \
       --eval_task rm \
       --pretrain ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/qwen2.5-7b-rm \
       --bf16 \
       --max_len 1024 \
       --dataset ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-rlhf_lora16_output_generate_vllm_current_t={current_t}.json  \
       --dataset_probs 1.0 \
       --zero_stage 0 \
       --post_processor make_dataset \
       --micro_batch_size 1 \
       --total_T 1 \
       --current_t 0 \
       --output_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_rm_evaluated_current_t={current_t}.json
    """
    return run_command(cmd)


def reward_dataset_dpo(gpu_nodes, current_t, total_T, **kwargs):
    """Create the dataset using DPO's generation and current reward model for the current iteration"""
    cmd = f"""
    export CUDA_VISIBLE_DEVICES={gpu_nodes}
    deepspeed --include=localhost:{gpu_nodes} --master_port 27012 --module openrlhf.cli.online_batch_inference \
       --eval_task rm \
       --pretrain NCSOFT/Llama-3-OffsetBias-RM-8B \
       --bf16 \
       --max_len 1024 \
       --dataset ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_output_generate_vllm_current_t={current_t}.json  \
       --dataset_probs 1.0 \
       --zero_stage 0 \
       --post_processor make_dataset \
       --micro_batch_size 1 \
       --total_T 1 \
       --current_t 0 \
       --output_path ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_rm_evaluated_current_t={current_t}.json
    """
    return run_command(cmd)


def collect_dpo_results(current_t, **kwargs):
    """Collect and save results from the current iteration"""
    cmd = f"""python -m openrlhf.utils.collect_results \
        --score_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_current_t={current_t}_output_DPO_dataset.json"""
    
    # Run command and capture output
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
        # Extract mean score from output
        mean_score = float(output.strip().split(": ")[-1])
        
        # Save to results file
        results_file = f"./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/mean_scores.txt"
        with open(results_file, "a") as f:
            f.write(f"Iteration {current_t}, mean of chosen and rejected scores: {mean_score:.4f}\n")
        
        return True
    except Exception as e:
        print(f"Failed to collect results: {e}")
        return False


def collect_results(current_t, **kwargs):
    """Collect and save results from the current iteration"""
    cmd = f"""python -m openrlhf.utils.collect_results \
        --score_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_current_t={current_t}_output_PPO_dataset.json"""
    
    # Run command and capture output
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
        # Extract mean score from output
        mean_score = float(output.strip().split(": ")[-1])
        
        # Save to results file
        results_file = f"./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/mean_scores.txt"
        with open(results_file, "a") as f:
            f.write(f"Iteration {current_t}, mean of chosen and rejected scores: {mean_score:.4f}\n")
        
        return True
    except Exception as e:
        print(f"Failed to collect results: {e}")
        return False


def generate_final_dataset_dpo(current_t, **kwargs):
    """Generate the final dataset by comparing rewards and converting to the required format"""
    # Compare rewards
    compare_cmd = f"""python -m openrlhf.utils.compare_rewarded_outputs \
        --eval_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_rm_evaluated_current_t={current_t}.json \
        --true_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_rm_oracle_current_t={current_t}.json \
        --output_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_rm_current_t={current_t}_comparison.json"""
    
    if not run_command(compare_cmd):
        return False
    
    # Convert to dataset
    convert_cmd = f"""python -m openrlhf.utils.convert_to_dataset \
        --eval_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_rm_evaluated_current_t={current_t}.json \
        --true_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_rm_oracle_current_t={current_t}.json \
        --output_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-dpo_current_t={current_t}_output_DPO_dataset.json \
        --strategy {args.rm_strategy}"""
    
    if not run_command(convert_cmd):
        return False
    
    # Collect results after dataset generation
    return collect_dpo_results(current_t)


def generate_final_dataset(current_t, **kwargs):
    """Generate the final dataset by comparing rewards and converting to the required format"""
    # Compare rewards
    compare_cmd = f"""python -m openrlhf.utils.compare_rewarded_outputs \
        --eval_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_rm_evaluated_current_t={current_t}.json \
        --true_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_rm_oracle_current_t={current_t}.json \
        --output_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_rm_current_t={current_t}_comparison.json"""
    
    if not run_command(compare_cmd):
        return False
    
    # Convert to dataset
    convert_cmd = f"""python -m openrlhf.utils.convert_to_dataset \
        --eval_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_rm_evaluated_current_t={current_t}.json \
        --true_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_rm_oracle_current_t={current_t}.json \
        --output_file ./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}/llama3.2-1b-ppo_current_t={current_t}_output_PPO_dataset.json \
        --strategy {args.rm_strategy}"""
    
    if not run_command(convert_cmd):
        return False
    
    # Collect results after dataset generation
    return collect_results(current_t)


def execute_step(step_name: str, step_params: Dict[str, Any], step_functions: Dict[str, Callable]) -> bool:
    """Execute a single step in the pipeline"""
    if step_name not in step_functions:
        print(f"Unknown step: {step_name}")
        return False
    
    time_current = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    RED = "\033[91m***"
    RESET = "\033[0m"
    print(f"\n{time_current} {RED}Executing step: {step_name}{RESET}")
    
    return step_functions[step_name](**step_params)


def process_recipe(recipe: List) -> List[Dict[str, Any]]:
    """Process recipe list into a list of step configurations
    
    Args:
        recipe: Recipe in format like ["SFT", ["RM", "PPO"]*5]
    
    Returns:
        List of dicts containing step name and current_t info
    
    Example input/output:
        Input: ["SFT", ["RM", "PPO", "generate"]*2]
        Output: [
            {"name": "SFT", "current_t": 0},
            {"name": "RM", "current_t": 0},
            {"name": "PPO", "current_t": 0}, 
            {"name": "generate", "current_t": 0},
            {"name": "RM", "current_t": 1},
            {"name": "PPO", "current_t": 1},
            {"name": "generate", "current_t": 1}
        ]
    """
    processed_steps = []
    current_t = 0
    
    for step in recipe:
        if isinstance(step, list):
            # For repeated steps, add them with current_t
            for sub_step in step:
                processed_steps.append({
                    "name"     : sub_step,
                    "current_t": current_t
                })
            # Increment current_t after completing all steps in the list
            current_t += 1
        else:
            # For single steps, add them with current_t
            processed_steps.append({
                "name"     : step,
                "current_t": current_t
            })
    
    return processed_steps


def save_recipe_checkpoint(checkpoint_dir: str, recipe: List, current_step: int):
    """Save the current recipe progress to a checkpoint file"""
    checkpoint_file = os.path.join(checkpoint_dir, "recipe_checkpoint.json")
    checkpoint_data = {
        "rlhf_recipe" : recipe,
        "current_step": current_step
    }
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def load_recipe_checkpoint(checkpoint_dir: str) -> Tuple[List, int]:
    """Load the recipe progress from checkpoint file if it exists"""
    checkpoint_file = os.path.join(checkpoint_dir, "recipe_checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)
        return checkpoint_data["rlhf_recipe"], checkpoint_data["current_step"]
    return None, None


def run_rlhf_recipe(args, rlhf_recipe, step_functions):
    # Get GPU configuration
    gpu_nodes, train_batch_size, tp_size, max_num_seqs = get_gpu_config()
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = f"./checkpoint_online_ultraFB_qwen2.5_{args.method}_{args.rm_type}_{args.rm_strategy}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Try to load checkpoint
    saved_recipe, saved_step = load_recipe_checkpoint(checkpoint_dir)
    if saved_recipe is not None:
        if saved_recipe == rlhf_recipe:
            print(f"\nResuming from step {saved_step}")
            start_step = saved_step
        else:
            print("\nWarning: Saved recipe differs from current recipe. Starting from beginning.")
            start_step = 0
    else:
        print("\nNo checkpoint. Starting from the beginning")
        start_step = 0
    
    # Process recipe
    recipe_steps = process_recipe(rlhf_recipe)
    current_t = -1
    # print(recipe_steps)
    
    RED = "\033[91m"
    RESET = "\033[0m"
    print(f"\n{RED}Recipe Steps: {recipe_steps}{RESET}")
    for step_idx, step in enumerate(recipe_steps):
        print(step_idx, step)
    
    # Execute recipe steps
    for step_idx, step in enumerate(recipe_steps):
        # Skip steps before the saved checkpoint
        if step_idx < start_step:
            continue
        
        # Only process steps within the requested iteration range
        if args.start_t <= step["current_t"] < args.stop_t:
            # Print iteration header when starting a new iteration
            if step["current_t"] != current_t:
                current_t = step["current_t"]
                time_current = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n=== Starting iteration {current_t}/{args.stop_t} at {time_current} ===")
            
            # Prepare step parameters
            step_params = {
                "current_t"       : step["current_t"],
                "total_T"         : args.total_T,
                "gpu_nodes"       : gpu_nodes,
                "train_batch_size": train_batch_size,
                "tp_size"         : tp_size,
                "max_num_seqs"    : max_num_seqs
            }
            
            # Execute the step
            if not execute_step(step["name"], step_params, step_functions):
                print(f"Failed at step {step['name']} in iteration {step['current_t']}")
                return
            
            # Save checkpoint after successful step execution
            save_recipe_checkpoint(checkpoint_dir, rlhf_recipe, step_idx + 1)
            
            # Print iteration footer when all steps in the iteration are complete
            if step == recipe_steps[-1] or recipe_steps[recipe_steps.index(step) + 1]["current_t"] != current_t:
                time_current = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n=== Completed iteration {current_t}/{args.stop_t} at {time_current} ===")


def main():
    global args
    parser = argparse.ArgumentParser(description='Online RLHF Pipeline for LLaMA 3.2 1B with UltraFeedback')
    parser.add_argument('--total_T', type=int, default=50, help='Total number of iterations')
    parser.add_argument('--start_t', type=int, default=0, help='Starting iteration')
    parser.add_argument('--stop_t', type=int, default=30, help='Stopping iteration')
    parser.add_argument('--dpo_initial_num', type=int, default=1000, help='Initial num of samples for train dpo.')
    parser.add_argument('--method', choices=['dpo', 'ppo'], default='ppo', help='Training method (DPO or PPO)')
    parser.add_argument('--rm_type', choices=['regular', 'hvp'], default='hvp', help='Reward model training type')
    parser.add_argument('--rm_strategy', choices=['best_worst', 'best_second', 'best_quartile', 'random'],
                        default='best_worst', help='Strategy for generating preference dataset')
    args = parser.parse_args()
    
    '''
    Generate Online_RLHF_Pipeline recipe.
    
    Recipe Format:
    1. A recipe is a list that can contain two types of elements:
       - Single step: A string representing a single step (e.g., "SFT")
       - Step group: A list of strings representing a group of steps that belong to one iteration
    
    2. Recipe Processing Rules:
       - Single steps will be executed at the current iteration (current_t)
       - Step groups (list) will be executed sequentially, and current_t will increment after all steps in the group are completed
    
    3. Example Recipes:
       a) PPO + HVP:
          ["SFT"] + [["RM", "PPO", "generate", "evaluate", "rewarding", "generate_dataset", "collect_results"]] * stop_t
          - "SFT" runs at t=0
          - Each list of steps runs as a group, incrementing current_t after completion
       
       b) DPO:
          ["SFT"] + [["DPO", "generate", "evaluate", "rewarding", "generate_dataset", "collect_results"]] * stop_t
          - "SFT" runs at t=0
          - DPO groups repeat for stop_t iterations
    
    You can modify the recipe to add or remove steps easily by following this format!
    '''
    ################################################################
    if args.method == 'ppo':
        if args.rm_type == 'hvp':
            rlhf_recipe = ["SFT"] + [["RM", "PPO", "generate", "evaluate", "rewarding", "generate_dataset", "collect_results"]] * args.stop_t
        elif args.rm_type == 'regular':
            # rlhf_recipe = ["SFT"] + [["RM", "PPO", "generate", "evaluate", "rewarding", "generate_dataset", "collect_results"]] * args.stop_t
            rlhf_recipe = ["SFT"] + \
                          [["RM0", "PPO", "generate", "evaluate", "rewarding", "generate_dataset", "collect_results"]] + \
                          [["RM0", "RM", "PPO", "generate", "evaluate", "rewarding", "generate_dataset", "collect_results"]] * (args.stop_t - 1)
        else:
            raise ValueError(f"Unknown RM type: {args.rm_type}")
    # elif args.method == 'dpo':
    #     rlhf_recipe = ["SFT"] + ["DPO"] + [["generate", "evaluate"]] * args.stop_t
    elif args.method == 'dpo':
        rlhf_recipe = ["SFT"] + [["DPO", "generate", "evaluate", "rewarding", "generate_dataset", "collect_results"]] * args.stop_t
    else:
        raise ValueError(f"Unknown method: {args.method}")
    #################################################################
    
    # print rlhf_recipe with red color:
    RED = "\033[91m"
    RESET = "\033[0m"
    print(f"\n{RED}Online RLHF Pipeline Recipe: {rlhf_recipe}{RESET}")
    
    step_functions = {
        "SFT"             : lambda **kwargs: True,  # Placeholder for SFT step (already SFT)
        "RM"              : train_reward_model if args.rm_type == 'regular' else train_reward_model_hvp,
        "RM0"             : train_reward_model_dataset0,
        "PPO"             : train_ppo,
        "DPO"             : train_dpo,
        "generate"        : generate_ppo_responses if args.method == 'ppo' else generate_dpo_responses,
        "evaluate"        : evaluate_ppo_responses if args.method == 'ppo' else evaluate_dpo_responses,
        "rewarding"       : reward_dataset_ppo if args.method == 'ppo' else reward_dataset_dpo,
        "generate_dataset": generate_final_dataset if args.method == 'ppo' else generate_final_dataset_dpo,
        "collect_results" : collect_results if args.method == 'ppo' else collect_dpo_results,
    }
    
    run_rlhf_recipe(args, rlhf_recipe, step_functions)


if __name__ == "__main__":
    main()
