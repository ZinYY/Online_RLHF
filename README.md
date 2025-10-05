# Online RLHF Pipeline: A Pytorch Implementation

A PyTorch implementation of the NeurIPS'25 paper "Provably Efficient Online RLHF with One-Pass Reward Modeling". This repository provides a flexible and modular approach to Online Reinforcement Learning from Human Feedback (Online RLHF).

This repository is forked from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) repo, and modified to implement our Online RLHF Pipeline.

## 📖 Reference & Cite

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{NeurIPS'25:OnlineRLHF,
    author = {Long-Fei Li and Yu-Yang Qian and Peng Zhao and Zhi-Hua Zhou},
    title = {Provably Efficient Online RLHF with One-Pass Reward Modeling},
    booktitle = {Advances in Neural Information Processing Systems 38 (NeurIPS)},
    year = {2025},
    pages = {to appear}
}
```

## 🌟 Cook Your Own Online-RLHF Recipe!

Think of Online-RLHF training as cooking - each step is an ingredient that contributes to the final dish. This implementation lets you mix and match different training components to create your perfect RLHF recipe!

### 🧑‍🍳 Recipe Format

```python
# A recipe is a list of ingredients (steps):
# 1. Single ingredient: A string representing one step (e.g., "SFT")
# 2. Ingredient group: A list of steps that cook together in one iteration

# Example Recipe for Online PPO:
rlhf_recipe = ["SFT"] + [["RM", "PPO", "generate", "evaluate", "rewarding", "generate_dataset", "collect_results"]] * stop_t
```

### 🥘 Available Ingredients

-   `"SFT"`: Supervised Fine-Tuning (base preparation)
-   `"RM"`: Reward Model Training (regular or HVP flavor)
-   `"PPO"`: Proximal Policy Optimization (main course)
-   `"DPO"`: Direct Preference Optimization (alternative main)
-   `"generate"`: Response Generation (plating)
-   `"evaluate"`: Response Evaluation (tasting)
-   `"rewarding"`: Reward Calculation (seasoning)
-   `"generate_dataset"`: Dataset Creation (meal prep)
-   `"collect_results"`: Results Collection (food review)

## 🚀 Quick Start

### Install the requirements:

```bash
pip install -r requirements.txt
```

### For the Passive Stage:

1. you should first run SFT to get the initial reward model's checkpoint,

```bash
export gpu_nodes="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$gpu_nodes
deepspeed --include=localhost:$gpu_nodes --master_port 27010 --module openrlhf.cli.train_sft \
    --max_len 2048 \
    --dataset HuggingFaceH4/ultrafeedback_binarized \
    --train_split train_sft \
    --input_key prompt \
    --output_key messages/-1/content \
    --train_batch_size 4
    --max_samples 50000 \
    --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
    --save_path ./checkpoint_ultraFB/llama3-8b-sft \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 3 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --learning_rate 5e-6 \
    --load_checkpoint \
    --gradient_checkpointing \
    --use_wandb XXXXX \
    --wandb_project openrlhf_sft_ultraFB
```

2. then run the passive training pipeline:

```bash
# Train Reward Model (SGD):
export gpu_nodes="0,1,2,3"
export train_batch_size=32
export CUDA_VISIBLE_DEVICES=$gpu_nodes
deepspeed --include=localhost:$gpu_nodes --master_port 27109 --module openrlhf.cli.train_rm_head \
 --save_path ./checkpoint_ultraFB/llama3-8b-rm \
 --save_steps -1 \
 --logging_steps 1 \
 --eval_steps 100 \
 --train_batch_size $train_batch_size \
 --micro_train_batch_size 8 \
 --max_len 1024 \
 --max_samples 30000 \
 --pretrain ./checkpoint_ultraFB/llama3-8b-sft \
 --bf16 \
 --max_epochs 1 \
 --zero_stage 0 \
 --learning_rate 1e-3 \
 --dataset HuggingFaceH4/ultrafeedback_binarized \
 --train_split train_prefs \
 --eval_split test_prefs \
 --apply_chat_template \
 --chosen_key chosen \
 --rejected_key rejected \
 --flash_attn \
 --packing_samples \
 --use_wandb XXXXX \
  --wandb_run_name llama3-8b-rm-head-SGD \
 --wandb_project openrlhf_RM_ultraFB_head


# Train Reward Model (Ours Method):
export gpu_nodes="0,1,2,3"
export train_batch_size=32
export CUDA_VISIBLE_DEVICES=$gpu_nodes
deepspeed --include=localhost:$gpu_nodes --master_port 29319 --module openrlhf.cli.train_rm_head_hvp \
 --save_path ./checkpoint_ultraFB/llama3-8b-rm \
 --save_steps -1 \
 --logging_steps 1 \
 --eval_steps 100 \
 --train_batch_size $train_batch_size \
 --micro_train_batch_size 8 \
 --max_len 1024 \
 --max_samples 30000 \
 --damping 0.8 \
 --damping_strategy linear \
 --damping_growth_rate 100 \
 --num_cg_steps 3 \
 --use_hvp \
 --pretrain ./checkpoint_ultraFB/llama3-8b-sft \
 --bf16 \
 --max_epochs 1 \
 --zero_stage 3 \
 --learning_rate 1e-3 \
 --dataset HuggingFaceH4/ultrafeedback_binarized \
 --train_split train_prefs \
 --eval_split test_prefs \
 --apply_chat_template \
 --chosen_key chosen \
 --rejected_key rejected \
 --flash_attn \
 --packing_samples \
 --use_wandb XXXXX \
 --wandb_project openrlhf_RM_ultraFB_head \
 --wandb_run_name llama3-8b-rm-head-hvp
```

### For the online RLHF with deployment-time adaptation:

1. Cook your favorite recipe style, then run:

```bash
# Cook with PPO + OMD (hvp)
python pipeline/Ultrafeedback/llama/online_deployment_ultrafeedback_llama.py \
    --method ppo \
    --rm_type hvp \
    --rm_strategy best_worst \
    --total_T 50 \
    --stop_T 20

```

## 📁 Code Structure

```
./
├── openrlhf/
│ ├── cli/ # Command line tools
│ ├── trainer/ # Training implementations
│ └── utils/ # Utility functions
├── pipeline/
│ └── Different scripts of RLHF pipeline
├── plot/ # Visualization tools
└── requirements.txt # Dependencies

```

## 🛠 Requirements

-   PyTorch
-   transformers==4.46.3
-   accelerate
-   bitsandbytes
-   peft
-   wandb
-   And more (see requirements.txt)

## 🎛 Configuration Options

-   `--method`: Choose between 'ppo' or 'dpo'
-   `--rm_type`: Select 'regular' or 'hvp' reward model
-   `--rm_strategy`: Pick from 'best_worst', 'best_second', 'best_quartile', 'random'
-   `--total_T`: Total number of training iterations
-   `--start_t`: Starting iteration
-   `--stop_t`: Stopping iteration
