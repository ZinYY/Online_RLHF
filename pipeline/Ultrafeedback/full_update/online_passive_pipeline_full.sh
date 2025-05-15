export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx

# Train SFT:
#get GPUs:
gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)
if echo "$gpu_model" | grep -i "A800" > /dev/null; then
   # 如果显卡是 A800 (1, 3号卡坏了，屏蔽)
   gpu_nodes="0,2,3,4"
   train_batch_size=4
else
   gpu_nodes="0,1,2,3"
   train_batch_size=4
fi
export CUDA_VISIBLE_DEVICES=$gpu_nodes
deepspeed --include=localhost:$gpu_nodes --master_port 27010 --module openrlhf.cli.train_sft \
  --max_len 2048 \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --train_split train_sft \
  --input_key prompt \
  --output_key messages/-1/content \
  --train_batch_size $train_batch_size \
  --micro_train_batch_size 1 \
  --max_samples 50000 \
  --pretrain meta-llama/Llama-3.2-1B-Instruct \
  --save_path ./checkpoint_ultraFB/llama3.2-1b-sft \
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
  --use_wandb XXXXXXXXXXXXXXX \
  --wandb_project openrlhf_sft_ultraFB


# Train Reward Model (SGD-full):
#get GPUs:
gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)
if echo "$gpu_model" | grep -i "A800" > /dev/null; then
    # 如果显卡是 A800 (1, 3号卡坏了，屏蔽)
    gpu_nodes="0"
    train_batch_size=1
else
    gpu_nodes="0,1,2,3"
    train_batch_size=4
fi
export CUDA_VISIBLE_DEVICES=$gpu_nodes
deepspeed --include=localhost:$gpu_nodes --master_port 27109 --module openrlhf.cli.train_rm \
 --save_path ./checkpoint_ultraFB/llama3.2-1b-rm \
 --save_steps -1 \
 --logging_steps 1 \
 --eval_steps 100 \
 --train_batch_size $train_batch_size \
 --micro_train_batch_size 1 \
 --max_len 1024 \
 --max_samples 30000 \
 --pretrain ./checkpoint_ultraFB/llama3.2-1b-sft \
 --bf16 \
 --seed 8888 \
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
 --use_wandb XXXXXXXXXXXXXXX \
  --wandb_run_name llama3.2-1b-rm-full-SGD \
 --wandb_project openrlhf_RM_ultraFB_full


# Train Reward Model (hvp-full):
#get GPUs:
gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)
if echo "$gpu_model" | grep -i "A800" > /dev/null; then
    # 如果显卡是 A800 (1, 3号卡坏了，屏蔽)
    gpu_nodes="0,2,3,4"
    train_batch_size=4
else
    gpu_nodes="0,1,2,3"
    train_batch_size=4
fi
export CUDA_VISIBLE_DEVICES=$gpu_nodes
deepspeed --include=localhost:$gpu_nodes --master_port 29319 --module openrlhf.cli.train_rm_hvp \
 --save_path ./checkpoint_ultraFB/llama3.2-1b-rm \
 --save_steps -1 \
 --logging_steps 1 \
 --eval_steps 100 \
 --train_batch_size $train_batch_size \
 --micro_train_batch_size 1 \
 --max_len 1024 \
 --max_samples 30000 \
 --damping 0.5 \
 --damping_strategy linear \
 --damping_growth_rate 100000 \
 --num_cg_steps 3 \
 --use_hvp \
 --pretrain ./checkpoint_ultraFB/llama3.2-1b-sft \
 --bf16 \
 --max_epochs 1 \
 --zero_stage 0 \
 --learning_rate 1e-5 \
 --dataset HuggingFaceH4/ultrafeedback_binarized \
 --train_split train_prefs \
 --eval_split test_prefs \
 --apply_chat_template \
 --chosen_key chosen \
 --rejected_key rejected \
 --packing_samples \
 --use_wandb XXXXXXXXXXXXXXX \
 --wandb_project openrlhf_RM_ultraFB_full \
 --wandb_run_name llama3.2-1b-rm-full-hvp
