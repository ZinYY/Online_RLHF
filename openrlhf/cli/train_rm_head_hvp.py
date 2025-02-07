import argparse
import math
import os
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import RewardDataset
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.trainer.rm_trainer_head_hvp import RewardModelTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


# from huggingface_hub import login

# login('XXXXX')

def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()
    
    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        init_value_head=True,
        value_head_prefix=args.value_head_prefix,
        packing_samples=args.packing_samples,
    )
    
    # Freeze all parameters except the value head
    value_head_prefix = getattr(model, "value_head_prefix", "score")
    for name, param in model.named_parameters():
        if not name.startswith(value_head_prefix): 
            param.requires_grad = False 
            # strategy.print(f"Freezing parameter: {name}")
    
    # Print trainable parameters
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    strategy.print(f"\033[91mTrainable parameters: {trainable_params}\033[0m")
    
    # Add debug prints
    strategy.print("\033[91mAll parameter shapes:\033[0m")
    for name, param in model.named_parameters():
        strategy.print(f"{name}: {param.shape}, requires_grad: {param.requires_grad}")
    
    # Print value head specific info
    value_head = getattr(model, value_head_prefix, None)
    if value_head is not None:
        strategy.print(f"\033[91mValue head structure: {value_head}\033[0m")
    
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    
    strategy.print(model)
    
    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
    
    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        stopping_strategy="all_exhausted",
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    train_dataset = RewardDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
    )
    eval_dataset = RewardDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
    )
    
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        False,
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    )
    
    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    
    # scheduler
    # scheduler = get_scheduler(
    #     "cosine_with_min_lr",
    #     optim,
    #     num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
    #     num_training_steps=max_steps,
    #     scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    # )
    scheduler = get_scheduler(
        "constant",
        optim,
        num_warmup_steps=0,
        num_training_steps=max_steps,
    )
    
    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
    
    # strategy prepare
    # if not args.use_hvp:
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))
    # else:
    #     model = strategy.prepare((model))
    
    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = RewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        loss=args.loss,
        cg_damping=args.damping,
        cg_max_steps=args.num_cg_steps,
        damping_strategy=args.damping_strategy,
        damping_growth_rate=args.damping_growth_rate,
        max_train_iter=args.max_train_iter,
        use_momentum=args.use_momentum,
        momentum_beta1=args.momentum_beta1,
        momentum_beta2=args.momentum_beta2,
        momentum_eps=args.momentum_eps
    )
    
    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)
    
    # Save value_head_prefix
    strategy.print("Save value_head_prefix in config")
    unwrap_model = strategy._unwrap_model(model)
    unwrap_model.config.value_head_prefix = args.value_head_prefix
    
    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_rm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    
    # DeepSpeed
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    
    # Models
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--value_head_prefix", type=str, default="score")
    
    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
             "It should be a divisor of the number of heads. "
             "A larger value may results in faster training but will consume more memory.",
    )
    
    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    
    # RM training
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--loss", type=str, default="sigmoid")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    
    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)
    
    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=512)
    
    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_rm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )
    
    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")
    
    # HVP parameters
    parser.add_argument("--use_hvp", action="store_true", default=False, help="Use HVP update")
    parser.add_argument("--use_optimizer", action="store_true", default=False, help="Use Adam optimizer to update")
    parser.add_argument("--damping", type=float, default=0.8, help="Base damping coefficient for HVP")
    parser.add_argument("--damping_strategy", type=str, choices=["", "constant", "log", "linear", "square", "cosine"], default="linear",
                        help="Strategy for selecting damping coefficient")
    parser.add_argument("--damping_growth_rate", type=float, default=100.0,
                        help="Growth rate for damping strategies.")
    parser.add_argument("--num_cg_steps", type=int, default=3, help="Number of conjugate gradient steps")
    parser.add_argument("--momentum_beta1", type=float, default=0.9, help="Beta1 coefficient for momentum")
    parser.add_argument("--momentum_beta2", type=float, default=0.999, help="Beta2 coefficient for second moment")
    parser.add_argument("--momentum_eps", type=float, default=1e-8, help="Epsilon for Adam-style update")
    parser.add_argument("--use_momentum", action="store_true", default=False, help="Use momentum in updates")
    
    # Add this to the argument parser section
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")
    parser.add_argument("--max_train_iter", type=int, default=-1, help="Maximum number of training iterations")
    
    args = parser.parse_args()
    
    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None
    
    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )
    
    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True
    
    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"
    
    train(args)
