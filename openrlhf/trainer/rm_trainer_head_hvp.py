from openrlhf.models import LogExpLoss, PairWiseLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
import os
from abc import ABC
import torch
from torch.optim import Optimizer
from tqdm import tqdm
import copy
from typing import List, Tuple, Optional
from torch import nn
import numpy as np
import deepspeed

'''
Only update the head layer of the reward model
'''


class RewardModelTrainer(ABC):
    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
        cg_max_steps: int = 3,
        cg_damping: float = 0.1, 
        damping_strategy: str = "", 
        damping_growth_rate: float = 5.0,  
        max_train_iter: int = -1,
        use_momentum: bool = False,
        momentum_beta1: float = 0.9,
        momentum_beta2: float = 0.999,
        momentum_eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.max_train_iter = max_train_iter
        
        
        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping  
        self.damping_strategy = damping_strategy 
        self.damping_growth_rate = damping_growth_rate  
        self.total_steps = 0  
        
        
        self.use_momentum = use_momentum
        self.momentum_beta1 = momentum_beta1
        self.momentum_beta2 = momentum_beta2
        self.momentum_eps = momentum_eps
        
        
        if self.use_momentum:
            self.m = {}  
            self.v = {}
            
            param_shapes = {}
            total_params = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param_shapes[name] = param.shape
                    total_params += param.numel()
            
            temp_direction = torch.zeros(total_params, device=next(model.parameters()).device)
            
            offset = 0
            for name, shape in param_shapes.items():
                param_size = np.prod(shape)
                self.m[name] = None
                self.v[name] = None
                offset += param_size
        
        if max_train_iter > 0:
            self.total_T = max_train_iter
        else:
            self.total_T = len(train_dataloader) * max_epochs
        
        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss")
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss")
        
        self.aux_loss = self.args.aux_loss_coef > 1e-8
        self.packing_samples = strategy.args.packing_samples
        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss
        
        
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb
            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)
        
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)
    
    def hessian_vector_product(self, params: List[nn.Parameter], loss: torch.Tensor,
                               flat_vector: torch.Tensor, flat_grad) -> torch.Tensor:
        
        # with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
        with torch.enable_grad():
            
            grad_vector_prod = torch.sum(flat_grad * flat_vector)
            
            
            hvp = torch.autograd.grad(grad_vector_prod, params, retain_graph=True)
            flat_hvp = torch.cat([g.reshape(-1).detach().clone() for g in hvp if g is not None])
        
        del flat_grad, grad_vector_prod, hvp
        for param in params:
            if param.grad is not None:
                param.grad = None
        torch.cuda.empty_cache()
        
        flat_hvp += self.cg_damping * flat_vector
        
        return flat_hvp.detach()
    
    def conjugate_gradient_solver(self, params: List[nn.Parameter], loss: torch.Tensor,
                                  flat_grad: torch.Tensor, max_iter=10, residual_tol=1e-10):
        # flat_grad = grad.view(-1)
        x = torch.zeros_like(flat_grad)  # initial guess
        # x = 1e-4 * torch.randn_like(flat_grad)  # initial guess
        r = flat_grad.detach().clone() - x  # residual
        p = r.detach().clone()  # search direction
        
        r_norm_sq = torch.sum(r * r)
        
        for i in range(max_iter):
            with torch.no_grad():
                Ap = self.hessian_vector_product(params, loss, p, flat_grad)
                
                alpha = r_norm_sq / (torch.sum(p * Ap) + 1e-8)
                x += alpha.detach() * p.detach()
                
                if i == max_iter - 1:
                    break
                
                r -= alpha * Ap
                r_norm_sq = torch.sum(r * r)
                
                beta = r_norm_sq / r_norm_sq
                
                if r_norm_sq < residual_tol:
                    break
                
                p = r + beta * p
                
                del Ap
                torch.cuda.empty_cache()
        
        if max_iter > 1:
            x = self.args.damping * flat_grad + (1.0 - self.args.damping) * x
        
        return x.detach()
    
    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")
        
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)
        
        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch",
                         disable=not self.strategy.is_rank_0())
        
        total_iter = 0
        
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )
            
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc=f"Train step of epoch {epoch}",
                disable=not self.strategy.is_rank_0()
            )
            
            self.model.train()
            acc_mean = 0
            loss_mean = 0
            for data in self.train_dataloader:
                total_iter += 1
                self.total_steps = total_iter
                if self.max_train_iter > 0 and total_iter > self.max_train_iter:
                    break
                
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, margin = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    
                    chosen_reward, reject_reward, aux_loss = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask
                    )
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, margin = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())
                    
                    chosen_reward, reject_reward, aux_loss = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens
                    )
                
                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                
                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()
                
                preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                preference_loss_item = preference_loss.item()
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0
                
                loss = preference_loss + aux_loss * self.args.aux_loss_coef
                del preference_loss, aux_loss
                
                if self.args.verbose:
                    print("1 loss", loss)
                
                params = []
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        params.append(param)
                
                for param in params:
                    if param.grad is not None:
                        param.grad = None
                
                if self.args.verbose:
                    print("2 params", len(params))
                
                # with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                with torch.enable_grad():
                    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
                
                if self.args.verbose:
                    print("3 flat_grad", grads[:10], grads.shape)
                
                current_damping = self.get_current_damping()
                if self.args.use_hvp and current_damping < 1.0:
                    update_direction = self.conjugate_gradient_solver(
                        params, loss, grads, max_iter=self.cg_max_steps
                    )
                    # clear memory:
                    del grads, loss
                    torch.cuda.empty_cache()
                    for param in params:
                        if param.grad is not None:
                            param.grad = None
                else:
                    update_direction = grads.view(-1)
                
                if self.args.verbose:
                    print("4 update_direction", update_direction.shape)
                    print("4.1 current_damping", current_damping)
                
                # Update model
                if self.args.use_optimizer:
                    with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                        offset = 0
                        for param in params:
                            param_size = param.numel()
                            param.grad = update_direction[offset:offset + param_size].reshape(param.shape)
                            offset += param_size
                    
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                else:
                    with torch.no_grad():
                        offset = 0
                        for name, param in self.model.named_parameters():
                            if not param.requires_grad:
                                continue
                            param_size = param.numel()
                            update = update_direction[offset:offset + param_size].reshape(param.shape)
                            
                            if self.use_momentum:
                                
                                if self.m[name] is not None:
                                    self.m[name] = self.momentum_beta1 * self.m[name] + (1 - self.momentum_beta1) * update
                                    self.v[name] = self.momentum_beta2 * self.v[name] + (1 - self.momentum_beta2) * (update ** 2)
                                else:
                                    self.m[name] = self.momentum_beta1 * update
                                    self.v[name] = self.momentum_beta2 * (update ** 2)
                                    
                                m_corrected = self.m[name] / (1 - self.momentum_beta1 ** (self.total_steps + 1))
                                v_corrected = self.v[name] / (1 - self.momentum_beta2 ** (self.total_steps + 1))


                                param.add_(
                                    -self.args.learning_rate * m_corrected / (torch.sqrt(v_corrected) + self.momentum_eps)
                                )
                            else:
                                param.add_(update, alpha=-self.args.learning_rate)
                            
                            offset += param_size
                
                if self.args.verbose:
                    print("5 update")
                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_mean = acc_mean * 0.9 + 0.1 * acc
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss_item
                if self.args.verbose:
                    print("6 acc", acc)
                # self.scheduler.step()
                
                logs_dict = {
                    "loss"         : preference_loss_item,
                    "acc"          : acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "loss_mean"    : loss_mean,
                    "acc_mean"     : acc_mean,
                    "lr"           : self.scheduler.get_last_lr()[0],
                    "current_damping": current_damping,
                }
                # if self.aux_loss:
                #     logs_dict["aux_loss"] = aux_loss.item()
                
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()
                
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)
                
                step += 1
                if step % 10 == 0:
                    torch.cuda.empty_cache()
            epoch_bar.update()
        
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()
    
    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
        
        # eval
        if global_step % args.eval_steps == 0:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )
    
    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards = []
            loss_sum = 0
            for data in eval_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, margin = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    
                    chosen_reward, reject_reward, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask
                    )
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, margin = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())
                    
                    chosen_reward, reject_reward, _ = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens
                    )
                
                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                
                loss = self.loss_fn(chosen_reward, reject_reward, margin)
                
                rewards += [chosen_reward.flatten(), reject_reward.flatten()]
                acc += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                step_bar.update()
            
            acc_mean = acc / self.eval_dataloader.__len__()
            loss_mean = loss_sum / self.eval_dataloader.__len__()
            
            rewards = torch.cat(rewards).float()
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)
            
            # save mean std
            self.strategy.print("Set reward mean std")
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()
            
            bar_dict = {
                "eval_loss"  : loss_mean,
                "acc_mean"   : acc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std" : reward_std.item(),
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)
            
            histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
            self.strategy.print("histgram")
            self.strategy.print(histgram)
            
            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state
    
    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0]:]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_rewards, rejected_rewards, aux_loss
    
    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        
        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )
        
        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks
    
    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens):
        all_values, output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        half_len = len(packed_seq_lens) // 2
        chosen_rewards = all_values[:half_len]
        rejected_rewards = all_values[half_len:]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        
        return chosen_rewards, rejected_rewards, aux_loss
    
    def get_current_damping(self) -> float:
        base_damping = self.cg_damping
        t = self.total_steps / self.total_T
        if self.damping_growth_rate > 1.0:
            t = self.total_steps / self.damping_growth_rate
        
        if self.damping_strategy == "constant":
            return 1.0 if t >= 1.0 else base_damping
        
        elif self.damping_strategy == "linear":
            return min((1.0 - base_damping) * t + base_damping, 1.0)
        
        elif self.damping_strategy == "log":
            return min(base_damping * (1 + np.log(1 + t)), 1.0)
        
        elif self.damping_strategy == "square":
            return min(base_damping * (1 + t ** 2), 1.0)
        
        elif self.damping_strategy == "cosine":
            return min(base_damping * (2 - np.cos(np.pi * t)), 1.0)
        
        else:
            return base_damping
