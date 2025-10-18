import os
import math
import time
import torch
import inspect
import tiktoken
import numpy as np
import torch.nn as nn
from datetime import datetime
import torch.distributed as dist
from datasets import load_dataset
from dataclasses import dataclass
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import save_model, load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# pip install tiktoken huggingface_hub safetensors

# torchrun --standalone --nproc_per_node=2 16-gpt-3-small-with-or-without-bias-and-gelu-relu-relu^2-silu-or-swiglu.py
# Note: torchrun sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE

###############################################
# 2 x H100 SXM | 56 vCPU 503 GB RAM | $5.38/h # 
###############################################

@dataclass
class GPTConfig:
    max_seq_len: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    up_proj_factor: int = 4
    pad_token_id: int = 50257
    norm_type: str = "rms" # "rms" or any other name for "layer"
    bias: bool = False
    pos_encoding_type: str = "rope" # "rope", "nope", or any other name for "absolute"
    rope_base_theta: int = 500_000
    fair: bool = True
    activation: str = "swiglu" # "gelu", "relu", "relu2" (relu^2), "silu" or "swiglu"

class Rotary(nn.Module):
    def __init__(self, head_size, base_theta, max_seq_len):
        super().__init__()
        assert head_size % 2 == 0, "RoPE needs even head_size"
        # head_size/2,
        # frequencies decay geometrically down to ~1/base_theta with ratio 1/base_theta^(2/head_size)
        inv_freq = 1.0 / (base_theta ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size))
        # register_buffer registers as a non-trainable buffer that moves/casts with the model (to device/dtype)
        # persistent=False avoids taking checkpoint space (won't be in state_dict)
        # buffers are broadcast to all ranks (broadcast_buffers=True), so this stays in sync
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        # seq_len, head_size/2
        freqs = torch.outer(t, self.inv_freq)
        # 1, 1, seq_len, head_size/2
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def get_cos_sin(self, seq_len, dtype, device):
        # slice to current length and cast on the fly to avoid
        # shape mutation of buffers
        cos = self.cos_cached[..., :seq_len, :].to(device=device, dtype=dtype)
        sin = self.sin_cached[..., :seq_len, :].to(device=device, dtype=dtype)
        return cos, sin

def apply_rotation(q, k, cos, sin):
    gpu_batch_size, n_heads, seq_len, head_size = q.shape

    # gpu_batch_size, n_heads, seq_len, head_size // 2, 2
    q_pairs = q.contiguous().reshape(gpu_batch_size, n_heads, seq_len, head_size // 2, 2)
    k_pairs = k.contiguous().reshape(gpu_batch_size, n_heads, seq_len, head_size // 2, 2)

    # gpu_batch_size, n_heads, seq_len, head_size/2 each
    q_x1, q_x2 = q_pairs[..., 0], q_pairs[..., 1]
    k_x1, k_x2 = k_pairs[..., 0], k_pairs[..., 1]

    # gpu_batch_size, n_heads, seq_len, head_size
    q_rot = torch.stack([q_x1 * cos - q_x2 * sin, q_x1 * sin + q_x2 * cos], dim=-1)
    k_rot = torch.stack([k_x1 * cos - k_x2 * sin, k_x1 * sin + k_x2 * cos], dim=-1)

    q = q_rot.reshape(gpu_batch_size, n_heads, seq_len, head_size)
    k = k_rot.reshape(gpu_batch_size, n_heads, seq_len, head_size)
    
    return q, k

class DataLoader:
    def __init__(self, gpu_batch_size, seq_len, ddp_world_size, ddp_rank, data_folder, split="train", 
                 epoch=0, current_shard_idx=0, grad_accum_mini_steps_per_shard_counter=0):
        # batch size each gpu can fit in
        self.gpu_batch_size = gpu_batch_size
        # seq len for each gpu
        self.seq_len = seq_len
        # total gpu count
        self.ddp_world_size = ddp_world_size
        # 'id' of gpu
        self.ddp_global_rank = ddp_rank
        self.split = split
        self.epoch = epoch

        shards = os.listdir(data_folder)
        shards = [shard for shard in shards if split in shard]
        sorted_shards = sorted(shards)
        sorted_shards = [os.path.join(data_folder, shard) for shard in sorted_shards]
        assert len(sorted_shards) > 0, f"no shards found for split {split}"
        if master_process:
            message = f"found {len(sorted_shards)} shards for split {split}"
            print(message)
            log_buffer.append(message)
        self.sorted_shards = sorted_shards
        self.current_shard_idx = current_shard_idx
        self.tokens = self.load_shard_tokens(sorted_shards[self.current_shard_idx])

        # there are ddp_world_size (e.g., 2) gpus,
        # each to get gpu_batch_size (e.g., 2) sequences,
        # each of seq_len (e.g., 1024) tokens
        # then for the current shard (loading the next shard upon token exhaustion):
        # we discard tokens that wouldn't fit into the final grad accumulation mini-step,
        # e.g., tokens in positions 12289-16000 if there were 16000 tokens total in 
        # the shard, keeping 0-12288 (12289 tokens, with 12288 fitting nicely into 3 
        # grad accum mini steps of 2 * 2 * 1024 = 4096 tokens each plus a final token
        # for y, since it is always one token ahead of x)
        self.grad_accum_tokens_per_mini_step = ddp_world_size * gpu_batch_size * seq_len
        # needing grad_accum_mini_steps_per_shard (e.g., 3) to exhaust tokens
        self.grad_accum_mini_steps_per_shard = len(self.tokens) // self.grad_accum_tokens_per_mini_step
        # e.g., take up to 3 * 4096 = 12288 and a final token since y is one ahead
        self.tokens = self.tokens[:self.grad_accum_tokens_per_mini_step * self.grad_accum_mini_steps_per_shard + 1]

        # precompute all possible x sequence start indices (stride = seq_len)
        # e.g., [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264]
        # to get tokens 0-1023, 1024-2047, 2048-3071, 3072-4095, 4096-5119, 5120-6143,
        # 6144-7167, 7168-8191, 8192-9215, 9216-10239, 10240-11263, 11264-12287
        self.x_seq_starts = list(range(0, len(self.tokens) - seq_len, seq_len))

        if master_process:
            self.show_shard_info()

        # shuffle x chunk starts, e.g.,
        # [0, 2048, 11264, 10240, 4096, 9216, 1024, 5120, 6144, 8192, 3072, 7168]
        self._shuffle_shard()

        # after this counter reaches self.grad_accum_mini_steps_per_shard, the next shard should replace self.tokens
        self.grad_accum_mini_steps_per_shard_counter = grad_accum_mini_steps_per_shard_counter
    
    def show_shard_info(self):
        messages = [
            f"total {self.split} shard {self.current_shard_idx} tokens: {len(self.tokens):,}",
            f"total gpus: {self.ddp_world_size}",
            f"gpu batch size: {self.gpu_batch_size:,} sequences",
            f"sequence length: {self.seq_len:,} tokens",
            f"tokens fed to gpu per grad accum mini-step: {(self.seq_len * self.gpu_batch_size):,} ({self.ddp_world_size:,} gpus, {self.grad_accum_tokens_per_mini_step:,} total tokens)",
            f"per-gpu grad accumulation mini-steps for {self.split} shard {self.current_shard_idx} (each mini-step processing {self.grad_accum_tokens_per_mini_step:,} tokens): {(len(self.tokens) // self.grad_accum_tokens_per_mini_step):,}"
        ]
        for message in messages:
            print(message)
            log_buffer.append(message)
    
    def _shuffle_shard(self):
        g = torch.Generator()
        g.manual_seed(self.current_shard_idx + self.epoch)
        # e.g., [0, 2048, 11264, 10240, 4096, 9216, 1024, 5120, 6144, 8192, 3072, 7168]
        shuffled_indices = torch.randperm(len(self.x_seq_starts), generator=g).tolist()
        self.x_seq_starts = [self.x_seq_starts[i] for i in shuffled_indices]
        # e.g., 12 * 0 / 2 = 0 for gpu0, 12 * 1 / 2 = 6 for gpu 1
        gpu_x_seq_start_idx = len(self.x_seq_starts) * self.ddp_global_rank // self.ddp_world_size
        # e.g., 0 + 12 / 2 = 6 for gpu0, 6 + 12 / 2 = 12 for gpu 1
        gpu_x_seq_end_idx = gpu_x_seq_start_idx + len(self.x_seq_starts) // self.ddp_world_size
        # e.g., [0, 2048, 11264, 10240, 4096, 9216] for gpu 0
        # [1024, 5120, 6144, 8192, 3072, 7168] for gpu 1
        self.gpu_x_seq_starts = self.x_seq_starts[gpu_x_seq_start_idx:gpu_x_seq_end_idx]

    def load_shard_tokens(self, shard_path):
        return torch.tensor(np.load(shard_path).astype(np.int32), dtype=torch.long)
    
    def reset(self):
        self.current_shard_idx = 0
        self.grad_accum_mini_steps_per_shard_counter = 0
        self.tokens = self.load_shard_tokens(self.sorted_shards[self.current_shard_idx])
        self._shuffle_shard()
    
    def next_batch(self):
        if self.grad_accum_mini_steps_per_shard_counter == self.grad_accum_mini_steps_per_shard:
            self.current_shard_idx += 1
            self.grad_accum_mini_steps_per_shard_counter = 0
            if self.current_shard_idx >= len(self.sorted_shards):
                self.current_shard_idx = 0
                self.epoch += 1
                if master_process:
                    message = f"--- starting epoch {self.epoch} ---"
                    print(message)
                    log_buffer.append(message)
                    self.show_shard_info()
            self.tokens = self.load_shard_tokens(self.sorted_shards[self.current_shard_idx])
            self._shuffle_shard()
        # e.g., 0 for the 1st grad accum mini-step, gpu_batch_size for the 2nd grad accum mini-step, etc.
        i = self.grad_accum_mini_steps_per_shard_counter * self.gpu_batch_size
        # e.g., for gpu 0: [0, 2048] for the 1st grad accum mini-step, [11264, 10240] for the 2nd grad accum mini-step, etc.
        #       for gpu 1: [1024, 5120] for the 1st grad accum mini-step, [6144, 8192] for the 2nd grad accum mini-step, etc.
        batch_starts = self.gpu_x_seq_starts[i:i + self.gpu_batch_size]
        # e.g., for gpu 0: [0-1023, 2048-3071] for the 1st grad accum mini-step
        #       for gpu 1: [1024-2047, 5120-6143] for the 1st grad accum mini-step
        x = torch.stack([self.tokens[start:start + self.seq_len] for start in batch_starts])
        y = torch.stack([self.tokens[start + 1:start + self.seq_len + 1] for start in batch_starts])
        # a full mini-step is to be done after this
        self.grad_accum_mini_steps_per_shard_counter += 1
        return x, y

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        assert config.d_model % config.n_heads == 0
        self.c_attn = nn.Linear(config.d_model, config.d_model * 3, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.rotary = Rotary(self.head_size, config.rope_base_theta, config.max_seq_len) if config.pos_encoding_type.lower() == "rope" else None

    def forward(self, x, attn_mask=None):
        gpu_batch_size, seq_len, d_model = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(gpu_batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(gpu_batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(gpu_batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        
        if self.rotary is not None:
            # apply RoPE
            cos, sin = self.rotary.get_cos_sin(seq_len, q.dtype, q.device)
            q, k = apply_rotation(q, k, cos, sin)

        # if a mask is passed
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            # if (gpu_batch_size, seq_len)
            if attn_mask.dim() == 2:
                # insert two dimensions to reach shape (gpu_batch_size, 1, seq_len, 1)
                q_valid = attn_mask[:, None, :, None]
                # insert two dimensions to reach shape (gpu_batch_size, 1, 1, seq_len)
                k_valid = attn_mask[:, None, None, :]
                allow_mask_4d = q_valid & k_valid
            # if (gpu_batch_size, seq_len, seq_len)
            elif attn_mask.dim() == 3:
                # insert one dimension to reach shape (gpu_batch_size, 1, seq_len, seq_len)
                allow_mask_4d = attn_mask[:, None, :, :]
            # else: keep as [gpu_batch_size, 1, seq_len, seq_len] or [gpu_batch_size, n_heads, seq_len, seq_len]
            elif attn_mask.dim() == 4:
                allow_mask_4d = attn_mask
            else:
                raise ValueError(f"attn_mask must be 2D/3D/4D, got shape {attn_mask.shape}")
            
            # upper triangular is set to zeros
            causal = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            # insert two dimensions to reach shape (1, 1, seq_len, seq_len)
            causal = causal[None, None, :, :]

            # combine masks, with the causal mask being broadcasted in the first dimension
            # to result in (gpu_batch_size, 1, seq_len, seq_len) -unless dim=4 is passed as
            # (gpu_batch_size, n_heads, seq_len, seq_len) in which case 2 broadcasts apply-
            allow_mask_4d = allow_mask_4d & causal
            # as per scaled_dot_product_attention docs: sent mask is negated then filled with -infinity where True
            # (after negation, i.e., what is False in our attn_mask is turned into -infinity)
            # attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            # so our attn_mask True's are kept, False's are disregarded, i.e., we need to buil a ** keep mask **
            attn_mask = allow_mask_4d

            is_causal = False

            # safety check
            assert attn_mask.shape[0] == gpu_batch_size and \
            (attn_mask.shape[1] == 1 or attn_mask.shape[1] == self.n_heads) and \
            attn_mask.shape[2:] == (seq_len, seq_len), \
                f"attn_mask shape mismatch: got {attn_mask.shape}, expected ({gpu_batch_size}, 1 or {self.n_heads}, {seq_len}, {seq_len})"
        # if no attention mask, use the default built-in mask (when is_causal=True)
        else:
            is_causal = True

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        y = y.transpose(1,2).contiguous().view(gpu_batch_size, seq_len, d_model)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_name = config.activation.lower()

        if self.activation_name == "swiglu":
            # without gating:
            # the up projection is (d_model, up_proj_factor * d_model) with -if used- up_proj_factor * d_model biases
            # the down projection is (up_proj_factor * d_model, d_model) with -if used- d_model biases
            # with gating:
            # the first matrix needs to start as d_model -i.e., (d_model, X)- 
            # and the last matrix end as d_model -i.e., (X, d_model)
            # we also need to match X (dimension) in both matrices for @
            # and split the output of the first matrix in X/2 -i.e., (d_model, X/2)- for gating
            # so the 2 up_proj_factor * d_model dimensions without gating need to be split 
            # for gating into 3 of size 2 / 3 * up_proj_factor * d_model, resulting in:
            # 2 up projection matrices of size (d_model, 2 / 3 * up_proj_factor * d_model) each, with -if used- 2 * 2 / 3 * up_proj_factor * d_model total biases
            # 1 down projection of size (2 / 3 * up_proj_factor * d_model, d_model) with -if used- d_model biases
            # the total number of weight parameters matches, but, if used, biases -slightly- differ as
            # up_proj_factor * d_model + d_model != 2 * 2 / 3 * up_proj_factor * d_model + d_model
            self.hidden_dim = int(round((2.0/3.0) * config.up_proj_factor * config.d_model)) if config.fair else config.up_proj_factor * config.d_model
            self.c_fc = nn.Linear(config.d_model, self.hidden_dim * 2, bias=config.bias)
            self.c_proj = nn.Linear(self.hidden_dim, config.d_model, bias=config.bias)
        else:
            self.hidden_dim = config.up_proj_factor * config.d_model
            self.c_fc = nn.Linear(config.d_model, self.hidden_dim, bias=config.bias)
            self.c_proj = nn.Linear(self.hidden_dim, config.d_model, bias=config.bias)

    def forward(self, x):
        # Gaussian Error Linear Unit
        if self.activation_name == "gelu":
            x = F.gelu(self.c_fc(x))
        # Rectified Linear Unit
        elif self.activation_name == "relu":
            x = F.relu(self.c_fc(x))
        # Rectified Linear Unit^2
        elif self.activation_name == "relu2":
            x = F.relu(self.c_fc(x)) ** 2
        # Sigmoid Linear Unit or Swish
        elif self.activation_name == "silu":
            x = F.silu(self.c_fc(x))
        # Swish Gated Linear Unit
        elif self.activation_name == "swiglu":
            # split the tensor in two along the last dimension
            x1, x2 = self.c_fc(x).chunk(2, dim=-1)
            x = F.silu(x1) * x2
        else:
            raise ValueError(f"unsupported activation function: {self.activation_name}")
        return self.c_proj(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.residual_scale = (2 * config.n_layers) ** -0.5

        if config.norm_type.lower() == "rms":
            self.ln_1 = nn.RMSNorm(config.d_model)
        else:
            self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        if config.norm_type.lower() == "rms":
            self.ln_2 = nn.RMSNorm(config.d_model)
        else:
            self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None):
        x = x + self.residual_scale * self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.residual_scale * self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.pad_token_id = config.pad_token_id
        self.pos_encoding_type = config.pos_encoding_type.lower()

        if self.pos_encoding_type == "rope" or self.pos_encoding_type == "nope":
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.d_model),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f = nn.RMSNorm(config.d_model) if config.norm_type.lower() == "rms" else nn.LayerNorm(config.d_model),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.max_seq_len, config.d_model),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.RMSNorm(config.d_model) if config.norm_type.lower() == "rms" else nn.LayerNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, lr, betas, eps, weight_decay, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            messages = [
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters",
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            ]
            for message in messages:
                print(message)
                log_buffer.append(message)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device_type
        if master_process:
            message = f"using fused AdamW: {use_fused}"
            print(message)
            log_buffer.append(message)
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=eps, fused=use_fused)
        return optimizer

    def forward(self, indices, targets=None, attn_mask=None):
        _, seq_len = indices.size()
        te = self.transformer.wte(indices)

        # if RoPE or NoPE, rotate later in attention or do nothing, respectively
        if self.pos_encoding_type == "rope" or self.pos_encoding_type == "nope":
            x = te
        # else, apply absolute position encoding
        else:
            pe = self.transformer.wpe(torch.arange(0, seq_len, dtype=torch.long, device=device))
            x = te + pe
        
        for block in self.transformer.h:
            x = block(x, attn_mask=attn_mask)
        x = self.transformer.ln_f(x)

        # (gpu_batch_size, seq_len, vocab_size)
        if torch.isnan(x).any():
            pass
        logits = self.lm_head(x)
            
        if targets is not None:
            loss_mask = (targets != self.pad_token_id)
            # logits are viewed as gpu_batch_size * seq_len, vocab_size
            # targets as gpu_batch_size * seq_len, as only 1 out of vocab_size tokens is the correct one
            # do not reduce to mean loss as some losses (from masked tokens) are not to be used
            # making the loss be of size (gpu_batch_size * seq_len) compared to a scalar if reduced
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            # instead, get the number of unmasked (non-False) tokens
            n_non_masked_tokens = loss_mask.sum()
            # multiply unreduced loss by the mask to ingore losses coming from masked tokens and sum them
            sum_non_masked_loss_tokens = (loss * loss_mask.view(-1)).sum()
            # then manually reduce to the mean loss
            loss = sum_non_masked_loss_tokens / n_non_masked_tokens
        else:
            loss = None

        return logits, loss

def get_sample_token_count(step, base=5, step_interval=1000, max_tokens=50):
    return min(base + (step // step_interval) * base, max_tokens)

def get_lr(step):
    if step < warmup_steps:
        return (step + 1) * max_lr / warmup_steps
    elif step >= warmup_and_cosine_steps:
        return min_lr_after_warmup
    else:
        coeff = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (warmup_and_cosine_steps - warmup_steps)))
        return min_lr_after_warmup + coeff * (max_lr - min_lr_after_warmup)

def sample(sample_sequences, max_new_tokens=5, temperature=1.0, top_k=None, top_p=None):
    gpt_model.eval()
    with torch.no_grad():
        # convert all prompts to ids,
        # e.g., 'AGI is' to [4760, 40, 318]
        # 'AGI is not' to [4760, 40, 318, 407]
        # and truncate them leaving max_new_tokens for generation
        max_allowed_input_len = raw_gpt_model.max_seq_len - max_new_tokens
        initial_input_ids_list = [
            tokenizer.encode(sequence)[:max_allowed_input_len] 
            for sequence in sample_sequences
        ]

        # find the max input length (in ids) in this batch
        max_input_len = max(len(ids) for ids in initial_input_ids_list)

        # get a length cap to pre-allocate the tensor while not needing to go to max_seq_len
        alloc_len = max_input_len + max_new_tokens

        # rounding up to a nice multiple for better tensor cores / GPU efficiency
        round_multiple = 8
        alloc_len = math.ceil(alloc_len / round_multiple) * round_multiple
        alloc_len = min(alloc_len, raw_gpt_model.max_seq_len)

        # pre-allocate tensor of size len(sample_sequences), alloc_len and fill with padding 
        # to significanly boost performance (versus a new tensor size every generation)!
        generated_sequences = torch.full(
            (len(sample_sequences), alloc_len),
            raw_gpt_model.pad_token_id,
            dtype=torch.long,
            device=device
        )

        # then replace the first padding tokens of each pre-allocated sequence with their original tokens
        for i, seq_ids in enumerate(initial_input_ids_list):
            generated_sequences[i, :len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long, device=device)

        # track actual (non-padding) sequence lengths
        actual_sequence_lengths = torch.tensor([len(seq) for seq in initial_input_ids_list], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # then, for each new token to generate, update the mask by effectively comparing if 0, ..., alloc_len < the non-padded length for each sequence
            # using unsqueeze(0) to add a new dimension of size 1 at the beginning to give size (1, alloc_len)
            # and unsqueeze(1) to add a new dimension of size 1 at the end to give size (len(sample_sequences), 1)
            attn_mask = torch.arange(alloc_len, device=device).unsqueeze(0) < actual_sequence_lengths.unsqueeze(1)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # and predict, resulting in (len(sample_sequences), seq_len, vocab_size)
                logits, _ = gpt_model(generated_sequences, attn_mask=attn_mask)
            
            # of which we take the vocab_size values for each sequence's continuation to the last non-pad token,
            # resulting in len(sample_sequences), vocab_size
            last_logits = logits[torch.arange(len(sample_sequences), device=device), actual_sequence_lengths - 1, :]

            if temperature == 0.0:
                # then if temperature is 0, we cannot divide by 0, so take the max logit from vocab_size
                next_token_ids = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                # else, divide by the temperature
                temp_adjusted_logits = last_logits / temperature

                # apply top_k filtering
                if top_k is not None and top_k > 0:
                    # topk returns the k largest elements of the given input tensor along a given dimension
                    # resulting in len(sample_sequences), vocab_size
                    sorted_values, _ = torch.topk(temp_adjusted_logits, min(top_k, temp_adjusted_logits.size(-1)), dim=-1, largest=True, sorted=True)
                    # get the k-th largest logit of each sequence (i.e., last in sorted_values)
                    # and add a new dimension of size 1 at the end to give size (len(sample_sequences), 1)
                    sequences_k_th_logit = sorted_values[:, -1].unsqueeze(1)
                    # mask out everything less than the top_k logit
                    temp_adjusted_logits[temp_adjusted_logits < sequences_k_th_logit] = -float('Inf')

                # apply top_p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    # apply softmax to get probabilities for the continuation to the last non-pad token
                    # resulting in (still) len(sample_sequences), vocab_size
                    probs = F.softmax(temp_adjusted_logits, dim=-1)
                    # sort probabilities in descending order
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                    # obtain the cumulative sums of the probabilities sorted in descending order
                    # e.g., cumulative probs of [0.3, 0.6, 0.8, 1.0] for [0.3, 0.3, 0.2, 0.2]
                    # with size len(sample_sequences), vocab_size
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    if master_process:
                        message = "first 2 sequences cumulative probs:"
                        print(message)
                        log_buffer.append(message)
                        for i in range(min(2, cumulative_probs.size(0))):
                            values = cumulative_probs[i, :10]
                            message = f"\tseq {i}: {values.tolist()}"
                            print(message)
                            log_buffer.append(message)
                    # obtain the indices of logits to remove,
                    # initially excluding the logit that makes the cumulative sum match top_p,
                    # as we then shift to the right one position
                    # cumulative_probs [0.3, 0.6, 0.8, 1.0], top_p 0.75 would thus give (for a single sequence):
                    # [False, False, True, True] which shifted to the right is [False, False, False, True]
                    # while cumulative_probs [0.3, 0.6, 0.8, 1.0], top_p 0.8:
                    # [False, False, True, True] which shifted to the right is [False, False, False, True],
                    # being in both instances the smallest possible set that has at least top_p cumulative probability
                    sorted_indices_to_remove = cumulative_probs >= top_p
                    # shift the indices to the right one position
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    # replacing the last index value, now in the first position, to False
                    sorted_indices_to_remove[..., 0] = False
                    mask = sorted_indices_to_remove.to(torch.bool)

                    scatter_mask = torch.zeros_like(temp_adjusted_logits, dtype=torch.bool)
                    scatter_mask = scatter_mask.scatter(dim=-1, index=sorted_indices, src=mask)
                    if master_process:
                        for i in range(min(2, scatter_mask.size(0))):
                            masked_count = scatter_mask[i].sum().item()
                            total_count = scatter_mask.size(1)
                            #print(f"sequence {i} mask: {masked_count}/{total_count} logits masked")
                            if masked_count == total_count:
                                message = f"all logits masked for sequence {i}!"
                                print(message)
                                log_buffer.append(message)
                    temp_adjusted_logits = temp_adjusted_logits.masked_fill(scatter_mask, -float('inf'))
                
                probs = F.softmax(temp_adjusted_logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)

            # replace the 'actual_sequence_length' position of each sequence with the selected token
            generated_sequences[torch.arange(len(sample_sequences), device=device), actual_sequence_lengths] = next_token_ids[:, 0]
            # increase the actual, non-padded sequence endings
            actual_sequence_lengths += 1

        # and decode, ignoring above 50256
        local_messages = []
        for i in range(len(sample_sequences)):
            decoded = tokenizer.decode([
                token for token in generated_sequences[i, :actual_sequence_lengths[i]].tolist()
                if token < raw_gpt_model.pad_token_id])
            message = f"[rank {ddp_rank}] seq {i} >>> {decoded}"
            print(message)
            local_messages.append(message)

        all_messages = [None for _ in range(ddp_world_size)]
        dist.all_gather_object(all_messages, local_messages)

        if master_process:
            for rank_messages in all_messages:
                log_buffer.extend(rank_messages)
        dist.barrier()
    gpt_model.train()

# ---------------------------------------------------------
# Normalizing token probabilities to account for sequence length and comparing continuations A-D has the issue of an 
# easy continuation achieving high scores because once the answer deviates, the continuation is easy to guess, e.g.,

# (extreme example)
# She is cooking dinner,
# A) but she would rather be dancing in LA with the bride's friends tonight -> hard to guess dancing, LA, bride's
# B) one two three four five six -> easy to guess continuation after one two
# Here, it would not matter B makes less sense than A, because once 'one two' are told to be the continuation
# to the model, it has no way to be surprised by 'three four five six'
# ---------------------------------------------------------

# ---------------------------------------------------------
# From Language Models are Few-Shot Learners
# GPT-3 achieves 78.1% accuracy in the one-shot setting and 79.3% accuracy in the few-shot setting, outperforming the
# 75.4% accuracy of a fine-tuned 1.5B parameter language model [ZHR+19] but still a fair amount lower than the overall
# SOTA of 85.6% achieved by the fine-tuned multi-task model ALUM.

# Zero-shot
# Small | Med  | Large | XL   | 2.7B  | 6.7B  | 13B  | 175B
# 33.7  | 43.6 |  51.0 | 54.7 |  62.8 |  67.4 | 70.9 | 78.9

# One-shot (we will evaluate one-shot, but with Answer: A, Answer: B, Answer: C, Answer: D token probabilities)
# Small | Med  | Large | XL   | 2.7B  | 6.7B  | 13B  | 175B
# 33.0  | 42.9 | 50.5  | 53.5 | 61.9  | 66.5  | 70.0 | 78.1

# ...
# ---------------------------------------------------------

# ---------------------------------------------------------
# The A, B, C, D answers are uniformly distributed:

# HellaSwag train split answer label distribution:
#   A: 9986 (25.02%)
#   B: 10031 (25.14%)
#   C: 9867 (24.73%)
#   D: 10021 (25.11%)

# HellaSwag validation split answer label distribution:
#   A: 2515 (25.04%)
#   B: 2485 (24.75%)
#   C: 2584 (25.73%)
#   D: 2458 (24.48%)

# which make even a model biased towards some token, e.g., A,
# as good as a random-guesser model
# ---------------------------------------------------------

# ---------------------------------------------------------
# The fact they are equi-probable allows us to evaluate performance by doing:

# batch sequence 1 passed to the model

# Sentence to pick the continuation of: A little girl in a room standing in front of some chairs is hitting a dora pinata. She hits it a few times and then its someone else's turn. an older girl
# Continuation options:
# A) with a pink apron including a brown jacket tries to stop her but she hits it again.
# B) comes into the room and gives her and the girl laughs while they continue to hit the pinata.
# C) is also there in the room playing and before long some other little girl is swinging her away and she hasn't seen anyone.
# D) hits it so hard that it doesn't break but the whole pinata falls down and the adults have to put it back up.
# Correct continuation: D)

# Sentence to pick the continuation of: She is cooking dinner,
# Continuation options:
# A) but she would rather be dancing in LA with the bride's friends tonight
# B) one two three four five six
# Correct continuation: A)

# batch sequence 2 passed to the model

# ... (one-shot example)

# Sentence to pick the continuation of: She is cooking dinner,
# Continuation options:
# A) but she would rather be dancing in LA with the bride's friends tonight
# B) one two three four five six
# Correct continuation: B)

# to evaluate the likelihood the model assigns to each continuation
# ---------------------------------------------------------

# ---------------------------------------------------------
# For efficiency, we could score the log-probability of:
# " a" (for A)
# " b" (for B)
# " c" (for C)
# " d" (for D)

# but we'd rely on these tokens existing on the tokenizer (could break if we change our tokenizer)
# plus why not add " a", " A", " first",  etc. -if existent-,, eventually deciding to pass 4 batch sequences
# with the continuations included, one per batch sequence, even if more
# wasteful compared to passing a single sequence per HellaSwag sentence
# ---------------------------------------------------------

# ---------------------------------------------------------
# For length-normalization (to avoid penalizing long sequences):
# In prob space, the product of probabilities needs to be raised to 1/len(tokens) -i.e., the geometric mean-,
# while in log-prob space the sum of log-probs needs to be divided by len(tokens),
# e.g., given seq_A with 1 token, seq_B with 3:

# P(seq_A)           = 0.5 (favored)                  | P(seq_B)           = 0.5 * 0.5 * 0.5 = 0.125
# norm(P(seq_A))     = (0.5) ^ 1/1 = 0.5 (fair)       | norm(P(seq_B))     = (0.5 * 0.5 * 0.5) ^ (1/3) = 0.5

# log-P(seq_A)       = −0.69314 (favored)             | log-P(seq_B)       = -2.07944
# norm(log-P(seq_A)) = −0.69314 / 1 = −0.69314 (fair) | norm(log-P(seq_B)) = (−0.69314 + −0.69314 + −0.69314) / 3 = −0.69314
# ---------------------------------------------------------

# ---------------------------------------------------------
# If using hard-targets (only one token is correct, all others are incorrect):

# average loss = (1/sequence length) * sum(-log(probability of correct token at each position))
# likelihood = product of probabilities
# normalized likelihood = product of probabilities ^ (1/sequence length)
# average log-likelihood = log ((product of probabilities) ^ (1/sequence length)) = (1/sequence length) * sum(log-probabilities)
# average negative log-likelihood (nll) = - (average log-likelihood)
# perplexity (ppl) = e ^ (average negative log-likelihood)
# log-ppl = average loss = average nll
# ppl = e ^ (log-ppl) = e ^ (average loss) = e ^ (nll)

# F.cross_entropy() calculates - F.log_softmax(), and then either it reduces to mean or we do so manually
# ---------------------------------------------------------

def evaluate_hellaswag_one_shot(hellaswag_examples_per_batch=16):
    hellaswag_examples_per_batch = max(1, hellaswag_examples_per_batch)
    max_gpu_sequences_per_batch = hellaswag_examples_per_batch * 4

    gpt_model.eval()
    total_correct_predictions = 0
    total_dataset_examples = len(local_hellaswag_val_dataset)
    processed_examples_counter = 0

    one_shot_example = hellaswag_train_dataset[0]
    one_shot_context = one_shot_example['ctx']
    one_shot_endings = one_shot_example['endings']
    one_shot_correct_label_char = chr(ord('A') + int(one_shot_example['label']))

    max_model_seq_len = raw_gpt_model.max_seq_len 
    
    pre_allocated_input_ids = torch.full(
        (max_gpu_sequences_per_batch, max_model_seq_len), 
        raw_gpt_model.pad_token_id, 
        dtype=torch.long, 
        device=device
    )
    pre_allocated_attn_mask = torch.zeros(
        (max_gpu_sequences_per_batch, max_model_seq_len), 
        dtype=torch.bool, 
        device=device
    )

    with torch.no_grad():
        num_example_batches = (total_dataset_examples + hellaswag_examples_per_batch - 1) // hellaswag_examples_per_batch

        for i in range(num_example_batches):
            batch_slice_start = i * hellaswag_examples_per_batch
            batch_slice_stop = min((i + 1) * hellaswag_examples_per_batch, total_dataset_examples)
            current_examples = local_hellaswag_val_dataset.select(range(batch_slice_start, batch_slice_stop))
            # if i == 0:
            #     for rank in range(ddp_world_size):
            #         dist.barrier()
            #         if ddp_rank == rank:
            #             print(f"[rank {ddp_rank}] first HellaSwag batch:")
            #             for j, ex in enumerate(current_examples):
            #                 print(f"[rank {ddp_rank}] example {j}:")
            #                 print(f"ctx: {ex['ctx']}")
            #                 print(f"endings:")
            #                 for k, ending in enumerate(ex['endings']):
            #                     print(f"\t{chr(ord('A') + k)}) {ending}")
            #                 print(f"label: {ex['label']}")
            #                 print("-" * 60)
            #         dist.barrier()

            all_seq_ids = []
            orig_example_indices = []
            prefix_lengths = []

            for ex_idx_in_batch, example in enumerate(current_examples):
                context = example['ctx']
                continuations = [example['endings'][j] for j in range(4)]
                
                common_prefix_text_base = (
                    f"Sentence to pick the continuation of: {one_shot_context}\n"
                    "Continuation options:\n"
                    f"A) {one_shot_endings[0]}\n"
                    f"B) {one_shot_endings[1]}\n"
                    f"C) {one_shot_endings[2]}\n"
                    f"D) {one_shot_endings[3]}\n"
                    f"Correct continuation: {one_shot_correct_label_char})\n\n"
                    f"Sentence to pick the continuation of: {context}\n"
                    "Continuation options:\n"
                    f"A) {continuations[0]}\n"
                    f"B) {continuations[1]}\n"
                    f"C) {continuations[2]}\n"
                    f"D) {continuations[3]}\n"
                    "Correct continuation: "
                )
                prefix_ids = tokenizer.encode(common_prefix_text_base)
                
                for j in range(4):
                    final_token_text = f"{chr(ord('A') + j)})"
                    final_token_ids = tokenizer.encode(final_token_text)
                    combined_seq_ids = prefix_ids + final_token_ids
                    prefix_lengths.append(len(prefix_ids))
                    
                    if len(combined_seq_ids) > max_model_seq_len:
                        raise ValueError(
                            f"sequence for HellaSwag example {batch_slice_start + ex_idx_in_batch} is too long ({len(combined_seq_ids)} > {max_model_seq_len})"
                        )
                    
                    all_seq_ids.append(combined_seq_ids)
                    orig_example_indices.append(ex_idx_in_batch)

            pre_allocated_input_ids.fill_(raw_gpt_model.pad_token_id)
            pre_allocated_attn_mask.zero_()

            for idx, seq_ids in enumerate(all_seq_ids):
                pre_allocated_input_ids[idx, :len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long, device=device)
                pre_allocated_attn_mask[idx, :len(seq_ids)] = True

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = gpt_model(pre_allocated_input_ids, attn_mask=pre_allocated_attn_mask)
            
            # log softmax: softmax followed by log (to sum log-probs vs. multiply low-value probs)
            log_probs = F.log_softmax(logits, dim=-1)

            example_scores = [[] for _ in range(len(current_examples))]

            for seq_idx in range(len(all_seq_ids)):
                original_ex_idx = orig_example_indices[seq_idx]
                
                true_len = pre_allocated_attn_mask[seq_idx].sum().item()
                prefix_len = prefix_lengths[seq_idx]
                
                seq_log_prob = 0.0
                num_tokens = 0

                for token_idx in range(prefix_len, true_len):
                    target_token_id = pre_allocated_input_ids[seq_idx, token_idx].item()
                    seq_log_prob += log_probs[seq_idx, token_idx - 1, target_token_id].item()
                    num_tokens += 1

                normalized_log_prob = seq_log_prob / num_tokens if num_tokens > 0 else 0.0

                example_scores[original_ex_idx].append(normalized_log_prob)

            for ex_idx, scores in enumerate(example_scores):
                correct_label_idx = int(current_examples[ex_idx]['label'])
                predicted_label_idx = max(enumerate(scores), key=lambda x: x[1])[0]

                if predicted_label_idx == correct_label_idx:
                    total_correct_predictions += 1
                
                processed_examples_counter += 1

                if processed_examples_counter % 200 == 0:
                    for rank in range(ddp_world_size):
                        dist.barrier()
                        if ddp_rank == rank:
                            print(f"[rank {ddp_rank}] processed {processed_examples_counter} examples: acc: {total_correct_predictions / processed_examples_counter:.4f}")
                    dist.barrier()

    local_accuracy = total_correct_predictions / total_dataset_examples
    local_message = f"[rank {ddp_rank}] HellaSwag local acc: {total_correct_predictions} / {total_dataset_examples} = {local_accuracy:.4f}"
    print(local_message)
    all_messages = [None for _ in range(ddp_world_size)]
    dist.all_gather_object(all_messages, local_message)
    if master_process:
        log_buffer.extend(all_messages)
    dist.barrier()

    # convert local totals to tensors for sum reduction across ranks
    correct_tensor = torch.tensor(total_correct_predictions, dtype=torch.long, device=device)
    count_tensor = torch.tensor(total_dataset_examples, dtype=torch.long, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    accuracy = correct_tensor.item() / count_tensor.item()
    gpt_model.train()
    return accuracy

def evaluate_hellaswag_standard(hellaswag_examples_per_batch=16):
    hellaswag_examples_per_batch = max(1, hellaswag_examples_per_batch)
    max_gpu_sequences_per_batch = hellaswag_examples_per_batch * 4

    gpt_model.eval()
    total_correct_predictions = 0
    total_dataset_examples = len(local_hellaswag_val_dataset)
    processed_examples_counter = 0

    one_shot_example = hellaswag_train_dataset[0]
    one_shot_context = one_shot_example['ctx']
    one_shot_endings = one_shot_example['endings']
    one_shot_correct_label_char = chr(ord('A') + int(one_shot_example['label']))

    max_model_seq_len = raw_gpt_model.max_seq_len 
    
    pre_allocated_input_ids = torch.full(
        (max_gpu_sequences_per_batch, max_model_seq_len), 
        raw_gpt_model.pad_token_id, 
        dtype=torch.long, 
        device=device
    )
    pre_allocated_attn_mask = torch.zeros(
        (max_gpu_sequences_per_batch, max_model_seq_len), 
        dtype=torch.bool, 
        device=device
    )

    with torch.no_grad():
        num_example_batches = (total_dataset_examples + hellaswag_examples_per_batch - 1) // hellaswag_examples_per_batch

        for i in range(num_example_batches):
            batch_slice_start = i * hellaswag_examples_per_batch
            batch_slice_stop = min((i + 1) * hellaswag_examples_per_batch, total_dataset_examples)
            current_examples = local_hellaswag_val_dataset.select(range(batch_slice_start, batch_slice_stop))

            all_seq_ids = []
            orig_example_indices = []
            prefix_lengths = []

            for ex_idx_in_batch, example in enumerate(current_examples):
                context = example['ctx']
                continuations = [example['endings'][j] for j in range(4)]
                
                common_context_text = (
                    f"Sentence to pick the continuation of: {one_shot_context}\n"
                    "Continuation options:\n"
                    f"A) {one_shot_endings[0]}\n"
                    f"B) {one_shot_endings[1]}\n"
                    f"C) {one_shot_endings[2]}\n"
                    f"D) {one_shot_endings[3]}\n"
                    f"Correct continuation: {one_shot_correct_label_char})\n\n"
                    f"Sentence to pick the continuation of: {context}\n"
                )
                
                context_ids = tokenizer.encode(common_context_text)
                
                for j in range(4):
                    continuation_text = f" {continuations[j]}"
                    continuation_ids = tokenizer.encode(continuation_text)
                    
                    combined_seq_ids = context_ids + continuation_ids
                    
                    if len(combined_seq_ids) > max_model_seq_len:
                        raise ValueError(
                            f"sequence for HellaSwag example {batch_slice_start + ex_idx_in_batch} is too long ({len(combined_seq_ids)} > {max_model_seq_len})"
                        )
                    
                    all_seq_ids.append(combined_seq_ids)
                    orig_example_indices.append(ex_idx_in_batch)
                    prefix_lengths.append(len(context_ids))

            pre_allocated_input_ids.fill_(raw_gpt_model.pad_token_id)
            pre_allocated_attn_mask.zero_()

            for idx, seq_ids in enumerate(all_seq_ids):
                pre_allocated_input_ids[idx, :len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long, device=device)
                pre_allocated_attn_mask[idx, :len(seq_ids)] = True

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = gpt_model(pre_allocated_input_ids, attn_mask=pre_allocated_attn_mask)
            
            log_probs = F.log_softmax(logits, dim=-1)

            example_scores = [[] for _ in range(len(current_examples))]

            for seq_idx in range(len(all_seq_ids)):
                original_ex_idx = orig_example_indices[seq_idx]
                prefix_len = prefix_lengths[seq_idx]
                true_len = pre_allocated_attn_mask[seq_idx].sum().item()
                
                seq_log_prob = 0.0
                num_continuation_tokens = 0
                for token_idx in range(prefix_len, true_len):
                    target_token_id = pre_allocated_input_ids[seq_idx, token_idx].item()
                    seq_log_prob += log_probs[seq_idx, token_idx - 1, target_token_id].item()
                    num_continuation_tokens += 1
                
                normalized_log_prob = seq_log_prob / num_continuation_tokens

                example_scores[original_ex_idx].append(normalized_log_prob)

            for ex_idx, scores in enumerate(example_scores):
                correct_label_idx = int(current_examples[ex_idx]['label'])
                predicted_label_idx = max(enumerate(scores), key=lambda x: x[1])[0]

                if predicted_label_idx == correct_label_idx:
                    total_correct_predictions += 1
                
                processed_examples_counter += 1

                if processed_examples_counter % 200 == 0:
                    for rank in range(ddp_world_size):
                        dist.barrier()
                        if ddp_rank == rank:
                            print(f"[rank {ddp_rank}] processed {processed_examples_counter} examples: acc: {total_correct_predictions / processed_examples_counter:.4f}")
                    dist.barrier()

    local_accuracy = total_correct_predictions / total_dataset_examples
    local_message = f"[rank {ddp_rank}] HellaSwag local acc: {total_correct_predictions} / {total_dataset_examples} = {local_accuracy:.4f}"
    print(local_message)
    all_messages = [None for _ in range(ddp_world_size)]
    dist.all_gather_object(all_messages, local_message)
    if master_process:
        log_buffer.extend(all_messages)
    dist.barrier()

    correct_tensor = torch.tensor(total_correct_predictions, dtype=torch.long, device=device)
    count_tensor = torch.tensor(total_dataset_examples, dtype=torch.long, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    accuracy = correct_tensor.item() / count_tensor.item()
    gpt_model.train()
    return accuracy

def save_config_info():
    with open(config_filename, "w") as f:
        f.write(f"timestamp: {timestamp}\n")

        f.write(f"ddp world size: {ddp_world_size}\n")

        f.write(f"total tokens per step: {total_tokens_per_step}\n")
        f.write(f"gpu batch size: {gpu_batch_size}\n")
        f.write(f"seq len: {seq_len}\n")
        f.write(f"total tokens per mini-step: {total_tokens_per_mini_step}\n")
        f.write(f"grad accum mini-steps: {grad_accum_mini_steps}\n")
        
        f.write(f"betas: {betas}\n")
        f.write(f"eps: {eps}\n")
        f.write(f"max lr: {max_lr}\n")
        f.write(f"min lr after warmup ratio: {min_lr_after_warmup_ratio}\n")
        f.write(f"warmup tokens: {warmup_tokens}\n")
        f.write(f"warmup and cosine tokens: {warmup_and_cosine_tokens}\n")
        f.write(f"max tokens: {max_tokens}\n")
        f.write(f"weight decay: {weight_decay}\n")

        # derived
        f.write(f"max steps: {max_steps}\n")
        f.write(f"min lr after warmup: {min_lr_after_warmup}\n")
        f.write(f"warmup steps: {warmup_steps}\n")
        f.write(f"warmup and cosine steps: {warmup_and_cosine_steps}\n")

        f.write(f"base seed: {base_seed}\n")
        f.write(f"device type: {device_type}\n")
        f.write(f"tokenizer: gpt2 (tiktoken)\n")

        f.write(f"val target: {val_target}\n")
        f.write(f"val steps: {val_steps}\n")
        f.write(f"val interval: {val_interval}\n")
        f.write(f"sample interval: {sample_interval}\n")

        for k, v in GPTConfig().__dict__.items():
            f.write(f"model config - {k}: {v}\n")

def keep_latest_checkpoints(checkpoint_dir):
    all_files = os.listdir(checkpoint_dir)

    # group files by type
    model_files = sorted(
        [f for f in all_files if f.endswith(".safetensors")],
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )
    state_files = sorted(
        [f for f in all_files if f.endswith(".pt")],
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )

    # for every file that is past the last max_checkpoints_to_keep, remove it,
    # resulting in empty list if max_checkpoints_to_keep >= len(files),
    # and thus preserving all files in that case

    # prune excess model files
    for old_file in model_files[:-max_checkpoints_to_keep]:
        path = os.path.join(checkpoint_dir, old_file)
        os.remove(path)
        message = f"removed model: {path}"
        print(message)
        log_buffer.append(message)

    # prune excess state files
    for old_file in state_files[:-max_checkpoints_to_keep]:
        path = os.path.join(checkpoint_dir, old_file)
        os.remove(path)
        message = f"removed training state: {path}"
        print(message)
        log_buffer.append(message)

def save_checkpoint(step, torch_rng_state_cpu, torch_rng_state_cuda, val_loss, train_loss, train_tokens_processed,
                    total_train_t, total_val_t, total_sample_t, total_hellaswag_t, total_t, best_val_loss, epoch,
                    current_shard_idx, grad_accum_mini_steps_per_shard_counter):

    # create the model path by adding its timestamp, and train and val losses, if available
    parts = [f"model", f"step_{step:07d}"]
    parts.append(f"val_{val_loss:.4f}")
    parts.append(f"train_{train_loss:.4f}")
    checkpoint_name = "_".join(parts) + ".safetensors"
    safetensors_path = os.path.join(checkpoint_dir, checkpoint_name)

    # save the model locally
    save_model(gpt_model.module, safetensors_path)
    message = f"saved model weights locally in: {safetensors_path}"
    print(message)
    log_buffer.append(message)

    # gather the training state
    training_state = {
        'step': step,
        'torch_rng_state_cpu': torch_rng_state_cpu,
        'torch_rng_state_cuda': torch_rng_state_cuda,
        'train_tokens_processed': train_tokens_processed,
        'total_train_t': total_train_t,
        'total_val_t': total_val_t,
        'total_sample_t': total_sample_t,
        'total_hellaswag_t': total_hellaswag_t,
        'total_t': total_t,
        'best_val_loss': best_val_loss,
        'epoch': epoch,
        'current_shard_idx': current_shard_idx,
        'grad_accum_mini_steps_per_shard_counter': grad_accum_mini_steps_per_shard_counter,
        'optimizer_state_dict': optimizer.state_dict()
    }

    # create the training state path
    state_path = os.path.join(checkpoint_dir, f"training_state_step_{step:07d}.pt")

    # save the training state locally
    torch.save(training_state, state_path)
    message = f"saved training state locally in: {state_path}"
    print(message)
    log_buffer.append(message)

    # retain last max_checkpoints_to_keep
    keep_latest_checkpoints(checkpoint_dir)

def load_checkpoint():
    model_path = hf_hub_download(
        repo_id=hub_repo_id,
        filename=resume_checkpoint_path,
        token=hf_token,
        repo_type="model"
    )
    training_state_path = hf_hub_download(
        repo_id=hub_repo_id,
        filename=resume_state_dict_path,
        token=hf_token,
        repo_type="model"
    )

    # load weights
    raw_state_dict = load_file(model_path, device='cpu')
    # print(list(raw_state_dict.keys())[:5])

    # strip `_orig_mod.` prefix from keys
    model_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()
    }
    # print(list(model_state_dict.keys())[:5])

    gpt_model = GPT(GPTConfig())
    gpt_model.load_state_dict(model_state_dict, strict=False)

    # tie weights, creating wte.weight
    gpt_model.transformer.wte.weight = gpt_model.lm_head.weight

    # load training state
    training_state = torch.load(training_state_path, map_location="cpu")

    return (
        # continue on the next step after val
        training_state['step'] + 1,
        training_state['torch_rng_state_cpu'],
        training_state['torch_rng_state_cuda'],
        training_state['train_tokens_processed'],
        training_state['total_train_t'],
        training_state['total_val_t'],
        training_state['total_sample_t'],
        training_state['total_hellaswag_t'],
        training_state['total_t'],
        training_state['best_val_loss'],
        training_state['epoch'],
        training_state['current_shard_idx'],
        training_state['grad_accum_mini_steps_per_shard_counter'],
        training_state['optimizer_state_dict'],
        gpt_model,
    )

init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0

# buffer to 'write to disk' only at checkpointing steps, to avoid e.g. stopping training at step 233,
# restoring a model from step 200 -last val step that improves val loss-  and having training logs up
# to 233 then continuing from 201 (e.g., 232, 233, 201, 202, ...).
# This ensures training losses are stored only after a checkpointing step happens.
log_buffer = []

total_tokens_per_step = 2**19 # 524,288 or ~500M tokens from Language Models are Few-Shot Learners
gpu_batch_size = 64
seq_len = 1024
total_tokens_per_mini_step = ddp_world_size * gpu_batch_size * seq_len
grad_accum_mini_steps = total_tokens_per_step // total_tokens_per_mini_step
assert total_tokens_per_step % total_tokens_per_mini_step == 0
if master_process:
    message = f"per-gpu gradient accumulation mini-steps: {grad_accum_mini_steps}"
    print(message)
    log_buffer.append(message)

betas = (0.9,0.95)
eps = 1e-8
max_lr = 2.6e-3 # 6e-4 | 1e-3 | 1.4e-3 | 1.8e-3 | 2.2e-3 | 2.6e-3 | 3e-3
min_lr_after_warmup_ratio = 0.1 # 0.1 | 0.2
warmup_tokens = 375*10**6 # 375*10**5 | 375*10**6
warmup_and_cosine_tokens = 1.5*10**9 # 3*10**9 | 2*10**9 | 1.5*10**9
max_tokens = 5*10**9
weight_decay = 0.1

max_steps = max_tokens // total_tokens_per_step
min_lr_after_warmup = min_lr_after_warmup_ratio * max_lr
warmup_steps = warmup_tokens // total_tokens_per_step
warmup_and_cosine_steps = warmup_and_cosine_tokens // total_tokens_per_step
if master_process:
    messages = [
        f"warmup steps: {warmup_steps:,}",
        f"warmup and cosine steps: {warmup_and_cosine_steps:,}",
        f"max steps: {max_steps:,}"
    ]
    for message in messages:
        print(message)
        log_buffer.append(message)

device_type = "cuda"

torch.set_float32_matmul_precision('high')

tokenizer = tiktoken.get_encoding("gpt2")

val_target = 3.28
val_tokens = 2 ** 21 * 5
val_steps = math.ceil(val_tokens / total_tokens_per_mini_step)
val_interval = 100
if master_process:
    print(f"{val_tokens:,} val tokens to be consumed in {val_steps:,} steps ({total_tokens_per_mini_step:,} tokens per val step)")

sample_interval = 100
sample_sequences = [
    "The universe has always been",
    "Who am I? I am a language model",
    "Artificial General Intelligence is",
    "Artificial General Intelligence is not",
    "2+2 is",
    "The quick brown fox jumps",
    "Earth is",
    "Could you tell me what time it is?"
]

hellaswag_interval = float('inf')

# checkpoint_interval = 1000
max_checkpoints_to_keep = 1
resume_from_checkpoint = False
resume_checkpoint_path = "model_step_0000125_val_6.8745_train_6.9019.safetensors"
resume_state_dict_path = "training_state_step_0000125.pt"

if not resume_from_checkpoint:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
else:
    timestamp = "20250801_085237"

config_and_log_dir = f"./configs_and_logs/{timestamp}"
log_filename = os.path.join(config_and_log_dir, f"log.txt")
config_filename = os.path.join(config_and_log_dir, f"config.txt")

checkpoint_dir = f"./checkpoints/{timestamp}"

if master_process:
    os.makedirs(config_and_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

hf_user = os.environ.get("hf_user")
hf_token = os.environ.get("hf_token")
hub_repo_id = f"{hf_user}/gpt-3-small_{timestamp}"

start_step = 0
train_tokens_processed = 0
total_train_t = 0.0
total_val_t = 0.0
total_sample_t = 0.0
total_hellaswag_t = 0.0
total_t = 0.0
best_val_loss = float('inf')

base_seed = 1337

epoch = 0
current_shard_idx = 0
grad_accum_mini_steps_per_shard_counter = 0

seed = base_seed + ddp_rank
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if resume_from_checkpoint:
    # get start_step, train dataloader config (epoch, grad_accum_mini_steps_per_shard_counter, etc.),
    # optimizer state and model
    (start_step, torch_rng_state_cpu, torch_rng_state_cuda, train_tokens_processed,
        total_train_t, total_val_t, total_sample_t, total_hellaswag_t, total_t, best_val_loss, epoch,
        current_shard_idx, grad_accum_mini_steps_per_shard_counter,
        optimizer_state_dict, gpt_model) = load_checkpoint()
    if master_process:
        message = f"loaded checkpoint from step {start_step - 1}"
        print(message)
        log_buffer.append(message)
    dist.barrier()

    # move the downloaded gpt_model to device, compile it, set up DDP, and set raw_gpt_model to create the optimizer
    # based on the downloaded model
    gpt_model.to(device)
    gpt_model = torch.compile(gpt_model)
    gpt_model = DDP(gpt_model, device_ids=[ddp_local_rank])
    raw_gpt_model = gpt_model.module
    optimizer = raw_gpt_model.configure_optimizers(max_lr, betas, eps, weight_decay, device_type)
    # use the downloaded optimizer_state_dict for the optimizer
    optimizer.load_state_dict(optimizer_state_dict)

    # restore the random number generator state
    torch.set_rng_state(torch_rng_state_cpu)
    torch.cuda.set_rng_state(torch_rng_state_cuda)
    # wait for all ranks to sync
    dist.barrier()
else:
    gpt_model = GPT(GPTConfig())
    gpt_model.to(device)
    gpt_model = torch.compile(gpt_model)
    gpt_model = DDP(gpt_model, device_ids=[ddp_local_rank])
    raw_gpt_model = gpt_model.module
    optimizer = raw_gpt_model.configure_optimizers(max_lr, betas, eps, weight_decay, device_type)

data_path = "./data/edu_fineweb10B"
# if resume_from_checkpoint: epoch, current_shard_idx, and grad_accum_mini_steps_per_shard_counter are overriden
# above and thus set to non-zero values in the DataLoader()
train_data_loader = DataLoader(gpu_batch_size, seq_len, ddp_world_size, ddp_rank, data_path, "train",
                               epoch=epoch, current_shard_idx=current_shard_idx,
                               grad_accum_mini_steps_per_shard_counter=grad_accum_mini_steps_per_shard_counter)
val_data_loader = DataLoader(gpu_batch_size, seq_len, ddp_world_size, ddp_rank, data_path, "val",
                             epoch=0, current_shard_idx=0,
                             grad_accum_mini_steps_per_shard_counter=0)

# load HellaSwag and split it across all GPUs similar to how the train DataLoader assigns different chunks to each GPU
hellaswag_train_dataset = load_dataset("hellaswag", split="train")
hellaswag_val_dataset = load_dataset("hellaswag", split="validation")
total_hellaswag_val_examples = len(hellaswag_val_dataset)
hellaswag_val_examples_per_rank = (total_hellaswag_val_examples + ddp_world_size - 1) // ddp_world_size
hellaswag_val_start_idx = ddp_rank * hellaswag_val_examples_per_rank
hellaswag_val_end_idx = min(hellaswag_val_start_idx + hellaswag_val_examples_per_rank, total_hellaswag_val_examples)
local_hellaswag_val_dataset = hellaswag_val_dataset.select(range(hellaswag_val_start_idx, hellaswag_val_end_idx))
local_message = (
    f"[rank {ddp_rank}] gets HellaSwag sentences "
    f"{hellaswag_val_start_idx:,} to {hellaswag_val_end_idx - 1:,}"
)
print(local_message)
all_messages = [None for _ in range(ddp_world_size)]
dist.all_gather_object(all_messages, local_message)
if master_process:
    log_buffer.extend(all_messages)
dist.barrier()

if master_process:
    save_config_info()

# 50304*768 + 1024*768 + 12*2*(768+768) + 12*(768*3*768 + 3*768) + 12*(768*768 + 768) + 12*(768*4*768 + 4*768) + 12*(768*4*768 + 768) + 768 + 768
if master_process:
    message = f"{sum(p.numel() for p in gpt_model.parameters() if p.requires_grad):,} parameters"
    print(message)
    log_buffer.append(message)

try:
    for step in range(start_step, max_steps):
        
        torch.cuda.synchronize()
        start_train_t = time.time()
        gpt_model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        # step_checkpointed = False

        # per-gpu grad accumulation mini-steps
        for mini_step in range(grad_accum_mini_steps):
            x_train, y_train = train_data_loader.next_batch()
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            gpt_model.require_backward_grad_sync = (mini_step == grad_accum_mini_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, step_train_loss = gpt_model(x_train, y_train)
            step_train_loss /= grad_accum_mini_steps
            train_loss += step_train_loss.detach()
            step_train_loss.backward()

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(gpt_model.parameters(),1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        tl = float(train_loss)

        torch.cuda.synchronize()
        end_train_t = time.time()
        train_step_t = end_train_t - start_train_t
        total_train_t += train_step_t
        total_t += train_step_t

        if master_process:
            train_tokens_processed += (grad_accum_mini_steps * total_tokens_per_mini_step)
            train_log_content = (
                f"step: {step:,} | train loss: {tl:.8f} | "
                f"train ppl: {math.exp(tl):,.2f} | "
                f"train step time: {1000*(train_step_t):,.2f} ms | "
                f"grad norm: {norm:.4f} | lr: {lr:.8f} | "
                f"tok/s: {ddp_world_size * grad_accum_mini_steps * gpu_batch_size * seq_len / train_step_t:,.2f} | "
                f"total toks: {train_tokens_processed:,} | total time: {total_t/60:,.2f} min"
            )
            print(train_log_content)
            # with open(log_filename, "a") as f:
            #     f.write(train_log_content + "\n")
            log_buffer.append(train_log_content)

        if (step % sample_interval == 0 and step > 0) or step == max_steps - 1:
            torch.cuda.synchronize()
            start_sample_t = time.time()
            max_new_tokens = get_sample_token_count(step)
            sample(sample_sequences, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            end_sample_t = time.time()
            sample_step_t = end_sample_t - start_sample_t
            total_sample_t += sample_step_t
            total_t += sample_step_t
            if master_process:
                message = f"step: {step:,} | sampling time: {(sample_step_t):,.2f} s"
                print(message)
                log_buffer.append(message)

        if (step % hellaswag_interval == 0 and step > 0) or step == max_steps - 1:
            torch.cuda.synchronize()
            start_hellaswag_t = time.time()
            accuracy = evaluate_hellaswag_standard()
            torch.cuda.synchronize()
            end_hellaswag_t = time.time()
            hellaswag_step_t = end_hellaswag_t - start_hellaswag_t
            total_hellaswag_t += hellaswag_step_t
            total_t += hellaswag_step_t
            if master_process:
                message = f"step: {step:,} | HellaSwag acc: {accuracy:.4f} | HellaSwag time: {(hellaswag_step_t):,.2f} s"
                print(message)
                log_buffer.append(message)

        if (step % val_interval == 0 and step > 0) or step == max_steps - 1:
            torch.cuda.synchronize()
            start_val_t = time.time()
            gpt_model.eval()
            if master_process:
                message = f"resetting val loader at step {step}"
                print(message)
                log_buffer.append(message)
            val_data_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_steps):
                    x_val, y_val = val_data_loader.next_batch()
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, step_val_loss = gpt_model(x_val, y_val)
                    val_loss += step_val_loss / val_steps
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if master_process:
                    # pass (exact values as) arguments to avoid modifying the counters before it saves
                    # as save_checkpoint must defer writing to disk and the training loop continues
                    # before variables such as step or train_tokens_processed are stored
                    save_checkpoint(step=step,
                        torch_rng_state_cpu=torch.get_rng_state(),
                        torch_rng_state_cuda=torch.cuda.get_rng_state(),
                        train_loss=train_loss.item(),
                        val_loss=val_loss.item(),
                        train_tokens_processed=train_tokens_processed,
                        total_train_t=total_train_t,
                        total_val_t=total_val_t,
                        total_sample_t=total_sample_t,
                        total_hellaswag_t=total_hellaswag_t,
                        total_t=total_t,
                        best_val_loss=best_val_loss,
                        epoch=train_data_loader.epoch,
                        current_shard_idx=train_data_loader.current_shard_idx,
                        grad_accum_mini_steps_per_shard_counter=train_data_loader.grad_accum_mini_steps_per_shard_counter)
                    message = f"new best val loss: {best_val_loss:.8f}"
                    print(message)
                    log_buffer.append(message)
                    # only after we improve val loss and a checkpoint is saved, we push the logs
                    with open(log_filename, "a") as f:
                        for line in log_buffer:
                            f.write(line + "\n")
                        log_buffer.clear()
                dist.barrier()
                # step_checkpointed = True

            torch.cuda.synchronize()
            end_val_t = time.time()
            val_step_t = end_val_t - start_val_t
            total_val_t += val_step_t
            total_t += val_step_t

            if master_process:
                val_log_content = f"step: {step:,} | val loss: {val_loss.item():.8f} | val ppl: {math.exp(val_loss.item()):,.2f} | val time: {1000*(val_step_t):,.2f} ms"
                print(val_log_content)
                # with open(log_filename, "a") as f:
                #     f.write(val_log_content + "\n")
                log_buffer.append(val_log_content)

            if val_loss <= val_target:
                if master_process:
                    print(f"val loss {val_loss.item():.8f} reached target {val_target}")
                break

        # if not step_checkpointed and step % checkpoint_interval == 0 and step > 0 and master_process:
        #     start_checkpointing_t = time.time()
        #     save_checkpoint(train_loss=train_loss)
        #     torch.cuda.synchronize()
        #     end_checkpointing_t = time.time()
        #     checkpointing_step_t = end_checkpointing_t - start_checkpointing_t
        #     total_t += checkpointing_step_t

except Exception as e:
    if master_process:
        print(f"[rank {ddp_rank}] unhandled exception: {e}")
        import traceback
        traceback.print_exc()
    dist.barrier()
    raise
finally:
    dist.barrier()
    destroy_process_group()