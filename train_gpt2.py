from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os
import inspect


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        # for initialization taking account of skip connection
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x
    

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)

        # for initialization taking account of skip connection
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x):
        B, T, D = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        reshape_fn = lambda x: x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)     # (B, nh, T, dh)
        q, k, v = reshape_fn(q), reshape_fn(k), reshape_fn(v)

        # using flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.c_proj(y)
        
        return y


class Block(nn.Module):

    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    max_seq_len: int = 1024  # max sequence length!!


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.max_seq_len, config.n_embd),
            h = nn.ModuleList(
                [Block(config) for _ in range(config.n_layer)]
            ),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        # initial params in the way gpt does
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2'}, "currently, only the smallest model is supported"
        print(f"loading pretrained weights from gpt: {model_type}")

        from transformers import GPT2LMHeadModel

        config_sets = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, max_seq_len=1024),
            # omit the rest of the models in the family 
        }

        config_args = config_sets[model_type]
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') or k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"keys len missmatch: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.max_seq_len, f"sequence length {T} exceed max_seq_len {self.config.max_seq_len}!"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    # add support for weight decay
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # all parameters that require grad
        param_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}

        # add parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [param for name, param in param_dict.items() if param.dim() >= 2]
        nodecay_params = [param for name, param in param_dict.items() if param.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # print the number of parameters
        num_decay_params = sum(param.numel() for param in decay_params)
        num_nodecay_params = sum(param.numel() for param in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"fused AdamW enabled: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
        

# --------------------
# DataLoader
import tiktoken

class DataLoader:

    def __init__(self, B, T, process_rank, num_processes, input_dir='input.txt'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open(input_dir, 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        if master_process:
            print(f"loaded {len(self.tokens)} tokens from {input_dir}")

        self.current_pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos: self.current_pos + B * T + 1].clone().detach()
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_pos += B * T * self.num_processes

        if self.current_pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_pos = self.B * self.T * self.process_rank

        return x, y


if __name__ == '__main__':
    # lauching
    # single gpu: python train_gpt2.py
    # multi-gpu: torchrun --standalone --nproc_per_node=8 train_gpt2.py

    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist


    # setup DDP
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "only CUDA is supported"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ.get('RANK'))
        ddp_world_size = int(os.environ.get('WORLD_SIZE'))
        ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = 1

        # auto detect the device
        device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
        print(f"using device: {device}")  


    num_return_sequences = 5
    max_length = 30

    # get the model
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig(vocab_size=50304))
    model.eval()
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # lr scheduler
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 50
    def get_lr(it):
        # linear warmup
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        
        # if it > max_steps, return min learning rate
        if it > max_steps:
            return min_lr
        
        # in between, use cosine decay down
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # get the chatgpt tokenizer
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')

    # dataloader
    total_batch_size = 524288   # 0.5M tokens
    B = 16                      # micro batch size
    T = 1024                    # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

    torch.set_float32_matmul_precision('high')

    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    for i in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()

        loss_total = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # scale the loss so that the gradients add up correctly
            loss = loss / grad_accum_steps
            loss_total += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()

        if ddp:
            dist.all_reduce(loss_total, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # clip the gradient

        # determine the learning rate
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # and finally update the parameters
        optimizer.step()

        # wait for gpu to finish work
        torch.cuda.synchronize()
        t1 = time.time()
        # dt = (t1 - t0)*1000
        # tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        # print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
        dt = t1 - t0 # time difference in seconds
        tokens_processed = total_batch_size
        tokens_per_sec = tokens_processed / dt

        if master_process:
            print(f"step {i:4d} | loss: {loss_total.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | norm: {norm:.2f} | lr: {lr:.6f}")   # print the norm of the gradient

    if ddp:
        destroy_process_group()