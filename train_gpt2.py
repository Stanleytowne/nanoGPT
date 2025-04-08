from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

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

        # calc the attention results
        att = torch.einsum("BnTd,BnSd->BnTS", q, k) / math.sqrt(self.head_dim)
        # apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # apply v_proj
        y = att @ v
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
    

    def forward(self, idx):
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
        return logits
    

if __name__ == '__main__':
    # autodetect the device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"using device: {device}")

    num_return_sequences = 5
    max_length = 30

    # get the model
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.eval()
    model.to(device)

    # get the chatgpt tokenizer
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')

    # get the inputs
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    # set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # generate
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # decode the generated tokens
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)