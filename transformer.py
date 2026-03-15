"""GPT-2 style decoder-only transformer for next-token prediction.

Following Shai et al. (2026): 4 layers, d_model=120, d_MLP=480, causal attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, max_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 120, n_heads: int = 4,
                 n_layers: int = 4, d_mlp: int = 480, max_len: int = 128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_mlp, max_len) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.d_model = d_model
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, return_residual=False):
        """Forward pass. If return_residual=True, also return residual stream at final layer."""
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.head(h)
        if return_residual:
            return logits, h
        return logits

    def residual_stream(self, x, layer: int = -1) -> torch.Tensor:
        """Extract residual stream activations after a given layer (before final LN).

        layer=-1 means after last block (before ln_f), layer=0..n_layers-1 for intermediate.
        """
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == layer:
                return h
        return h  # layer == -1 or n_layers-1
