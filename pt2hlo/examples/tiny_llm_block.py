import math

import torch
import torch.nn as nn


class TinyLLMBlock(nn.Module):
    def __init__(self, hidden_size: int = 128, num_heads: int = 4, ffn_mult: int = 4):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden_size * ffn_mult, hidden_size),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, S, H] -> [B, heads, S, head_dim]
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, heads, S, head_dim] -> [B, S, H]
        bsz, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bsz, seq_len, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention block + residual.
        h = self.ln1(x)
        q = self._split_heads(self.q_proj(h))
        k = self._split_heads(self.k_proj(h))
        v = self._split_heads(self.v_proj(h))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = self._merge_heads(attn_out)
        x = x + self.o_proj(attn_out)

        # Pre-LN FFN + residual.
        h = self.ln2(x)
        x = x + self.ffn(h)
        return x


def build_model() -> nn.Module:
    return TinyLLMBlock(hidden_size=128, num_heads=4, ffn_mult=4)
