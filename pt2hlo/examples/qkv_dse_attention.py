import torch
import torch.nn as nn


class QKVDSEAttention(nn.Module):
    """A backend-friendly LLM-style block.

    This intentionally stays close to the currently workable ACT/QKV_DSE HLO:
    dot -> add -> exp -> reduce_sum -> divide -> dot -> add -> add
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)

        scores = q_proj + k_proj
        scores_exp = torch.exp(scores)
        probs = scores_exp / scores_exp.sum(dim=1, keepdim=True)

        ctx = self.o_proj(probs)
        res = ctx + v_proj
        return res + x


def build_model() -> nn.Module:
    return QKVDSEAttention(hidden_size=64)
