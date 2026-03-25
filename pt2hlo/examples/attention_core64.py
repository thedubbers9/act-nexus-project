import torch
import torch.nn as nn


class AttentionCore64(nn.Module):
    """Minimal attention core shaped for ACT-friendly HLO export.

    This keeps the computation close to the ATTN_TILE64 reference workload:
    transpose -> dot -> exponential -> reduce_sum -> divide -> dot
    """

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores_exp = torch.exp(scores)
        probs = scores_exp / scores_exp.sum(dim=-1, keepdim=True)
        return torch.matmul(probs, v)


def build_model() -> nn.Module:
    return AttentionCore64()
