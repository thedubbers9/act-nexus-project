import torch
import torch.nn as nn


class AttentionBlock64(nn.Module):
    """A slightly richer ATTN_TILE64-friendly attention block.

    This stays inside the currently supported HLO/ISA subset while producing
    a more varied instruction mix than the minimal attention core:

    transpose -> dot -> exponential -> reduce_sum -> divide -> dot -> dot -> add

    Inputs:
      q, k, v, proj, residual: all bf16[64,64]
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        proj: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores_exp = torch.exp(scores)
        probs = scores_exp / scores_exp.sum(dim=-1, keepdim=True)
        ctx = torch.matmul(probs, v)
        proj_out = torch.matmul(ctx, proj)
        residual_view = residual.reshape(residual.shape)
        return proj_out + residual_view


def build_model() -> nn.Module:
    return AttentionBlock64()
