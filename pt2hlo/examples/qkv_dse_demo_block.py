import torch
import torch.nn as nn


class QKVDSEDemoBlock(nn.Module):
    """Demo block intentionally shaped for the current QKV_DSE ISA.

    Expected HLO flavor:
    add -> exp -> reduce -> broadcast -> divide -> dot -> add -> dot -> add

    Keep this model simple on purpose. `nn.Linear` and more realistic attention
    blocks tend to lower into extra transpose / reshape / multiply patterns that
    the current backend does not accept. This version stays close to the ISA we
    actually have today.
    """

    def __init__(self, hidden_size: int = 64, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.mix_1 = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype))
        self.mix_2 = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype))
        self.bias_1 = nn.Parameter(torch.randn(1, hidden_size, dtype=dtype))
        self.bias_2 = nn.Parameter(torch.randn(1, hidden_size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = x + self.bias_1
        scores_exp = torch.exp(scores)
        probs = scores_exp / scores_exp.sum(dim=1, keepdim=True)
        ctx = torch.matmul(probs, self.mix_1)
        hidden = ctx + x
        proj = torch.matmul(hidden + self.bias_2, self.mix_2)
        return proj + hidden


def build_model() -> nn.Module:
    return QKVDSEDemoBlock(hidden_size=64)
