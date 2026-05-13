from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.register_buffer("class_weight", weight.detach().float())
        self.gamma = float(gamma)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(reduction)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.class_weight, reduction="none")
        pt = torch.exp(-ce).clamp(min=1e-7, max=1.0 - 1e-7)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
