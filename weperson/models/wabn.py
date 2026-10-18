"""Weighted Adaptive BatchNorm (ωαBN) — refactored from lihe404/WePerson SQA/wabn.py."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class WABN2d(nn.Module):
    """Weighted Batch Normalization for single-query statistics."""

    def __init__(self, bn: nn.BatchNorm2d, w: float = 0.5, n: int = 8, alpha: float = 0.9):
        super().__init__()
        self.w = float(w)
        self.n = int(n)
        self.alpha = float(alpha)
        self.eps = float(bn.eps)
        self.affine = bool(bn.affine)
        self.register_buffer("running_mean", deepcopy(bn.running_mean.detach()))
        self.register_buffer("running_var", deepcopy(bn.running_var.detach()))
        if self.affine:
            self.weight = nn.Parameter(deepcopy(bn.weight.detach()))
            self.bias = nn.Parameter(deepcopy(bn.bias.detach()))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.training and self.n > 0 and x.size(0) > 1:
                x_t = x[0]
                x_aug = x[1:]
                mu = self.w * x_t.mean([1, 2]) + (1.0 - self.w) * x_aug.mean([0, 2, 3])
                var_t = ((x_t - mu[:, None, None]) ** 2).sum([1, 2]) / (x_t.size(1) * x_t.size(2) - 1)
                var_aug = ((x_aug - mu[:, None, None]) ** 2).sum([0, 2, 3]) / (
                    x_aug.size(0) * x_aug.size(2) * x_aug.size(3) - 1
                )
                var = self.w * var_t + (1.0 - self.w) * var_aug
                self.running_mean = self.alpha * self.running_mean + (1.0 - self.alpha) * mu
                self.running_var = self.alpha * self.running_var + (1.0 - self.alpha) * var
            mu = self.running_mean
            var = self.running_var
        x = (x - mu[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x
