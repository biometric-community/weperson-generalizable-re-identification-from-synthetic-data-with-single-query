"""ResNet-50 BNNeck baseline used as the SQA host network."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BaselineReID(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = 2048, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # GAP included
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.backbone(x).flatten(1)
        bn = self.bn(f)
        logits = self.classifier(bn)
        return {"feat": F.normalize(bn, dim=1), "logits": logits, "pool": f}


def build_baseline(cfg: dict, num_classes: int) -> BaselineReID:
    m = cfg.get("model", {})
    return BaselineReID(
        num_classes=num_classes,
        feat_dim=int(m.get("feat_dim", 2048)),
        pretrained=bool(m.get("pretrained", True)),
    )
