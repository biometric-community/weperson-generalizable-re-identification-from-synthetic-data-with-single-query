"""Single Query Adaptation — refactored from lihe404/WePerson SQA/SQA.py."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T

from .wabn import WABN2d


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def set_module(model: nn.Module, submodule_key: str, module: nn.Module) -> None:
    tokens = submodule_key.split(".")
    cur = model
    for s in tokens[:-1]:
        cur = getattr(cur, s)
    setattr(cur, tokens[-1], module)


def configure_model(model: nn.Module, w: float = 0.5, alpha: float = 0.9) -> nn.Module:
    """Replace BatchNorm2d with WABN2d; freeze non-BN params."""
    model.train()
    model.requires_grad_(False)
    for nm, m in list(model.named_modules()):
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            set_module(model, nm, WABN2d(m, w=w, alpha=alpha))
    return model


def collect_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, WABN2d):
            for np, p in m.named_parameters():
                if np in ("weight", "bias") and p is not None:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def aug_samples(x: torch.Tensor, aug_n: int) -> torch.Tensor:
    """Stack raw query + aug_n random augmentations (B=1+aug_n)."""
    trans = [
        T.ColorJitter(0.2, 0.1, 0.5),
        T.RandomHorizontalFlip(1),
        T.RandomVerticalFlip(1),
        T.GaussianBlur(3),
        T.RandomErasing(1),
        T.RandomRotation(15),
        T.Grayscale(3),
        T.RandomPerspective(p=1),
    ]
    chosen = random.sample(trans, min(aug_n, len(trans)))
    raw = x[0]
    out = [raw]
    for t in chosen:
        out.append(t(raw))
    while len(out) < aug_n + 1:
        out.append(random.choice(chosen)(raw))
    return torch.stack(out, 0)


@torch.enable_grad()
def forward_and_adapt(
    x: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lambd: float,
    aug_n: int,
) -> torch.Tensor:
    raw = x
    xa = aug_samples(x, aug_n)
    model.train()
    out = model(xa)
    feat = out["feat"] if isinstance(out, dict) else out
    z_t = feat[0].repeat(aug_n, 1)
    z_aug = feat[1 : 1 + aug_n]
    c = z_t.T @ z_aug / aug_n
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        out2 = model(raw)
    return out2["feat"] if isinstance(out2, dict) else out2


class SQA(nn.Module):
    """Episodic single-query BN adaptation wrapper."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lambd: float = 0.0051,
        aug_n: int = 8,
        steps: int = 1,
        episodic: bool = True,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lambd = float(lambd)
        self.aug_n = int(aug_n)
        self.steps = int(steps)
        self.episodic = bool(episodic)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

    def reset(self) -> None:
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()
        feat = None
        for _ in range(self.steps):
            feat = forward_and_adapt(x, self.model, self.optimizer, self.lambd, self.aug_n)
        return feat
