from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SyntheticTask:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 12),
            nn.Tanh(),
            nn.Linear(12, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_synthetic_task(seed: int = 0, n_train: int = 128, n_val: int = 128) -> SyntheticTask:
    g = torch.Generator().manual_seed(seed)

    def sample(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand((n, 2), generator=g) * 2.0 - 1.0
        # Nonlinear but smooth binary boundary.
        score = x[:, 0] * x[:, 1] + 0.35 * x[:, 0] - 0.15 * x[:, 1]
        y = (score > 0.0).long()
        return x, y

    x_train, y_train = sample(n_train)
    x_val, y_val = sample(n_val)
    return SyntheticTask(x_train, y_train, x_val, y_val)


def make_tiny_model(seed: int) -> TinyMLP:
    torch.manual_seed(seed)
    return TinyMLP()


def make_fullbatch_callbacks(task: SyntheticTask):
    criterion = nn.CrossEntropyLoss()

    def real_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, epoch_index: int):
        del epoch_index
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(task.x_train)
        loss = criterion(logits, task.y_train)
        loss.backward()
        optimizer.step()
        return float(loss.detach()), 1

    def validate(model: nn.Module):
        model.eval()
        with torch.no_grad():
            logits = model(task.x_val)
            loss = criterion(logits, task.y_val)
            acc = (logits.argmax(dim=1) == task.y_val).float().mean()
        return float(loss), float(acc)

    return real_epoch, validate
