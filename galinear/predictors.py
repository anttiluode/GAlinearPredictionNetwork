from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class Forecast:
    parameters: torch.Tensor
    rank: int
    history_count: int


def _stack_recent(states: Sequence[torch.Tensor], history: int) -> torch.Tensor:
    if history < 2:
        raise ValueError("history must be at least 2")
    if len(states) < 2:
        raise ValueError("at least two observed states are required")
    recent = states[-min(history, len(states)) :]
    width = recent[0].numel()
    if any(s.numel() != width for s in recent):
        raise ValueError("all states must have equal size")
    return torch.stack([s.detach().reshape(-1) for s in recent], dim=0)


class ScalarLinearPredictor:
    def __init__(self, history: int = 5) -> None:
        if history < 2:
            raise ValueError("history must be at least 2")
        self.history = history

    def predict(self, states: Sequence[torch.Tensor], skip: int) -> Forecast:
        if skip < 1:
            raise ValueError("skip must be >= 1")
        X = _stack_recent(states, self.history)
        n = X.shape[0]
        t = torch.arange(n, dtype=X.dtype, device=X.device)
        tc = t - t.mean()
        denom = torch.sum(tc * tc)
        slope = (tc[:, None] * (X - X.mean(dim=0, keepdim=True))).sum(dim=0) / denom
        target_t = (n - 1) + skip
        pred = X.mean(dim=0) + slope * (target_t - t.mean())
        return Forecast(parameters=pred, rank=1, history_count=n)


class JointOperatorPredictor:
    def __init__(self, history: int = 5, max_rank: int = 8, ridge: float = 1e-6) -> None:
        if history < 3:
            raise ValueError("history must be at least 3")
        if max_rank < 1:
            raise ValueError("max_rank must be >= 1")
        if ridge < 0:
            raise ValueError("ridge must be nonnegative")
        self.history = history
        self.max_rank = max_rank
        self.ridge = ridge

    def predict(self, states: Sequence[torch.Tensor], skip: int) -> Forecast:
        if skip < 1:
            raise ValueError("skip must be >= 1")
        X = _stack_recent(states, self.history)
        if X.shape[0] < 3:
            raise ValueError("joint operator requires at least three observed states")

        center = X.mean(dim=0)
        centered = X - center
        # Vh rows are orthonormal directions in full parameter space.
        _, s, Vh = torch.linalg.svd(centered, full_matrices=False)
        if s.numel() == 0 or float(s[0]) == 0.0:
            return Forecast(parameters=X[-1].clone(), rank=0, history_count=X.shape[0])
        tol = torch.finfo(s.dtype).eps * max(centered.shape) * s[0]
        numerical_rank = int((s > tol).sum().item())
        rank = min(self.max_rank, numerical_rank, X.shape[0] - 1)
        if rank == 0:
            return Forecast(parameters=X[-1].clone(), rank=0, history_count=X.shape[0])

        basis = Vh[:rank].T
        z = centered @ basis
        zin = z[:-1]
        zout = z[1:]
        ones = torch.ones((zin.shape[0], 1), dtype=X.dtype, device=X.device)
        design = torch.cat([zin, ones], dim=1)
        gram = design.T @ design
        if self.ridge:
            reg = torch.eye(gram.shape[0], dtype=X.dtype, device=X.device) * self.ridge
            reg[-1, -1] = 0.0
            gram = gram + reg
        rhs = design.T @ zout
        coeff = torch.linalg.pinv(gram) @ rhs
        A = coeff[:-1]
        b = coeff[-1]

        cur = z[-1].clone()
        for _ in range(skip):
            cur = cur @ A + b
        pred = center + basis @ cur
        return Forecast(parameters=pred, rank=rank, history_count=X.shape[0])
