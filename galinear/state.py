from __future__ import annotations

import torch
from torch import nn


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    parts = [p.detach().reshape(-1) for p in model.parameters()]
    if not parts:
        return torch.empty(0)
    return torch.cat(parts).clone()


def load_flat_parameters(model: nn.Module, flat: torch.Tensor) -> None:
    flat = flat.detach()
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            n = param.numel()
            if offset + n > flat.numel():
                raise ValueError("flat vector is too short for model parameters")
            chunk = flat[offset : offset + n].to(device=param.device, dtype=param.dtype)
            param.copy_(chunk.view_as(param))
            offset += n
    if offset != flat.numel():
        raise ValueError("flat vector has extra entries beyond model parameters")
