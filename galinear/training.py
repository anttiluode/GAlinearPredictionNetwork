from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import time
from typing import Callable, Protocol, Sequence

import torch
from torch import nn

from .predictors import JointOperatorPredictor, ScalarLinearPredictor
from .state import flatten_parameters, load_flat_parameters


ValidationFn = Callable[[nn.Module], tuple[float, float]]
RealEpochFn = Callable[[nn.Module, torch.optim.Optimizer, int], tuple[float, int]]


class Predictor(Protocol):
    def predict(self, states: Sequence[torch.Tensor], skip: int): ...


@dataclass
class Counters:
    backward_passes: int = 0
    forward_checks: int = 0
    accepted_jumps: int = 0
    rejected_jumps: int = 0
    nonfinite_proposals: int = 0
    nominal_skipped_steps: int = 0
    predictor_seconds: float = 0.0


@dataclass(frozen=True)
class RunConfig:
    max_real_epochs: int = 30
    history: int = 5
    skip: int = 5
    max_rank: int = 4
    ridge: float = 1e-6
    guard_rel_tol: float = 0.0
    target_accuracy: float | None = None

    def __post_init__(self) -> None:
        if self.max_real_epochs < 1:
            raise ValueError("max_real_epochs must be >= 1")
        if self.history < 2:
            raise ValueError("history must be >= 2")
        if self.skip < 1:
            raise ValueError("skip must be >= 1")
        if self.max_rank < 1:
            raise ValueError("max_rank must be >= 1")
        if self.guard_rel_tol < 0:
            raise ValueError("guard_rel_tol must be nonnegative")


@dataclass(frozen=True)
class JumpResult:
    accepted: bool
    reason: str
    rank: int
    candidate_loss: float | None = None
    candidate_accuracy: float | None = None


@dataclass
class RunReceipt:
    arm: str
    initial_model_digest: str
    trained_epochs: int
    effective_epoch_equivalent: int
    backward_passes: int
    forward_checks: int
    accepted_jumps: int
    rejected_jumps: int
    nonfinite_proposals: int
    nominal_skipped_steps: int
    predictor_seconds: float
    total_seconds: float
    final_loss: float
    final_accuracy: float
    reached_target: bool
    status: str
    trajectory: list[dict[str, float | int | str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _sync(model: nn.Module) -> None:
    device = _device_of(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_digest(model: nn.Module) -> str:
    flat = flatten_parameters(model).detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(str(flat.dtype).encode())
    h.update(str(tuple(flat.shape)).encode())
    h.update(flat.numpy().tobytes())
    return h.hexdigest()


def attempt_jump(
    *,
    model: nn.Module,
    predictor: Predictor,
    states: Sequence[torch.Tensor],
    skip: int,
    counters: Counters,
    validation_fn: ValidationFn,
    guarded: bool,
    current_val_loss: float,
    guard_rel_tol: float,
) -> JumpResult:
    """Propose a future parameter state without executing a backward pass.

    The model is restored exactly when a guarded proposal is rejected or when
    the candidate is non-finite.  This function never calls backward().
    """
    before = flatten_parameters(model)
    _sync(model)
    t0 = time.perf_counter()
    forecast = predictor.predict(states, skip=skip)
    _sync(model)
    counters.predictor_seconds += time.perf_counter() - t0
    candidate = forecast.parameters.to(device=before.device, dtype=before.dtype)

    if not bool(torch.isfinite(candidate).all()):
        counters.nonfinite_proposals += 1
        counters.rejected_jumps += 1
        load_flat_parameters(model, before)
        return JumpResult(False, "nonfinite", int(forecast.rank))

    load_flat_parameters(model, candidate)

    if not guarded:
        counters.accepted_jumps += 1
        counters.nominal_skipped_steps += skip
        return JumpResult(True, "accepted", int(forecast.rank))

    counters.forward_checks += 1
    candidate_loss, candidate_accuracy = validation_fn(model)
    finite_metrics = torch.isfinite(torch.tensor([candidate_loss, candidate_accuracy])).all().item()
    allowed = current_val_loss * (1.0 + guard_rel_tol)
    if (not finite_metrics) or candidate_loss > allowed:
        counters.rejected_jumps += 1
        if not finite_metrics:
            counters.nonfinite_proposals += 1
        load_flat_parameters(model, before)
        return JumpResult(
            False,
            "guard_reject_nonfinite" if not finite_metrics else "guard_reject_loss",
            int(forecast.rank),
            float(candidate_loss),
            float(candidate_accuracy),
        )

    counters.accepted_jumps += 1
    counters.nominal_skipped_steps += skip
    return JumpResult(
        True,
        "accepted",
        int(forecast.rank),
        float(candidate_loss),
        float(candidate_accuracy),
    )


def _make_predictor(arm: str, config: RunConfig) -> Predictor | None:
    if arm == "plain":
        return None
    if arm == "scalar_linear":
        return ScalarLinearPredictor(history=config.history)
    if arm in {"joint_operator", "joint_guarded"}:
        return JointOperatorPredictor(
            history=max(3, config.history), max_rank=config.max_rank, ridge=config.ridge
        )
    raise ValueError(f"unknown arm: {arm}")


def train_arm(
    *,
    arm: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    real_epoch_fn: RealEpochFn,
    validation_fn: ValidationFn,
    config: RunConfig,
) -> RunReceipt:
    predictor = _make_predictor(arm, config)
    guarded = arm == "joint_guarded"
    counters = Counters()
    initial_digest = model_digest(model)
    trained_epoch = 0
    observed_states: list[torch.Tensor] = []
    current_loss, current_accuracy = validation_fn(model)
    trajectory: list[dict[str, float | int | str]] = [
        {
            "trained_epoch": 0,
            "effective_epoch": 0,
            "action": "initial",
            "loss": float(current_loss),
            "accuracy": float(current_accuracy),
        }
    ]
    reached_target = (
        config.target_accuracy is not None and current_accuracy >= config.target_accuracy
    )
    status = "ok"

    _sync(model)
    total_start = time.perf_counter()

    while trained_epoch < config.max_real_epochs and not reached_target:
        can_forecast = (
            predictor is not None
            and len(observed_states) >= config.history
        )

        if can_forecast:
            jump = attempt_jump(
                model=model,
                predictor=predictor,
                states=observed_states,
                skip=config.skip,
                counters=counters,
                validation_fn=validation_fn,
                guarded=guarded,
                current_val_loss=float(current_loss),
                guard_rel_tol=config.guard_rel_tol,
            )
            if jump.accepted:
                if guarded:
                    current_loss = float(jump.candidate_loss)
                    current_accuracy = float(jump.candidate_accuracy)
                else:
                    current_loss, current_accuracy = validation_fn(model)
                trajectory.append(
                    {
                        "trained_epoch": trained_epoch,
                        "effective_epoch": trained_epoch + counters.nominal_skipped_steps,
                        "action": "jump",
                        "loss": float(current_loss),
                        "accuracy": float(current_accuracy),
                        "rank": int(jump.rank),
                    }
                )
                # Do not train the next predictor on its own imagined states.
                observed_states = []
                reached_target = (
                    config.target_accuracy is not None
                    and current_accuracy >= config.target_accuracy
                )
                continue
            if jump.reason == "nonfinite" and not guarded:
                status = "failed_nonfinite_forecast"
                break

        train_loss, backwards = real_epoch_fn(model, optimizer, trained_epoch)
        counters.backward_passes += int(backwards)
        trained_epoch += 1
        current_loss, current_accuracy = validation_fn(model)
        observed_states.append(flatten_parameters(model))
        if len(observed_states) > config.history:
            observed_states = observed_states[-config.history :]
        trajectory.append(
            {
                "trained_epoch": trained_epoch,
                "effective_epoch": trained_epoch + counters.nominal_skipped_steps,
                "action": "real",
                "train_loss": float(train_loss),
                "loss": float(current_loss),
                "accuracy": float(current_accuracy),
            }
        )
        if not torch.isfinite(torch.tensor([current_loss, current_accuracy])).all().item():
            status = "failed_nonfinite_training"
            break
        reached_target = (
            config.target_accuracy is not None
            and current_accuracy >= config.target_accuracy
        )

    _sync(model)
    total_seconds = time.perf_counter() - total_start
    return RunReceipt(
        arm=arm,
        initial_model_digest=initial_digest,
        trained_epochs=trained_epoch,
        effective_epoch_equivalent=trained_epoch + counters.nominal_skipped_steps,
        backward_passes=counters.backward_passes,
        forward_checks=counters.forward_checks,
        accepted_jumps=counters.accepted_jumps,
        rejected_jumps=counters.rejected_jumps,
        nonfinite_proposals=counters.nonfinite_proposals,
        nominal_skipped_steps=counters.nominal_skipped_steps,
        predictor_seconds=counters.predictor_seconds,
        total_seconds=total_seconds,
        final_loss=float(current_loss),
        final_accuracy=float(current_accuracy),
        reached_target=bool(reached_target),
        status=status,
        trajectory=trajectory,
    )
