from __future__ import annotations

import copy

import torch

from galinear.tasks import make_fullbatch_callbacks, make_synthetic_task, make_tiny_model
from galinear.training import RunConfig, train_arm


ARMS = ("plain", "scalar_linear", "joint_operator", "joint_guarded")


def run_smoke(seed: int = 7):
    task = make_synthetic_task(seed=seed)
    real_epoch, validate = make_fullbatch_callbacks(task)
    base = make_tiny_model(seed=seed + 1000)
    initial = copy.deepcopy(base.state_dict())
    config = RunConfig(
        max_real_epochs=24,
        history=5,
        skip=3,
        max_rank=4,
        ridge=1e-5,
        guard_rel_tol=0.02,
    )
    receipts = {}
    for arm in ARMS:
        model = make_tiny_model(seed=seed + 1000)
        model.load_state_dict(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.35)
        receipts[arm] = train_arm(
            arm=arm,
            model=model,
            optimizer=optimizer,
            real_epoch_fn=real_epoch,
            validation_fn=validate,
            config=config,
        )
    return receipts


def main() -> None:
    receipts = run_smoke()
    print("CPU smoke benchmark")
    print("arm              backprops  jumps  rejects  loss      accuracy")
    for arm, r in receipts.items():
        print(
            f"{arm:17s} {r.backward_passes:9d} {r.accepted_jumps:6d} "
            f"{r.rejected_jumps:8d} {r.final_loss:8.5f} {r.final_accuracy:9.4f}"
        )


if __name__ == "__main__":
    main()
