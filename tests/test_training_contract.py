import copy

import torch
from torch import nn

from galinear.predictors import Forecast
from galinear.state import flatten_parameters
from galinear.training import Counters, RunConfig, attempt_jump, model_digest, train_arm


class FixedPredictor:
    def __init__(self, params):
        self.params = params

    def predict(self, states, skip):
        return Forecast(self.params.clone(), rank=1, history_count=len(states))


def make_model():
    torch.manual_seed(3)
    return nn.Linear(1, 1)


def validate(model):
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    y = 2.0 * x
    with torch.no_grad():
        pred = model(x)
        loss = torch.mean((pred - y) ** 2).item()
        acc = float(loss < 0.05)
    return loss, acc


def real_epoch(model, optimizer, epoch_index):
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    y = 2.0 * x
    optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((model(x) - y) ** 2)
    loss.backward()
    optimizer.step()
    return float(loss.detach()), 1


def test_attempt_jump_never_increments_backward_counter():
    model = make_model()
    counters = Counters()
    current = flatten_parameters(model)
    predictor = FixedPredictor(current * 0.9)
    before = counters.backward_passes
    result = attempt_jump(
        model=model,
        predictor=predictor,
        states=[current, current * 0.95],
        skip=2,
        counters=counters,
        validation_fn=validate,
        guarded=False,
        current_val_loss=validate(model)[0],
        guard_rel_tol=0.0,
    )
    assert result.accepted
    assert counters.backward_passes == before


def test_guard_rejection_restores_parameters_before_fallback_training():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original = flatten_parameters(model).clone()
    awful = torch.full_like(original, 1e5)
    counters = Counters()
    result = attempt_jump(
        model=model,
        predictor=FixedPredictor(awful),
        states=[original, original + 0.01],
        skip=2,
        counters=counters,
        validation_fn=validate,
        guarded=True,
        current_val_loss=validate(model)[0],
        guard_rel_tol=0.0,
    )
    assert not result.accepted
    assert torch.allclose(flatten_parameters(model), original)
    real_epoch(model, optimizer, 0)
    assert not torch.allclose(flatten_parameters(model), original)


def test_train_arm_uses_identical_initial_state_digest():
    base = make_model()
    initial_state = copy.deepcopy(base.state_dict())
    receipts = []
    for arm in ("plain", "scalar_linear", "joint_operator", "joint_guarded"):
        model = make_model()
        model.load_state_dict(initial_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        receipt = train_arm(
            arm=arm,
            model=model,
            optimizer=optimizer,
            real_epoch_fn=real_epoch,
            validation_fn=validate,
            config=RunConfig(max_real_epochs=6, history=3, skip=2, max_rank=2),
        )
        receipts.append(receipt)
    assert len({r.initial_model_digest for r in receipts}) == 1
    assert receipts[0].initial_model_digest == model_digest(base)


def test_forecasts_do_not_consume_real_epoch_budget():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)
    receipt = train_arm(
        arm="joint_guarded",
        model=model,
        optimizer=optimizer,
        real_epoch_fn=real_epoch,
        validation_fn=validate,
        config=RunConfig(
            max_real_epochs=12,
            history=4,
            skip=2,
            max_rank=2,
            guard_rel_tol=0.25,
        ),
    )
    assert receipt.trained_epochs == 12
    assert receipt.backward_passes == receipt.trained_epochs
    assert receipt.effective_epoch_equivalent == receipt.trained_epochs + receipt.nominal_skipped_steps
    assert receipt.forward_checks == receipt.accepted_jumps + receipt.rejected_jumps


def test_nonfinite_unguarded_candidate_is_visible_failure():
    model = make_model()
    counters = Counters()
    current = flatten_parameters(model)
    nan_candidate = current.clone()
    nan_candidate[0] = float("nan")
    result = attempt_jump(
        model=model,
        predictor=FixedPredictor(nan_candidate),
        states=[current, current + 0.01],
        skip=2,
        counters=counters,
        validation_fn=validate,
        guarded=False,
        current_val_loss=validate(model)[0],
        guard_rel_tol=0.0,
    )
    assert not result.accepted
    assert result.reason == "nonfinite"
    assert counters.nonfinite_proposals == 1
    assert torch.isfinite(flatten_parameters(model)).all()

def test_first_forecast_waits_for_full_real_history_window():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    receipt = train_arm(
        arm="scalar_linear",
        model=model,
        optimizer=optimizer,
        real_epoch_fn=real_epoch,
        validation_fn=validate,
        config=RunConfig(max_real_epochs=6, history=5, skip=5, max_rank=2),
    )
    jump_events = [e for e in receipt.trajectory if e["action"] == "jump"]
    assert jump_events
    assert jump_events[0]["trained_epoch"] == 5
