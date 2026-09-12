import pytest
import torch
from torch import nn

from galinear.state import flatten_parameters, load_flat_parameters
from galinear.predictors import ScalarLinearPredictor, JointOperatorPredictor


def test_flat_parameter_roundtrip():
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    original = flatten_parameters(model).clone()
    modified = original + torch.linspace(0.0, 0.1, original.numel())
    load_flat_parameters(model, modified)
    assert torch.allclose(flatten_parameters(model), modified)
    load_flat_parameters(model, original)
    assert torch.allclose(flatten_parameters(model), original)


def test_scalar_linear_exact_linear_sequence():
    states = [torch.tensor([1.0, -2.0]) + i * torch.tensor([0.5, 0.25]) for i in range(5)]
    predictor = ScalarLinearPredictor(history=5)
    pred = predictor.predict(states, skip=3)
    expected = states[-1] + 3 * torch.tensor([0.5, 0.25])
    assert torch.allclose(pred.parameters, expected, atol=1e-6)
    assert pred.history_count == 5


def test_joint_operator_exact_affine_recurrence():
    A = torch.tensor([[0.9, 0.2], [-0.1, 0.8]], dtype=torch.float64)
    b = torch.tensor([0.05, -0.02], dtype=torch.float64)
    z = torch.tensor([0.4, -0.3], dtype=torch.float64)
    basis = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]], dtype=torch.float64)
    offset = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    states = []
    for _ in range(10):
        states.append(offset + basis @ z)
        z = A @ z + b
    true = z.clone()
    for _ in range(3):
        true = A @ true + b
    expected = offset + basis @ true
    predictor = JointOperatorPredictor(history=10, max_rank=2, ridge=1e-12)
    pred = predictor.predict(states, skip=4)
    assert pred.rank <= 2
    assert torch.allclose(pred.parameters, expected, atol=2e-5, rtol=2e-5)


def test_predictors_reject_too_short_history():
    with pytest.raises(ValueError):
        ScalarLinearPredictor(history=5).predict([torch.zeros(3)], skip=1)
    with pytest.raises(ValueError):
        JointOperatorPredictor(history=5, max_rank=2).predict([torch.zeros(3)], skip=1)
