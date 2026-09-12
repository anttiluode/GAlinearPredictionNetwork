import csv
import json
from pathlib import Path

import torch

from galinear.reporting import write_receipts
from galinear.training import RunReceipt
from scripts.run_cifar10 import VanillaCNN, build_parser


def fake_receipt(arm="plain"):
    return RunReceipt(
        arm=arm,
        initial_model_digest="abc",
        trained_epochs=3,
        effective_epoch_equivalent=3,
        backward_passes=144,
        forward_checks=0,
        accepted_jumps=0,
        rejected_jumps=0,
        nonfinite_proposals=0,
        nominal_skipped_steps=0,
        predictor_seconds=0.0,
        total_seconds=1.25,
        final_loss=1.1,
        final_accuracy=0.61,
        reached_target=True,
        status="ok",
        trajectory=[],
    )


def test_cifar_parser_defaults_match_public_wnn_short_term_recipe():
    args = build_parser().parse_args([])
    assert args.epochs == 50
    assert args.history == 5
    assert args.skip == 5
    assert args.batch_size == 1024
    assert args.lr == 1e-3
    assert args.target_accuracy == 0.59
    assert args.max_rank == 4


def test_vanilla_cnn_has_expected_shape():
    model = VanillaCNN()
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)
    conv_channels = [m.out_channels for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert conv_channels == [8, 16, 32, 32]


def test_receipt_writer_emits_json_and_csv(tmp_path: Path):
    receipts = {"plain": fake_receipt("plain"), "joint_guarded": fake_receipt("joint_guarded")}
    json_path, csv_path = write_receipts(
        output_dir=tmp_path,
        run_name="unit",
        config={"seed": 1, "history": 5},
        metadata={"device": "cpu"},
        receipts=receipts,
    )
    payload = json.loads(json_path.read_text())
    assert payload["config"]["history"] == 5
    assert payload["receipts"]["joint_guarded"]["arm"] == "joint_guarded"
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["arm"] for r in rows} == {"plain", "joint_guarded"}
    assert "backward_passes" in rows[0]
    assert "total_seconds" in rows[0]
