from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from .training import RunReceipt


SUMMARY_FIELDS = [
    "arm",
    "status",
    "reached_target",
    "trained_epochs",
    "effective_epoch_equivalent",
    "backward_passes",
    "forward_checks",
    "accepted_jumps",
    "rejected_jumps",
    "nonfinite_proposals",
    "nominal_skipped_steps",
    "predictor_seconds",
    "total_seconds",
    "final_loss",
    "final_accuracy",
    "initial_model_digest",
]


def write_receipts(
    *,
    output_dir: str | Path,
    run_name: str,
    config: Mapping,
    metadata: Mapping,
    receipts: Mapping[str, RunReceipt],
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{run_name}.json"
    csv_path = output / f"{run_name}.csv"

    payload = {
        "config": dict(config),
        "metadata": dict(metadata),
        "receipts": {name: receipt.to_dict() for name, receipt in receipts.items()},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for name in receipts:
            row = receipts[name].to_dict()
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})
    return json_path, csv_path
