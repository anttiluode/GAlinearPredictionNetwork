import math

from scripts.run_smoke import run_smoke


def test_smoke_runs_all_four_matched_arms():
    receipts = run_smoke(seed=7)
    assert set(receipts) == {"plain", "scalar_linear", "joint_operator", "joint_guarded"}
    digests = {r.initial_model_digest for r in receipts.values()}
    assert len(digests) == 1
    for receipt in receipts.values():
        assert receipt.status.startswith("ok") or receipt.status.startswith("failed_")
        assert math.isfinite(receipt.final_loss)
        assert math.isfinite(receipt.final_accuracy)
    assert receipts["plain"].accepted_jumps == 0
    assert receipts["plain"].rejected_jumps == 0
    assert receipts["plain"].backward_passes == receipts["plain"].trained_epochs
    assert receipts["joint_guarded"].forward_checks == (
        receipts["joint_guarded"].accepted_jumps + receipts["joint_guarded"].rejected_jumps
    )
