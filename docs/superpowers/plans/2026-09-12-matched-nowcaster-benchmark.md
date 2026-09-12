# Matched Nowcaster Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible PyTorch benchmark comparing ordinary training, per-parameter linear nowcasting, a joint reduced trajectory operator, and a forward-guarded joint operator.

**Architecture:** Keep the predictor math independent from the training loop. A single benchmark runner owns deterministic data order, exact counters, state snapshots, timing, receipts, and the four matched arms. CI uses a tiny synthetic task; CIFAR-10/CUDA is opt-in from the command line.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, torchvision for the heavy CIFAR-10 benchmark, pytest for CPU tests.

**Spec:** `docs/superpowers/specs/2026-09-12-matched-nowcaster-benchmark-design.md`

## Global Constraints

- Heavy CIFAR-10/GPU benchmark is never required by CI.
- No pretrained forecaster is used.
- Forecasts may use only already-observed parameter states.
- Skipped steps must not call backward.
- Forward-only guard checks are counted separately.
- NaN/Inf is reported as failure and never triggers silent reseeding.
- Wall-clock CUDA timing uses synchronization.

---

### Task 1: Predictor primitives and chronology contract

**Files:**
- Create: `galinear/predictors.py`
- Create: `galinear/state.py`
- Create: `tests/test_predictors.py`

**Interfaces:**
- Produces `flatten_parameters(model) -> Tensor`, `load_flat_parameters(model, flat) -> None`.
- Produces `ScalarLinearPredictor(history: int)` with `predict(states, skip) -> Tensor`.
- Produces `JointOperatorPredictor(history: int, max_rank: int, ridge: float)` with `predict(states, skip) -> Forecast`.
- `Forecast` contains `parameters`, `rank`, and `history_count`.

- [ ] **Step 1: Write failing predictor tests** for exact scalar linear extrapolation, exact low-dimensional affine recurrence recovery, flat-parameter roundtrip, and rejection of too-short histories.
- [ ] **Step 2: Run `pytest -q tests/test_predictors.py`** and confirm failure because production modules do not exist.
- [ ] **Step 3: Implement minimal predictor/state code** using `torch.linalg.lstsq`/SVD and ridge solve in reduced coordinates.
- [ ] **Step 4: Run `pytest -q tests/test_predictors.py`** and require all green.
- [ ] **Step 5: Commit predictor primitives.**

### Task 2: Matched training engine and counters

**Files:**
- Create: `galinear/training.py`
- Create: `tests/test_training_contract.py`

**Interfaces:**
- Produces `RunConfig`, `RunReceipt`, `train_arm(...)`.
- Arm names are exactly `plain`, `scalar_linear`, `joint_operator`, `joint_guarded`.
- Counters include `backward_passes`, `forward_checks`, `accepted_jumps`, `rejected_jumps`, `nominal_skipped_steps`, and `predictor_seconds`.

- [ ] **Step 1: Write failing contract tests** proving a forecast jump does not increment `backward_passes`, a rejected guarded jump restores parameters before the real optimizer step, all arms start from identical serialized model/optimizer states, and nonfinite candidates become visible rejected/failed events.
- [ ] **Step 2: Run `pytest -q tests/test_training_contract.py`** and verify RED.
- [ ] **Step 3: Implement the training engine** with explicit snapshot history and forward-only guard.
- [ ] **Step 4: Run the contract tests** and require GREEN.
- [ ] **Step 5: Commit the matched engine.**

### Task 3: CPU smoke benchmark

**Files:**
- Create: `galinear/tasks.py`
- Create: `scripts/run_smoke.py`
- Create: `tests/test_smoke_benchmark.py`

**Interfaces:**
- Produces deterministic `make_synthetic_task(seed)` returning train/validation tensors and a tiny MLP factory.
- `run_smoke(seed=...) -> dict[str, RunReceipt]` executes all four arms cheaply.

- [ ] **Step 1: Write a failing smoke test** checking all four arms execute, receipts are finite, plain performs only real backwards, and predictor arms expose skip/guard counters.
- [ ] **Step 2: Run `pytest -q tests/test_smoke_benchmark.py`** and verify RED.
- [ ] **Step 3: Implement the synthetic task and smoke runner.**
- [ ] **Step 4: Run `pytest -q`** and require the complete CPU suite green.
- [ ] **Step 5: Commit the smoke benchmark.**

### Task 4: CIFAR-10/CUDA benchmark and receipts

**Files:**
- Create: `scripts/run_cifar10.py`
- Create: `galinear/reporting.py`
- Create: `requirements-benchmark.txt`
- Create: `README.md`

**Interfaces:**
- CLI accepts `--device`, `--seed`, `--epochs`, `--history`, `--skip`, `--max-rank`, `--guard-rel-tol`, `--batch-size`, `--target-accuracy`, `--output-dir`, `--data-dir`.
- Writes timestamped JSON and CSV receipts.

- [ ] **Step 1: Write source-level CLI/report tests** that do not download CIFAR-10 but validate parser defaults, receipt schema, and CSV serialization.
- [ ] **Step 2: Run those tests and verify RED.**
- [ ] **Step 3: Implement VanillaCNN, CIFAR-10 loaders, synchronized timing, CLI, JSON/CSV receipt writer, and README commands.**
- [ ] **Step 4: Run `pytest -q` and `python scripts/run_smoke.py`** locally.
- [ ] **Step 5: Commit the heavy benchmark.**

### Task 5: CI and final verification

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Add CI** for Python 3.11 and 3.12 installing CPU PyTorch plus pytest and running `pytest -q` and `python scripts/run_smoke.py`.
- [ ] **Step 2: Run local full verification** with the same commands.
- [ ] **Step 3: Push branch and verify GitHub Actions green.**
- [ ] **Step 4: Open a PR summarizing exactly what is tested locally and what remains for the user's GPU run.**
