# Matched Nowcaster Benchmark Design

## Goal

Test whether an online low-rank joint trajectory predictor can beat a WNN-style per-parameter linear nowcaster and ordinary training on the same training run, with no pretrained forecaster.

## Scientific question

For a fixed model, optimizer, data order, initialization, history length, and skip schedule, compare four arms:

1. `plain` — ordinary optimizer steps only.
2. `scalar_linear` — per-parameter linear extrapolation from recent history, analogous to the simple linear curve-fitting baseline discussed alongside Weight Nowcaster Network.
3. `joint_operator` — fit one reduced affine operator on the observed joint parameter trajectory and jump in its learned low-dimensional subspace.
4. `joint_guarded` — the same joint operator, but commit a jump only when a forward-only validation check does not worsen the current validation loss beyond a frozen tolerance.

The benchmark must not claim to reproduce the full learned WNN model. It is a matched comparison against a WNN-style linear baseline plus the proposed GAx-derived online joint operator.

## Benchmark task

Primary benchmark: CIFAR-10 image classification with a small VanillaCNN implemented in PyTorch. The heavy benchmark is intended for a local CUDA machine and is not run in CI.

CI/smoke benchmark: a tiny synthetic classification dataset and tiny MLP, exercising the exact same predictor and training-loop interfaces without network access or GPU.

## Fairness rules

- All arms start from byte-identical model and optimizer states.
- All arms consume the same deterministic batch schedule for real optimizer steps.
- A skipped optimizer step does not consume a gradient or increment the backward-pass counter.
- Forecast fitting uses only parameter states already observed in that arm.
- The guarded arm may perform forward-only validation checks; these are counted separately.
- Target metrics are wall-clock seconds to a frozen validation-accuracy threshold, backward passes to threshold, forward-only checks, skip acceptance/rejection counts, final accuracy, and final loss.
- CUDA timing must synchronize before and after measured regions.
- The benchmark records predictor-fit time separately from training time.
- No arm may silently retry with a different seed.
- NaN/Inf is a visible failed run, not a reseed trigger.

## Predictors

### Scalar linear baseline

For each flattened parameter coordinate independently, fit

`theta_j(t) = a_j * t + b_j`

by least squares over the latest `history` states and evaluate at `t + skip`.

This intentionally represents a cheap per-coordinate curve-fit baseline, not the learned WNN network.

### Joint low-rank operator

Given the latest `history` flattened parameter vectors, center them, compute a truncated SVD basis `U`, project snapshots to reduced coordinates `z_t`, and fit

`z_{t+1} = A z_t + b`

by ridge-regularized least squares. Apply the reduced affine map repeatedly for `skip` steps and reconstruct the predicted full parameter vector.

The retained rank is bounded by `max_rank` and the numerical rank of the centered history.

### Guard

Before committing a predicted jump, load the candidate parameters into the model and evaluate validation loss without gradients. Accept when

`candidate_loss <= current_reference_loss * (1 + guard_rel_tol)`.

If rejected, restore the pre-jump parameters and take one real optimizer step. Forward checks are counted.

## Schedule

Default heavy run uses a warm-up/history period of real training followed by periodic forecast opportunities. The default comparison should start with `history=5`, `skip=5`, matching the short-history/short-skip regime commonly associated with weight-nowcasting experiments. CLI flags expose these values without changing source.

## Outputs

Every run writes a JSON receipt containing configuration, environment, device, seed, per-arm counters, wall-clock timings, accuracy/loss trajectories, and a verdict section. A CSV summary is also written for easy comparison.

The program prints the exact command and receipt path so a result can be reproduced.

## Claim boundary

A win is earned only if `joint_guarded` reaches the same frozen validation-accuracy threshold faster in wall-clock time than both `plain` and `scalar_linear` on the same benchmark configuration. Fewer backward passes alone is interesting but is not sufficient for a practical speedup claim.

This repository does not claim to beat the learned WNN or NiNo systems unless their published benchmark conditions are independently reproduced closely enough for a defensible direct comparison.