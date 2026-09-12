# GAlinearPredictionNetwork

> **Conclusion: trajectory forecasting is a useful negative control, not the main architecture.**

This repository began as a matched test of whether recent weight history could be used to fast-forward neural-network training. The first real CUDA run gave a clean answer for the simple per-parameter linear nowcaster: **it made training worse, not faster.**

## Frozen seed-0 result

Command:

```powershell
python3.13 -m scripts.run_cifar10 --device cuda --seed 0
```

| arm | reached 59% | real epochs | backprops | forecast jumps | accuracy at stop | wall time |
|---|---:|---:|---:|---:|---:|---:|
| `plain` | yes | **30** | **1440** | 0 | 0.5907 | **788.495 s** |
| `scalar_linear` | yes | 40 | 1920 | 7 | 0.5948 | 1071.952 s |

Relative to plain training, the scalar nowcaster required:

- **1.36x the wall time** — about **36% slower**;
- **1.33x the backward passes** — 1920 instead of 1440;
- ten additional real training epochs even after seven forecast jumps.

The run was stopped when `joint_operator` began. **No result is claimed for `joint_operator` or `joint_guarded`.**

This is a one-seed result on one benchmark. It does **not** disprove Weight Nowcaster Network, and this repository does not reproduce WNN's learned forecaster. It does establish the thing we needed to know locally: **blindly extending a recent parameter trajectory is not automatically a training accelerator, even when the forecast itself is cheap enough to try.**

## Why keep this repository?

Because the failure separates two questions that looked similar earlier in the day:

### Forecasting

> Where is the optimizer likely to go next?

A trajectory model tries to infer

\[
\theta_{t+h} \approx F(\theta_{t-k:t}).
\]

That can be useful when training dynamics are smooth, but it treats the recent path as if it were one coherent object to extrapolate.

### Compatibility

> How does a proposed change collide with computations that must remain viable?

That is the stronger question exposed by `ThirdWay`. Instead of replacing the optimizer with a predictor, let the optimizer propose a change

\[
\Delta_t,
\]

then decompose what the existing structure can safely absorb from what remains as signed geometric debt:

\[
\Delta_t = \Delta_t^{\mathrm{compatible}} + r_t.
\]

Repeated coherent residual is then evidence for **structural growth / route separation**, not evidence that the system should average conflicting trajectories.

That is the conceptual boundary this repository now records:

```text
trajectory forecast
    predicts where weights may go

compatibility mechanism
    decides how a proposed change may safely become structure
```

The second problem is the one carried forward into `ThirdWay`.

## The lethal-average test this repo points toward

The next useful experiment is not a better nowcaster. Construct two viable computations, A and B, such that both work individually but their parameter-space average falls into dead space:

\[
W_A \text{ works},\qquad W_B \text{ works},\qquad
\frac{W_A+W_B}{2} \text{ fails}.
\]

Then feed alternating or conflicting useful update directions to three systems:

1. direct shared update;
2. trajectory prediction / averaging;
3. ThirdWay compatibility decomposition with signed residual and structural growth.

The question becomes whether the third system recognizes that the apparent temporal fluctuation is actually **two incompatible demands on shared structure** and grows/separates rather than averaging them away.

That experiment belongs in `ThirdWay`, not here.

## What is still in this repository

The code remains as an auditable benchmark/negative control with four arms:

| arm | behavior |
|---|---|
| `plain` | ordinary training only |
| `scalar_linear` | independent linear fit to each parameter over recent real epochs |
| `joint_operator` | reduced joint trajectory operator |
| `joint_guarded` | joint operator with a counted forward-only validation guard |

The anti-cheating contracts remain useful:

- predictors only see already-observed real training states;
- forecast jumps never call `backward()`;
- forecast-generated states are not fed back as observed history;
- rejected guarded jumps restore the exact pre-jump parameters;
- NaN/Inf is visible and never silently reseeded;
- backward passes, guard checks, predictor time, jumps, validation metrics, and wall time are counted separately.

## Reproduce the CPU contracts

```bash
python -m pip install -e '.[test,benchmark]'
python -m pytest -q
python -m scripts.run_smoke
```

The smoke benchmark is only a chronology/counter test. It is not evidence for acceleration.

## Historical motivation

Jang et al., *Learning to Boost Training by Periodic Nowcasting Near Future Weights* (ICML 2023), showed that a learned Weight Nowcaster Network can periodically forecast future weights and accelerate some training runs. This repo originally translated their small CIFAR-10 target-model recipe to PyTorch and compared simple per-coordinate extrapolation with an online joint reduced operator.

Paper: https://proceedings.mlr.press/v202/jang23b.html

Public code: https://github.com/jjh6297/WNN

The frozen negative result above is **not** a reproduction or refutation of WNN. It is the result for this repo's simple scalar linear control.

## Bottom line

\[
\boxed{\text{Prediction is not preservation.}}
\]

The seed-0 run made that distinction concrete. The nowcaster asked where the training trajectory was going and lost time. The architectural work continues with the different question: **which part of a proposed update is compatible with the computations already present, and what should happen to the incompatible residual?**

A static summary suitable for GitHub Pages lives at [`docs/index.html`](docs/index.html).
