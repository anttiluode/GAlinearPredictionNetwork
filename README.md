# GAlinearPredictionNetwork

A matched, auditable test of a simple question:

> Can an **online joint low-rank model of the current weight trajectory** skip useful chunks of neural-network training more effectively than per-parameter linear curve fitting?

This repo does **not** start by claiming to beat Weight Nowcaster Network (WNN). It builds the comparison needed to find out.

## The four matched arms

All arms start from the same model weights, optimizer recipe, seed, and real-epoch data stream.

| arm | behavior |
|---|---|
| `plain` | ordinary training only |
| `scalar_linear` | fit an independent line to each weight over the last 5 real epochs and predict 5 epochs ahead |
| `joint_operator` | discover the low-rank joint trajectory subspace, fit an affine operator there, and predict 5 epochs ahead |
| `joint_guarded` | same joint predictor, but pay for a forward-only validation check and reject a harmful jump |

A forecast never calls `backward()`. Actual backward passes, predictor time, forward guard checks, accepted/rejected jumps, validation accuracy, and total wall-clock time are counted separately.

## Why this benchmark

Jang et al., **Learning to Boost Training by Periodic Nowcasting Near Future Weights** (ICML 2023), report on CIFAR-10/VanillaCNN that ordinary training reached 59% validation accuracy in 52.82 s, linear curve fitting in 39.71 s (1.33x), and learned WNN in 25.27 s (2.09x) on a TITAN Xp. Their ablation found a 5-epoch history and 5-epoch forecast horizon best among the tested choices.

Paper: https://proceedings.mlr.press/v202/jang23b.html

Public code: https://github.com/jjh6297/WNN

The public CIFAR-10 script uses four 3x3 convolution layers with channels `[8,16,32,32]`, max-pooling after each, a 64-unit dense layer, Adam at `1e-3`, batch size 1024, per-pixel training-mean subtraction, and augmentation with rotation 10 degrees, width/height shift 0.15, and zoom 0.3. This repo translates that recipe to PyTorch and uses the same 5-real-epochs -> forecast-5-epochs periodic schedule.

It is still **not an exact bit-for-bit WNN reproduction**: the original target model is TensorFlow/Keras, augmentation implementations differ slightly, and this repo does not run the learned WNN network itself. The fair direct comparison here is `scalar_linear` versus the new joint predictor on the same PyTorch run. The paper's WNN numbers remain an external target until reproduced under sufficiently matched conditions.

## Fast CPU contract test

```bash
python -m pip install -e '.[test,benchmark]'
python -m pytest -q
python -m scripts.run_smoke
```

The smoke run uses a tiny deterministic synthetic task. It is there to catch cheating and broken counters, not to establish a speed record.

## Main CIFAR-10 GPU test

Install a CUDA-enabled PyTorch build appropriate for your machine, then install this repo without replacing it.

```bash
python -m pip install -e '.[test,benchmark]'
python -m scripts.run_cifar10 --device cuda --seed 0
```

Defaults intentionally follow the public WNN short-term CIFAR setup where practical:

```text
real-epoch cap     50
history             5
forecast horizon    5
batch size        1024
Adam LR          1e-3
target accuracy   0.59
```

The run writes JSON and CSV receipts under `results/`.

For the five-trial comparison used in the WNN paper, run seeds 0 through 4. In PowerShell:

```powershell
0..4 | ForEach-Object { python -m scripts.run_cifar10 --device cuda --seed $_ }
```

## What counts as a win

The primary result is **not** parameter forecast error and not merely fewer backward passes.

For a practical acceleration claim, `joint_guarded` should reach the frozen 59% accuracy threshold in less **measured wall-clock time** than both:

1. `plain`, and
2. `scalar_linear`.

The useful summary is

```text
plain seconds / joint_guarded seconds
```

on the same hardware, same seed, same data recipe.

A result above the WNN paper's 2.09x figure would be interesting, but it is not by itself a direct claim of beating WNN unless the benchmark conditions are close enough. A strong result across five seeds is much more meaningful than a single lucky run.

## Non-cheating rules

- predictors receive only already-observed real-epoch weight states;
- the first forecast waits for five complete real epochs;
- after a forecast, history is cleared and rebuilt from new real training before another forecast;
- forecast jumps never increment the backward counter;
- `joint_guarded` restores the exact pre-jump weights on rejection;
- NaN/Inf proposals are reported and never silently reseeded;
- CUDA synchronization surrounds measured predictor/whole-run timing boundaries;
- all arms are re-seeded so the kth **real** training epoch receives the same shuffle/augmentation stream.

## Core math

Scalar baseline, independently for every weight:

\[
\theta_j(t) \approx a_jt+b_j.
\]

Joint operator: stack the latest weight vectors, center them, compute a small observed SVD basis, and fit

\[
z_{t+1}\approx z_tA+b,
\qquad
\theta_t\approx \bar\theta + Uz_t.
\]

Only the small reduced state is propagated five steps. The full weight vector is reconstructed once at the end.

The experiment therefore tests whether **cross-parameter trajectory structure contains useful predictive information that element-wise curve fitting throws away**.
