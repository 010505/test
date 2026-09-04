# Diffusion inheritance and speed experiments

## Protocol

- Dataset: SHREC'17 official 14-class split, 1,960 training samples and 840
  official-test samples.
- Formal learned-gate experiments: 40 epochs, effective batch size 32,
  micro-batch size 4, seeds 42/43/44 and validation-prefix-AUC checkpoint
  selection.
- All candidates retain the same future-free 25/50/65/80/100% prefixes and
  `stem_agcrn_control` encoder.
- Existing fixed-diffusion checkpoints and results are never overwritten.

## 1. Fixed-inheritance sensitivity

Existing checkpoints trained with gamma=0.50 were evaluated with
`gamma in {0, 0.25, 0.50, 0.75, 1.00}`. This is a post-training sensitivity
screen, not an independently trained comparison.

| Gamma | Prefix AUC | Full accuracy | Decision accuracy | ADR | Final recovery | Correct retention |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 74.04% +/- 0.18% | 79.25% | 78.53% | 0.629 | 61.08% | 92.46% |
| 0.25 | 74.05% +/- 0.18% | 79.33% | 78.41% | 0.610 | 61.17% | 92.53% |
| 0.50 | **74.05% +/- 0.17%** | **79.37%** | 78.41% | **0.609** | 61.08% | 92.67% |
| 0.75 | 74.03% +/- 0.17% | 79.29% | **79.21%** | 0.658 | 60.79% | **92.73%** |
| 1.00 | 73.99% +/- 0.17% | 79.29% | 79.17% | 0.658 | 60.79% | **92.73%** |

AUC changes by only 0.06 pp across the complete range. Fixed gamma is not a
major performance bottleneck, and 0.50 remains the best balanced setting.

## 2. Unsupervised learned inheritance gate

The first learned gate receives the previous posterior, a direct current
evidence distribution, observation ratio, both entropies and their
Jensen-Shannon divergence:

```text
g_u = sigmoid(MLP([p_(u-1), e_u, r_u, H(p_(u-1)), H(e_u), JS]))
pi_u = g_u p_(u-1) + (1 - g_u) U
```

| Model | Prefix AUC | Full accuracy | Decision accuracy | ADR | Final recovery | Correct retention |
|---|---:|---:|---:|---:|---:|---:|
| Fixed gamma=0.50 | **74.05% +/- 0.17%** | **79.37%** | **78.41%** | **0.609** | **61.08%** | **92.67%** |
| Learned gate | 73.40% +/- 0.85% | 78.29% | 78.21% | 0.658 | 59.60% | 92.25% |

The gate collapses to approximately 0.9998 at every update. On samples that
both models misclassify at 25%, its final recovery is 3.86 +/- 1.94 pp lower.
This candidate is retained as a negative ablation and is not selected.

## 3. Reliability-supervised bounded gate

The second gate is bounded to `[0.05, 0.95]` and is trained to estimate the
previous posterior's probability assigned to the true class. It learns a real
sample-dependent policy:

| Observation | Mean gate |
|---:|---:|
| 50% | 0.561 +/- 0.025 |
| 65% | 0.746 +/- 0.016 |
| 80% | 0.781 +/- 0.016 |
| 100% | 0.798 +/- 0.017 |

At 50%, the seed-42 gate averages 0.683 when the previous prediction is correct
and 0.383 when it is wrong. The mechanism therefore works as designed, but the
recognition result remains negative:

| Model | Prefix AUC | Full accuracy | Decision accuracy | ADR | Final recovery | Correct retention |
|---|---:|---:|---:|---:|---:|---:|
| Fixed gamma=0.50 | **74.05% +/- 0.17%** | **79.37%** | 78.41% | **0.609** | **61.08%** | 92.67% |
| Reliability gate | 73.54% +/- 0.83% | 78.45% | **78.41%** | 0.753 | 59.63% | **92.70%** |

On the common 25%-error subset, recovery is 3.24 +/- 2.34 pp lower than the
fixed model. This model is also retained but not selected. The result shows
that learnable inheritance is feasible, but inheritance is not the limiting
component in the current four-step model.

## 4. Reverse-step truncation

The denoiser already predicts the clean class at every reverse step. Therefore
inference can stop after `K` reverse updates and return the exact clean-class
marginal at that point. Four steps reproduce the original reverse chain within
numerical tolerance. No retraining or checkpoint replacement is required.

| Reverse steps | Prefix AUC | Full accuracy | Decision accuracy | ADR | Final recovery | Online/update | Relative latency reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 73.72% +/- 0.10% | 79.09% | 77.78% | **0.602** | 60.21% | **2.637 ms** | **24.8%** |
| 2 | 73.84% +/- 0.13% | 79.05% | 77.86% | 0.607 | 60.04% | 2.910 ms | 17.1% |
| 3 | 73.87% +/- 0.06% | 79.09% | **78.53%** | 0.610 | 60.39% | 3.194 ms | 8.9% |
| 4 | **74.05% +/- 0.17%** | **79.37%** | 78.41% | 0.609 | **61.08%** | 3.508 ms | 0.0% |

Absolute latency varies with GPU load and clock state; the paired reduction in
the same benchmark is the reliable comparison.

## Selection

- Accuracy/research default: retain the original four-step
  `04_class_diffusion` model.
- Conservative deployment: three steps, reducing online latency by 8.9% with
  a 0.18 pp AUC change and no observed decision-accuracy loss.
- Balanced deployment: two steps, reducing online latency by 17.1% for a
  0.21 pp AUC and 0.32 pp full-accuracy change.
- Aggressive deployment: one step, reducing latency by 24.8% for a 0.33 pp AUC
  change.

The recommended deployment setting is two reverse steps. The original
four-step result remains the reported accuracy model.

## Artifacts

- Inheritance sweep: `runs/diffusion_inheritance_sweep/analysis.json`.
- Unsupervised gate: `runs/gated_diffusion_aggregate/analysis.json` and
  `runs/progressive_gated_seed{42,43,44}`.
- Reliability gate: `runs/reliability_gated_diffusion_aggregate/analysis.json`
  and `runs/progressive_reliability_seed{42,43,44}`.
- Step sweep: `runs/diffusion_step_sweep/analysis.json`.

