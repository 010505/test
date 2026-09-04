# GestureGraph Lab complete verified experiment results

## Result scope

This document includes all locally verified formal results currently selected
for comparison. Unless a table explicitly says otherwise, results use the
DD-Net median-filtered SHREC'17 official split with 1960 training and 840 test
sequences, 64 frames, 22 joints and 14 classes.

Three-seed experiments use seeds 42/43/44, 40 epochs, batch size 32, AdamW,
cosine scheduling, label smoothing 0.05, a class-stratified validation split
and best-validation checkpoint selection. Values are official-test accuracy;
standard deviations are sample standard deviations.

Archived pre-eigenvalue fixes, additive GAT, non-edge-only residual adjacency,
pre-QKV-replacement runs and smoke datasets are retained on disk but excluded
from the formal tables. The velocity/time-gating results in the supplied
`MY_IMPROVEMENTS_REPORT.md` were produced in another environment and are not
presented as locally verified results here.

## 1. Foundational backbone ablations

This is the original single-run SHREC'17 diagnostic benchmark.

| Model | Validation | Official test | Delta vs full ST-GCN |
|---|---:|---:|---:|
| Full ST-GCN | 77.75% | **70.95%** | - |
| Flattened MLP | 67.52% | 63.81% | -7.14 pp |
| ST-GCN without graph neighbours | 74.68% | 68.93% | -2.02 pp |
| Single-frame ST-GCN | 37.08% | 32.02% | -38.93 pp |

The temporal signal is the dominant component, while explicit physical graph
structure supplies a smaller but measurable gain.

### Joint-group masking of the full ST-GCN

| Masked group | Accuracy | Drop from 70.95% |
|---|---:|---:|
| Wrist and palm | 60.71% | 10.24 pp |
| Thumb | 64.29% | 6.67 pp |
| Index finger | 67.86% | 3.10 pp |
| Middle finger | 71.79% | -0.83 pp |
| Ring finger | 71.55% | -0.60 pp |
| Little finger | 70.24% | 0.71 pp |

## 2. Original unified spectral-PE backbone comparison

These models inject `Z_l = X_l + (SE diag(alpha)) W_pe,l` directly at every
spatial layer. Delta is paired against the same-seed 3-channel ST-GCN control.

| Model | Parameters | Seed 42 | Seed 43 | Seed 44 | Mean +/- std | Paired delta |
|---|---:|---:|---:|---:|---:|---:|
| ST-GCN control | 208,978 | 70.60% | 68.21% | 69.17% | 69.33% +/- 1.20% | - |
| Unified PE + ST-GCN | 210,538 | 67.14% | 69.52% | 69.64% | 68.77% +/- 1.41% | -0.56 +/- 2.54 pp |
| Unified PE + QKV | 268,286 | 60.24% | 73.33% | 71.55% | 68.37% +/- 7.10% | -0.95 +/- 8.26 pp |
| Unified PE + Graph WaveNet | 282,703 | 70.71% | 72.26% | 74.17% | 72.38% +/- 1.73% | +3.06 +/- 2.59 pp |
| Unified PE + joint AGCRN W_v | 930,178 | 80.95% | 80.71% | 80.00% | **80.56% +/- 0.50%** | **+11.23 +/- 1.13 pp** |

The original QKV is unstable. Joint AGCRN improves all three seeds and is the
lowest-variance 80%+ model in the verified experiments.

## 3. Stem-first semantic-SE ablation

All models first lift XYZ from 3 to 32 channels. SE uses the first K nontrivial
normalised-Laplacian eigenvectors and their eigenvalues.

| Model | Parameters | Seed 42 | Seed 43 | Seed 44 | Mean +/- std | Delta vs stem |
|---|---:|---:|---:|---:|---:|---:|
| Stem ST-GCN control | 217,330 | 70.83% | 70.83% | 72.02% | 71.23% +/- 0.69% | - |
| Linear SE K=8 | 219,126 | 69.88% | 71.07% | 71.31% | 70.75% +/- 0.77% | -0.48 +/- 0.63 pp |
| Linear SE K=16 | 220,918 | 69.88% | 70.60% | 69.52% | 70.00% +/- 0.55% | -1.23 +/- 1.16 pp |
| Linear SE K=21 | 222,038 | 72.14% | 69.88% | 71.19% | 71.07% +/- 1.14% | -0.16 +/- 1.27 pp |
| Residual-MLP SE K=21 | 242,010 | 74.76% | 72.26% | 74.29% | **73.77% +/- 1.33%** | **+2.54 +/- 1.27 pp** |
| Residual-MLP + spectral gate | 242,094 | 72.62% | 71.79% | 74.76% | 73.06% +/- 1.54% | +1.83 +/- 0.89 pp |

Selected SE design: K=21 fixed eigenvalue weighting, direct linear path plus a
residual two-layer ReLU MLP. The learnable spectral gate is functional but does
not improve the three-seed mean.

## 4. Stem-first operator transfer using joint support aggregation

QKV uses full-joint four-head attention. GWN and AGCRN use the common joint
diffusion representation:

```text
H = [X, A_phys X, A_phys^2 X, A_dyn X, A_dyn^2 X]
```

GWN applies shared W; AGCRN generates W_v for every joint. The final GWN/AGCRN
values below were independently rerun and exactly reproduce the preserved v1
aggregate.

| Model | Parameters | Seed 42 | Seed 43 | Seed 44 | Mean +/- std | Paired SE delta |
|---|---:|---:|---:|---:|---:|---:|
| QKV, no SE | 278,790 | 74.05% | 79.05% | 76.79% | 76.63% +/- 2.50% | - |
| QKV + semantic SE | 303,470 | 80.24% | 81.55% | 81.19% | **80.99% +/- 0.68%** | **+4.37 +/- 1.85 pp** |
| Joint GWN, no SE | 294,570 | 72.14% | 73.21% | 74.76% | 73.37% +/- 1.32% | - |
| Joint GWN + semantic SE | 319,250 | 74.05% | 75.00% | 75.12% | 74.72% +/- 0.59% | +1.35 +/- 0.86 pp |
| Joint AGCRN, no SE | 988,010 | 81.19% | **83.93%** | 80.83% | **81.98% +/- 1.69%** | - |
| Joint AGCRN + semantic SE | 1,012,690 | 79.17% | 82.50% | 79.17% | 80.28% +/- 1.92% | -1.71 +/- 0.30 pp |

AGCRN minus matched GWN is +8.61 pp without SE and +5.56 pp with SE. The
absolute-performance selection is joint AGCRN without SE. The selected
attention model is QKV + semantic SE.

## 5. Stable QKV and gated dual-branch structural ablation

Stable QKV adds pre-QKV LayerNorm, positive learned per-head temperatures and a
unit-initialised attention residual gain. Gated GWN/AGCRN use separate physical
and adaptive branches with complementary fusion. Only the adaptive AGCRN branch
has W_v.

| Model | Parameters | Seed 42 | Seed 43 | Seed 44 | Mean +/- std | Paired SE delta |
|---|---:|---:|---:|---:|---:|---:|
| Stable QKV, no SE | 279,254 | 74.17% | 75.60% | 74.05% | 74.60% +/- 0.86% | - |
| Stable QKV + semantic SE | 303,934 | 74.29% | 77.62% | 78.69% | 76.87% +/- 2.30% | +2.26 +/- 2.27 pp |
| Gated GWN, no SE | 309,934 | 72.62% | 72.62% | 74.05% | 73.10% +/- 0.82% | - |
| Gated GWN + semantic SE | 334,614 | 73.93% | 73.21% | 74.52% | 73.89% +/- 0.66% | +0.79 +/- 0.45 pp |
| Dynamic-only AGCRN, no SE | 726,894 | 78.33% | 77.14% | 78.81% | 78.10% +/- 0.86% | - |
| Dynamic-only AGCRN + semantic SE | 751,574 | 80.60% | 78.33% | 80.71% | **79.88% +/- 1.34%** | **+1.79 +/- 0.55 pp** |

Stable QKV reduces variation for the no-SE control but loses 2.02 pp in mean.
It does not transfer successfully to the already stable semantic-SE QKV, which
loses 4.13 pp and increases its standard deviation. It is retained as a failed
but complete ablation, not as the selected attention model.

Dynamic-only AGCRN exceeds its structurally matched gated GWN by +5.00 pp
without SE and +5.99 pp with SE. Learned adaptive-branch gates average about
15-16%, confirming that the dynamic residual is active but conservative.

## 6. Pure QKV removal of the physical soft bias

This strict ablation keeps the same stem, four attention heads and residual
scale, but changes the score from
`QK^T/sqrt(d) + b_head A_physical` to exactly `QK^T/sqrt(d)`. Semantic SE is
still added to X before the Q/K/V projections.

| Model | Parameters | Seed 42 | Seed 43 | Seed 44 | Mean +/- std | Paired SE delta |
|---|---:|---:|---:|---:|---:|---:|
| Pure QKV, no SE | 278,774 | 73.57% | 77.74% | 76.90% | 76.07% +/- 2.20% | - |
| Pure QKV + semantic SE | 303,454 | 77.98% | 79.76% | 81.07% | **79.60% +/- 1.55%** | **+3.53 +/- 1.31 pp** |

Semantic SE improves all three pure-QKV seeds, confirming that the selected
node encoding transfers to unconstrained attention. However, removing the
physical soft bias loses 0.56 pp without SE and 1.39 pp with SE relative to the
matched biased QKV. The pure variant is retained as a completed ablation; it
does not replace `stem_semantic_qkv`.

## 7. Joint aggregation versus gated dual branches

| Joint aggregation minus gated v2 | No semantic SE | With semantic SE |
|---|---:|---:|
| Graph WaveNet | +0.28 pp | +0.83 pp |
| AGCRN | **+3.89 pp** | +0.40 pp |

Joint aggregation has little effect on shared-W GWN. It substantially improves
AGCRN because node-specific W_v also transforms reliable physical-graph
features. This recovers the 80%+ result but makes spectral node identity more
redundant, explaining the negative SE delta.

## 8. Progressive causal-prefix and class-diffusion recognition

All five models use the selected joint-support AGCRN encoder. Causal inputs are
constructed before resampling, never access future frames and use mask-aware
pooling. Prefix AUC integrates official-test accuracy over 25/50/65/80/100%
observation.

| Model | 25% | 50% | 65% | 80% | 100% | Prefix AUC | ADR | Decision accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-sequence baseline | 34.05% | 61.47% | 71.71% | 78.37% | **80.16%** | 65.38% +/- 1.54% | 1.000 | 80.16% |
| Mild temporal crop | 28.65% | 59.05% | 70.08% | 75.40% | 76.55% | 62.34% +/- 1.05% | 1.000 | 76.55% |
| Causal prefix | 55.08% | 72.70% | 76.98% | 77.74% | 76.55% | 72.31% +/- 1.99% | 0.598 | 75.48% |
| GRU evidence | 55.99% | 71.59% | 75.32% | 76.79% | 76.55% | 71.61% +/- 0.52% | 0.597 | 75.16% |
| Four-step class diffusion | **57.90%** | **73.77%** | **78.17%** | **79.33%** | 79.37% | **74.05% +/- 0.17%** | **0.609** | **78.41%** |

Class diffusion improves prefix AUC by 8.67 +/- 1.41 pp over the full-sequence
baseline while changing 100% accuracy by -0.79 +/- 1.94 pp. Against the direct
causal-prefix control, it improves AUC by 1.74 +/- 2.07 pp, full accuracy by
2.82 +/- 2.45 pp and calibrated decision accuracy by 2.94 +/- 0.84 pp. The
selected closed-set early-recognition model is `04_class_diffusion`. Unknown
rejection remains unverified because SHREC'17 supplies only the 14 known labels.

An additional official-test trajectory audit conditions on samples misclassified
at 25%. Diffusion recovers 61.08% +/- 0.71% of its initial errors versus
53.20% +/- 0.39% for GRU. On the stricter paired subset that both models get
wrong at 25%, diffusion recovers 57.34% +/- 2.20% and GRU 47.34% +/- 2.59%
(+10.00 +/- 3.27 pp). Batch-one state-preserving RTX 3090 latency is 8.087 ms
per diffusion update versus 5.310 ms for GRU (1.52x); calibrated early-exit
compute is estimated at 21.787 ms versus 13.874 ms. Full trajectories and
protocol are stored in `runs/progressive_recovery/analysis.json`.

Inheritance optimization was also tested. An unconstrained learned gate
collapses to full inheritance and reaches 73.40% +/- 0.85% prefix AUC. A
bounded reliability-supervised gate learns nontrivial values but reaches
73.54% +/- 0.83%; both are below the fixed-gamma result and remain negative
ablations. Without retraining, two reverse steps retain 73.84% +/- 0.13% AUC
and 79.05% full accuracy while reducing paired online latency by 17.1%. Four
steps remain the accuracy result; two steps are selected for balanced
deployment.

## Final selections

| Objective | Selected model | Result |
|---|---|---:|
| Highest three-seed mean | Joint-support stem AGCRN, no SE | **81.98% +/- 1.69%** |
| Lowest variance among 80%+ models | Original unified-PE AGCRN | **80.56% +/- 0.50%** |
| Best attention model | Stem semantic-SE QKV | **80.99% +/- 0.68%** |
| Best isolated semantic-SE gain | Stem residual-MLP SE vs stem ST-GCN | **+2.54 +/- 1.27 pp** |
| Strict dynamic-branch AGCRN interpretation | Gated dynamic-only AGCRN + SE | **79.88% +/- 1.34%** |
| Best closed-set early recognition | Four-step class diffusion over AGCRN | **74.05% +/- 0.17% prefix AUC; ADR 0.609** |

The final accuracy-oriented backbone is `stem_agcrn_control`. The final
attention-oriented backbone is `stem_semantic_qkv`. Stable QKV and gated
dual-branch models remain reproducible structural ablations.

## Verification and artifact locations

- Current standard checks: environment 9/9, JavaScript 8/8, Python 35/35.
- Stem SE formal audit: 18 checkpoints with complete 40-epoch histories.
- Operator-transfer audit: 18 checkpoints with complete 40-epoch histories.
- Stable/gated v2 audit: 18 checkpoints with complete 40-epoch histories.
- Fresh joint-aggregation recheck: 12 checkpoints; all reload to finite `[1,14]` logits.
- Pure-QKV audit: 6 checkpoints with complete 40-epoch histories.
- Progressive recognition audit: 15 checkpoints with complete 40-epoch histories.
- Learned-inheritance audit: 6 checkpoints with complete 40-epoch histories;
  all reload to finite `[1,5,14]` outputs.
- Foundational report: `runs/shrec17_benchmark/REPORT.md`.
- Original backbone aggregate: `runs/improvement_backbones_ddnet_aggregate/aggregate.json`.
- Stem SE aggregate: `runs/se_semantic_ablation_aggregate/aggregate.json`.
- Joint aggregation recheck: `runs/joint_aggregation_recheck_aggregate/aggregate.json`.
- Stable/gated v2 aggregate: `runs/se_transfer_v2_aggregate/aggregate.json`.
- Pure-QKV aggregate: `runs/pure_qkv_aggregate/aggregate.json`.
- Progressive aggregate: `runs/progressive_aggregate/aggregate.json`.
- Progressive recovery and online latency: `runs/progressive_recovery/analysis.json`.
- Diffusion inheritance/speed optimization: `DIFFUSION_INHERITANCE_AND_SPEED_EXPERIMENTS.md`.
- Complete definitions: `SE_SEMANTIC_EXPERIMENTS.md`,
  `SE_TRANSFER_V2_EXPERIMENTS.md`, `JOINT_AGGREGATION_RECHECK.md` and
  `PURE_QKV_EXPERIMENT.md` and `PROGRESSIVE_CLASS_DIFFUSION_EXPERIMENT.md`.
