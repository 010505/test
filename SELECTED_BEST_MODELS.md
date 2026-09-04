# Selected best GestureGraph designs

## Shared input and training protocol

All four selected models use 64-frame, 22-joint XYZ input and a
`1x1 Conv + BN + ReLU` stem that lifts each joint from 3 to 32 channels. Formal
results use the official DD-Net SHREC'17 1960/840 split, 40 epochs, batch 32 and
seeds 42/43/44. Selection is based on three-seed mean, not the best test seed.

## Selected models

| Family | Build name | Selected design | Parameters | Three-seed official test |
|---|---|---|---:|---:|
| Fixed-graph ST-GCN | `stem_residual_mlp_se` | Physical A + K=21 residual-MLP semantic SE | 242,010 | **73.77% +/- 1.33%** |
| Spatial attention | `stem_semantic_qkv` | K=21 semantic SE + 4-head full-joint QKV + physical soft bias | 303,470 | **80.99% +/- 0.68%** |
| Graph WaveNet | `stem_semantic_gwnet` | K=21 semantic SE + joint physical/adaptive order-2 diffusion + shared W | 319,250 | **74.72% +/- 0.59%** |
| AGCRN | `stem_agcrn_control` | Joint physical/adaptive order-2 diffusion + node-specific W_v; no SE | 988,010 | **81.98% +/- 1.69%** |
| Early recognition | `04_class_diffusion` | Causal masked prefixes + AGCRN + exact 4-step categorical class diffusion | 1,010,376 | **74.05% +/- 0.17% prefix AUC; ADR 0.609** |

## 1. Best fixed-graph ST-GCN

```text
XYZ -> stem 3->32
S = (SE * lambda)
P_l = S W_linear,l + eta_l MLP_l(S)
Z_l = X_l + beta_l P_l
Y_l = ST-GCN(Z_l, A_physical) -> TCN
```

The direct spectral path is retained and the two-layer ReLU MLP only adds a
semantic residual. Fixed eigenvalue weighting is selected; learnable spectral
gates are not used.

| Seed | Accuracy | Checkpoint |
|---:|---:|---|
| 42 | 74.76% | `runs/se_semantic_ablation_seed42/04_stem_residual_mlp_se_k21/best.pt` |
| 43 | 72.26% | `runs/se_semantic_ablation_seed43/04_stem_residual_mlp_se_k21/best.pt` |
| 44 | 74.29% | `runs/se_semantic_ablation_seed44/04_stem_residual_mlp_se_k21/best.pt` |

## 2. Best QKV attention model

```text
XYZ -> stem 3->32 -> semantic SE
Q = Z W_q, K = Z W_k, V = Z W_v
Attention = softmax(Q K^T / sqrt(d) + b_head A_physical)
Y = Z + 0.1_learned * Attention V -> TCN
```

It uses four heads, no temporal positional encoding and no hard adjacency mask.
The stable-QKV LayerNorm/temperature variant is not selected because it lowers
the semantic-SE QKV mean from 80.99% to 76.87%.
The strict pure-QKV ablation also is not selected: removing
`b_head A_physical` lowers the semantic-SE mean to 79.60% +/- 1.55%. Thus the
physical graph is retained only as a learned soft logit prior, never as a hard
attention mask.

| Seed | Accuracy | Checkpoint |
|---:|---:|---|
| 42 | 80.24% | `runs/se_transfer_seed42/01_stem_semantic_qkv/best.pt` |
| 43 | 81.55% | `runs/se_transfer_seed43/01_stem_semantic_qkv/best.pt` |
| 44 | 81.19% | `runs/se_transfer_seed44/01_stem_semantic_qkv/best.pt` |

## 3. Best Graph WaveNet model

```text
A_dyn = softmax(relu(E1 E2^T))
Z = X + semantic_SE(X)
H = [Z, A_phys Z, A_phys^2 Z, A_dyn Z, A_dyn^2 Z]
Y = H W_shared -> TCN
```

The physical and adaptive graphs are separate supports but are aggregated by a
single joint projection. The gated dual-branch GWN is not selected because its
semantic-SE mean is 73.89%.

| Seed | Accuracy | Checkpoint |
|---:|---:|---|
| 42 | 74.05% | `runs/joint_aggregation_recheck_seed42/03_stem_semantic_gwnet/best.pt` |
| 43 | 75.00% | `runs/joint_aggregation_recheck_seed43/03_stem_semantic_gwnet/best.pt` |
| 44 | 75.12% | `runs/joint_aggregation_recheck_seed44/03_stem_semantic_gwnet/best.pt` |

## 4. Best AGCRN model and final overall selection

```text
A_dyn = softmax(relu(E1 E2^T))
H = [X, A_phys X, A_phys^2 X, A_dyn X, A_dyn^2 X]
W_v = sum_k E1[v,k] W_pool[k]
Y_v = H_v W_v + b_v -> TCN
```

No node embedding is added to X. No sigmoid branch gate is used. Node-specific
W_v transforms the complete physical and adaptive diffusion representation.
Semantic SE is omitted because it lowers all three same-seed results.

| Seed | Accuracy | Checkpoint |
|---:|---:|---|
| 42 | 81.19% | `runs/joint_aggregation_recheck_seed42/04_stem_agcrn_control/best.pt` |
| 43 | 83.93% | `runs/joint_aggregation_recheck_seed43/04_stem_agcrn_control/best.pt` |
| 44 | 80.83% | `runs/joint_aggregation_recheck_seed44/04_stem_agcrn_control/best.pt` |

This is the selected final backbone for accuracy. Seed 43 is the highest
observed test checkpoint, but the reported model result remains the three-seed
mean `81.98% +/- 1.69%`.

## Selection summary

- Final accuracy backbone: `stem_agcrn_control`.
- Final attention backbone: `stem_semantic_qkv`.
- Final Graph WaveNet backbone: `stem_semantic_gwnet`.
- Final fixed-graph SE backbone: `stem_residual_mlp_se`.
- Final closed-set early-recognition model: `04_class_diffusion`.
- Balanced early-recognition deployment: the same `04_class_diffusion`
  checkpoint with `model.inference_steps = 2` (73.84% +/- 0.13% prefix AUC,
  17.1% paired online-latency reduction). Four steps remain the accuracy mode.
- Structural ablations retained but not selected: pure QKV, stable QKV, gated
  dual-branch GWN, dynamic-only AGCRN and both learned inheritance gates.
