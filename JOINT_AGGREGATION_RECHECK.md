# Joint-support Graph WaveNet and AGCRN recheck

## Definition

This recheck restores the original joint-support aggregation for both models.
They use the same learned Graph WaveNet support and second-order diffusion:

```text
A_dyn = softmax(relu(E1 E2^T))
H = [X, A_phys X, A_phys^2 X, A_dyn X, A_dyn^2 X]
```

Graph WaveNet uses one node-shared projection over the complete representation:

```text
Y_v = H_v W
```

AGCRN changes only that projection to a node-specific weight generated from a
shared pool:

```text
W_v = sum_k E1[v,k] W_pool[k]
Y_v = H_v W_v + b_v
```

There is no static/dynamic branch gate and no node embedding is added to X.
Unlike gated v2, AGCRN node-specific weights operate on physical and adaptive
diffusion features together.

## Protocol

- DD-Net SHREC'17 official 1960/840 split; 20% stratified validation from train.
- 64 frames, 22 joints, XYZ; stem lifts 3 channels to 32 before spatial blocks.
- 40 epochs, batch 32, AdamW, cosine schedule and label smoothing 0.05.
- Seeds 42, 43 and 44; best-validation checkpoint evaluated on the official test.
- Semantic SE uses K=21 fixed eigenvalue weighting plus residual MLP, when enabled.

## Three-seed official-test results

| Model | Seed 42 | Seed 43 | Seed 44 | Mean +/- std | SE delta |
|---|---:|---:|---:|---:|---:|
| Joint GWN, no SE | 72.14% | 73.21% | 74.76% | 73.37% +/- 1.32% | - |
| Joint GWN + SE | 74.05% | 75.00% | 75.12% | 74.72% +/- 0.59% | +1.35 +/- 0.86 pp |
| Joint AGCRN, no SE | 81.19% | **83.93%** | 80.83% | **81.98% +/- 1.69%** | - |
| Joint AGCRN + SE | 79.17% | 82.50% | 79.17% | 80.28% +/- 1.92% | -1.71 +/- 0.30 pp |

AGCRN improves over the matched GWN by +8.61 pp without SE and +5.56 pp with
SE. A completely fresh rerun reproduces the preserved v1 aggregate exactly.

## Comparison with gated dual branches

| Joint aggregation minus gated v2 | No semantic SE | With semantic SE |
|---|---:|---:|
| Graph WaveNet | +0.28 pp | +0.83 pp |
| AGCRN | +3.89 pp | +0.40 pp |

Joint aggregation has little effect on GWN because both variants retain shared
weights. Its main benefit is in AGCRN: W_v can adapt not only latent dynamic
connections but also the reliable physical-graph features. This explains the
return to 80%+ accuracy. The no-SE joint AGCRN is selected for absolute
performance. Semantic SE is not selected for this model because it loses in all
three seeds, consistent with redundant node identity information.

## Reproduction and artifacts

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$python = 'E:\Anaconda_envs\envs\test_transform2\python.exe'

foreach ($seed in 42,43,44) {
  & $python -m gesturegraph.se_transfer_benchmark `
    --dataset shrec17_npz --data data\shrec17_ddnet_npz `
    --output "runs\joint_aggregation_recheck_seed$seed" `
    --epochs 40 --batch-size 32 --frames 64 --seed $seed --device cuda `
    --pe-dim 21 --stem-channels 32 --semantic-hidden 64 --adaptive-dim 10 `
    --only 02_stem_gwnet_control 03_stem_semantic_gwnet `
           04_stem_agcrn_control 05_stem_semantic_agcrn
}

& $python -m gesturegraph.aggregate_joint_aggregation `
  --runs runs\joint_aggregation_recheck_seed42 `
         runs\joint_aggregation_recheck_seed43 `
         runs\joint_aggregation_recheck_seed44 `
  --output runs\joint_aggregation_recheck_aggregate `
  --gated-aggregate runs\se_transfer_v2_aggregate\aggregate.json
```

- Formal artifacts: 3 run roots, 12 checkpoints and complete 40-epoch histories.
- Reload audit: all 12 checkpoints return finite `[1,14]` logits.
- Aggregate: `runs/joint_aggregation_recheck_aggregate/aggregate.json`.
- GWN heatmap: `gwnet_joint_seed44_adjacency.{png,svg,pdf}` in the aggregate directory.
- AGCRN heatmap: `agcrn_joint_seed43_adjacency.{png,svg,pdf}` in the aggregate directory.

