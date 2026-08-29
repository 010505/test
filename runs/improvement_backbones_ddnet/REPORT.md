# GestureGraph independent backbone experiments

All experiments use the official SHREC'17 split, the same seed, frame count, optimiser, augmentation, and epoch budget.
Dataset mode: `shrec17_npz`.

| Experiment | Backbone | Parameters | Best validation | Official test | Delta vs ST-GCN |
|---|---|---:|---:|---:|---:|
| Baseline | stgcn | 208,978 | 80.05% | 70.60% | 0.00 pp |
| 01_spectral_pe | spectral_pe_stgcn | 210,538 | 78.01% | 67.14% | -3.45 pp |
| 02_qkv_spatial_attention | spectral_pe_qkv | 268,286 | 67.77% | 60.24% | -10.36 pp |
| 03_gwnet_adaptive_support | gwnet_adaptive_support | 282,703 | 76.47% | 70.71% | +0.12 pp |
| 04_agcrn_factorized_adjacency | agcrn_factorized_adjacency | 930,178 | 86.96% | 80.95% | +10.36 pp |

Experiment 3 stores the fixed physical and full learned adaptive supports in 
`03_gwnet_adaptive_support/adjacency_matrices.npz` for reproducible visualisation.

This run uses the DD-Net copy of the official 1960/840 skeleton split.
DD-Net median-filtered every coordinate channel before serialisation; the
same-run ST-GCN control is therefore the only valid baseline for deltas.

## Per-class official-test accuracy

| Class | 00_stgcn_control | 01_spectral_pe | 02_qkv_spatial_attention | 03_gwnet_adaptive_support | 04_agcrn_factorized_adjacency |
|---|---:|---:|---:|---:|---:|
| expand | 55.7% | 44.3% | 75.4% | 80.3% | 72.1% |
| grab | 67.2% | 74.1% | 51.7% | 72.4% | 87.9% |
| pinch | 78.2% | 69.1% | 80.0% | 92.7% | 92.7% |
| rotation_ccw | 89.1% | 85.5% | 96.4% | 90.9% | 94.5% |
| rotation_cw | 68.6% | 68.6% | 80.4% | 84.3% | 86.3% |
| shake | 83.6% | 79.5% | 97.3% | 78.1% | 91.8% |
| swipe_down | 65.6% | 55.7% | 50.8% | 32.8% | 67.2% |
| swipe_left | 68.5% | 72.2% | 59.3% | 72.2% | 81.5% |
| swipe_plus | 60.3% | 60.3% | 5.2% | 51.7% | 82.8% |
| swipe_right | 69.4% | 72.6% | 72.6% | 67.7% | 80.6% |
| swipe_up | 57.4% | 47.1% | 41.2% | 67.6% | 70.6% |
| swipe_v | 80.7% | 82.5% | 38.6% | 70.2% | 89.5% |
| swipe_x | 53.6% | 40.6% | 5.8% | 43.5% | 47.8% |
| tap | 94.8% | 96.6% | 96.6% | 94.8% | 96.6% |
