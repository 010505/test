# GestureGraph independent backbone experiments

All experiments use the official SHREC'17 split, the same seed, frame count, optimiser, augmentation, and epoch budget.
Dataset mode: `shrec17_npz`.

| Experiment | Backbone | Parameters | Best validation | Official test | Delta vs ST-GCN |
|---|---|---:|---:|---:|---:|
| Baseline | stgcn | 208,978 | 69.82% | 69.17% | 0.00 pp |
| 01_spectral_pe | spectral_pe_stgcn | 210,538 | 71.36% | 69.64% | +0.48 pp |
| 02_qkv_spatial_attention | spectral_pe_qkv | 268,286 | 70.33% | 71.55% | +2.38 pp |
| 03_gwnet_adaptive_support | gwnet_adaptive_support | 282,703 | 75.96% | 74.17% | +5.00 pp |
| 04_agcrn_factorized_adjacency | agcrn_factorized_adjacency | 930,178 | 81.59% | 80.00% | +10.83 pp |

Experiment 3 stores the fixed physical and full learned adaptive supports in 
`03_gwnet_adaptive_support/adjacency_matrices.npz` for reproducible visualisation.

This run uses the DD-Net copy of the official 1960/840 skeleton split.
DD-Net median-filtered every coordinate channel before serialisation; the
same-run ST-GCN control is therefore the only valid baseline for deltas.

## Per-class official-test accuracy

| Class | 00_stgcn_control | 01_spectral_pe | 02_qkv_spatial_attention | 03_gwnet_adaptive_support | 04_agcrn_factorized_adjacency |
|---|---:|---:|---:|---:|---:|
| expand | 68.9% | 65.6% | 86.9% | 72.1% | 73.8% |
| grab | 72.4% | 74.1% | 60.3% | 70.7% | 70.7% |
| pinch | 74.5% | 80.0% | 90.9% | 85.5% | 90.9% |
| rotation_ccw | 89.1% | 90.9% | 92.7% | 94.5% | 92.7% |
| rotation_cw | 76.5% | 76.5% | 86.3% | 76.5% | 80.4% |
| shake | 82.2% | 79.5% | 90.4% | 82.2% | 87.7% |
| swipe_down | 41.0% | 41.0% | 65.6% | 59.0% | 75.4% |
| swipe_left | 70.4% | 72.2% | 85.2% | 79.6% | 83.3% |
| swipe_plus | 60.3% | 63.8% | 20.7% | 55.2% | 67.2% |
| swipe_right | 77.4% | 59.7% | 71.0% | 80.6% | 79.0% |
| swipe_up | 57.4% | 50.0% | 57.4% | 64.7% | 75.0% |
| swipe_v | 71.9% | 75.4% | 71.9% | 82.5% | 93.0% |
| swipe_x | 40.6% | 58.0% | 36.2% | 46.4% | 59.4% |
| tap | 93.1% | 96.6% | 94.8% | 96.6% | 96.6% |
