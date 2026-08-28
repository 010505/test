# GestureGraph independent backbone experiments

All experiments use the official SHREC'17 split, the same seed, frame count, optimiser, augmentation, and epoch budget.
Dataset mode: `shrec17_npz`.

| Experiment | Backbone | Parameters | Best validation | Official test | Delta vs ST-GCN |
|---|---|---:|---:|---:|---:|
| Baseline | stgcn | 208,978 | 75.96% | 68.21% | 0.00 pp |
| 01_spectral_pe | spectral_pe_stgcn | 210,538 | 76.73% | 69.52% | +1.31 pp |
| 02_qkv_spatial_attention | spectral_pe_qkv | 268,286 | 78.01% | 73.33% | +5.12 pp |
| 03_gwnet_adaptive_support | gwnet_adaptive_support | 282,703 | 81.07% | 72.26% | +4.05 pp |
| 04_agcrn_factorized_adjacency | agcrn_factorized_adjacency | 930,178 | 85.93% | 80.71% | +12.50 pp |

Experiment 3 stores the fixed physical and full learned adaptive supports in 
`03_gwnet_adaptive_support/adjacency_matrices.npz` for reproducible visualisation.

This run uses the DD-Net copy of the official 1960/840 skeleton split.
DD-Net median-filtered every coordinate channel before serialisation; the
same-run ST-GCN control is therefore the only valid baseline for deltas.

## Per-class official-test accuracy

| Class | 00_stgcn_control | 01_spectral_pe | 02_qkv_spatial_attention | 03_gwnet_adaptive_support | 04_agcrn_factorized_adjacency |
|---|---:|---:|---:|---:|---:|
| expand | 52.5% | 59.0% | 85.2% | 70.5% | 80.3% |
| grab | 74.1% | 72.4% | 67.2% | 81.0% | 87.9% |
| pinch | 74.5% | 78.2% | 89.1% | 80.0% | 96.4% |
| rotation_ccw | 81.8% | 87.3% | 92.7% | 89.1% | 92.7% |
| rotation_cw | 66.7% | 64.7% | 88.2% | 78.4% | 88.2% |
| shake | 80.8% | 87.7% | 95.9% | 91.8% | 94.5% |
| swipe_down | 54.1% | 59.0% | 47.5% | 39.3% | 54.1% |
| swipe_left | 79.6% | 63.0% | 77.8% | 70.4% | 81.5% |
| swipe_plus | 55.2% | 51.7% | 36.2% | 51.7% | 69.0% |
| swipe_right | 75.8% | 77.4% | 80.6% | 85.5% | 72.6% |
| swipe_up | 55.9% | 48.5% | 72.1% | 66.2% | 80.9% |
| swipe_v | 80.7% | 77.2% | 64.9% | 78.9% | 87.7% |
| swipe_x | 37.7% | 55.1% | 37.7% | 39.1% | 52.2% |
| tap | 93.1% | 94.8% | 96.6% | 94.8% | 98.3% |
