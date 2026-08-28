# Three-seed GestureGraph comparison

Mean ± sample standard deviation across seeds. Delta is paired against the
ST-GCN control trained with the same seed and validation split.

| Experiment | Parameters | Official test | Paired delta vs ST-GCN |
|---|---:|---:|---:|
| 00_stgcn_control | 208,978 | 69.33% ± 1.20% | +0.00 ± 0.00 pp |
| 01_spectral_pe | 210,538 | 68.77% ± 1.41% | -0.56 ± 2.54 pp |
| 02_qkv_spatial_attention | 268,286 | 68.37% ± 7.10% | -0.95 ± 8.26 pp |
| 03_gwnet_adaptive_support | 282,703 | 72.38% ± 1.73% | +3.06 ± 2.59 pp |
| 04_agcrn_factorized_adjacency | 930,178 | 80.56% ± 0.50% | +11.23 ± 1.13 pp |

Seeds: 42, 43, 44.
