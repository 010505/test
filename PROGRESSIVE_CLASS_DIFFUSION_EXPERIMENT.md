# Progressive causal-prefix and class-diffusion experiment

## Scope

This experiment verifies whether GestureGraph can recognise an ongoing gesture
before all 64 frames have been observed. It is a closed-set 14-class study;
unknown-gesture rejection is not claimed because the available SHREC'17 data
does not contain a valid open-set training/test protocol.

All new runs are independent from the preserved backbone experiments. The
shared encoder is the selected joint-support AGCRN without semantic SE. Formal
runs use the DD-Net median-filtered SHREC'17 1960/840 official split, seeds
42/43/44, 40 epochs, effective batch size 32 and best-validation checkpoint
selection.

## Future-free prefix construction

For observation ratio rho, only the raw causal prefix is accessible:

```text
X_rho = X[0 : ceil(rho * T)]
```

The prefix is normalised independently, resampled to `ceil(rho * 64)` valid
positions, then padded to 64 by repeating its last observed frame. Mask-aware
pooling excludes the padding after the two stride-two temporal blocks. Changing
any coordinate after the prefix has no effect on the model input; this property
is covered by a unit test.

The fixed evaluation ratios are 25%, 50%, 65%, 80% and 100%. Prefix AUC is the
normalised trapezoidal area under accuracy versus observation ratio.

## Models

| Experiment | Training input | Decision head | Parameters | Checkpoint selection |
|---|---|---|---:|---|
| `00_full_sequence` | Complete sequence only | Linear progress-conditioned head | 989,830 | Full validation accuracy |
| `01_mild_temporal_crop` | Energy-aware 50-100% random crop, resampled to 64 | Same direct head | 989,830 | Full validation accuracy |
| `02_causal_prefix` | Random future-free causal prefix | Same direct head | 989,830 | Validation prefix AUC |
| `03_gru_evidence` | All five causal prefixes in order | 128-D GRU evidence state | 1,089,272 | Validation prefix AUC |
| `04_class_diffusion` | All five causal prefixes in order | Four-step categorical diffusion | 1,010,376 | Validation prefix AUC |

The mild crop is an independent implementation motivated by the supplied
reports; their referenced source code and checkpoints were not available in
this workspace, so this is not claimed as an exact reproduction.

## Four-step class diffusion

For 14 classes, every forward transition is

```text
Q_r = (1 - beta_r) I + beta_r U
beta = [0.25, 0.45, 0.65, 0.90]
```

where U is the uniform 14x14 transition. The denoiser predicts the clean class
from the current noisy class, causal AGCRN feature, observation ratio and step
embedding. Training combines clean-class cross entropy under sampled forward
noise with the differentiable exact reverse marginal. At inference, every
reverse posterior is computed exactly over all 14 states; the next video update
starts from a 50/50 mixture of the previous posterior and the uniform prior.

## Formal three-seed results

| Model | 25% | 50% | 65% | 80% | 100% | Prefix AUC | ADR | Decision accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-sequence baseline | 34.05% | 61.47% | 71.71% | 78.37% | **80.16%** | 65.38% +/- 1.54% | 1.000 | 80.16% |
| Mild temporal crop | 28.65% | 59.05% | 70.08% | 75.40% | 76.55% | 62.34% +/- 1.05% | 1.000 | 76.55% |
| Causal prefix | 55.08% | 72.70% | 76.98% | 77.74% | 76.55% | 72.31% +/- 1.99% | 0.598 | 75.48% |
| GRU evidence | 55.99% | 71.59% | 75.32% | 76.79% | 76.55% | 71.61% +/- 0.52% | 0.597 | 75.16% |
| Four-step class diffusion | **57.90%** | **73.77%** | **78.17%** | **79.33%** | 79.37% | **74.05% +/- 0.17%** | **0.609 +/- 0.068** | **78.41% +/- 1.40%** |

ADR is calibrated independently on each validation split. Acceptance requires
the same top class for two consecutive updates plus validation-calibrated
confidence and top-two margin thresholds. The target validation decision
accuracy is within one percentage point of the model's full-observation
accuracy. The diffusion mean ADR 0.609 corresponds to about 39.0 of 64 frames.

## Paired conclusions

- Diffusion minus full-sequence baseline: prefix AUC +8.67 +/- 1.41 pp,
  100% accuracy -0.79 +/- 1.94 pp and ADR -39.07 +/- 6.84 percentage points.
- Diffusion minus direct causal-prefix model: prefix AUC +1.74 +/- 2.07 pp,
  100% accuracy +2.82 +/- 2.45 pp and decision accuracy +2.94 +/- 0.84 pp.
- Diffusion minus GRU evidence model: prefix AUC +2.44 +/- 0.66 pp,
  100% accuracy +2.82 +/- 1.64 pp and decision accuracy +3.25 +/- 1.92 pp.
- Mild random cropping lowers both prefix AUC and full accuracy in all three
  AGCRN seeds. It is retained as a negative transfer ablation and is not
  selected.

The selected early-recognition model is `04_class_diffusion`. The result
supports the new class-space diffusion idea for closed-set early recognition:
it improves every tested observation ratio below 100%, preserves almost all
full-sequence accuracy and needs about 61% of the gesture on average. It does
not yet establish open-set Unknown rejection.

## Early-error recovery and online latency

To test the proposed error-inertia explanation, each trained checkpoint was
replayed on the same 840 official-test samples. Recovery is conditioned on an
incorrect prediction at the 25% prefix; it therefore measures whether later
evidence can escape an early error rather than merely reporting overall
accuracy.

| Model | Initial errors | Correct at 50% | Correct at 65% | Correct at 80% | Final recovery | Initially-correct retention |
|---|---:|---:|---:|---:|---:|---:|
| Causal prefix | 377.3 | 45.65% | 56.48% | 59.21% | 58.31% +/- 0.42% | 91.37% |
| GRU evidence | 369.7 | 41.10% | 50.12% | 53.63% | 53.20% +/- 0.39% | **94.89%** |
| Four-step class diffusion | 353.7 | **46.55%** | **57.67%** | **60.31%** | **61.08% +/- 0.71%** | 92.67% |

Because each model makes different 25% errors, the primary paired comparison
uses only samples that both GRU and diffusion initially misclassify. Across the
three seeds there are 287.0 +/- 9.5 such samples. GRU recovers 47.34% +/- 2.59%
by the final observation, while diffusion recovers 57.34% +/- 2.20%, a paired
gain of **10.00 +/- 3.27 pp**. This supports stronger empirical correction of
early errors. It does not by itself prove that the Markov structure is the sole
cause; soft posterior propagation, current-feature conditioning and 50% uniform
prior mixing are all part of the implemented mechanism.

True online latency was additionally measured on the RTX 3090 with batch size
1, retaining the GRU hidden state or diffusion class posterior between updates:

| Model | Mean per update | Five updates | Mean updates to decision | Estimated decision compute |
|---|---:|---:|---:|---:|
| Causal prefix | 5.235 +/- 0.029 ms | 26.176 ms | 2.62 | 13.735 ms |
| GRU evidence | 5.310 +/- 0.156 ms | 26.551 ms | 2.62 | 13.874 ms |
| Four-step class diffusion | 8.087 +/- 0.204 ms | 40.436 ms | 2.69 | 21.787 ms |

Diffusion is 1.52x slower per online update than GRU, but the absolute cost is
about 8.1 ms per checkpoint update and about 21.8 ms of accumulated model
compute at the calibrated decision point. The five updates occur across the
gesture, so 40.4 ms is cumulative compute rather than a single 40.4 ms response
stall. These timings exclude input capture, preprocessing and transfer costs.

## Inheritance and reverse-step optimization

A post-training gamma sweep shows only 0.06 pp total AUC variation over
`gamma=0...1`; fixed inheritance is not the current bottleneck. Two independent
40-epoch learned-gate models were then trained. The unconstrained gate collapses
to approximately 1.0 and reaches 73.40% +/- 0.85% AUC. A bounded,
reliability-supervised gate learns meaningful sample-dependent values but still
reaches only 73.54% +/- 0.83%, versus 74.05% +/- 0.17% for fixed gamma=0.50.
Neither gate is selected.

Inference-step truncation is more useful. Returning the denoiser's clean-class
marginal after two reverse steps preserves 73.84% +/- 0.13% AUC and 79.05% full
accuracy while reducing paired online latency by 17.1%. Therefore the original
four-step checkpoint remains the accuracy default, while `inference_steps=2`
is the selected balanced deployment setting. Full results are in
`DIFFUSION_INHERITANCE_AND_SPEED_EXPERIMENTS.md`.

## Reproduction

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$py='E:\Anaconda_envs\envs\test_transform2\python.exe'
foreach ($seed in 42,43,44) {
  & $py -m gesturegraph.progressive_benchmark `
    --data data\shrec17_ddnet_npz `
    --output "runs\progressive_seed$seed" `
    --epochs 40 --batch-size 32 --sequence-batch-size 4 `
    --eval-batch-size 8 --seed $seed --device cuda
}

& $py -m gesturegraph.aggregate_progressive `
  --runs runs\progressive_seed42 runs\progressive_seed43 runs\progressive_seed44 `
  --output runs\progressive_aggregate

& $py scripts\audit_progressive_results.py `
  --runs runs\progressive_seed42 runs\progressive_seed43 runs\progressive_seed44 `
  --expected-epochs 40

& $py -m scripts.analyze_progressive_recovery `
  --data data\shrec17_ddnet_npz `
  --runs runs\progressive_seed42 runs\progressive_seed43 runs\progressive_seed44 `
  --output runs\progressive_recovery --device cuda --latency-iterations 50
```

Machine-readable aggregate: `runs/progressive_aggregate/aggregate.json`.
Recovery/latency details: `runs/progressive_recovery/analysis.json` and
`runs/progressive_recovery/REPORT.md`.
Every seed directory contains five independent `best.pt`, `history.json` and
`model.json` artifacts.
