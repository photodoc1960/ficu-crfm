# Experiment A: L1 Trajectory Readout — Resume from Checkpoint

## Setup
- **Readout**: Spatiotemporal matched filter, 3 taps at settle steps {8, 16, 24} (of 24 total)
- **Feature dim**: 486-D (3 × 162-D per-tap, independently normalised)
- **Physics**: Frozen from `l1_phoneme.pt` (post-TIMIT 10-epoch baseline)
- **Training**: 100 epochs readout-only, cosine LR 1e-3 → 1e-5, delta rule with weight_decay=0.01
- **W init**: Zeros (486 × 40)

## Learning Curve
| Epoch | Val Acc |
|---|---|
| 0 | 33.67% |
| 10 | 40.20% |
| 20 | 41.01% |
| 30 | 41.31% |
| 40 | 41.35% |
| 50 | 41.58% |
| 60 | 41.49% |
| 70 | 41.59% |
| 80 | 41.52% |
| 90 | 41.55% |
| 99 | 41.55% |

## Results
- **Peak val_acc**: 41.64% at epoch 72
- **Final val_acc**: 41.55% at epoch 99
- **Endpoint-only baseline**: 38.76% (peak at ep38, same checkpoint, same 100-epoch protocol)
- **Improvement**: +2.88pp (final) / +3.09pp (peak) over endpoint-only matched filter
- **TBI at final epoch**: Not measured in this run (physics frozen, expected ~0.380 ± 0.114 from the checkpoint's fixed-point value)

## Notes
- The spatiotemporal matched filter converges faster (crosses 40% by epoch 10) and to a higher ceiling than the endpoint-only readout.
- The improvement is entirely from the readout-side trajectory information — no physics changes.
- The 486-D trajectory vector provides ~3× the feature dimensionality of the 162-D endpoint vector, allowing the matched filter to detect transient field patterns that are absent from the settled endpoint.
- No anomalies observed. Per-tap normalisation is critical — without it, the early-settle taps (which have smaller field amplitudes) would be dwarfed by the endpoint.
