# Experiment B: L1 Trajectory Readout — Full Retraining

## Setup
- **Phase 1**: 10 epochs joint EP + readout training from random initialisation.
  Trajectory readout (3 taps at steps {8, 16, 24}), 486-D feature vector with
  per-tap Welford normalisation. EP physics updates via PETU gate.
- **Phase 2**: 100 epochs readout-only training from Phase 1 checkpoint.
  Physics frozen. Cosine LR 1e-3 → 1e-5.

## Phase 1 Learning Curve (EP + Readout)
| Epoch | Val Acc | TBI | update_frac |
|---|---|---|---|
| 0 | 38.34% | 0.561 | 0.691 |
| 1 | 39.83% | 0.548 | 0.607 |
| 2 | 40.19% | 0.542 | 0.585 |
| 3 | 40.72% | 0.535 | 0.581 |
| 5 | 40.94% | 0.533 | 0.576 |
| 7 | 40.73% | 0.533 | 0.578 |
| 9 | 40.58% | 0.532 | 0.584 |

## Phase 2 Learning Curve (Readout Only)
| Epoch | Val Acc |
|---|---|
| 0 | 41.55% |
| 10 | 41.35% |
| 20 | 41.40% |
| 30 | 41.49% |
| 50 | 41.32% |
| 55 | **41.69%** (peak) |
| 70 | 41.56% |
| 90 | 41.60% |
| 99 | 41.59% |

## Results
- **Peak val_acc**: 41.69% at epoch 55 (Phase 2)
- **Final val_acc**: 41.59%
- **Endpoint-only baseline**: 38.76% (same protocol, holographic readout)
- **Improvement**: +2.93pp (peak) / +2.83pp (final) over endpoint-only
- **TBI at Phase 1 convergence**: 0.532 ± 0.114 (comparable to holographic 0.524 ± 0.108)
- **PETU update_fraction**: 0.584 at ep9 (holographic: 0.650) — better automaticity

## Comparison to Experiment A
| | Exp A (resume) | Exp B (retrain) |
|---|---|---|
| Physics source | Holographic checkpoint | Trajectory-trained |
| Phase 1 peak | N/A (frozen) | 40.94% (10 ep) |
| Phase 2 peak | 41.64% (ep72) | 41.69% (ep55) |
| Δ vs endpoint | +2.88pp | +2.93pp |

- The difference is +0.05pp — within noise. Physics training source does not
  meaningfully affect the trajectory readout ceiling.
- Both experiments converge to the same ~41.7% plateau, consistent with this
  being the information ceiling of the 3-tap spatiotemporal feature vector under
  the delta-rule readout.

## Notes
- Phase 1 demonstrates that the trajectory readout also works well during joint
  EP+readout training — val_acc crosses 40% by epoch 2 (faster than the
  holographic readout which needed ~10 epochs to reach 38.5%).
- PETU dynamics are healthier with trajectory readout: update_fraction decreases
  faster (0.69→0.58 over 10 epochs vs 0.98→0.65 for holographic over 10 epochs),
  suggesting that the richer trajectory features help the field achieve
  automaticity earlier.
- No anomalies.
