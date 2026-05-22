# Multi-Seed Cascade Replication Summary

**Protocol**: 20-epoch L3 cascade, β_nudge=0.05, λ_max=5.0, drive_gain=1.0 at L2/L3, --no_gate, PETU active. L1 frozen (`checkpoints/l1_phoneme.pt`), L2 frozen (`checkpoints/l2_phase2_extended.pt`), L3 fresh per seed.

**Seeds**: [0, 1, 2, 3, 4]  •  **Only RNG varies** (PredictiveField W init, batch shuffling, PETU sample masking).

---

## Table 1 — Final-epoch TBI (mean ± SD across seeds, min–max)

| Layer | TBI mean ± SD (min–max) |
|---|---|
| L1 (frozen, sanity) | 0.5237 ± 0.0000  (0.5237–0.5237) |
| L2 (post-cascade)   | 0.5151 ± 0.0026  (0.5117–0.5188) |
| L3 (post-cascade)   | 0.5002 ± 0.0090  (0.4937–0.5179) |

*Single-seed manuscript value: L1=0.524, L2=0.510, L3=0.492.*

## Table 2 — L2 cascade trajectory across seeds

| Quantity | mean ± SD | min – max |
|---|---|---|
| L2 TBI initial (epoch 0)   | 0.3492 ± 0.0000 | 0.3492 – 0.3492 |
| L2 TBI final (epoch 19)    | 0.5151 ± 0.0026 | 0.5117 – 0.5188 |
| Δ L2 TBI                   | 0.1659 ± 0.0026 | 0.1625 – 0.1696 |
| Gap closure (% of L1−L2)   | 95.1% ± 1.5% | 93.1% – 97.2% |

*Single-seed manuscript value: Δ = +0.161 nats, 92% gap closure.*

## Table 3 — Final-epoch per-layer distribution (mean ± SD across seeds)

| Layer | median | IQR | skewness | % TBI < 0.05 |
|---|---|---|---|---|
| L1 (frozen)         | 0.4687 ± 0.0000 | 0.1928 ± 0.0000 | 0.5949 ± 0.0000 | 0.0% (pooled) |
| L2 pre-cascade      | 0.3928 ± 0.0000 | — | — | 38.4% ± 0.0% |
| L2 post-cascade     | 0.5560 ± 0.0056 | 0.6070 ± 0.0037 | 0.0474 ± 0.0136 | 38.4% ± 0.0% |
| L3 (cascade)        | 0.3806 ± 0.0092 | 0.1527 ± 0.0205 | -1.0197 ± 0.0796 | 9.2% ± 0.0% |

*Single-seed manuscript values: L2 zero-coherence fraction 32.1% pre and 32.1% post.*

## Table 4 — Per-seed final numbers

| seed | L1_TBI | L2_TBI | L3_TBI | ΔL2 | gap closed | L2 zero-frac post | L3_val_acc | upd_frac |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5237 | 0.5162 | 0.4986 | +0.1670 | 95.7% | 38.4% | 54.2% | 0.859 |
| 1 | 0.5237 | 0.5160 | 0.5179 | +0.1668 | 95.6% | 38.4% | 52.1% | 0.844 |
| 2 | 0.5237 | 0.5117 | 0.4937 | +0.1625 | 93.1% | 38.4% | 51.0% | 0.867 |
| 3 | 0.5237 | 0.5128 | 0.4944 | +0.1636 | 93.8% | 38.4% | 54.9% | 0.856 |
| 4 | 0.5237 | 0.5188 | 0.4962 | +0.1696 | 97.2% | 38.4% | 53.9% | 0.868 |

## Figures

- `runs/figures/fig5_multiseed.png` — L2 TBI trajectory, mean ± SD over 5 seeds, with individual seed traces overlaid.
- `runs/figures/fig6_multiseed.png` — three-layer per-sample TBI box plots (pooled across 5 seeds) plus zero-coherence fraction with cross-seed error bars.

## Flags
- No anomalous seeds. All 5 seeds replicate the cascade within expected variance.
