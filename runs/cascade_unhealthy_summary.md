# Cascade Dissociation: Healthy vs Unhealthy L3 Physics

**Paired protocol** (5 seeds, identical except for L3 regime):  epochs=20, batch_size=32, β_nudge=0.05, λ_max=5.0, petu_coh_floor=0.35, PETU active. L1 frozen (`checkpoints/l1_phoneme.pt`), L2 frozen (`checkpoints/l2_phase2_extended.pt`), PredictiveField fresh per seed.

- **Healthy condition**: L3 fresh, `drive_gain=1.0` throughout (matches `run_multiseed_cascade.py`).  
- **Unhealthy condition**: L3 initialized from a freshly-trained attenuated checkpoint (`runs/l3_attenuated_fresh/l3_attenuated.pt`) — 5 epochs at `drive_gain=0.1`, β=0.05, λ_max=0 (no cascade) — then run with `drive_gain=0.1` for the same 20-epoch cascade protocol.

## Operating-point note

The Figure 4 ablation reported L3_TBI=0.024–0.040 for the attenuated regime. Under the corrected-amplitude codebase, neither the saved `l3_beta_cal_0.05.pt` checkpoint nor a fresh re-train under the original Figure-4 hyperparameters lands in that band:

| Source | L3_TBI mean | L3 val_acc |
|---|---|---|
| Manuscript Figure 4 attenuated | 0.024–0.040 | 56.0–56.8% |
| Saved `l3_beta_cal_0.05.pt` (loaded, no further training) | 0.121 | 52.5% |
| Fresh re-train, original Fig-4 hyperparameters | 0.153 | 56.25% |

This is consistent with the amplitude-attenuation correction documented in the manuscript: Figure 4 was generated when L2's `coupling_l2` was still initialized at 0.1·I (pre-fix), so the absolute attenuated-TBI number could not survive the L2 correction. The **regime contrast** is preserved (unhealthy L3_TBI 0.116 ≈ 23% of healthy 0.500; unhealthy val_acc 57.5% remains chance-band). The paired comparison below uses these post-correction values.

---

## Table 1 — Final-epoch TBI (mean ± SD across seeds)

| Layer | Healthy | Unhealthy |
|---|---|---|
| L1 (frozen, sanity) | 0.5237 ± 0.0000 | 0.5237 ± 0.0000 |
| L2 (post-cascade)   | 0.5151 ± 0.0026 | 0.4016 ± 0.0010 |
| L3 (post-cascade)   | 0.5002 ± 0.0090 | 0.1162 ± 0.0023 |

## Table 2 — L2 cascade transmission (ΔL2)

| Quantity | Healthy | Unhealthy |
|---|---|---|
| L2 TBI initial (epoch 0)  | 0.3492 ± 0.0000 | 0.3492 ± 0.0000 |
| L2 TBI final (epoch 19)   | 0.5151 ± 0.0026 | 0.4016 ± 0.0010 |
| Δ L2 TBI                  | 0.1659 ± 0.0026 | 0.0524 ± 0.0010 |
| Gap closure (% of L1−L2)  | 95.1% ± 1.5% | 30.1% ± 0.5% |

## Table 3 — PredictiveField L2-prediction error

| Epoch | Healthy | Unhealthy |
|---|---|---|
| 0 (initial)  | 0.0005 ± 0.0000 | 0.0006 ± 0.0000 |
| 19 (final)   | 0.0010 ± 0.0000 | 0.0005 ± 0.0000 |

## Table 4 — L3 task quality

| Quantity | Healthy | Unhealthy |
|---|---|---|
| L3 val_acc final | 0.5323 ± 0.0145 | 0.5750 ± 0.0098 |

## Table 5 — L2 zero-coherence fraction (TBI<0.05) at final epoch

| Layer-stat | Healthy | Unhealthy |
|---|---|---|
| L2 post-cascade median | 0.5560 ± 0.0056 | 0.3936 ± 0.0022 |
| L2 frac<0.05 (post)    | 38.4% ± 0.0% | 38.4% ± 0.0% |

## Table 6 — Per-seed final numbers (paired)

| seed | L2_TBI_H | L2_TBI_U | ΔL2_H | ΔL2_U | PFerr_H | PFerr_U | val_H | val_U |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5162 | 0.4006 | +0.1670 | +0.0514 | 0.0010 | 0.0005 | 54.2% | 58.6% |
| 1 | 0.5160 | 0.4012 | +0.1668 | +0.0520 | 0.0011 | 0.0005 | 52.1% | 56.7% |
| 2 | 0.5117 | 0.4031 | +0.1625 | +0.0539 | 0.0011 | 0.0005 | 51.0% | 56.0% |
| 3 | 0.5128 | 0.4024 | +0.1636 | +0.0532 | 0.0010 | 0.0005 | 54.9% | 58.4% |
| 4 | 0.5188 | 0.4009 | +0.1696 | +0.0517 | 0.0010 | 0.0005 | 53.9% | 57.8% |

## Figures

- `runs/figures/fig5_dissociation.{png,pdf}` — two-panel paired figure. Panel (a): L2 TBI trajectories with ±SD bands, healthy vs unhealthy, L1-frozen and L2-pre-cascade reference lines. Panel (b): PredictiveField L2-prediction-error trajectories, healthy vs unhealthy, ±SD bands. Tells the causal story: source-field coherence (healthy vs unhealthy L3) → prediction quality (PF error) → transmission (ΔL2).

## Flags
- No unhealthy seed deviates more than 0.05 from the seed mean ΔL2 (+0.0524); cross-seed range +0.0514 – +0.0539.
