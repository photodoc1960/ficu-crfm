# FICU-CRFM — Hierarchical Phase-Coherence Cascade

Reference implementation for the manuscript

> **Hierarchical Automaticity Emerges from Prediction-Error-Triggered Learning
> in Continuous Wave Fields Trained by Equilibrium Propagation**

Three-layer Landau–Ginzburg complex-valued wave-field hierarchy
(L1 phoneme · L2 word · L3 sentence) trained on TIMIT with
equilibrium propagation, prediction-error-triggered updates (PETU),
a learned L3→L2 PredictiveField, and a holographic matched-filter
readout.

## Repository layout

| Path | Contents |
|---|---|
| `architecture/` | L1/L2/L3 field modules, PredictiveField, readouts |
| `training/` | EP training loops for each layer, L3 cascade orchestrator |
| `dataset/` | TIMIT loader (phoneme / word / sentence splits) |
| `metrics/` | TBI (temporal-binding index) and related diagnostics |
| `figures/` | Paper figures + LaTeX/Markdown tables |
| `reports/` | Methods documentation, trajectory-readout experiment reports |
| `runs/` | Multi-seed replication artifacts (see below) |

## Multi-seed cascade experiments

The 5-seed paired cascade replication and its symmetric dissociation
control live in `runs/`:

- `runs/run_multiseed_cascade.py` — healthy cascade orchestrator
  (drive_gain=1.0 at L2 and L3, 20 epochs × 5 seeds)
- `runs/run_multiseed_cascade_unhealthy.py` — unhealthy/attenuated
  cascade (drive_gain=0.1 at L3; L1, L2 unchanged) paired to the same
  seeds for a like-for-like comparison
- `runs/verify_attenuated_l3.py` — operating-point verification script
  for the attenuated-regime checkpoint
- `runs/cascade_multiseed_summary.md` — healthy-only summary tables
- `runs/cascade_unhealthy_summary.md` — paired healthy-vs-unhealthy
  summary tables (the dissociation result)
- `runs/figures/fig5_multiseed.{png,pdf}` — multi-seed L2-trajectory band
- `runs/figures/fig6_multiseed.{png,pdf}` — three-layer per-sample TBI
  distributions and zero-coherence-mode bar chart
- `runs/figures/fig5_dissociation.{png,pdf}` — two-panel dissociation
  figure (L2 cascade trajectory bands + PF prediction-error trajectories,
  healthy vs unhealthy)
- `runs/cascade_seed{0..4}/` and `runs/cascade_unhealthy_seed{0..4}/` —
  per-seed CSV logs, per-sample TBI arrays (.npz), distribution
  statistics (JSON), and stdout

## Withheld artifacts (available on request)

Several large binary artifacts are intentionally **not committed** to
keep the repository lightweight (~250 MB of model state across the
multi-seed runs):

| Path pattern | What | Why withheld |
|---|---|---|
| `runs/*/cascade.pt` | Per-seed final L3 + L2 + PredictiveField state dicts (≈21 MB each) | Regenerable from the orchestrator scripts; weights are 5× redundant in the multi-seed design |
| `runs/l3_attenuated_fresh/l3_attenuated.pt` | Fresh attenuated-regime L3 source checkpoint used to seed the unhealthy cascade | Regenerable in ~90 s from `training/train_l3.py` with the documented hyperparameters |
| `checkpoints/*.pt` | Frozen L1 and L2 backbone checkpoints used throughout | Originally produced from the L1/L2 training scripts; preserved off-repo |

All per-sample TBI arrays (`per_sample_tbi.npz`, ~20 KB each), per-epoch
CSV logs, distribution statistics, run logs, and final figures **are**
committed — the dissociation result is fully reproducible from the
committed materials plus a single `python -u -m
runs.run_multiseed_cascade*` invocation.

If you'd like the full set of `.pt` checkpoints (frozen L1/L2 backbones
plus the per-seed cascade weights) for direct loading without
retraining, please open a GitHub issue or email — they will be
provided as a release asset or via a separate file share.

## Reproducing the dissociation result

Inside the GPU container (NVIDIA PyTorch image, see `launch_gpu.sh`):

```bash
# Healthy 5-seed cascade
python -u -m runs.run_multiseed_cascade

# Symmetric unhealthy control (requires the L1/L2 checkpoints and the
# fresh attenuated L3, regenerable as documented in
# runs/cascade_unhealthy_summary.md)
python -u -m runs.run_multiseed_cascade_unhealthy
```

Each orchestrator is idempotent — completed seeds are skipped on
re-invocation, so individual seeds can be regenerated without
re-running the full sweep.

## Archived snapshot and full checkpoint bundle

A versioned snapshot of this repository, together with all of the
checkpoints withheld above (frozen L1/L2 backbones, 5-seed healthy
cascade, paired 5-seed unhealthy cascade, and the attenuated-regime
L3 source), is archived on Zenodo:

> **DOI:** [10.5281/zenodo.2034701](https://doi.org/10.5281/zenodo.2034701)

The Zenodo record contains:

- `ficu-crfm-code-0489063.tar.gz` — clean `git archive` of commit
  [`0489063`](https://github.com/photodoc1960/ficu-crfm/commit/0489063)
- `ficu-crfm-checkpoints-v1.tar.gz` — all `.pt` checkpoints, figure
  files (PNG + PDF), per-seed CSV logs, per-sample TBI arrays
  (`.npz`), distribution-statistics JSON, summary markdowns, and a
  `MANIFEST.txt` with per-file sha256 sums

### How to cite

```bibtex
@software{slater_ficu_crfm_2026,
  author    = {Slater, Jeremy},
  title     = {{FICU-CRFM v1.0 — code, checkpoints, and figure data
                for ``Hierarchical Automaticity Emerges from
                Prediction-Error-Triggered Learning in Continuous
                Wave Fields Trained by Equilibrium Propagation''}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.2034701},
  url       = {https://doi.org/10.5281/zenodo.2034701}
}
```
