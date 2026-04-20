# Methods Documentation — FICU-CRFM Implementation Details

Raw documentation for the Methods section. Each answer is factual and
references source file paths and line numbers. Synthesize into prose
separately.

---

## Section 1: Field grid and settling parameters

### Q1.1 Grid sizes per layer (before pooling)
- L1: 94 x 64 (HEIGHT=TARGET_FRAMES=94, WIDTH=N_MELS=64)
  - Source: `architecture/ficu_l1.py:406-407` (references `feature_extractor.py:23-24`)
- L2: 47 x 32
  - Source: `architecture/ficu_l2.py:28-29`
- L3: 24 x 16
  - Source: `architecture/ficu_l3.py:31-32`
- All layers: CHANNELS = 3
- Pool size (for readout coarse-graining): 3 at all layers

### Q1.2 Settling timesteps and dt
- **All three layers**: n_settle_steps = 24, dt = 0.07
  - L1: `ficu_l1.py:409` (constructor default args)
  - L2: `ficu_l2.py:34`
  - L3: `ficu_l3.py:35`
- Note: The settle loop iterates 24 discrete steps, NOT 120. Total settling time = 24 * 0.07 = 1.68 time units. The paper should state 24 steps, not 120.

### Q1.3 Settling window across layers
- Identical across all three layers (24 steps, dt=0.07). No layer uses a longer window.
- Note: L2 and L3 receive pooled inputs from the layer below, so their spatial resolution is lower but temporal resolution is the same.

### Q1.4 LG equation parameters per layer

All three layers share **identical** initial parameter values. Differences arise only in:
- Mexican Hat kernel size and sigma
- Coupling matrix initialisation
- Drive gain multiplier

**Per-channel initial values (same across L1/L2/L3):**

| Parameter | Channel 0 | Channel 1 | Channel 2 | Activation function | Source |
|---|---|---|---|---|---|
| gamma | 0.025 | 0.050 | 0.100 | sigmoid(·) * 0.2 | L1:421, L2:42, L3:43 |
| omega | 0.800 | 1.265 | 2.000 | abs(·) + 0.1 | L1:424, L2:44, L3:45 (logspace) |
| diffusion | 0.100 | 0.100 | 0.100 | sigmoid(·) * 0.5 | L1:427, L2:46, L3:47 |
| beta_per_channel | 0.060 | 0.110 | 0.210 | abs(·) + 0.01 | L1:429, L2:47, L3:48 |

**Scalar parameters (same across layers):**
| Parameter | Init | Activation | Source |
|---|---|---|---|
| chi_spm | 0.15 | sigmoid(·) * 0.5 | L1:432, L2:50, L3:50 |
| chi_xpm | 0.10 | sigmoid(·) * 0.3 | L1:433, L2:51, L3:51 |
| raw_evanescent | 0.01 | softplus(·) * 0.1 | L1:434, L2:52, L3:52 |

**Mexican Hat lateral inhibition (differs per layer):**
| Layer | Kernel size | sigma_exc | sigma_inh | Source |
|---|---|---|---|---|
| L1 | 9 x 9 | 1.0 | 3.0 | `ficu_l1.py:436` |
| L2 | 7 x 7 | 1.5 | 4.0 | `ficu_l2.py:57` |
| L3 | 5 x 5 | 1.0 | 2.5 | `ficu_l3.py:56` |

**Amplitude clamp (all layers):**
`scale = clamp(10.0 / amplitude, max=1.0)` — per-element magnitude bounded at 10.
Source: `ficu_l1.py:521-523`, `ficu_l2.py:118-120`, `ficu_l3.py:116-118`

---

## Section 2: Sensory front-end and inter-layer projection

### Q2.1 L1 acoustic front-end
- **Features**: Mel spectrogram (power=2.0, Slaney-normalised filterbank) + delta + delta-delta.
- **Sample rate**: 16,000 Hz
- **Window size**: 400 samples = 25 ms (Hann window)
- **Hop size**: 160 samples = 10 ms
- **Frequency bins**: 64 mel bands
- **Output**: [B, 3, 94, 64] — 3 channels (mel, delta, delta-delta), 94 frames, 64 mel bins
- **Normalization**: per-utterance mean/std normalization applied separately to each channel.
- **Implementation**: pure PyTorch (torch.stft + handbuilt filterbank), NOT torchaudio. NGC containers don't ship torchaudio; we reimplemented the Slaney mel filterbank and ComputeDeltas kernel.
- Source: `dataset/feature_extractor.py` (entire file, ~170 lines)

### Q2.2 L1 input projection
- **Direct**: the 3-channel mel output [B, 3, 94, 64] is used directly as the real part of the drive signal. Imaginary drive is zero. No linear projection, convolutional encoder, or reshape.
- `drive_r = feats; drive_i = torch.zeros_like(feats)` then `dZ_r += 0.1 * drive_r` in field_step.
- Source: `ficu_l1.py:555-557,515`

### Q2.3 L2 input from L1
- L1 outputs its settled field state: [B, 3, 94, 64]
- L2 receives this via `drive_from_l1()`:
  1. Spatial 2x avg_pool2d: [B, 3, 94, 64] → [B, 3, 47, 32]
  2. Channel coupling via learned 3x3 matrix `coupling_l1` (init 0.1 * I): `einsum('bchw,cd->bdhw', Z1_pooled, coupling_l1.T)`
  3. Output: [B, 3, 47, 32] drive tensor
- Source: `ficu_l2.py:141-152`

### Q2.4 L3 input from L2
- L2 outputs its settled field state: [B, 3, 47, 32]
- L3 receives this via `drive_from_l2_mean()`:
  1. Spatial adaptive_avg_pool2d: [B, 3, 47, 32] → [B, 3, 24, 16]
  2. Channel coupling via learned 3x3 matrix `coupling_l2` (init 1.0 * I): `einsum('bchw,cd->bdhw', Z2_pooled, coupling_l2.T)`
  3. Output: [B, 3, 24, 16] drive tensor
- Source: `ficu_l3.py:138-148`
- Note: `coupling_l2` was initialised at 0.1 * I originally; changed to 1.0 * I during the amplitude-attenuation fix. See Discrepancies section.

### Q2.5 Drive gain parameter
- **Single scalar per layer**, hardcoded in `field_step()`, shared across channels and spatial locations. Not learned.
- **Values used in reported experiments:**
  - L1: 0.1 (unchanged throughout). Source: `ficu_l1.py:515`
  - L2: 1.0 (changed from 0.1 during amplitude-attenuation fix). Source: `ficu_l2.py:112`
  - L3: 1.0 (changed from 0.1 during amplitude-attenuation fix). Source: `ficu_l3.py:110`
- Note: The L2 checkpoint `l2_phase2_extended.pt` was trained with drive_gain=0.1. The L2 readout accuracy of 38.85% is from this original regime. The drive_gain=1.0 was needed for the cascade experiments. Readout accuracy under drive_gain=1.0 is lower (~24.5%) unless the readout is retrained.

---

## Section 3: TBI computation

### Q3.1 Exact mathematical form

The TBI is the magnitude of the spatially-averaged unit phasor of the cross-correlation between the free and nudged field states:

For each spatial location (i,j) and channel c:
```
phasor(b,c,i,j) = [Z_nudge * conj(Z_free)] / (|Z_nudge| * |Z_free|)
```
This is a unit complex number encoding the phase difference between free and nudged fields.

Average over space:
```
mean_phasor(b,c) = mean_{i,j}[ phasor(b,c,i,j) ]
```

Take the magnitude (Kuramoto-style order parameter):
```
coherence(b,c) = |mean_phasor(b,c)|
```

Average over channels:
```
TBI(b) = mean_c[ coherence(b,c) ]
```

**Answers to specific sub-questions:**
- **Per-location → averaged**: yes. Computed at each grid point, averaged over space, then over channels. NOT aggregate on a single scalar.
- **Computed from endpoint field state** (the final Z after all 24 settle steps). NOT from trajectory.
- **phi**: there is no fixed reference phase phi in the implementation. The TBI measures the phase DIFFERENCE between nudge and free, not alignment with a reference. The draft formula TBI ~ gamma * cos(theta - phi) should be revised.
- **gamma(t)**: NOT the LG damping parameter. The "gamma" in TBI is implicitly the spatial coherence (magnitude of the average phasor). There is no explicit gamma term — the formula IS the magnitude.
- **theta(t)**: the phase of `Z_nudge * conj(Z_free)` at each spatial location — i.e., the local phase difference between nudge and free fields.

Source: `metrics/tbi.py:21-40`

### Q3.2 Per-sample vs batch
- Computed **per-sample**: returns [B] tensor.
- Epoch-level reporting: `tbi_all = torch.cat(all_batch_tbis)` then `.mean()` and `.std()` over all samples in the epoch.
- Source: training loops accumulate per-batch TBI tensors, concatenate at epoch end.

### Q3.3 Source location
- `metrics/tbi.py:compute_TBI()` (lines 21-40)

### Q3.4 Reported mean and std
- **Within-epoch statistics across all training samples.** Each batch yields [B] TBI values; these are concatenated into one vector of ~177k values (L1) or ~40k values (L2), and mean/std are computed once over the full vector.
- Not across-epoch statistics (those would require tracking epoch-level means over multiple epochs).

---

## Section 4: Training hyperparameters

### Q4.1 Batch size
- **32 at all layers.** Same across L1, L2, L3.
- Source: train_l1.py:281, train_l2.py:353, train_l3.py:447

### Q4.2 EP physics learning rate and schedule
- **lr_physics = 0.001** at all layers. **Constant** (no cosine, no warmup).
- Applied via `model.apply_ep_update(obs_free, obs_nudge, lr_physics=0.001)`.
- Each observable delta is clamped to [-0.01, 0.01] before the lr scaling.
- Source: train_l1.py:314, ficu_l1.py:727 (clamp=0.01)

### Q4.3 Readout template learning rate and schedule
- **lr_templates = 0.005** at all layers. **Constant** during standard training.
- For the extended 100-epoch readout runs, a **cosine schedule** from 1e-3 → 1e-5 was used.
- Source: train_l1.py:315, extended_accuracy.py (cosine_lr function)

### Q4.4 EP physics optimizer
- **No optimizer** (not SGD, not Adam). EP physics parameters are updated via direct in-place addition of clamped, lr-scaled observable deltas. No momentum, no adaptive learning rate. The update rule is:
  ```
  param.data += lr_physics * clamp(-(1/beta) * (obs_nudge - obs_free), -0.01, 0.01)
  ```
- Source: ficu_l1.py:724-755 (apply_ep_update)

### Q4.5 Delta rule readout
- **Pure delta rule** (not Adam, not SGD). Deterministic per-batch update:
  ```
  error = target_onehot - softmax(logits)
  delta_W = (error.T @ normalized_features) / B - weight_decay * W
  W += lr * delta_W
  ```
- Weight decay = 0.01. No momentum.
- Source: ficu_l1.py:58-86 (HolographicAssociativeReadout.update)

### Q4.6 Training data per epoch
- **L1 phoneme**: 177,080 segments per epoch (train split)
- **L2 word**: 39,834 segments per epoch (train split)
- **L3 sentence**: 4,620 segments per epoch (train split)
- **501 words**: the 500 most frequent words across all TIMIT training utterances, plus one '<unk>' token for all others. Built by `_build_word_vocab()` counting word occurrences in .WRD files.
- **3 sentence types**: SA (dialect, prefix "sa"), SI (diverse, prefix "si"), SX (compact, prefix "sx"). Determined by the first 2 characters of the TIMIT utterance filename.
- Source: timit_loader.py:184-200 (word vocab), timit_loader.py:203-210 (sentence types)

### Q4.7 TIMIT data split
- **Standard LDC TIMIT train/test split**: TRAIN/ and TEST/ directories.
  - Train: 4,620 utterances (462 speakers)
  - Test: 1,680 utterances (168 speakers)
- **No separate held-out validation**: "val" in the code uses the TEST split directly.
- For the L2 cascade experiment (train_l2.py), a speaker-held-out val split from train-clean-100 was available via LibriSpeech, but the final L2 numbers use the standard TIMIT test split.
- Source: timit_loader.py:126-130 (split_dir = root / split.upper())

### Q4.8 Wall-clock times (approximate, NVIDIA GB10 single GPU)
| Experiment | Epochs | Time/epoch | Total |
|---|---|---|---|
| L1 10-epoch TIMIT (EP+readout) | 10 | ~6.5 min | ~65 min |
| L1 100-epoch readout-only (cosine LR) | 100 | ~2.5 min | ~4.2 hr |
| L2 15-epoch phase 2 | 15 | ~40 s | ~10 min |
| L3 5-epoch calibration | 5 | ~18 s | ~90 s |
| L3 cascade 20-epoch (λ_max=5) | 20 | ~18 s | ~6 min |
| L3 cascade 50-epoch (λ_max=5) | 50 | ~18 s | ~15 min |
| MLP baseline 100-epoch | 100 | ~7 s | ~12 min |

---

## Section 5: Novel mechanisms

### Q5.1 PETU implementation
- **Per-sample mask**: each sample in the batch independently passes or fails the PETU gate. Only passing samples contribute to the EP physics update.
- **Gate fires (update=True) when**: `(prediction_error > threshold) OR (coherence < coherence_floor)`
  - i.e., updates when **high CE loss** (poorly classified) **OR low TBI** (low phase coherence)
  - This matches: "updates occur only while the system remains in a high-uncertainty regime"
- **Threshold**: `0.5 * ln(K)` where K is the number of classes. Derived from chance-level CE = ln(K); the cutoff is "performing worse than half-trained."
  - L1: 0.5 * ln(40) = 1.84 nats
  - L2: 0.5 * ln(501) = 3.11 nats
  - L3: 0.5 * ln(3) = 0.55 nats
- **Coherence floor**: 0.45 (L1), 0.35 (L2/L3)
- Source: `training/petu.py:27-42`

### Q5.2 LevelGate implementation
- **Threshold**: 0.60 for L1 (never triggered in practice), 0.35 for L2.
- **Checked per epoch** (after each full val pass).
- **Logic**: if val_acc >= threshold for `patience` consecutive epochs, freeze all `physics_parameters()`.
- **Disabled in cascade experiments**: `--no_gate` flag added to train_l2.py and train_l3.py. This was used for the L2 extended 15-epoch run and all L3 runs.
- Source: `architecture/level_gate.py` (entire file), train_l2.py flag at line 393

### Q5.3 PredictiveField implementation
- **Architecture**: single linear transformation W (no nonlinearity).
  - W shape: [4512, 1152] (l2_dim=3*47*32, l3_dim=3*24*16)
  - Total parameters: 4512 * 1152 = 5,197,824
- **Initialization**: `randn * 0.001`. `requires_grad = False` (delta-rule only, not gradient-based).
- **Lambda schedule**: linear 0 → λ_max over the training epochs.
  - 20-epoch cascade: λ_max = 5.0
  - 50-epoch cascade: λ_max = 5.0
- **Update rule**: delta rule on (actual L2 - predicted L2): `dW = (error.T @ L3_flat) / B; W += lr * dW`
- Source: `architecture/predictive_field.py` (entire file, 95 lines)

---

## Section 6: Cascade experiment protocol

### Q6.1 Forward/backward ordering
The cascade uses a **two-pass settle per batch**:

**Pass 1 (feed-forward):**
1. L1 settle (free, no top-down): Z1_initial
2. L2 settle from Z1_initial: Z2_initial
3. L3 settle from Z2_initial: Z3_free

**Pass 2 (top-down recurrence):**
4. L3 → L2 prediction: `pred_l2 = PredictiveField.predict(Z3_free)` (scaled by λ)
5. L2 → L1 top-down bias: `bias_l1 = L2.l2_to_l1_topdown_bias(Z2_initial)` (scaled by λ)
6. L1 re-settle with top-down bias: Z1_td
7. L2 re-settle from Z1_td with predicted_init from step 4: Z2_td
8. L3 re-settle from Z2_td: Z3_td_free (this is the cascade-refined free state)

**EP nudge phase:**
9. L3 nudge settle from Z2_td with sentence label: Z3_nudge
10. L3 TBI from (Z3_td_free, Z3_nudge)
11. PETU mask + L3 EP update
12. PredictiveField delta-rule update from (Z3_td_free, Z2_td)
13. L3 readout delta-rule update

Source: `training/train_l3.py:cascade_forward()` (lines 155-176) and the training loop (lines 247-280)

### Q6.2 Which parameters are frozen during cascade
- **L1**: fully frozen (all parameters, `requires_grad=False`)
- **L2**: loaded from checkpoint, `eval()` mode, parameters technically un-frozen BUT **no L2 EP updates are run** in the L3 training loop (the label spaces don't match — L2 was trained on words, L3 on sentences). L2's lambda_td_L2_L1 is scheduled externally.
- **L3**: fresh initialisation, all parameters trainable, EP updates via PETU.

### Q6.3 L2 readout during cascade
- **L2 readout is NOT updated** during cascade training. Only L3's readout updates.
- L2's readout was pre-trained in phase 2 and is not touched.

### Q6.4 PredictiveField injection into L2
- The prediction is injected as `predicted_init` in L2's settle():
  ```python
  Z2_td = L2.settle(Z1_td, Z1i_td, predicted_init=(pred_l2_r, pred_l2_i))
  ```
- This replaces the **initial field state** (normally zeros) with the prediction. It is NOT added to the drive term and NOT added to the coupling term.
- Source: `ficu_l2.py:167-168` (predicted_init replaces Z_r, Z_i initialization)

---

## Section 7: Readout comparison experiments

### Q7.1 Coherent matched filter v2 (raw field)
- Functional form: `score_k = logit_scale * |sum_{c,i,j} conj(Z[c,i,j]) * T_k[c,i,j]|^2`
- Z is the raw field [B, 3, 94, 64], T_k is a complex template per class [K, 3, 94, 64]
- Template norm clipped to max 1.0 per class (max_template_norm=1.0)
- logit_scale = 0.1
- Training: per-class EMA on unit-norm-clipped nudged field states (not delta rule)
- Duration: 10 epochs
- Result: 17.6% (ep0), plateau at ~17.5%
- Source: `ficu_l1.py` (ComplexCoherentReadout class, the v2 raw-field variant was replaced by v3 but the v2 experiment was run from this code)

### Q7.2 Coherent matched filter v3 (81-D complex pooled features)
- 81-D = 9 channels * 3 * 3 pool (same spatial layout as holographic, but treated as complex: `feat_c = complex(feat_r_flat, feat_i_flat)`)
- Templates: [K, 81] complex64
- logit_scale = 0.01
- Result: 2.6% (chance). The 40-class chance level is 1/40 = 2.5%. The 2.6% is within noise of chance — NOT "with unknown-class masking." The readout simply didn't learn.
- Source: logs/l1_coherent_v3.csv

### Q7.3 Whitening variant
- Running mean/var per feature dimension (complex mean, real variance), same incremental Welford update as holographic readout
- Added MIN_VAR = 0.01 floor on variance to prevent 1/std explosion in low-variance dimensions
- Result: 5.3% → 3.0% (worse than unwhitened, drifting toward chance)
- Source: `ficu_l1.py` (ComplexCoherentReadout._whiten), logs/l1_coherent_v3w.csv

---

## Section 8: Hardware, compute, and reproducibility

### Q8.1 GPU
- **NVIDIA GB10** (Thor/Blackwell-class iGPU, compute capability sm_121)
- Single GPU, no multi-GPU
- CUDA 13.0, Driver 580.142
- Container: nvcr.io/nvidia/pytorch:25.11-py3 (PyTorch 2.10.0a0 nightly)
- Note: NGC 24.01 and 25.02 containers do NOT support sm_121; 25.11 was the minimum working container.

### Q8.2 Total compute
- Approximate total GPU-hours for ALL experiments in this conversation: ~15-20 GPU-hours on GB10
- Breakdown: L1 extended 100ep (~4hr), all L1 beta-nudge runs (~4hr), L2 runs (~1hr), L3 runs (~1hr), cascade runs (~0.5hr), MLP baseline (~0.2hr), coherent readout experiments (~1hr), miscellaneous (~3hr)

### Q8.3 Random seed handling
- **Single run per experiment, no fixed seed.** PyTorch default initialization (different each run).
- Exception: some diagnostic scripts use `torch.manual_seed(0)` for reproducibility of shape checks.
- The paper should note: "Results are from single runs; error bars from seed variation are not reported."

### Q8.4 Code availability
- Not yet determined. Current code lives at `/home/slater/code_projects/FICU/ficu_crfm/`.
- No public repo or license yet.

### Q8.5 Checkpoint availability
- Checkpoints exist at `ficu_crfm/checkpoints/`:
  - `l1_phoneme.pt` (38.55% L1 baseline)
  - `l2_phase2_extended.pt` (38.85% L2 baseline)
  - `l3_cascade_retest.pt` (cascade-confirmed L3)
  - Various experimental checkpoints
- Release plan not yet determined.

---

## Section 9: Software dependencies

### Q9.1 Framework
- **PyTorch 2.10.0a0** (NGC nightly, nv25.11 build)
- Host Python 3.13.5 (miniconda); container Python 3.12

### Q9.2 Non-standard dependencies
- `tgt` (TextGrid parser) — used only for LibriSpeech MFA alignment loading, not for TIMIT
- `soundfile` — for reading NIST SPHERE and FLAC audio
- **No torchaudio** — reimplemented mel spectrogram in pure PyTorch due to NGC container incompatibility
- Custom Welford running-statistics implementation (inline in `HolographicAssociativeReadout.update()`)

---

## Section 10: Figures data

### Fig 1: Architecture diagram
- Conceptual; no data file needed. Configuration values documented above.

### Fig 2: L1 TBI dynamics
- `logs/l1_timit_diagnostic.csv` — 10 epochs, columns include TBI_mean, TBI_std per epoch
- `logs/l1_runc_extended.csv` — 30-epoch extended Run C with TBI basin-hopping dynamics

### Fig 3: Amplitude attenuation fix
- **Before fix**: L2_TBI baseline from `logs/l2_phase2_extended.csv` (TBI ~0.225-0.233, drive_gain=0.1)
- **After fix**: L3 calibration runs `logs/l3_beta_cal2_*.csv` show L2_TBI=0.349 (drive_gain=1.0)
- Amplitude measurements were printed to stdout (not CSV): L1 amp=0.0993, L2 amp=0.0013 (before) / 0.0126 (after), L3 amp=0.0001 (before) / 0.0075 (after)

### Fig 4: Classification/cascade separation
- `logs/l3_beta_cal2_0.01.csv` through `logs/l3_beta_cal2_1.0.csv` — L3 val_acc vs L3_TBI at 5 beta values
- Key data: val_acc ~55-56% regardless of TBI (0.22 to 0.59), demonstrating decoupling

### Fig 5: Cascade convergence
- **20-epoch cascade**: `logs/l3_cascade_retest.csv` — L2_TBI rises 0.349 → 0.510
- **50-epoch cascade**: `logs/extended_l3_accuracy.csv` — L2_TBI rises to 0.552
- Both show L1_TBI constant at 0.5237

### Fig 6: Three-layer TBI at final epoch
- From `logs/l3_cascade_retest.csv` epoch 19: L1=0.524, L2=0.510, L3=0.492
- From 50-epoch extended: L1=0.524, L2=0.552, L3=0.518

---

## Discrepancies flagged

### D1: N_PHONEMES is 40, not 39
- The code variable is named `PHONEMES_39` but contains **40 elements** (39 phonemes + silence marker `h#`).
- `N_PHONEMES = len(PHONEMES_39) = 40`
- All training uses 40 classes. The paper should say "40-class phoneme recognition" (or "39 phonemes plus silence").
- Source: `timit_loader.py:48-50`

### D2: n_settle_steps is 24, not 120
- The paper draft may reference "120 settling steps." The actual implementation uses 24 steps at dt=0.07.
- Total settling time = 24 * 0.07 = 1.68 time units.
- Source: all three layers' constructor defaults

### D3: Drive gain was changed mid-project
- Original: 0.1 at all layers
- Final for cascade experiments: L1=0.1, L2=1.0, L3=1.0
- The L2 readout accuracy of 38.85% was measured under drive_gain=0.1
- The cascade result (L2_TBI 0.349→0.510) was measured under drive_gain=1.0
- These two numbers come from DIFFERENT regimes and should be reported with this caveat
- Source: ficu_l2.py:112, ficu_l3.py:110

### D4: coupling_l2 was changed mid-project
- Original: 0.1 * I (same as coupling_l1)
- Final: 1.0 * I (to fix amplitude attenuation at L3)
- The L3 calibration and cascade results use coupling_l2 = 1.0 * I
- Source: ficu_l3.py:54

### D5: TBI formula does not match draft
- Draft says TBI ~ gamma(t) * cos(theta(t) - phi)
- Implementation computes: |E_{spatial}[exp(i * phase_diff)]| averaged over channels
- There is no reference phase phi, no explicit gamma factor. The formula is a Kuramoto-style spatial order parameter of the phase difference between free and nudge fields.

### D6: PETU coherence_floor varies by layer
- L1: 0.45
- L2, L3: 0.35
- The paper should document this per-layer if it discusses PETU thresholds.

### D7: LevelGate was disabled in most reported experiments
- The cascade results, extended L2 runs, and L3 runs all used `--no_gate`
- Only the initial L2 3-epoch run used LevelGate (which triggered immediately at 35% threshold)
