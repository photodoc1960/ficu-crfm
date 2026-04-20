"""Measure per-sample TBI distributions at cascade endpoint for all 3 layers.

Loads the cascade-trained checkpoint (l3_cascade_retest.pt), runs the full
cascade forward on the TIMIT test set, and saves per-sample TBI arrays +
summary statistics for Figure 6 error bars.

Usage:
    python figures/measure_cascade_tbi_distributions.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))

import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ficu_crfm.architecture.ficu_l1 import FICUL1Phoneme
from ficu_crfm.architecture.ficu_l2 import FICUL2Word
from ficu_crfm.architecture.ficu_l3 import FICUL3Sentence
from ficu_crfm.architecture.predictive_field import PredictiveField
from ficu_crfm.dataset.timit_loader import (
    build_dataset, collate_phonemes, collate_words, collate_sentences,
    N_PHONEMES, N_SENTENCE_TYPES,
)
from ficu_crfm.dataset.feature_extractor import SAMPLE_RATE
from ficu_crfm.metrics.tbi import compute_TBI

TIMIT = '/home/slater/data/timit/TIMIT'
CKPT = _ROOT / 'checkpoints'
OUT = _ROOT / 'figures'


def word_to_l1(wave, target=6400):
    B, L = wave.shape
    if L >= target:
        s = (L - target) // 2
        return wave[:, s:s + target]
    return F.pad(wave, ((target - L) // 2, target - L - (target - L) // 2))


def sentence_to_l1(wave, target=32000):
    B, L = wave.shape
    if L >= target:
        s = (L - target) // 2
        return wave[:, s:s + target]
    return F.pad(wave, ((target - L) // 2, target - L - (target - L) // 2))


def distribution_stats(arr, name):
    q5, q25, q50, q75, q95 = np.percentile(arr, [5, 25, 50, 75, 95])
    iqr = q75 - q25
    skew = float(((arr - arr.mean())**3).mean() / (arr.std()**3 + 1e-12))
    frac_zero = float((arr < 0.05).mean())
    stats = {
        'name': name,
        'n': len(arr),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'median': float(q50),
        'q25': float(q25),
        'q75': float(q75),
        'q5': float(q5),
        'q95': float(q95),
        'iqr': float(iqr),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'skewness': skew,
        'frac_tbi_lt_005': frac_zero,
    }
    return stats


def print_stats(s):
    print(f"  {s['name']} (n={s['n']}):")
    print(f"    mean={s['mean']:.4f}  std={s['std']:.4f}")
    print(f"    median={s['median']:.4f}  Q25={s['q25']:.4f}  Q75={s['q75']:.4f}  IQR={s['iqr']:.4f}")
    print(f"    Q5={s['q5']:.4f}  Q95={s['q95']:.4f}")
    print(f"    min={s['min']:.4f}  max={s['max']:.4f}")
    print(f"    skewness={s['skewness']:.3f}")
    print(f"    frac(TBI < 0.05) = {s['frac_tbi_lt_005']:.3f} ({s['frac_tbi_lt_005']*100:.1f}%)")


def main():
    device = torch.device('cpu')  # fast enough for ~1-2k samples
    print("Loading models...", flush=True)

    # L1 frozen
    l1 = FICUL1Phoneme(n_classes=N_PHONEMES, n_settle_steps=24).to(device)
    l1.load_state_dict(torch.load(CKPT / 'l1_phoneme.pt', map_location=device))
    l1.eval()

    # L2 from cascade checkpoint
    cascade_ckpt = torch.load(CKPT / 'l3_cascade_retest.pt', map_location=device)
    l2_state = cascade_ckpt.get('l2', None)
    if l2_state is None:
        # Fall back to the pre-cascade L2
        l2_state = torch.load(CKPT / 'l2_phase2_extended.pt', map_location=device)
        print("  WARNING: cascade checkpoint has no 'l2' key, using pre-cascade L2")
    n_l2 = l2_state['readout.W'].shape[0]
    l2 = FICUL2Word(n_classes=n_l2, n_settle_steps=24).to(device)
    l2.load_state_dict(l2_state, strict=False)
    l2.eval()

    # L3 from cascade checkpoint
    l3_state = cascade_ckpt.get('l3', cascade_ckpt)
    l3 = FICUL3Sentence(n_classes=N_SENTENCE_TYPES, n_settle_steps=24, beta=0.05).to(device)
    l3.load_state_dict(l3_state, strict=False)
    l3.eval()

    # PredictiveField
    pred_state = cascade_ckpt.get('pred', None)
    pred = PredictiveField(
        l3_shape=(3, 24, 16), l2_shape=(3, 47, 32),
        lr=0.01, lambda_td=5.0, handshake_threshold=1.0,
    ).to(device)
    if pred_state is not None:
        pred.load_state_dict(pred_state, strict=False)
    pred.eval()

    # Set lambda to cascade endpoint value
    with torch.no_grad():
        l2.lambda_td_L2_L1.fill_(5.0)
    pred.lambda_td = 5.0

    # ---- L1 TBI (frozen baseline, phoneme test set) ----
    print("\nMeasuring L1 TBI on phoneme test set...", flush=True)
    ph_ds = build_dataset('phoneme', 'test', TIMIT)
    ph_loader = DataLoader(ph_ds, batch_size=64, collate_fn=collate_phonemes)
    l1_tbis = []
    with torch.no_grad():
        for i, batch in enumerate(ph_loader):
            if i >= 20: break
            wave, label = batch['waveform'], batch['label']
            zfr, zfi = l1.settle(wave)
            znr, zni = l1.settle(wave, target_class=label)
            l1_tbis.append(compute_TBI(zfr, zfi, znr, zni))
    l1_tbi = torch.cat(l1_tbis).numpy()
    l1_stats = distribution_stats(l1_tbi, 'L1 (frozen, phonemes)')
    print_stats(l1_stats)

    # ---- L2 TBI with cascade (word test set, full cascade forward) ----
    print("\nMeasuring L2 TBI with cascade on word test set...", flush=True)
    wd_ds = build_dataset('word', 'test', TIMIT)
    wd_loader = DataLoader(wd_ds, batch_size=64, collate_fn=collate_words)
    l2_tbis = []
    with torch.no_grad():
        for i, batch in enumerate(wd_loader):
            if i >= 20: break
            wave = word_to_l1(batch['waveform'], 6400)
            label = batch['label']
            # Full cascade forward
            z1r, z1i = l1.settle(wave)
            z2r_init, z2i_init = l2.settle(z1r, z1i)
            z3r, z3i = l3.settle(z2r_init, z2i_init)
            pred_l2_r, pred_l2_i = pred.predict(z3r, z3i)
            bias_l1_r, bias_l1_i = l2.l2_to_l1_topdown_bias(z2r_init, z2i_init)
            z1r_td, z1i_td = l1.settle(wave, topdown_bias=(bias_l1_r, bias_l1_i))
            z2r_free, z2i_free = l2.settle(z1r_td, z1i_td,
                                            predicted_init=(pred_l2_r, pred_l2_i))
            z2r_nudge, z2i_nudge = l2.settle(z1r_td, z1i_td, target_class=label,
                                              predicted_init=(pred_l2_r, pred_l2_i))
            l2_tbis.append(compute_TBI(z2r_free, z2i_free, z2r_nudge, z2i_nudge))
    l2_tbi = torch.cat(l2_tbis).numpy()
    l2_stats = distribution_stats(l2_tbi, 'L2 (cascade endpoint, words)')
    print_stats(l2_stats)

    # ---- Also measure L2 PRE-cascade for comparison ----
    print("\nMeasuring L2 TBI PRE-cascade on word test set...", flush=True)
    l2_pre_tbis = []
    with torch.no_grad():
        for i, batch in enumerate(wd_loader):
            if i >= 20: break
            wave = word_to_l1(batch['waveform'], 6400)
            label = batch['label']
            z1r, z1i = l1.settle(wave)
            z2r_free, z2i_free = l2.settle(z1r, z1i)
            z2r_nudge, z2i_nudge = l2.settle(z1r, z1i, target_class=label)
            l2_pre_tbis.append(compute_TBI(z2r_free, z2i_free, z2r_nudge, z2i_nudge))
    l2_pre_tbi = torch.cat(l2_pre_tbis).numpy()
    l2_pre_stats = distribution_stats(l2_pre_tbi, 'L2 (pre-cascade, words)')
    print_stats(l2_pre_stats)

    # ---- L3 TBI with cascade (sentence test set) ----
    print("\nMeasuring L3 TBI with cascade on sentence test set...", flush=True)
    st_ds = build_dataset('sentence', 'test', TIMIT)
    st_loader = DataLoader(st_ds, batch_size=32, collate_fn=collate_sentences)
    l3_tbis = []
    with torch.no_grad():
        for i, batch in enumerate(st_loader):
            if i >= 20: break
            wave = sentence_to_l1(batch['waveform'], 32000)
            label = batch['label']
            z1r, z1i = l1.settle(wave)
            z2r, z2i = l2.settle(z1r, z1i)
            z3r_free, z3i_free = l3.settle(z2r, z2i)
            z3r_nudge, z3i_nudge = l3.settle(z2r, z2i, target_class=label)
            l3_tbis.append(compute_TBI(z3r_free, z3i_free, z3r_nudge, z3i_nudge))
    l3_tbi = torch.cat(l3_tbis).numpy()
    l3_stats = distribution_stats(l3_tbi, 'L3 (cascade endpoint, sentences)')
    print_stats(l3_stats)

    # Save all arrays and stats
    np.savez(OUT / 'cascade_tbi_distributions.npz',
             l1_tbi=l1_tbi, l2_tbi=l2_tbi, l2_pre_tbi=l2_pre_tbi, l3_tbi=l3_tbi)

    all_stats = {
        'l1': l1_stats,
        'l2_cascade': l2_stats,
        'l2_pre_cascade': l2_pre_stats,
        'l3': l3_stats,
    }
    with open(OUT / 'cascade_tbi_stats.json', 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nSaved distributions to {OUT / 'cascade_tbi_distributions.npz'}")
    print(f"Saved stats to {OUT / 'cascade_tbi_stats.json'}")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Cascade-endpoint TBI distributions")
    print("="*70)
    print(f"{'Layer':<30} {'Median':>8} {'IQR':>12} {'Skew':>8} {'%<0.05':>8}")
    print("-"*70)
    for s in [l1_stats, l2_pre_stats, l2_stats, l3_stats]:
        print(f"{s['name']:<30} {s['median']:8.4f} [{s['q25']:.3f},{s['q75']:.3f}] {s['skewness']:8.3f} {s['frac_tbi_lt_005']*100:7.1f}%")


if __name__ == '__main__':
    main()
