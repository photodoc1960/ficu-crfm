"""Phase 3: train L3 sentence FICU + PredictiveField top-down, with the
full L3 → L2 → L1 cascade enabled.

This rewrite differs from the original in three key ways:
  1. L1 is **frozen** (parameters) but its FORWARD is re-run each batch with
     a top-down bias from L2 — wired through `L2.l2_to_l1_topdown_bias()`,
     scaled by `L2.lambda_td_L2_L1`.
  2. L3 → L2 prediction is wired through `PredictiveField.predict()` and
     applied as L2's `predicted_init` (existing path).
  3. Both lambdas are **scheduled** linearly from 0 → `lambda_max` over the
     run rather than learned (the EP gradient for these scalars isn't
     defined; a schedule is the simplest clean test of "does TD feedback
     drive L2 TBI up").

Logging: per epoch we record L1_TBI (frozen baseline on phonemes), L2_TBI
(measured on a held-out word sample with the current cascade configuration),
L3_TBI (from the sentence training loop), L2 word-task val accuracy,
L3 sentence-task val accuracy, plus both lambda values.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Allow `python -m training.train_l3` and the dotted variant.
_THIS = Path(__file__).resolve()
for _p in (_THIS.parents[1], _THIS.parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ficu_crfm.architecture.ficu_l1 import FICUL1Phoneme
from ficu_crfm.architecture.ficu_l2 import FICUL2Word
from ficu_crfm.architecture.ficu_l3 import FICUL3Sentence
from ficu_crfm.architecture.predictive_field import PredictiveField
from ficu_crfm.dataset.timit_loader import (
    build_dataset, collate_sentences, collate_phonemes, collate_words,
    N_PHONEMES, N_SENTENCE_TYPES,
)
from ficu_crfm.dataset.feature_extractor import SAMPLE_RATE
from ficu_crfm.metrics.tbi import compute_TBI
from ficu_crfm.metrics.handshake import handshake_success_rate
from ficu_crfm.training.petu import should_update_physics, update_fraction


CSV_FIELDS = [
    'epoch', 'L3_val_acc', 'L2_val_acc',
    'L1_TBI_mean', 'L1_TBI_std',
    'L2_TBI_mean', 'L2_TBI_std',
    'L3_TBI_mean', 'L3_TBI_std',
    'lambda_td_L2_L1', 'lambda_td_L3_L2',
    'update_fraction', 'prediction_error',
    'epoch_time_s',
]


def _sentence_to_l1_window(wave, target_samples):
    B, L = wave.shape
    if L >= target_samples:
        start = (L - target_samples) // 2
        return wave[:, start:start + target_samples]
    pad = target_samples - L
    return F.pad(wave, (pad // 2, pad - pad // 2))


def _word_to_l1_window(wave, target_samples):
    B, L = wave.shape
    if L >= target_samples:
        start = (L - target_samples) // 2
        return wave[:, start:start + target_samples]
    pad = target_samples - L
    return F.pad(wave, (pad // 2, pad - pad // 2))


# ---------------------------------------------------------------------------
# Cascade forward
# ---------------------------------------------------------------------------

def cascade_forward(l1, l2, l3, pred, wave, sentence_target_samples,
                    use_topdown=True):
    """One full L1 → L2 → L3 → (TD) → L1 → L2 → L3 cascade pass.

    Returns (Z1_td_r, Z1_td_i, Z2_td_r, Z2_td_i, Z3_free_r, Z3_free_i).
    With `use_topdown=False`, the recurrent step is skipped — equivalent to
    the original feed-forward path. Used to disambiguate cascade-induced
    changes from baseline behaviour.
    """
    wave = _sentence_to_l1_window(wave, sentence_target_samples)
    # Initial feed-forward
    Z1_r, Z1_i = l1.settle(wave)                                  # frozen L1
    Z2_r, Z2_i = l2.settle(Z1_r, Z1_i)                            # initial L2
    Z3_r, Z3_i = l3.settle(Z2_r, Z2_i)                            # initial L3

    if not use_topdown:
        return Z1_r, Z1_i, Z2_r, Z2_i, Z3_r, Z3_i

    # Top-down recurrent step
    pred_l2_r, pred_l2_i = pred.predict(Z3_r, Z3_i)               # L3 → L2 bias
    bias_l1_r, bias_l1_i = l2.l2_to_l1_topdown_bias(Z2_r, Z2_i)   # L2 → L1 bias
    Z1_td_r, Z1_td_i = l1.settle(wave, topdown_bias=(bias_l1_r, bias_l1_i))
    Z2_td_r, Z2_td_i = l2.settle(
        Z1_td_r, Z1_td_i, predicted_init=(pred_l2_r, pred_l2_i)
    )
    Z3_td_r, Z3_td_i = l3.settle(Z2_td_r, Z2_td_i)
    return Z1_td_r, Z1_td_i, Z2_td_r, Z2_td_i, Z3_td_r, Z3_td_i


# ---------------------------------------------------------------------------
# Per-epoch baseline measurements
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_l1_baseline_tbi(l1, ph_loader, device):
    """L1 phoneme TBI without any topdown bias — frozen sanity check."""
    l1.eval()
    tbis = []
    for batch in ph_loader:
        wave = batch['waveform'].to(device)
        label = batch['label'].to(device)
        Z_r_free, Z_i_free = l1.settle(wave)
        Z_r_nudge, Z_i_nudge = l1.settle(wave, target_class=label)
        tbis.append(compute_TBI(Z_r_free, Z_i_free, Z_r_nudge, Z_i_nudge))
    tbi = torch.cat(tbis) if tbis else torch.zeros(1)
    return tbi.mean().item(), tbi.std().item()


@torch.no_grad()
def measure_l2_word_tbi_and_acc(l1, l2, l3, pred, word_loader, device,
                                word_target_samples, use_topdown):
    """L2 TBI on a held-out word sample, measured under the current cascade
    config. Also returns word-classification accuracy on the same sample.

    The L2 free state is computed via the full cascade: L1 → L2 → L3 →
    L3-derived L2-bias → L2 (with predicted_init). The L2 nudge state uses
    the same cascade but with the ground-truth word label as nudge target.
    """
    l1.eval(); l2.eval(); l3.eval()
    tbis, correct, total = [], 0, 0
    for batch in word_loader:
        wave = batch['waveform'].to(device)
        label = batch['label'].to(device)
        wave_w = _word_to_l1_window(wave, word_target_samples)
        Z1r, Z1i = l1.settle(wave_w)
        Z2r, Z2i = l2.settle(Z1r, Z1i)
        Z3r, Z3i = l3.settle(Z2r, Z2i)

        if use_topdown:
            pred_l2_r, pred_l2_i = pred.predict(Z3r, Z3i)
            bias_l1_r, bias_l1_i = l2.l2_to_l1_topdown_bias(Z2r, Z2i)
            Z1r_td, Z1i_td = l1.settle(wave_w, topdown_bias=(bias_l1_r, bias_l1_i))
            Z2r_free, Z2i_free = l2.settle(
                Z1r_td, Z1i_td, predicted_init=(pred_l2_r, pred_l2_i)
            )
            Z2r_nudge, Z2i_nudge = l2.settle(
                Z1r_td, Z1i_td, target_class=label,
                predicted_init=(pred_l2_r, pred_l2_i),
            )
        else:
            Z2r_free, Z2i_free = Z2r, Z2i
            Z2r_nudge, Z2i_nudge = l2.settle(Z1r, Z1i, target_class=label)

        tbis.append(compute_TBI(Z2r_free, Z2i_free, Z2r_nudge, Z2i_nudge))

        # Word classification accuracy via L2 readout on free state
        feat_r, feat_i = l2._phase_features(Z2r_free, Z2i_free)
        logits = l2.readout(feat_r, feat_i)
        correct += (logits.argmax(1) == label).sum().item()
        total += label.size(0)

    tbi = torch.cat(tbis) if tbis else torch.zeros(1)
    acc = correct / max(total, 1)
    return tbi.mean().item(), tbi.std().item(), acc


@torch.no_grad()
def validate_l3(l1, l2, l3, pred, val_loader, device,
                sentence_target_samples, use_topdown):
    """Sentence-class validation accuracy via the full cascade."""
    l1.eval(); l2.eval(); l3.eval()
    correct = total = 0
    for batch in val_loader:
        wave = batch['waveform'].to(device)
        label = batch['label'].to(device)
        _, _, _, _, Z3r, Z3i = cascade_forward(
            l1, l2, l3, pred, wave, sentence_target_samples,
            use_topdown=use_topdown,
        )
        feat_r, feat_i = l3._phase_features(Z3r, Z3i)
        logits = l3.readout(feat_r, feat_i)
        correct += (logits.argmax(1) == label).sum().item()
        total += label.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _build_datasets(args):
    train_sent = build_dataset('sentence', 'train', args.timit_root,
                               synthetic_n=args.synthetic_n)
    val_sent = build_dataset('sentence', 'test', args.timit_root,
                             synthetic_n=args.synthetic_n // 2)

    # Held-out phoneme sample for the L1 frozen baseline.
    ph_ds = build_dataset('phoneme', 'test', args.timit_root,
                          synthetic_n=args.synthetic_n // 4)
    if hasattr(ph_ds, '__len__') and len(ph_ds) > args.tbi_sample_n:
        step = max(1, len(ph_ds) // args.tbi_sample_n)
        idx = list(range(0, len(ph_ds), step))[:args.tbi_sample_n]
        ph_ds = Subset(ph_ds, idx)

    # Held-out word sample for the L2 cascade measurement.
    word_ds = build_dataset('word', 'test', args.timit_root,
                            synthetic_n=args.synthetic_n // 4)
    if hasattr(word_ds, '__len__') and len(word_ds) > args.tbi_sample_n:
        step = max(1, len(word_ds) // args.tbi_sample_n)
        idx = list(range(0, len(word_ds), step))[:args.tbi_sample_n]
        word_ds = Subset(word_ds, idx)

    # Cap val sentence set
    if hasattr(val_sent, '__len__') and len(val_sent) > args.max_val_items:
        step = max(1, len(val_sent) // args.max_val_items)
        idx = list(range(0, len(val_sent), step))[:args.max_val_items]
        val_sent = Subset(val_sent, idx)

    return train_sent, val_sent, ph_ds, word_ds


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def train_l3(args):
    # Seed all RNGs for statistical replication across seeds. We do NOT
    # set cudnn.deterministic=True — that would double wall-clock for
    # bitwise reproducibility we don't need. Seeding torch + CUDA + numpy
    # + python random + PYTHONHASHSEED is enough to pin: PredictiveField
    # W_pf init, batch shuffling, PETU mask draws, and any other
    # RNG-driven code paths.
    if getattr(args, 'seed', None) is not None:
        import os, random
        import numpy as np
        seed = int(args.seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[train_l3] seed={seed} (cudnn.deterministic=False)", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train_l3] device: {device}", flush=True)

    train_ds, val_ds, ph_ds, word_ds = _build_datasets(args)
    print(f"[train_l3] sent train={len(train_ds)} val={len(val_ds)} | "
          f"phoneme sample={len(ph_ds)} | word sample={len(word_ds)}",
          flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_sentences,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=collate_sentences,
                            num_workers=max(args.num_workers // 2, 0),
                            pin_memory=(device.type == 'cuda'))
    ph_loader = DataLoader(ph_ds, batch_size=args.batch_size,
                           collate_fn=collate_phonemes)
    word_loader = DataLoader(word_ds, batch_size=args.batch_size,
                             collate_fn=collate_words)

    # ---- L1 frozen ----
    l1 = FICUL1Phoneme(n_classes=N_PHONEMES,
                       n_settle_steps=args.n_settle_steps).to(device)
    if os.path.exists(args.l1_checkpoint):
        l1.load_state_dict(torch.load(args.l1_checkpoint, map_location=device))
        print(f"[train_l3] loaded L1 from {args.l1_checkpoint}", flush=True)
    else:
        raise SystemExit(f"L1 checkpoint not found: {args.l1_checkpoint}")
    l1.eval()
    for p in l1.parameters():
        p.requires_grad = False

    # ---- L2 (NOT frozen — but EP physics not updated in this loop;
    # L2 still adapts via its lambda_td_L2_L1 schedule and via inputs
    # changing under the cascade) ----
    l2_state = torch.load(args.l2_checkpoint, map_location=device)
    n_classes_l2 = l2_state['readout.W'].shape[0]
    l2 = FICUL2Word(n_classes=n_classes_l2,
                    n_settle_steps=args.n_settle_steps).to(device)
    l2.load_state_dict(l2_state)
    print(f"[train_l3] loaded L2 ({n_classes_l2} classes) from "
          f"{args.l2_checkpoint}", flush=True)
    l2.eval()  # eval mode but params still requires_grad
    # Initialize the L2 → L1 lambda to zero; we'll schedule it per epoch.
    with torch.no_grad():
        l2.lambda_td_L2_L1.fill_(0.0)

    # ---- L3 fresh + PredictiveField ----
    l3 = FICUL3Sentence(n_classes=N_SENTENCE_TYPES,
                        n_settle_steps=args.n_settle_steps,
                        beta=args.beta_nudge,
                        drive_gain=args.l3_drive_gain).to(device)
    if args.l3_resume_from is not None:
        state = torch.load(args.l3_resume_from, map_location=device,
                           weights_only=True)
        l3_sd = state.get('l3', state)
        m_sd = l3.state_dict()
        compat = {k: v for k, v in l3_sd.items()
                  if k in m_sd and v.shape == m_sd[k].shape}
        missing, unexpected = l3.load_state_dict(compat, strict=False)
        print(f"[train_l3] L3 resumed from {args.l3_resume_from} "
              f"(loaded={len(compat)}/{len(l3_sd)} "
              f"missing={len(missing)} unexpected={len(unexpected)} "
              f"drive_gain={l3.drive_gain})", flush=True)
    pred = PredictiveField(
        l3_shape=(FICUL3Sentence.CHANNELS, FICUL3Sentence.HEIGHT, FICUL3Sentence.WIDTH),
        l2_shape=(FICUL2Word.CHANNELS, FICUL2Word.HEIGHT, FICUL2Word.WIDTH),
        lr=args.predictor_lr,
        lambda_td=0.0,  # initialize at 0; we'll schedule per epoch
        handshake_threshold=args.handshake_threshold,
    ).to(device)

    sentence_target = int(args.sentence_window_ms * SAMPLE_RATE / 1000)
    word_target = int(args.word_window_ms * SAMPLE_RATE / 1000)
    print(f"[train_l3] sentence window: {args.sentence_window_ms}ms = "
          f"{sentence_target} samples; word window: {args.word_window_ms}ms = "
          f"{word_target} samples", flush=True)

    # CSV logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, args.log_name)
    csv_file = open(log_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    print(f"[train_l3] logging → {log_path}", flush=True)

    # Print a one-time L1 baseline TBI (sanity on the resume).
    l1m, l1s = measure_l1_baseline_tbi(l1, ph_loader, device)
    print(f"[train_l3] L1 baseline TBI (initial): {l1m:.4f} ± {l1s:.4f}",
          flush=True)

    n_train_batches = max(1, len(train_loader))
    progress_every = max(1, n_train_batches // 10)

    for epoch in range(args.epochs):
        t0 = time.time()
        l3.train()

        # Schedule both lambdas linearly from 0 to lambda_max across epochs.
        # ep0 gets lambda 0 (no top-down), ep(epochs-1) gets lambda_max.
        if args.epochs > 1:
            frac = epoch / (args.epochs - 1)
        else:
            frac = 1.0
        cur_lambda = args.lambda_max * frac
        pred.lambda_td = cur_lambda
        with torch.no_grad():
            l2.lambda_td_L2_L1.fill_(cur_lambda)

        tbi_acc = []
        update_acc = []
        perr_acc = []

        for step, batch in enumerate(train_loader):
            wave = batch['waveform'].to(device)
            lab = batch['label'].to(device)

            with torch.no_grad():
                # Cascade forward (free)
                Z1r, Z1i, Z2r, Z2i, Z3r_free, Z3i_free = cascade_forward(
                    l1, l2, l3, pred, wave, sentence_target,
                    use_topdown=True,
                )
                feat_r, feat_i = l3._phase_features(Z3r_free, Z3i_free)
                logits_free = l3.readout(feat_r, feat_i)
                err_per_sample = F.cross_entropy(logits_free, lab, reduction='none')
                l3.readout.update(feat_r, feat_i, lab, lr=args.lr_templates)

                # Predictive-field error and update
                perr = pred.prediction_error(Z3r_free, Z3i_free, Z2r, Z2i)
                perr_acc.append(perr)
                pred.update(Z3r_free, Z3i_free, Z2r, Z2i)

                # L3 nudge from the cascade-refined L2 state
                Z3r_nudge, Z3i_nudge = l3.settle(Z2r, Z2i, target_class=lab)
                tbi = compute_TBI(Z3r_free, Z3i_free, Z3r_nudge, Z3i_nudge)
                tbi_acc.append(tbi)

                mask = should_update_physics(
                    err_per_sample, tbi,
                    num_classes=N_SENTENCE_TYPES,
                    threshold_frac=args.petu_threshold_frac,
                    coherence_floor=args.petu_coh_floor,
                )
                update_acc.append(mask)

                if mask.any():
                    obs_free = l3.ep_observables(Z3r_free[mask], Z3i_free[mask])
                    obs_nudge = l3.ep_observables(Z3r_nudge[mask], Z3i_nudge[mask])
                    l3.apply_ep_update(obs_free, obs_nudge,
                                       lr_physics=args.lr_physics)

            if (step + 1) % progress_every == 0:
                pct = 100.0 * (step + 1) / n_train_batches
                rt = tbi_acc[-1].mean().item() if tbi_acc else 0.0
                ru = update_acc[-1].float().mean().item() if update_acc else 0.0
                print(f"  [L3 ep{epoch:2d}] step {step+1}/{n_train_batches} "
                      f"({pct:4.1f}%) λ={cur_lambda:.3f} L3_TBI~{rt:.3f} "
                      f"upd~{ru:.2f}", flush=True)

        # Aggregate L3 metrics
        tbi_all = torch.cat(tbi_acc) if tbi_acc else torch.zeros(1)
        update_all = torch.cat(update_acc) if update_acc else torch.zeros(1)
        perr_all = torch.cat(perr_acc) if perr_acc else torch.zeros(1)

        # Per-epoch measurements
        l1_tbi_mean, l1_tbi_std = measure_l1_baseline_tbi(l1, ph_loader, device)
        l2_tbi_mean, l2_tbi_std, l2_val_acc = measure_l2_word_tbi_and_acc(
            l1, l2, l3, pred, word_loader, device, word_target,
            use_topdown=True,
        )
        l3_val_acc = validate_l3(l1, l2, l3, pred, val_loader, device,
                                 sentence_target, use_topdown=True)

        elapsed = time.time() - t0

        row = {
            'epoch': epoch,
            'L3_val_acc': f'{l3_val_acc:.4f}',
            'L2_val_acc': f'{l2_val_acc:.4f}',
            'L1_TBI_mean': f'{l1_tbi_mean:.4f}',
            'L1_TBI_std': f'{l1_tbi_std:.4f}',
            'L2_TBI_mean': f'{l2_tbi_mean:.4f}',
            'L2_TBI_std': f'{l2_tbi_std:.4f}',
            'L3_TBI_mean': f'{tbi_all.mean().item():.4f}',
            'L3_TBI_std': f'{tbi_all.std().item():.4f}',
            'lambda_td_L2_L1': f'{l2.lambda_td_L2_L1.item():.4f}',
            'lambda_td_L3_L2': f'{pred.lambda_td:.4f}',
            'update_fraction': f'{update_fraction(update_all):.4f}',
            'prediction_error': f'{perr_all.mean().item():.4f}',
            'epoch_time_s': f'{elapsed:.1f}',
        }
        writer.writerow(row)
        csv_file.flush()
        print(
            f"Epoch {epoch}: L3_val={l3_val_acc*100:5.1f}% L2_val={l2_val_acc*100:5.1f}% | "
            f"L1_TBI={l1_tbi_mean:.3f}±{l1_tbi_std:.3f} | "
            f"L2_TBI={l2_tbi_mean:.3f}±{l2_tbi_std:.3f} | "
            f"L3_TBI={tbi_all.mean().item():.3f}±{tbi_all.std().item():.3f} | "
            f"λ_L3L2={cur_lambda:.3f} λ_L2L1={cur_lambda:.3f} | "
            f"upd={update_fraction(update_all):.2f} ({elapsed:.0f}s)",
            flush=True,
        )

    torch.save({'l3': l3.state_dict(), 'l2': l2.state_dict(), 'pred': pred.state_dict()},
               os.path.join(args.checkpoint_dir, args.checkpoint_name))
    print("=== PHASE 3 COMPLETE ===", flush=True)
    csv_file.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--timit_root', default=None)
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--synthetic_n', type=int, default=64)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=None,
                   help='RNG seed for statistical replication. Sets '
                        'torch/CUDA/numpy/random/PYTHONHASHSEED but leaves '
                        'cudnn.deterministic=False to avoid 2x wall-clock cost.')
    p.add_argument('--n_settle_steps', type=int, default=24)
    p.add_argument('--beta_nudge', type=float, default=0.1,
                   help='EP nudge magnitude for L3')
    p.add_argument('--l3_drive_gain', type=float, default=1.0,
                   help='Sensory drive gain at L3 (1.0 = healthy regime, '
                        '0.1 = attenuated regime matching the Figure 4 ablation)')
    p.add_argument('--l3_resume_from', default=None,
                   help='Optional checkpoint to load L3 starting state from '
                        '(physics + readout). Used to plug a pre-calibrated '
                        'L3 into the cascade without retraining from scratch.')
    p.add_argument('--petu_threshold_frac', '--petu_threshold',
                   dest='petu_threshold_frac', type=float, default=0.5)
    p.add_argument('--petu_coh_floor', type=float, default=0.35)
    p.add_argument('--lr_physics', type=float, default=0.001)
    p.add_argument('--lr_templates', type=float, default=0.005)
    p.add_argument('--predictor_lr', type=float, default=0.01)
    p.add_argument('--lambda_max', type=float, default=0.5,
                   help='Max value of lambda_td for both L3→L2 and L2→L1 — '
                        'both are scheduled linearly from 0 → lambda_max '
                        'over `--epochs` epochs.')
    p.add_argument('--handshake_threshold', type=float, default=1.0)
    p.add_argument('--sentence_window_ms', type=float, default=2000.0)
    p.add_argument('--word_window_ms', type=float, default=400.0)
    p.add_argument('--max_val_items', type=int, default=2000)
    p.add_argument('--tbi_sample_n', type=int, default=256,
                   help='# of held-out phonemes/words used for the L1 and L2 '
                        'baseline TBI measurements each epoch')
    p.add_argument('--l1_checkpoint',
                   default='ficu_crfm/checkpoints/l1_phoneme.pt')
    p.add_argument('--l2_checkpoint',
                   default='ficu_crfm/checkpoints/l2_phase2_extended.pt')
    p.add_argument('--log_dir', default='ficu_crfm/logs')
    p.add_argument('--log_name', default='l3_training.csv')
    p.add_argument('--checkpoint_dir', default='ficu_crfm/checkpoints')
    p.add_argument('--checkpoint_name', default='l3_sentence.pt')
    args = p.parse_args()
    if args.synthetic:
        args.timit_root = None
    elif args.timit_root is None:
        raise SystemExit("train_l3: --timit_root is required for real training")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    return args


if __name__ == '__main__':
    train_l3(parse_args())
