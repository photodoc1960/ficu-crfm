"""Extended accuracy runs for the paper — L1/L2/L3 readout maximization + MLP baseline.

Usage (from the container):
    python -u -m training.extended_accuracy \
        --timit_root /data/timit/TIMIT \
        --phase {l1,l2,l3,mlp,all}
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
for _p in (_THIS.parents[1], _THIS.parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn as nn
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
from ficu_crfm.dataset.feature_extractor import MelFrontEnd, SAMPLE_RATE
from ficu_crfm.metrics.tbi import compute_TBI


def cosine_lr(step, total_steps, lr_max, lr_min):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / total_steps))


# ---------------------------------------------------------------------------
# Phase 1: Extended L1 readout (phoneme, physics frozen)
# ---------------------------------------------------------------------------

def phase_l1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nPhase 1: Extended L1 readout (phoneme)\n{'='*60}", flush=True)

    train_ds = build_dataset('phoneme', 'train', args.timit_root)
    val_ds = build_dataset('phoneme', 'test', args.timit_root)
    print(f"  train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_phonemes, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=collate_phonemes, num_workers=2,
                            pin_memory=True)

    model = FICUL1Phoneme(n_classes=N_PHONEMES, n_settle_steps=24,
                          readout_mode=args.readout_mode).to(device)
    state = torch.load(args.l1_checkpoint, map_location=device)
    # Filter for shape compatibility (trajectory readout has different W shape).
    model_sd = model.state_dict()
    compat = {k: v for k, v in state.items()
              if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(compat, strict=False)
    print(f"  loaded {len(compat)}/{len(state)} params from {args.l1_checkpoint} "
          f"(readout_mode={args.readout_mode})", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    log_path = os.path.join(args.log_dir, 'extended_l1_accuracy.csv')
    csv_f = open(log_path, 'w', newline='')
    writer = csv.DictWriter(csv_f, fieldnames=['epoch', 'val_acc', 'lr', 'epoch_time_s'])
    writer.writeheader()

    total_steps = args.l1_epochs * len(train_loader)
    global_step = 0

    for epoch in range(args.l1_epochs):
        t0 = time.time()
        for batch in train_loader:
            wave = batch['waveform'].to(device)
            label = batch['label'].to(device)
            lr = cosine_lr(global_step, total_steps, args.lr_max, args.lr_min)
            with torch.no_grad():
                Z_r, Z_i = model.settle(wave)
                if args.readout_mode == 'trajectory':
                    model.readout.update(model._trajectory, label, lr=lr)
                else:
                    feat_r, feat_i = model._phase_features(Z_r, Z_i)
                    model.readout.update(feat_r, feat_i, label, lr=lr)
            global_step += 1

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                wave = batch['waveform'].to(device)
                label = batch['label'].to(device)
                logits = model(wave)
                correct += (logits.argmax(1) == label).sum().item()
                total += label.size(0)
        val_acc = correct / max(total, 1)
        elapsed = time.time() - t0

        row = {'epoch': epoch, 'val_acc': f'{val_acc:.4f}',
               'lr': f'{lr:.6f}', 'epoch_time_s': f'{elapsed:.1f}'}
        writer.writerow(row)
        csv_f.flush()

        if epoch % 10 == 0 or epoch == args.l1_epochs - 1:
            print(f"  L1 ep{epoch:3d}: val_acc={val_acc*100:5.1f}% lr={lr:.6f} ({elapsed:.0f}s)",
                  flush=True)

    csv_f.close()
    torch.save(model.state_dict(),
               os.path.join(args.checkpoint_dir, 'l1_extended.pt'))
    print(f"  L1 final val_acc: {val_acc*100:.1f}%", flush=True)
    return val_acc


# ---------------------------------------------------------------------------
# Phase 2: Extended L2 readout (word, L1 frozen)
# ---------------------------------------------------------------------------

def _word_to_l1(wave, target_samples):
    B, L = wave.shape
    if L >= target_samples:
        start = (L - target_samples) // 2
        return wave[:, start:start + target_samples]
    pad = target_samples - L
    return F.pad(wave, (pad // 2, pad - pad // 2))


def phase_l2(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nPhase 2: Extended L2 readout (word)\n{'='*60}", flush=True)

    train_ds = build_dataset('word', 'train', args.timit_root)
    val_ds = build_dataset('word', 'test', args.timit_root)
    if hasattr(train_ds, 'word_vocab'):
        n_classes = len(train_ds.word_vocab)
    else:
        n_classes = train_ds.n_words + 1
    print(f"  vocab={n_classes} train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_words, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=collate_words, num_workers=2,
                            pin_memory=True)

    l1 = FICUL1Phoneme(n_classes=N_PHONEMES, n_settle_steps=24).to(device)
    l1.load_state_dict(torch.load(args.l1_checkpoint, map_location=device))
    l1.eval()
    for p in l1.parameters():
        p.requires_grad = False

    l2_state = torch.load(args.l2_checkpoint, map_location=device)
    l2 = FICUL2Word(n_classes=n_classes, n_settle_steps=24).to(device)
    l2.load_state_dict(l2_state, strict=False)
    l2.eval()
    for p in l2.parameters():
        p.requires_grad = False

    word_target = int(400 * SAMPLE_RATE / 1000)

    # Cache L1 states
    print("  Caching L1 states...", flush=True)
    from ficu_crfm.training.train_l2 import cache_l1_states
    train_cache = cache_l1_states(l1, train_loader, device, word_target)
    val_cache = cache_l1_states(l1, val_loader, device, word_target)
    print(f"  Cached train={tuple(train_cache[0].shape)} val={tuple(val_cache[0].shape)}", flush=True)

    from ficu_crfm.training.train_l2 import iter_cached, validate_cached

    log_path = os.path.join(args.log_dir, 'extended_l2_accuracy.csv')
    csv_f = open(log_path, 'w', newline='')
    writer = csv.DictWriter(csv_f, fieldnames=['epoch', 'val_acc', 'L2_TBI_mean', 'lr', 'epoch_time_s'])
    writer.writeheader()

    n_batches = max(1, train_cache[0].shape[0] // args.batch_size)
    total_steps = args.l2_epochs * n_batches
    global_step = 0

    for epoch in range(args.l2_epochs):
        t0 = time.time()
        tbi_acc = []
        for zr, zi, lab in iter_cached(train_cache, args.batch_size, shuffle=True):
            zr, zi, lab = zr.to(device), zi.to(device), lab.to(device)
            lr = cosine_lr(global_step, total_steps, args.lr_max, args.lr_min)
            with torch.no_grad():
                Z2r, Z2i = l2.settle(zr, zi)
                feat_r, feat_i = l2._phase_features(Z2r, Z2i)
                l2.readout.update(feat_r, feat_i, lab, lr=lr)
                # TBI every 10th batch
                if global_step % 10 == 0:
                    Z2rn, Z2in = l2.settle(zr, zi, target_class=lab)
                    tbi_acc.append(compute_TBI(Z2r, Z2i, Z2rn, Z2in))
            global_step += 1

        val_acc = validate_cached(l2, val_cache, args.batch_size, device)
        tbi_all = torch.cat(tbi_acc) if tbi_acc else torch.zeros(1)
        elapsed = time.time() - t0

        row = {'epoch': epoch, 'val_acc': f'{val_acc:.4f}',
               'L2_TBI_mean': f'{tbi_all.mean().item():.4f}',
               'lr': f'{lr:.6f}', 'epoch_time_s': f'{elapsed:.1f}'}
        writer.writerow(row)
        csv_f.flush()

        if epoch % 10 == 0 or epoch == args.l2_epochs - 1:
            print(f"  L2 ep{epoch:3d}: val_acc={val_acc*100:5.1f}% "
                  f"TBI={tbi_all.mean().item():.3f} lr={lr:.6f} ({elapsed:.0f}s)",
                  flush=True)

    csv_f.close()
    torch.save(l2.state_dict(),
               os.path.join(args.checkpoint_dir, 'l2_extended.pt'))
    print(f"  L2 final val_acc: {val_acc*100:.1f}%", flush=True)
    return val_acc


# ---------------------------------------------------------------------------
# Phase 3: Extended L3 cascade
# ---------------------------------------------------------------------------

def phase_l3(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nPhase 3: Extended L3 cascade\n{'='*60}", flush=True)

    # Re-use the cascade training script
    from ficu_crfm.training.train_l3 import train_l3

    class L3Args:
        pass
    a = L3Args()
    a.timit_root = args.timit_root
    a.synthetic = False
    a.synthetic_n = 64
    a.epochs = args.l3_epochs
    a.batch_size = args.batch_size
    a.num_workers = 4
    a.n_settle_steps = 24
    a.beta_nudge = 0.05
    a.petu_threshold_frac = 0.5
    a.petu_coh_floor = 0.35
    a.lr_physics = 0.001
    a.lr_templates = 0.005
    a.predictor_lr = 0.01
    a.lambda_max = 5.0
    a.handshake_threshold = 1.0
    a.sentence_window_ms = 2000.0
    a.word_window_ms = 400.0
    a.max_val_items = 2000
    a.tbi_sample_n = 256
    a.l1_checkpoint = args.l1_checkpoint
    a.l2_checkpoint = args.l2_checkpoint
    a.log_dir = args.log_dir
    a.log_name = 'extended_l3_accuracy.csv'
    a.checkpoint_dir = args.checkpoint_dir
    a.checkpoint_name = 'l3_extended.pt'
    a.no_gate = True

    train_l3(a)


# ---------------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, n_classes, hidden1=512, hidden2=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def phase_mlp(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nMLP Baseline (word classification)\n{'='*60}", flush=True)

    train_ds = build_dataset('word', 'train', args.timit_root)
    val_ds = build_dataset('word', 'test', args.timit_root)
    if hasattr(train_ds, 'word_vocab'):
        n_classes = len(train_ds.word_vocab)
    else:
        n_classes = train_ds.n_words + 1
    print(f"  vocab={n_classes} train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_words, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=collate_words, num_workers=2,
                            pin_memory=True)

    mel = MelFrontEnd().to(device)
    word_target = int(400 * SAMPLE_RATE / 1000)

    # Determine input dim from a single sample
    with torch.no_grad():
        sample_wave = torch.randn(1, word_target, device=device)
        sample_feat = mel(sample_wave)
        input_dim = sample_feat.flatten(1).shape[1]
    print(f"  MLP input_dim={input_dim} (mel features flattened)", flush=True)

    model = MLPBaseline(input_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    total_steps = args.mlp_epochs * len(train_loader)

    log_path = os.path.join(args.log_dir, 'mlp_baseline.csv')
    csv_f = open(log_path, 'w', newline='')
    writer = csv.DictWriter(csv_f, fieldnames=['epoch', 'val_acc', 'train_loss', 'lr', 'epoch_time_s'])
    writer.writeheader()

    global_step = 0
    for epoch in range(args.mlp_epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            wave = batch['waveform'].to(device)
            label = batch['label'].to(device)

            # Crop/pad to fixed word window, compute mel features
            wave = _word_to_l1(wave, word_target)
            with torch.no_grad():
                feats = mel(wave).flatten(1)  # [B, input_dim]

            logits = model(feats)
            loss = F.cross_entropy(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Cosine LR
            lr = cosine_lr(global_step, total_steps, args.lr_max, args.lr_min)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                wave = batch['waveform'].to(device)
                label = batch['label'].to(device)
                wave = _word_to_l1(wave, word_target)
                feats = mel(wave).flatten(1)
                logits = model(feats)
                correct += (logits.argmax(1) == label).sum().item()
                total += label.size(0)
        val_acc = correct / max(total, 1)
        elapsed = time.time() - t0

        row = {'epoch': epoch, 'val_acc': f'{val_acc:.4f}',
               'train_loss': f'{total_loss/n_batches:.4f}',
               'lr': f'{lr:.6f}', 'epoch_time_s': f'{elapsed:.1f}'}
        writer.writerow(row)
        csv_f.flush()

        if epoch % 10 == 0 or epoch == args.mlp_epochs - 1:
            print(f"  MLP ep{epoch:3d}: val_acc={val_acc*100:5.1f}% "
                  f"loss={total_loss/n_batches:.3f} lr={lr:.6f} ({elapsed:.0f}s)",
                  flush=True)

    csv_f.close()
    torch.save(model.state_dict(),
               os.path.join(args.checkpoint_dir, 'mlp_baseline.pt'))
    print(f"  MLP final val_acc: {val_acc*100:.1f}%", flush=True)
    return val_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--timit_root', required=True)
    p.add_argument('--phase', choices=['l1', 'l2', 'l3', 'mlp', 'all'], default='all')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--l1_epochs', type=int, default=100)
    p.add_argument('--l2_epochs', type=int, default=100)
    p.add_argument('--l3_epochs', type=int, default=50)
    p.add_argument('--mlp_epochs', type=int, default=100)
    p.add_argument('--lr_max', type=float, default=1e-3)
    p.add_argument('--lr_min', type=float, default=1e-5)
    p.add_argument('--readout_mode', choices=['holographic', 'trajectory'],
                   default='holographic')
    p.add_argument('--l1_checkpoint',
                   default='ficu_crfm/checkpoints/l1_phoneme.pt')
    p.add_argument('--l2_checkpoint',
                   default='ficu_crfm/checkpoints/l2_phase2_extended.pt')
    p.add_argument('--log_dir', default='ficu_crfm/logs')
    p.add_argument('--checkpoint_dir', default='ficu_crfm/checkpoints')
    args = p.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    return args


def main():
    args = parse_args()
    results = {}

    if args.phase in ('l1', 'all'):
        results['L1_phoneme'] = phase_l1(args)
    if args.phase in ('l2', 'all'):
        results['L2_word'] = phase_l2(args)
    if args.phase in ('l3', 'all'):
        phase_l3(args)
    if args.phase in ('mlp', 'all'):
        results['MLP_word'] = phase_mlp(args)

    if results:
        print(f"\n{'='*60}\nFinal accuracies:\n{'='*60}")
        for name, acc in results.items():
            print(f"  {name}: {acc*100:.1f}%")


if __name__ == '__main__':
    main()
