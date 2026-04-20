"""Phase 2: train L2 word-binding FICU on top of frozen L1.

Pre-requirement: a saved checkpoint at ficu_crfm/checkpoints/l1_phoneme.pt.

The L2 layer reads cached L1 field states. Words are converted to fixed-size
mel windows by center-cropping/padding the word's waveform to TARGET_FRAMES
mel frames (re-using the L1 mel front-end), then settling L1 to get its field.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Allow `python -m training.train_l2` and `python -m ficu_crfm.training.train_l2`.
_THIS = Path(__file__).resolve()
for _p in (_THIS.parents[1], _THIS.parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ficu_crfm.architecture.ficu_l1 import FICUL1Phoneme
from ficu_crfm.architecture.ficu_l2 import FICUL2Word
from ficu_crfm.architecture.level_gate import LevelGate
from ficu_crfm.dataset.timit_loader import (
    build_dataset, LibriSpeechMFADataset, collate_words, collate_phonemes,
    N_PHONEMES,
)
from ficu_crfm.dataset.feature_extractor import SAMPLE_RATE
from ficu_crfm.metrics.tbi import compute_TBI
from ficu_crfm.training.petu import should_update_physics, update_fraction


CSV_FIELDS = ['epoch', 'val_acc',
              'L1_TBI_mean', 'L1_TBI_std',
              'L2_TBI_mean', 'L2_TBI_std', 'L2_TBI_std_of_std',
              'update_fraction', 'physics_frozen',
              'lambda_td_L2_L1', 'epoch_time_s']


def _word_to_l1_input(wave: torch.Tensor, target_samples: int) -> torch.Tensor:
    """Center-crop / pad a [B, L] waveform to [B, target_samples]."""
    B, L = wave.shape
    if L == target_samples:
        return wave
    if L > target_samples:
        start = (L - target_samples) // 2
        return wave[:, start:start + target_samples]
    pad = target_samples - L
    return F.pad(wave, (pad // 2, pad - pad // 2))


def _build_l2_datasets(args):
    """Word-level datasets, plus a small phoneme sample for L1 TBI tracking.

    Returns (train_word_ds, val_word_ds, l1_phoneme_sample) where the third
    element is a list of (waveform, phoneme_label) pairs used to recompute
    L1 TBI each epoch as a frozen-baseline sanity check.
    """
    if args.loader == 'librispeech':
        root = Path(args.timit_root)
        audio_root = root / 'audio' / 'LibriSpeech' / 'train-clean-100'
        align_root = root / 'alignments' / 'train-clean-100'
        if not audio_root.exists() or not align_root.exists():
            raise SystemExit(
                f"train_l2: LibriSpeech audio/alignments missing under {root}"
            )
        speakers = sorted(p.name for p in audio_root.iterdir() if p.is_dir())
        stride = max(int(round(1.0 / max(args.val_speaker_frac, 1e-6))), 2)
        val_speakers = set(speakers[::stride])
        print(f"[train_l2] librispeech: {len(speakers)} speakers, "
              f"{len(val_speakers)} held out for val")
        train_word_ds = LibriSpeechMFADataset(
            audio_root=str(audio_root), alignment_root=str(align_root),
            split='train-clean-100', level='word',
            speaker_exclude=val_speakers, max_utterances=args.max_utterances,
        )
        val_word_ds = LibriSpeechMFADataset(
            audio_root=str(audio_root), alignment_root=str(align_root),
            split='train-clean-100', level='word',
            word_vocab=train_word_ds.word_vocab,
            speaker_include=val_speakers,
        )
        # Phoneme sample for L1 TBI baseline.
        l1_ph_ds = LibriSpeechMFADataset(
            audio_root=str(audio_root), alignment_root=str(align_root),
            split='train-clean-100', level='phoneme',
            speaker_include=val_speakers, max_utterances=10,
        )
    else:  # timit
        train_word_ds = build_dataset('word', 'train', args.timit_root,
                                      synthetic_n=args.synthetic_n)
        val_word_ds = build_dataset('word', 'test', args.timit_root,
                                    synthetic_n=args.synthetic_n // 2,
                                    word_vocab=getattr(train_word_ds, 'word_vocab', None))
        l1_ph_ds = build_dataset('phoneme', 'test', args.timit_root,
                                 synthetic_n=args.synthetic_n // 4)

    # Cap val word set for tractable val time.
    if hasattr(val_word_ds, '__len__') and len(val_word_ds) > args.max_val_items:
        step = max(1, len(val_word_ds) // args.max_val_items)
        idx = list(range(0, len(val_word_ds), step))[:args.max_val_items]
        val_word_ds = Subset(val_word_ds, idx)

    # Cap the L1 phoneme sample for the TBI baseline pass (256 items default).
    if hasattr(l1_ph_ds, '__len__') and len(l1_ph_ds) > args.l1_tbi_sample_n:
        step = max(1, len(l1_ph_ds) // args.l1_tbi_sample_n)
        idx = list(range(0, len(l1_ph_ds), step))[:args.l1_tbi_sample_n]
        l1_ph_ds = Subset(l1_ph_ds, idx)

    return train_word_ds, val_word_ds, l1_ph_ds


@torch.no_grad()
def measure_l1_tbi(l1_model, l1_ph_ds, batch_size, device):
    """Run the L1 phoneme sample through free + nudge settles and return
    (mean, std) of TBI. Used as a frozen-baseline sanity check during L2
    training — these numbers should be (numerically) constant across epochs
    if L1 is truly frozen.
    """
    l1_model.eval()
    loader = DataLoader(l1_ph_ds, batch_size=batch_size,
                        collate_fn=collate_phonemes)
    tbis = []
    for batch in loader:
        wave = batch['waveform'].to(device)
        label = batch['label'].to(device)
        Z_r_free, Z_i_free = l1_model.settle(wave)
        Z_r_nudge, Z_i_nudge = l1_model.settle(wave, target_class=label)
        tbi = compute_TBI(Z_r_free, Z_i_free, Z_r_nudge, Z_i_nudge)
        tbis.append(tbi)
    tbi_all = torch.cat(tbis) if tbis else torch.zeros(1)
    return tbi_all.mean().item(), tbi_all.std().item()


def cache_l1_states(l1_model, loader, device, word_target_samples):
    Zr, Zi, labels, speakers = [], [], [], []
    l1_model.eval()
    with torch.no_grad():
        for batch in loader:
            wave = _word_to_l1_input(batch['waveform'], word_target_samples).to(device)
            zr, zi = l1_model.settle(wave)
            Zr.append(zr.cpu())
            Zi.append(zi.cpu())
            labels.append(batch['label'])
            speakers.extend(batch['speaker'])
    return (torch.cat(Zr), torch.cat(Zi), torch.cat(labels), speakers)


def iter_cached(cache, batch_size, shuffle):
    Zr, Zi, lab, _ = cache
    n = Zr.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, batch_size):
        sl = idx[s:s + batch_size]
        yield Zr[sl], Zi[sl], lab[sl]


def validate_cached(l2_model, cache, batch_size, device):
    l2_model.eval()
    correct = total = 0
    for zr, zi, lab in iter_cached(cache, batch_size, shuffle=False):
        zr, zi, lab = zr.to(device), zi.to(device), lab.to(device)
        logits, _ = l2_model(zr, zi)
        correct += (logits.argmax(1) == lab).sum().item()
        total += lab.size(0)
    return correct / max(total, 1)


def train_l2(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train_l2] device: {device} | loader={args.loader}", flush=True)

    train_ds, val_ds, l1_ph_ds = _build_l2_datasets(args)

    if hasattr(train_ds, 'word_vocab'):
        n_classes = len(train_ds.word_vocab)
    else:
        n_classes = train_ds.n_words + 1
    print(f"[train_l2] word vocab={n_classes} train={len(train_ds)} "
          f"val={len(val_ds)} l1_phoneme_sample={len(l1_ph_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_words,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=collate_words,
                            num_workers=max(args.num_workers // 2, 0),
                            pin_memory=(device.type == 'cuda'))

    l1 = FICUL1Phoneme(n_classes=N_PHONEMES,
                       n_settle_steps=args.n_settle_steps).to(device)
    if os.path.exists(args.l1_checkpoint):
        l1.load_state_dict(torch.load(args.l1_checkpoint, map_location=device))
        print(f"[train_l2] loaded L1 baseline from {args.l1_checkpoint}", flush=True)
    else:
        print(f"[train_l2] WARNING no L1 checkpoint at {args.l1_checkpoint} — "
              f"using untrained L1 (smoke-test only)", flush=True)
    l1.eval()
    for p in l1.parameters():
        p.requires_grad = False

    # Initial L1 baseline TBI measurement (sanity reference for the cascade).
    l1_tbi_mean0, l1_tbi_std0 = measure_l1_tbi(
        l1, l1_ph_ds, args.batch_size, device,
    )
    print(f"[train_l2] L1 baseline TBI: {l1_tbi_mean0:.4f} ± {l1_tbi_std0:.4f} "
          f"(measured on {len(l1_ph_ds)} held-out phonemes)", flush=True)

    word_target = int(args.word_window_ms * SAMPLE_RATE / 1000)
    print(f"[train_l2] word window: {args.word_window_ms} ms = {word_target} samples",
          flush=True)

    print("[train_l2] caching L1 states for words...", flush=True)
    train_cache = cache_l1_states(l1, train_loader, device, word_target)
    val_cache = cache_l1_states(l1, val_loader, device, word_target)
    print(f"[train_l2] cached train={tuple(train_cache[0].shape)} "
          f"val={tuple(val_cache[0].shape)}", flush=True)

    l2 = FICUL2Word(n_classes=n_classes,
                    n_settle_steps=args.n_settle_steps).to(device)
    gate = LevelGate(threshold=args.threshold, patience=args.patience, name='L2')

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, args.log_name)
    csv_file = open(log_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    print(f"[train_l2] logging → {log_path}", flush=True)

    n_train_batches = max(1, train_cache[0].shape[0] // args.batch_size)
    progress_every = max(1, n_train_batches // 10)

    for epoch in range(args.epochs):
        t0 = time.time()
        l2.train()
        tbi_acc, update_acc = [], []

        for step, (zr, zi, lab) in enumerate(
            iter_cached(train_cache, args.batch_size, shuffle=True)
        ):
            zr, zi, lab = zr.to(device), zi.to(device), lab.to(device)

            with torch.no_grad():
                Z2_r_free, Z2_i_free = l2.settle(zr, zi)
                feat_r, feat_i = l2._phase_features(Z2_r_free, Z2_i_free)
                logits_free = l2.readout(feat_r, feat_i)
                err_per_sample = F.cross_entropy(logits_free, lab, reduction='none')
                l2.readout.update(feat_r, feat_i, lab, lr=args.lr_templates)

                Z2_r_nudge, Z2_i_nudge = l2.settle(zr, zi, target_class=lab)
                tbi = compute_TBI(Z2_r_free, Z2_i_free, Z2_r_nudge, Z2_i_nudge)
                tbi_acc.append(tbi)

                mask = should_update_physics(
                    err_per_sample, tbi,
                    num_classes=n_classes,
                    threshold_frac=args.petu_threshold_frac,
                    coherence_floor=args.petu_coh_floor,
                )
                update_acc.append(mask)

                if mask.any() and not gate.frozen:
                    obs_free = l2.ep_observables(Z2_r_free[mask], Z2_i_free[mask])
                    obs_nudge = l2.ep_observables(Z2_r_nudge[mask], Z2_i_nudge[mask])
                    l2.apply_ep_update(
                        obs_free, obs_nudge,
                        zr[mask], zi[mask],
                        Z2_r_free[mask], Z2_i_free[mask],
                        Z2_r_nudge[mask], Z2_i_nudge[mask],
                        lr_physics=args.lr_physics,
                    )

            if (step + 1) % progress_every == 0:
                pct = 100.0 * (step + 1) / n_train_batches
                recent_tbi = tbi_acc[-1].mean().item() if tbi_acc else 0.0
                recent_upd = update_acc[-1].float().mean().item() if update_acc else 0.0
                print(f"  [L2 ep{epoch:2d}] step {step+1}/{n_train_batches} "
                      f"({pct:4.1f}%) TBI~{recent_tbi:.3f} upd~{recent_upd:.2f}",
                      flush=True)

        tbi_all = torch.cat(tbi_acc) if tbi_acc else torch.zeros(1)
        update_all = torch.cat(update_acc) if update_acc else torch.zeros(1)

        # Per-batch within-sample std, then std of THAT across batches —
        # validates whether L2's population homogenisation (if any) is real
        # rather than a sampling artifact. Same metric as L1's.
        per_batch_stds = torch.stack([b.std(unbiased=False) for b in tbi_acc]) \
            if len(tbi_acc) > 1 else torch.zeros(1)
        l2_std_of_std = per_batch_stds.std(unbiased=False).item()

        val_acc = validate_cached(l2, val_cache, args.batch_size, device)

        # Frozen-L1 sanity baseline: should be (numerically) constant across
        # epochs. If it drifts, L1 parameters are getting touched somewhere.
        l1_tbi_mean, l1_tbi_std = measure_l1_tbi(
            l1, l1_ph_ds, args.batch_size, device,
        )

        elapsed = time.time() - t0

        row = {
            'epoch': epoch,
            'val_acc': f'{val_acc:.4f}',
            'L1_TBI_mean': f'{l1_tbi_mean:.4f}',
            'L1_TBI_std': f'{l1_tbi_std:.4f}',
            'L2_TBI_mean': f'{tbi_all.mean().item():.4f}',
            'L2_TBI_std': f'{tbi_all.std().item():.4f}',
            'L2_TBI_std_of_std': f'{l2_std_of_std:.4f}',
            'update_fraction': f'{update_fraction(update_all):.4f}',
            'physics_frozen': int(gate.frozen),
            'lambda_td_L2_L1': f'{l2.lambda_td_L2_L1.item():.4f}',
            'epoch_time_s': f'{elapsed:.1f}',
        }
        writer.writerow(row)
        csv_file.flush()
        print(f"Epoch {epoch}: val_acc={val_acc*100:5.1f}% | "
              f"L1_TBI={l1_tbi_mean:.3f}±{l1_tbi_std:.3f} | "
              f"L2_TBI={tbi_all.mean().item():.3f}±{tbi_all.std().item():.3f} | "
              f"upd_frac={update_fraction(update_all):.2f} | "
              f"frozen={gate.frozen} ({elapsed:.0f}s)", flush=True)

        if not args.no_gate and gate.update(val_acc, l2):
            print("=== PHASE 2 COMPLETE ===", flush=True)
            torch.save(l2.state_dict(),
                       os.path.join(args.checkpoint_dir, args.checkpoint_name))
            break

    if not gate.frozen:
        torch.save(l2.state_dict(),
                   os.path.join(args.checkpoint_dir, args.checkpoint_name))
        print("[L2] reached max epochs without crossing threshold; checkpoint saved",
              flush=True)
    csv_file.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--timit_root', default=None,
                   help='TCD-TIMIT root, or LibriSpeech root when --loader librispeech')
    p.add_argument('--loader', choices=['timit', 'librispeech'], default='timit')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--synthetic_n', type=int, default=64)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--n_settle_steps', type=int, default=24)
    # LevelGate: default 0.35 reflects the realistic L2 ceiling on this
    # architecture/data — calibrated similarly to L1's adjusted gate.
    p.add_argument('--levelgate_threshold', '--threshold', dest='threshold',
                   type=float, default=0.35,
                   help='val-acc threshold (0-1) to freeze L2 physics')
    p.add_argument('--levelgate_patience', '--patience', dest='patience',
                   type=int, default=3)
    # PETU: same num_classes-aware calibration as L1. For ~500 word classes,
    # ln(501)≈6.22, so threshold_frac=0.5 gives an effective CE cutoff of ~3.11.
    p.add_argument('--petu_threshold_frac', '--petu_threshold',
                   dest='petu_threshold_frac', type=float, default=0.5,
                   help='fraction of log(num_classes) to use as the CE cutoff for PETU')
    p.add_argument('--petu_coh_floor', type=float, default=0.35)
    p.add_argument('--lr_physics', type=float, default=0.001)
    p.add_argument('--lr_templates', type=float, default=0.005)
    p.add_argument('--word_window_ms', type=float, default=400.0)
    p.add_argument('--l1_checkpoint', default='ficu_crfm/checkpoints/l1_phoneme.pt')
    p.add_argument('--log_dir', default='ficu_crfm/logs')
    p.add_argument('--log_name', default='l2_training.csv')
    p.add_argument('--checkpoint_dir', default='ficu_crfm/checkpoints')
    p.add_argument('--checkpoint_name', default='l2_word.pt')
    # LibriSpeech-specific knobs
    p.add_argument('--val_speaker_frac', type=float, default=0.05)
    p.add_argument('--max_val_items', type=int, default=4000)
    p.add_argument('--max_utterances', type=int, default=None)
    # L1 TBI baseline tracking
    p.add_argument('--l1_tbi_sample_n', type=int, default=256,
                   help='# of held-out phoneme segments used to recompute L1 TBI '
                        'each epoch as a frozen-baseline sanity check')
    p.add_argument('--no_gate', action='store_true',
                   help='Disable LevelGate entirely — physics keeps updating for '
                        'all `--epochs` regardless of val_acc. Use to observe '
                        'full TBI trajectories without early stop.')
    args = p.parse_args()

    # LevelGate threshold convenience: percentages ≥1 are treated as percent.
    if args.threshold > 1.0:
        args.threshold /= 100.0

    if args.synthetic:
        args.timit_root = None
    elif args.timit_root is None:
        raise SystemExit("train_l2: --timit_root is required for real training")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    return args


if __name__ == '__main__':
    train_l2(parse_args())
