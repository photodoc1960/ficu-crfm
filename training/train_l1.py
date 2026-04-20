"""Phase 1: train L1 phoneme FICU with PETU + LevelGate.

Run as:
    python -m ficu_crfm.training.train_l1 --timit_root /path/to/TCDTIMIT
or with synthetic data (smoke test only — accuracy stays at chance):
    python -m ficu_crfm.training.train_l1 --synthetic
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Support invocation as either:
#   python -m ficu_crfm.training.train_l1  (from repo parent)
#   python -m training.train_l1            (from inside ficu_crfm/)
_THIS = Path(__file__).resolve()
for _p in (_THIS.parents[1], _THIS.parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ficu_crfm.architecture.ficu_l1 import FICUL1Phoneme
from ficu_crfm.architecture.level_gate import LevelGate
from ficu_crfm.dataset.timit_loader import (
    build_dataset, LibriSpeechMFADataset, collate_phonemes, N_PHONEMES,
)
from ficu_crfm.metrics.tbi import compute_TBI
from ficu_crfm.training.petu import should_update_physics, update_fraction


CSV_FIELDS = ['epoch', 'val_acc', 'TBI_mean', 'TBI_std', 'TBI_std_of_std',
              'update_fraction', 'physics_frozen', 'epoch_time_s']

MIN_PHONEME_SEGMENTS = 1000


def validate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            wave = batch['waveform'].to(device)
            label = batch['label'].to(device)
            logits = model(wave)
            correct += (logits.argmax(1) == label).sum().item()
            total += label.size(0)
    return correct / max(total, 1)


def _build_librispeech_datasets(args):
    """Build LibriSpeech phoneme datasets with a speaker-held-out val split.

    `args.timit_root` is reused as the LibriSpeech root. We expect the
    standard unpacked layout produced by the `train-clean-100.tar.gz` tarball
    (`<root>/audio/LibriSpeech/train-clean-100/...`) plus the MFA TextGrids
    under `<root>/alignments/train-clean-100/...`.
    """
    root = Path(args.timit_root)
    audio_root = root / 'audio' / 'LibriSpeech' / 'train-clean-100'
    align_root = root / 'alignments' / 'train-clean-100'
    if not audio_root.exists():
        raise SystemExit(f"train_l1: LibriSpeech audio missing: {audio_root}")
    if not align_root.exists():
        raise SystemExit(f"train_l1: LibriSpeech alignments missing: {align_root}")

    # Held-out val speakers: deterministic stride over sorted speaker ids.
    speakers = sorted(p.name for p in audio_root.iterdir() if p.is_dir())
    stride = max(int(round(1.0 / max(args.val_speaker_frac, 1e-6))), 2)
    val_speakers = set(speakers[::stride])
    print(f"[train_l1] librispeech speakers: total={len(speakers)} "
          f"val={len(val_speakers)} (stride={stride})")

    train_ds = LibriSpeechMFADataset(
        audio_root=str(audio_root), alignment_root=str(align_root),
        split='train-clean-100', level='phoneme',
        phoneme_target_samples=1600,
        speaker_exclude=val_speakers,
        max_utterances=args.max_utterances,
    )
    val_ds_full = LibriSpeechMFADataset(
        audio_root=str(audio_root), alignment_root=str(align_root),
        split='train-clean-100', level='phoneme',
        phoneme_target_samples=1600,
        speaker_include=val_speakers,
    )
    # Cap val items so the val loop doesn't dominate runtime.
    if len(val_ds_full) > args.max_val_items:
        step = max(1, len(val_ds_full) // args.max_val_items)
        indices = list(range(0, len(val_ds_full), step))[:args.max_val_items]
        val_ds = Subset(val_ds_full, indices)
    else:
        val_ds = val_ds_full
    return train_ds, val_ds


def train_l1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train_l1] device: {device} | loader={args.loader}", flush=True)

    if args.loader == 'librispeech':
        train_ds, val_ds = _build_librispeech_datasets(args)
    else:
        train_ds = build_dataset('phoneme', 'train', args.timit_root,
                                 synthetic_n=args.synthetic_n)
        val_ds = build_dataset('phoneme', 'test', args.timit_root,
                               synthetic_n=args.synthetic_n // 2)

    print(f"[train_l1] train={len(train_ds)} val={len(val_ds)}", flush=True)

    # Data guard — refuse to train on suspiciously small datasets.
    if len(train_ds) < MIN_PHONEME_SEGMENTS:
        raise RuntimeError(
            f"Insufficient data — check --timit_root path "
            f"(got {len(train_ds)} train phoneme segments, "
            f"need ≥ {MIN_PHONEME_SEGMENTS})"
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_phonemes,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=collate_phonemes,
                            num_workers=max(args.num_workers // 2, 0),
                            pin_memory=(device.type == 'cuda'))

    model = FICUL1Phoneme(n_classes=N_PHONEMES,
                          n_settle_steps=args.n_settle_steps,
                          beta=args.beta_nudge,
                          readout_mode=args.readout_mode).to(device)
    if args.resume_from is not None:
        state = torch.load(args.resume_from, map_location=device)
        # Filter out keys with shape mismatches (e.g., holographic readout.W
        # [K, 162] vs trajectory readout.W [K, 486]). Physics parameters load
        # cleanly; readout parameters are skipped and reinitialised from scratch.
        model_sd = model.state_dict()
        compat = {k: v for k, v in state.items()
                  if k in model_sd and v.shape == model_sd[k].shape}
        missing, unexpected = model.load_state_dict(compat, strict=False)
        n_loaded = len(compat)
        n_skipped = len(state) - n_loaded
        print(f"[train_l1] resumed from {args.resume_from} "
              f"(loaded={n_loaded} skipped_shape={n_skipped} "
              f"missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    gate = LevelGate(threshold=args.threshold, patience=args.patience, name='L1')

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, args.log_name)
    csv_file = open(log_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    print(f"[train_l1] logging → {log_path}", flush=True)

    n_train_batches = len(train_loader)
    progress_every = max(1, n_train_batches // 10)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        tbi_acc = []
        update_acc = []

        for step, batch in enumerate(train_loader):
            wave = batch['waveform'].to(device)
            label = batch['label'].to(device)

            with torch.no_grad():
                Z_r_free, Z_i_free = model.settle(wave)

                if model.readout_mode == 'trajectory':
                    logits_free = model.readout(model._trajectory)
                    err_per_sample = F.cross_entropy(logits_free, label, reduction='none')
                    model.readout.update(model._trajectory, label,
                                         lr=args.lr_templates)
                else:
                    feat_r_free, feat_i_free = model._phase_features(Z_r_free, Z_i_free)
                    logits_free = model.readout(feat_r_free, feat_i_free)
                    err_per_sample = F.cross_entropy(logits_free, label, reduction='none')

                    # Holographic readout: delta-rule update on FREE phase features.
                    if model.readout_mode == 'holographic':
                        model.readout.update(feat_r_free, feat_i_free, label,
                                             lr=args.lr_templates)

                Z_r_nudge, Z_i_nudge = model.settle(wave, target_class=label)
                tbi = compute_TBI(Z_r_free, Z_i_free, Z_r_nudge, Z_i_nudge)
                tbi_acc.append(tbi)

                # Coherent readout: template EMA on NUDGED phase features.
                if model.readout_mode == 'coherent':
                    feat_r_nudge, feat_i_nudge = model._phase_features(
                        Z_r_nudge, Z_i_nudge
                    )
                    model.readout.update(feat_r_nudge, feat_i_nudge, label,
                                         lr=args.lr_templates)

                mask = should_update_physics(
                    err_per_sample, tbi,
                    num_classes=N_PHONEMES,
                    threshold_frac=args.petu_threshold_frac,
                    coherence_floor=args.petu_coh_floor,
                )
                update_acc.append(mask)

                if mask.any() and not gate.frozen:
                    sel = mask
                    obs_free = model.ep_observables(Z_r_free[sel], Z_i_free[sel])
                    obs_nudge = model.ep_observables(Z_r_nudge[sel], Z_i_nudge[sel])
                    model.apply_ep_update(obs_free, obs_nudge,
                                          lr_physics=args.lr_physics)

            if (step + 1) % progress_every == 0:
                pct = 100.0 * (step + 1) / n_train_batches
                recent_tbi = tbi_acc[-1].mean().item() if tbi_acc else 0.0
                recent_upd = update_acc[-1].float().mean().item() if update_acc else 0.0
                print(f"  [L1 ep{epoch:2d}] step {step+1}/{n_train_batches} "
                      f"({pct:4.1f}%) TBI~{recent_tbi:.3f} upd~{recent_upd:.2f}",
                      flush=True)

        tbi_all = torch.cat(tbi_acc) if tbi_acc else torch.zeros(1)
        update_all = torch.cat(update_acc) if update_acc else torch.zeros(1)

        # Per-batch within-sample std, then std-of-that across batches.
        # Low std_of_std means batches agree on how spread their samples are —
        # i.e. the population homogenization is structural, not a sampling
        # artifact of whichever phonemes happened to land in each batch.
        per_batch_stds = torch.stack([b.std(unbiased=False) for b in tbi_acc]) \
            if len(tbi_acc) > 1 else torch.zeros(1)
        tbi_std_of_std = per_batch_stds.std(unbiased=False).item()

        val_acc = validate(model, val_loader, device)
        elapsed = time.time() - t0

        row = {
            'epoch': epoch,
            'val_acc': f'{val_acc:.4f}',
            'TBI_mean': f'{tbi_all.mean().item():.4f}',
            'TBI_std': f'{tbi_all.std().item():.4f}',
            'TBI_std_of_std': f'{tbi_std_of_std:.4f}',
            'update_fraction': f'{update_fraction(update_all):.4f}',
            'physics_frozen': int(gate.frozen),
            'epoch_time_s': f'{elapsed:.1f}',
        }
        writer.writerow(row)
        csv_file.flush()
        print(f"Epoch {epoch}: val_acc={val_acc*100:5.1f}% | "
              f"TBI={tbi_all.mean().item():.3f}±{tbi_all.std().item():.3f} | "
              f"TBI_std_of_std={tbi_std_of_std:.4f} | "
              f"update_fraction={update_fraction(update_all):.2f} | "
              f"frozen={gate.frozen} ({elapsed:.0f}s)", flush=True)

        if gate.update(val_acc, model):
            print("=== PHASE 1 COMPLETE ===")
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, args.checkpoint_name))
            break

    if not gate.frozen:
        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint_dir, args.checkpoint_name))
        print("[L1] reached max epochs without crossing threshold; checkpoint saved")
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
    p.add_argument('--beta_nudge', type=float, default=0.1,
                   help='EP nudge magnitude (scales the readout-error '
                        'projection applied inside settle()). Larger β means '
                        'stronger directional phase pull; EP physics deltas '
                        'scale as 1/β so the overall update is self-consistent.')
    p.add_argument('--resume_from', default=None,
                   help='path to an existing l1_phoneme.pt checkpoint to '
                        'initialize from (no weight reset)')
    p.add_argument('--readout_mode', choices=['holographic', 'coherent', 'trajectory'],
                   default='holographic',
                   help='L1 classifier head: legacy holographic linear '
                        'readout on pooled phase features, or the new '
                        'ComplexCoherentReadout matched-filter readout '
                        'operating directly on the raw complex field.')
    # LevelGate: --levelgate_threshold and --levelgate_patience are the
    # user-facing names; --threshold / --patience remain as legacy aliases.
    p.add_argument('--levelgate_threshold', '--threshold', dest='threshold',
                   type=float, default=0.60,
                   help='val-acc threshold (0-1) to freeze L1 physics')
    p.add_argument('--levelgate_patience', '--patience', dest='patience',
                   type=int, default=3)
    # PETU: threshold is derived from num_classes (fraction of ln(K)) rather
    # than an absolute CE cutoff — for 39 classes, ln(39)≈3.66, so the
    # default 0.5 fraction gives an effective threshold of ~1.83 nats.
    p.add_argument('--petu_threshold_frac', '--petu_threshold',
                   dest='petu_threshold_frac', type=float, default=0.5,
                   help='fraction of log(num_classes) to use as the "learned '
                        'enough" CE cutoff for PETU')
    p.add_argument('--petu_coh_floor', type=float, default=0.45,
                   help='TBI below this triggers physics update regardless of CE')
    p.add_argument('--lr_physics', type=float, default=0.001)
    p.add_argument('--lr_templates', type=float, default=0.005)
    p.add_argument('--log_dir', default='ficu_crfm/logs')
    p.add_argument('--log_name', default='l1_training.csv')
    p.add_argument('--checkpoint_dir', default='ficu_crfm/checkpoints')
    p.add_argument('--checkpoint_name', default='l1_phoneme.pt')
    # LibriSpeech val split + subsampling knobs
    p.add_argument('--val_speaker_frac', type=float, default=0.05,
                   help='fraction of speakers reserved for val (speaker-held-out)')
    p.add_argument('--max_val_items', type=int, default=2000,
                   help='cap val phoneme segments so the val loop stays fast')
    p.add_argument('--max_utterances', type=int, default=None,
                   help='optional cap on training utterances (debugging)')
    args = p.parse_args()

    # LevelGate threshold convenience: accept percentages ≥1 (e.g. 60 → 0.60)
    if args.threshold > 1.0:
        args.threshold /= 100.0

    if args.synthetic:
        args.timit_root = None
    elif args.loader == 'timit':
        _assert_real_timit(args.timit_root)
    else:  # librispeech
        if args.timit_root is None or not Path(args.timit_root).exists():
            raise SystemExit(
                f"train_l1: --timit_root {args.timit_root} not found for librispeech loader"
            )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    return args


def _assert_real_timit(timit_root):
    """Refuse to start a real training run without TCD-TIMIT data.

    Pass --synthetic to bypass (smoke tests only — accuracy stays at chance).
    """
    if timit_root is None:
        raise SystemExit(
            "train_l1: --timit_root is required for real training. "
            "Pass --synthetic to run a smoke test instead."
        )
    root = Path(timit_root)
    if not root.exists():
        raise SystemExit(f"train_l1: --timit_root {timit_root} does not exist.")
    try:
        next(root.rglob('*.WAV'))
    except StopIteration:
        raise SystemExit(
            f"train_l1: no .WAV files found under {timit_root}. "
            "Did you run extractTCDTIMITaudio.py to extract the corpus?"
        )


if __name__ == '__main__':
    train_l1(parse_args())
