"""Verify that the saved attenuated L3 checkpoint (drive_gain=0.1, beta=0.05)
loads correctly and reproduces the published operating point:
  - L3_TBI mean in [0.024, 0.040]
  - L3 val_acc  in [56.0%, 56.8%]
No training updates. Pure forward measurement.
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ficu_crfm.architecture.ficu_l1 import FICUL1Phoneme
from ficu_crfm.architecture.ficu_l2 import FICUL2Word
from ficu_crfm.architecture.ficu_l3 import FICUL3Sentence
from ficu_crfm.architecture.predictive_field import PredictiveField
from ficu_crfm.dataset.timit_loader import (
    build_dataset, collate_sentences, N_PHONEMES, N_SENTENCE_TYPES,
)
from ficu_crfm.metrics.tbi import compute_TBI
from ficu_crfm.training.train_l3 import cascade_forward, validate_l3

SAMPLE_RATE = 16000


@torch.no_grad()
def measure_l3_tbi(l1, l2, l3, pred, loader, device, sentence_target, use_topdown):
    """Free vs nudge TBI on the L3 (sentence) output."""
    l1.eval(); l2.eval(); l3.eval()
    tbis = []
    for batch in loader:
        wave = batch['waveform'].to(device)
        label = batch['label'].to(device)
        _, _, _, _, Z3r_free, Z3i_free = cascade_forward(
            l1, l2, l3, pred, wave, sentence_target, use_topdown=use_topdown,
        )
        # Recompute Z2 with the same cascade flow so the nudge phase sees the
        # same L2 state as the free phase did.
        if use_topdown:
            wave_l1 = wave[:, :32000] if wave.shape[1] >= 32000 else wave
            from ficu_crfm.training.train_l3 import _sentence_to_l1_window
            w = _sentence_to_l1_window(wave, sentence_target)
            Z1r, Z1i = l1.settle(w)
            Z2r, Z2i = l2.settle(Z1r, Z1i)
            pred_l2_r, pred_l2_i = pred.predict(Z3r_free, Z3i_free)
            bias_l1_r, bias_l1_i = l2.l2_to_l1_topdown_bias(Z2r, Z2i)
            Z1r_td, Z1i_td = l1.settle(w, topdown_bias=(bias_l1_r, bias_l1_i))
            Z2r_td, Z2i_td = l2.settle(Z1r_td, Z1i_td,
                                       predicted_init=(pred_l2_r, pred_l2_i))
            Z3r_nudge, Z3i_nudge = l3.settle(Z2r_td, Z2i_td, target_class=label)
        else:
            from ficu_crfm.training.train_l3 import _sentence_to_l1_window
            w = _sentence_to_l1_window(wave, sentence_target)
            Z1r, Z1i = l1.settle(w)
            Z2r, Z2i = l2.settle(Z1r, Z1i)
            Z3r_nudge, Z3i_nudge = l3.settle(Z2r, Z2i, target_class=label)
        tbis.append(compute_TBI(Z3r_free, Z3i_free, Z3r_nudge, Z3i_nudge))
    return torch.cat(tbis) if tbis else torch.zeros(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--l1_checkpoint', required=True)
    p.add_argument('--l2_checkpoint', required=True)
    p.add_argument('--l3_checkpoint', required=True)
    p.add_argument('--timit_root', required=True)
    p.add_argument('--drive_gain', type=float, default=0.1)
    p.add_argument('--beta_nudge', type=float, default=0.05)
    p.add_argument('--n_settle_steps', type=int, default=24)
    p.add_argument('--sentence_window_ms', type=int, default=1500)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_items', type=int, default=2048)
    p.add_argument('--split', default='test', choices=['train', 'test'])
    p.add_argument('--use_topdown', action='store_true')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    l1 = FICUL1Phoneme(n_classes=N_PHONEMES, n_settle_steps=args.n_settle_steps).to(device)
    s1 = torch.load(args.l1_checkpoint, map_location=device, weights_only=True)
    l1.load_state_dict(s1.get('l1', s1))
    for p_ in l1.parameters():
        p_.requires_grad = False
    l1.eval()

    s2 = torch.load(args.l2_checkpoint, map_location=device, weights_only=True)
    l2_sd = s2.get('l2', s2)
    n_classes_l2 = l2_sd['readout.W'].shape[0]
    l2 = FICUL2Word(n_classes=n_classes_l2, n_settle_steps=args.n_settle_steps).to(device)
    l2.load_state_dict(l2_sd)
    with torch.no_grad():
        l2.lambda_td_L2_L1.fill_(0.0)
    l2.eval()

    l3 = FICUL3Sentence(
        n_classes=N_SENTENCE_TYPES,
        n_settle_steps=args.n_settle_steps,
        beta=args.beta_nudge,
        drive_gain=args.drive_gain,
    ).to(device)
    s3 = torch.load(args.l3_checkpoint, map_location=device, weights_only=True)
    l3_sd = s3.get('l3', s3)
    m_sd = l3.state_dict()
    compat = {k: v for k, v in l3_sd.items() if k in m_sd and v.shape == m_sd[k].shape}
    miss, unex = l3.load_state_dict(compat, strict=False)
    print(f"[verify] L3 loaded={len(compat)}/{len(l3_sd)} miss={len(miss)} unex={len(unex)} "
          f"drive_gain={l3.drive_gain} coupling_l2[0,0]={l3.coupling_l2[0,0].item():.4f}",
          flush=True)
    print(f"[verify] L3 gamma={l3.gamma_per_channel.tolist()}", flush=True)
    print(f"[verify] L3 beta={l3.beta_per_channel.tolist()}", flush=True)
    l3.eval()

    pred = PredictiveField(
        l3_shape=(FICUL3Sentence.CHANNELS, FICUL3Sentence.HEIGHT, FICUL3Sentence.WIDTH),
        l2_shape=(FICUL2Word.CHANNELS, FICUL2Word.HEIGHT, FICUL2Word.WIDTH),
        lr=0.001,
        lambda_td=0.0,
        handshake_threshold=0.0,
    ).to(device)
    if 'pred' in s3:
        try:
            pred.load_state_dict(s3['pred'])
            print('[verify] PredictiveField restored from checkpoint', flush=True)
        except Exception as e:
            print(f'[verify] PF load failed: {e}', flush=True)
    print(f"[verify] pred.lambda_td={pred.lambda_td}", flush=True)

    ds = build_dataset('sentence', args.split, args.timit_root)
    if hasattr(ds, '__len__') and len(ds) > args.max_items:
        step = max(1, len(ds) // args.max_items)
        idx = list(range(0, len(ds), step))[:args.max_items]
        ds = Subset(ds, idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_sentences)

    sentence_target = int(args.sentence_window_ms * SAMPLE_RATE / 1000)
    print(f"[verify] split={args.split} n={len(ds)} batch_size={args.batch_size} "
          f"use_topdown={args.use_topdown}", flush=True)

    tbi = measure_l3_tbi(l1, l2, l3, pred, loader, device, sentence_target,
                        use_topdown=args.use_topdown)
    print(f"[verify] L3 TBI mean={tbi.mean().item():.4f}  std={tbi.std().item():.4f}  "
          f"(target band: 0.024–0.040)", flush=True)

    acc = validate_l3(l1, l2, l3, pred, loader, device, sentence_target,
                     use_topdown=args.use_topdown)
    print(f"[verify] L3 val_acc={acc*100:.2f}%  (target band: 56.0–56.8%)", flush=True)


if __name__ == '__main__':
    main()
