"""End-to-end shape/no-error smoke test on a tiny synthetic batch.

This is NOT training. It only verifies that:
  - The mel front-end produces the expected [B, 3, 94, 64] shape.
  - L1 / L2 / L3 forward and EP nudge passes run without error.
  - LevelGate freezes physics when fed a passing accuracy.
  - PredictiveField predicts and updates without shape mismatches.
  - PETU mask + TBI return sensible values.
"""

from __future__ import annotations

import os
import sys
import torch

# Allow running as a script from inside the directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ficu_crfm.architecture.ficu_l1 import FICUL1Phoneme
from ficu_crfm.architecture.ficu_l2 import FICUL2Word
from ficu_crfm.architecture.ficu_l3 import FICUL3Sentence
from ficu_crfm.architecture.level_gate import LevelGate
from ficu_crfm.architecture.predictive_field import PredictiveField
from ficu_crfm.dataset.timit_loader import (
    SyntheticTIMITDataset, collate_phonemes, collate_words, collate_sentences,
    N_PHONEMES, N_SENTENCE_TYPES,
)
from ficu_crfm.dataset.feature_extractor import MelFrontEnd, TARGET_FRAMES, N_MELS
from ficu_crfm.metrics.tbi import compute_TBI
from ficu_crfm.training.petu import should_update_physics, update_fraction
import torch.nn.functional as F


def header(s):
    print(f"\n=== {s} ===")


def main():
    torch.manual_seed(0)
    device = torch.device('cpu')
    B = 4
    print(f"Using device: {device}, batch size: {B}")

    # ---- mel front-end ----
    header("Mel front-end")
    mel = MelFrontEnd().to(device)
    wave = torch.randn(B, 1600)
    feats = mel(wave)
    assert feats.shape == (B, 3, TARGET_FRAMES, N_MELS), f"got {feats.shape}"
    print(f"  mel({tuple(wave.shape)}) -> {tuple(feats.shape)} OK")

    # ---- L1 forward + nudge + EP observables ----
    header("L1 phoneme")
    l1 = FICUL1Phoneme(n_classes=N_PHONEMES, n_settle_steps=4).to(device)
    p_ds = SyntheticTIMITDataset('phoneme', n_items=B)
    batch = collate_phonemes([p_ds[i] for i in range(B)])
    Z1r_free, Z1i_free = l1.settle(batch['waveform'])
    assert Z1r_free.shape == (B, 3, 94, 64), f"got {Z1r_free.shape}"
    print(f"  L1 free settled -> {tuple(Z1r_free.shape)}")

    Z1r_nudge, Z1i_nudge = l1.settle(batch['waveform'], target_class=batch['label'])
    print(f"  L1 nudge settled -> {tuple(Z1r_nudge.shape)}")

    obs_free = l1.ep_observables(Z1r_free, Z1i_free)
    obs_nudge = l1.ep_observables(Z1r_nudge, Z1i_nudge)
    deltas = l1.apply_ep_update(obs_free, obs_nudge, lr_physics=0.001)
    print(f"  L1 EP deltas: {deltas}")

    # ---- TBI + PETU mask ----
    header("TBI + PETU at L1")
    tbi = compute_TBI(Z1r_free, Z1i_free, Z1r_nudge, Z1i_nudge)
    feat_r, feat_i = l1._phase_features(Z1r_free, Z1i_free)
    logits_l1 = l1.readout(feat_r, feat_i)
    err = F.cross_entropy(logits_l1, batch['label'], reduction='none')
    mask = should_update_physics(err, tbi, threshold=0.5, coherence_floor=0.3)
    print(f"  TBI mean: {tbi.mean().item():.3f} ± {tbi.std().item():.3f}")
    print(f"  per-sample CE loss: {err.tolist()}")
    print(f"  PETU mask: {mask.tolist()} (update fraction = {update_fraction(mask):.2f})")

    # ---- LevelGate dry run ----
    header("LevelGate dry run")
    gate = LevelGate(threshold=0.6, patience=2, name='L1')
    gate.update(0.55, l1)            # below
    gate.update(0.65, l1)            # above (1)
    triggered = gate.update(0.70, l1) # above (2) -> freeze
    print(f"  triggered={triggered} frozen={gate.frozen}")
    assert triggered and gate.frozen
    for p in l1.physics_parameters():
        assert p.requires_grad is False, "physics_parameters should be frozen"
    print("  physics_parameters all requires_grad=False")

    # ---- L2 forward + nudge ----
    header("L2 word")
    l2 = FICUL2Word(n_classes=33, n_settle_steps=4).to(device)
    Z2r_free, Z2i_free = l2.settle(Z1r_free, Z1i_free)
    assert Z2r_free.shape == (B, 3, 47, 32), f"got {Z2r_free.shape}"
    print(f"  L2 free settled -> {tuple(Z2r_free.shape)}")
    fake_word_labels = torch.randint(0, 33, (B,))
    Z2r_nudge, Z2i_nudge = l2.settle(Z1r_free, Z1i_free, target_class=fake_word_labels)
    feat_r2, feat_i2 = l2._phase_features(Z2r_free, Z2i_free)
    l2_logits = l2.readout(feat_r2, feat_i2)
    print(f"  L2 logits -> {tuple(l2_logits.shape)}")
    obs_free2 = l2.ep_observables(Z2r_free, Z2i_free)
    obs_nudge2 = l2.ep_observables(Z2r_nudge, Z2i_nudge)
    deltas2 = l2.apply_ep_update(obs_free2, obs_nudge2,
                                 Z1r_free, Z1i_free,
                                 Z2r_free, Z2i_free,
                                 Z2r_nudge, Z2i_nudge,
                                 lr_physics=0.001)
    print(f"  L2 EP deltas: {deltas2}")

    # ---- L3 forward + nudge ----
    header("L3 sentence")
    l3 = FICUL3Sentence(n_classes=N_SENTENCE_TYPES, n_settle_steps=4).to(device)
    Z3r_free, Z3i_free = l3.settle(Z2r_free, Z2i_free)
    assert Z3r_free.shape == (B, 3, 24, 16), f"got {Z3r_free.shape}"
    print(f"  L3 free settled -> {tuple(Z3r_free.shape)}")
    fake_sent_labels = torch.randint(0, N_SENTENCE_TYPES, (B,))
    Z3r_nudge, Z3i_nudge = l3.settle(Z2r_free, Z2i_free, target_class=fake_sent_labels)
    feat_r3, feat_i3 = l3._phase_features(Z3r_free, Z3i_free)
    l3_logits = l3.readout(feat_r3, feat_i3)
    print(f"  L3 logits -> {tuple(l3_logits.shape)}")
    obs_free3 = l3.ep_observables(Z3r_free, Z3i_free)
    obs_nudge3 = l3.ep_observables(Z3r_nudge, Z3i_nudge)
    deltas3 = l3.apply_ep_update(obs_free3, obs_nudge3, lr_physics=0.001)
    print(f"  L3 EP deltas: {deltas3}")

    # ---- PredictiveField ----
    header("PredictiveField (L3 -> L2)")
    pred = PredictiveField(
        l3_shape=(3, 24, 16), l2_shape=(3, 47, 32),
        lr=0.01, lambda_td=0.5, handshake_threshold=10.0,
    )
    pred_r, pred_i = pred.predict(Z3r_free, Z3i_free)
    assert pred_r.shape == (B, 3, 47, 32)
    print(f"  predicted L2 -> {tuple(pred_r.shape)}")
    err_pred = pred.prediction_error(Z3r_free, Z3i_free, Z2r_free, Z2i_free)
    print(f"  per-sample prediction error: {err_pred.tolist()}")
    rate = pred.handshake_success_rate(err_pred)
    print(f"  handshake success rate: {rate:.2f}")
    pred.update(Z3r_free, Z3i_free, Z2r_free, Z2i_free)

    # ---- L2 settle WITH predicted bias init ----
    header("L2 with predicted-init bias")
    Z2r_b, Z2i_b = l2.settle(Z1r_free, Z1i_free, predicted_init=(pred_r, pred_i))
    assert Z2r_b.shape == (B, 3, 47, 32)
    print(f"  L2 with bias -> {tuple(Z2r_b.shape)}")

    # ---- Synthetic dataset items ----
    header("Synthetic dataset shapes")
    w_ds = SyntheticTIMITDataset('word', n_items=B)
    s_ds = SyntheticTIMITDataset('sentence', n_items=B)
    print(f"  phoneme[0].waveform = {tuple(p_ds[0].waveform.shape)}")
    print(f"  word[0].waveform    = {tuple(w_ds[0].waveform.shape)}")
    print(f"  sentence[0].waveform= {tuple(s_ds[0].waveform.shape)}")
    collate_words([w_ds[i] for i in range(B)])
    collate_sentences([s_ds[i] for i in range(B)])

    print("\nAll smoke checks passed.")


if __name__ == '__main__':
    main()
