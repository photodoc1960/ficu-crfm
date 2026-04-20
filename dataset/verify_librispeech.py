"""Tensor-shape verification for LibriSpeechMFADataset.

Streams a handful of train.clean.100 utterances from HuggingFace
(`openslr/librispeech_asr`, decode=False so we don't need torchcodec), writes
them as FLACs into a LibriSpeech-layout cache, then exercises the loader at
phoneme / word / sentence levels.

Run:
    HF_HOME=/home/slater/data/hf_cache \
    python -m ficu_crfm.dataset.verify_librispeech
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ficu_crfm.dataset.timit_loader import (
    LibriSpeechMFADataset,
    N_PHONEMES,
    N_SENTENCE_TYPES,
    collate_phonemes,
    collate_words,
    collate_sentences,
)
from ficu_crfm.dataset.feature_extractor import MelFrontEnd, TARGET_FRAMES, N_MELS

ALIGN_ROOT = Path('/home/slater/data/librispeech/alignments/train-clean-100')
AUDIO_CACHE = Path('/home/slater/data/librispeech/audio/train-clean-100')

# Target N utterances to verify. The loader only loads utterances whose flac
# exists on disk, so we cache at least this many from the HF stream.
N_TARGET = 15


def _stage_audio(n_target: int) -> int:
    """Pull n_target utterances from HF and write them as flac files.

    Only caches utterances that have a matching TextGrid alignment so we
    guarantee the loader sees at least n_target usable items.
    """
    from datasets import load_dataset, Audio

    AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
    ds = load_dataset('openslr/librispeech_asr', 'clean',
                      split='train.100', streaming=True)
    ds = ds.cast_column('audio', Audio(decode=False))

    saved = 0
    for item in ds:
        utt_id = item['id']
        spk = str(item['speaker_id'])
        chap = str(item['chapter_id'])
        tg_path = ALIGN_ROOT / spk / chap / f'{utt_id}.TextGrid'
        if not tg_path.exists():
            continue
        out_dir = AUDIO_CACHE / spk / chap
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{utt_id}.flac'
        if not out_path.exists():
            data, sr = sf.read(io.BytesIO(item['audio']['bytes']))
            assert sr == 16000, f'expected 16 kHz, got {sr}'
            sf.write(str(out_path), data, sr, subtype='PCM_16')
        saved += 1
        if saved >= n_target:
            break
    return saved


def main():
    print(f"Staging audio from HF → {AUDIO_CACHE}")
    n = _stage_audio(N_TARGET)
    print(f"  staged {n} utterances")
    assert n >= 10, f"need at least 10 utterances, staged {n}"

    mel = MelFrontEnd()

    # ---- phoneme level ----
    print("\n=== phoneme level ===")
    ph_ds = LibriSpeechMFADataset(
        audio_root=str(AUDIO_CACHE),
        alignment_root=str(ALIGN_ROOT),
        split='train-clean-100',
        level='phoneme',
        max_utterances=N_TARGET,
    )
    print(f"  utterances indexed: {len(ph_ds._utterances)}")
    print(f"  phoneme items: {len(ph_ds)}")
    assert len(ph_ds) >= 10

    for i in range(3):
        seg = ph_ds[i]
        assert seg.waveform.shape == (1600,), f"got {seg.waveform.shape}"
        assert 0 <= seg.label < N_PHONEMES
        print(f"  [{i}] wave={tuple(seg.waveform.shape)} label={seg.label} "
              f"spk={seg.speaker} utt={seg.sentence_id}")

    batch = collate_phonemes([ph_ds[i] for i in range(10)])
    assert batch['waveform'].shape == (10, 1600)
    assert batch['label'].shape == (10,)
    feats = mel(batch['waveform'])
    assert feats.shape == (10, 3, TARGET_FRAMES, N_MELS), f"mel -> {tuple(feats.shape)}"
    print(f"  collate(10) waveform={tuple(batch['waveform'].shape)} "
          f"labels={batch['label'].tolist()}")
    print(f"  mel-front-end -> {tuple(feats.shape)} OK")

    # ---- word level ----
    print("\n=== word level ===")
    wd_ds = LibriSpeechMFADataset(
        audio_root=str(AUDIO_CACHE),
        alignment_root=str(ALIGN_ROOT),
        split='train-clean-100',
        level='word',
        vocab_size=200,
        max_utterances=N_TARGET,
    )
    print(f"  word items: {len(wd_ds)}  vocab_size: {len(wd_ds.word_vocab)}")
    assert len(wd_ds) >= 10
    for i in range(3):
        seg = wd_ds[i]
        assert seg.waveform.ndim == 1
        assert 0 <= seg.label <= len(wd_ds.word_vocab)
        print(f"  [{i}] wave={tuple(seg.waveform.shape)} "
              f"label={seg.label} spk={seg.speaker}")
    wb = collate_words([wd_ds[i] for i in range(10)])
    assert wb['waveform'].shape[0] == 10 and wb['waveform'].ndim == 2
    print(f"  collate(10) waveform={tuple(wb['waveform'].shape)} "
          f"labels={wb['label'].tolist()}")

    # ---- sentence level ----
    print("\n=== sentence level ===")
    st_ds = LibriSpeechMFADataset(
        audio_root=str(AUDIO_CACHE),
        alignment_root=str(ALIGN_ROOT),
        split='train-clean-100',
        level='sentence',
        max_utterances=N_TARGET,
    )
    print(f"  sentence items: {len(st_ds)}")
    assert len(st_ds) >= 10
    for i in range(3):
        seg = st_ds[i]
        assert seg.waveform.ndim == 1
        assert 0 <= seg.label < N_SENTENCE_TYPES
        dur_s = seg.waveform.numel() / 16000
        print(f"  [{i}] wave={tuple(seg.waveform.shape)} "
              f"({dur_s:.2f}s) label={seg.label} spk={seg.speaker}")
    sb = collate_sentences([st_ds[i] for i in range(10)])
    assert sb['waveform'].shape[0] == 10 and sb['waveform'].ndim == 2
    print(f"  collate(10) waveform={tuple(sb['waveform'].shape)} "
          f"labels={sb['label'].tolist()}")

    print("\nAll LibriSpeech shape checks passed.")


if __name__ == '__main__':
    main()
