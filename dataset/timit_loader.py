"""TCD-TIMIT loader with synthetic fallback for smoke tests.

TCD-TIMIT (https://sigmedia.tcd.ie/TCDTIMIT/) is a freely available dataset
that uses the identical sentence set, phoneme annotation format, and 39-phoneme
reduction as the LDC TIMIT corpus, so the loader works against either source.

The loader expects an extracted TCD-TIMIT tree at `timit_root` (after running
extractTCDTIMITaudio.py from https://github.com/matthijsvk/TIMITspeech) with
files like:
    TRAIN/DR1/FCJF0/SA1.WAV     (16 kHz mono PCM, RIFF or NIST header)
    TRAIN/DR1/FCJF0/SA1.PHN     (phoneme alignments: start_sample end_sample phn)
    TRAIN/DR1/FCJF0/SA1.WRD     (word alignments)
    TRAIN/DR1/FCJF0/SA1.TXT     (orthographic transcription)

If `timit_root` is None or missing, the loader emits a synthetic dataset of
phoneme/word/sentence-shaped tensors with random labels — enough to exercise
the full L1→L2→L3 architecture but NOT to train.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


# Standard TIMIT 61→39 phoneme reduction (Lee & Hon 1989).
TIMIT_61 = [
    'iy','ih','eh','ae','ix','ax','ah','uw','ux','uh','ao','aa','ey','ay','oy',
    'aw','ow','l','el','r','y','w','er','axr','m','em','n','en','nx','ng','eng',
    'ch','jh','dh','b','d','dx','g','p','t','k','z','zh','v','f','th','s','sh',
    'hh','hv','pcl','tcl','kcl','qcl','bcl','dcl','gcl','epi','h#','pau','q'
]
PHONEME_61_TO_39 = {p: p for p in TIMIT_61}
_REDUCTIONS = {
    'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'hv': 'hh', 'ix': 'ih',
    'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng',
    'zh': 'sh', 'ux': 'uw',
    'pcl': 'h#', 'tcl': 'h#', 'kcl': 'h#', 'qcl': 'h#',
    'bcl': 'h#', 'dcl': 'h#', 'gcl': 'h#',
    'epi': 'h#', 'pau': 'h#', 'q': 'h#',
}
PHONEME_61_TO_39.update(_REDUCTIONS)
PHONEMES_39 = sorted(set(PHONEME_61_TO_39.values()))
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEMES_39)}
N_PHONEMES = len(PHONEMES_39)  # 39

SENTENCE_TYPES = ['sa', 'si', 'sx']
SENTENCE_TO_IDX = {s: i for i, s in enumerate(SENTENCE_TYPES)}
N_SENTENCE_TYPES = len(SENTENCE_TYPES)


@dataclass
class PhonemeSegment:
    waveform: torch.Tensor   # [1600] mono float32, padded/truncated to 100ms
    label: int               # phoneme class
    speaker: str
    sentence_id: str         # for held-out speaker analysis


@dataclass
class WordSegment:
    waveform: torch.Tensor   # variable length, will run through frozen L1
    label: int               # word index (top-N words; rest -> unknown)
    speaker: str
    sentence_id: str


@dataclass
class SentenceSegment:
    waveform: torch.Tensor   # full sentence
    label: int               # sentence type 0/1/2 (sa/si/sx)
    speaker: str
    sentence_id: str


# ---------------------------------------------------------------------------
# Real TIMIT loader
# ---------------------------------------------------------------------------


def _read_nist_wav(path: str) -> torch.Tensor:
    """Read a TIMIT NIST SPHERE-style 16-bit mono wav.

    TIMIT files have a 1024-byte text header followed by raw int16 PCM.
    Returns float32 in [-1, 1].
    """
    import numpy as np
    with open(path, 'rb') as f:
        header = f.read(1024)
        # Some redistributions ship plain RIFF wav; fall back to torchaudio.
        if not header.startswith(b'NIST'):
            f.seek(0)
            import soundfile as sf
            data, sr = sf.read(path, dtype='int16')
            if sr != 16000:
                raise ValueError(f"unexpected sample rate {sr} in {path}")
            return torch.from_numpy(data.astype('float32') / 32768.0)
        raw = f.read()
    arr = np.frombuffer(raw, dtype='<i2').astype('float32') / 32768.0
    return torch.from_numpy(arr)


class TIMITRealDataset(Dataset):
    """Lazy TIMIT loader.

    `level` selects which segment type to emit: 'phoneme', 'word', or 'sentence'.
    `split` is 'train' or 'test'. Word vocabulary is built on first construction
    and cached on the dataset object.
    """

    def __init__(self, timit_root: str, split: str, level: str,
                 word_vocab: Optional[dict] = None, vocab_size: int = 500,
                 phoneme_target_samples: int = 1600):
        super().__init__()
        self.root = Path(timit_root)
        self.split = split.lower()
        self.level = level
        self.vocab_size = vocab_size
        self.phoneme_target_samples = phoneme_target_samples

        split_dir = self.root / self.split.upper()
        if not split_dir.exists():
            raise FileNotFoundError(f"TIMIT split not found: {split_dir}")

        self._utterances = self._index_utterances(split_dir)

        if level == 'phoneme':
            self._items = self._build_phoneme_index()
        elif level == 'word':
            if word_vocab is None:
                word_vocab = self._build_word_vocab()
            self.word_vocab = word_vocab
            self._items = self._build_word_index()
        elif level == 'sentence':
            self._items = self._build_sentence_index()
        else:
            raise ValueError(f"unknown level: {level}")

    def _index_utterances(self, split_dir: Path):
        items = []
        for wav in split_dir.rglob('*.WAV'):
            base = wav.with_suffix('')
            phn = base.with_suffix('.PHN')
            wrd = base.with_suffix('.WRD')
            if not (phn.exists() and wrd.exists()):
                continue
            speaker = wav.parent.name
            sentence_id = wav.stem  # e.g. SA1, SI1234, SX42
            items.append({
                'wav': wav, 'phn': phn, 'wrd': wrd,
                'speaker': speaker, 'sentence_id': sentence_id,
            })
        if not items:
            raise FileNotFoundError(f"No usable utterances under {split_dir}")
        return items

    @staticmethod
    def _read_alignment(path: Path):
        out = []
        for line in path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start, end, label = int(parts[0]), int(parts[1]), parts[2]
            out.append((start, end, label))
        return out

    def _build_phoneme_index(self):
        items = []
        for u in self._utterances:
            for start, end, phn in self._read_alignment(u['phn']):
                phn39 = PHONEME_61_TO_39.get(phn, None)
                if phn39 is None or phn39 not in PHONEME_TO_IDX:
                    continue
                items.append({**u, 'start': start, 'end': end,
                              'label': PHONEME_TO_IDX[phn39]})
        return items

    def _build_word_vocab(self):
        from collections import Counter
        counter = Counter()
        for u in self._utterances:
            for _s, _e, w in self._read_alignment(u['wrd']):
                counter[w.lower()] += 1
        most = [w for w, _ in counter.most_common(self.vocab_size)]
        vocab = {w: i for i, w in enumerate(most)}
        vocab['<unk>'] = len(vocab)
        return vocab

    def _build_word_index(self):
        items = []
        for u in self._utterances:
            for start, end, w in self._read_alignment(u['wrd']):
                idx = self.word_vocab.get(w.lower(), self.word_vocab['<unk>'])
                items.append({**u, 'start': start, 'end': end, 'label': idx})
        return items

    def _build_sentence_index(self):
        items = []
        for u in self._utterances:
            sid = u['sentence_id'].lower()
            kind = sid[:2]
            if kind not in SENTENCE_TO_IDX:
                continue
            items.append({**u, 'start': None, 'end': None,
                          'label': SENTENCE_TO_IDX[kind]})
        return items

    def __len__(self):
        return len(self._items)

    def _load_wave(self, item):
        wave = _read_nist_wav(str(item['wav']))
        if item['start'] is None:
            return wave
        return wave[item['start']:item['end']]

    def __getitem__(self, idx):
        item = self._items[idx]
        wave = self._load_wave(item)

        if self.level == 'phoneme':
            tgt = self.phoneme_target_samples
            if wave.numel() >= tgt:
                start = (wave.numel() - tgt) // 2
                wave = wave[start:start + tgt]
            else:
                pad = tgt - wave.numel()
                wave = torch.nn.functional.pad(wave, (pad // 2, pad - pad // 2))
            return PhonemeSegment(
                waveform=wave, label=item['label'],
                speaker=item['speaker'], sentence_id=item['sentence_id'],
            )

        if self.level == 'word':
            return WordSegment(
                waveform=wave, label=item['label'],
                speaker=item['speaker'], sentence_id=item['sentence_id'],
            )

        return SentenceSegment(
            waveform=wave, label=item['label'],
            speaker=item['speaker'], sentence_id=item['sentence_id'],
        )


# ---------------------------------------------------------------------------
# Synthetic fallback dataset
# ---------------------------------------------------------------------------


class SyntheticTIMITDataset(Dataset):
    """Random tensors shaped like TIMIT segments.

    Used only for smoke tests — the labels are uniformly random so accuracy
    will sit at chance forever. NEVER use this for real training.
    """

    def __init__(self, level: str, n_items: int = 64, seed: int = 0,
                 phoneme_target_samples: int = 1600,
                 word_target_samples: int = 4000,
                 sentence_target_samples: int = 32000,
                 n_words: int = 32, n_speakers: int = 8):
        super().__init__()
        self.level = level
        self.n_items = n_items
        self.phoneme_target_samples = phoneme_target_samples
        self.word_target_samples = word_target_samples
        self.sentence_target_samples = sentence_target_samples
        self.n_words = n_words
        self.n_speakers = n_speakers
        self.word_vocab = {f'word_{i}': i for i in range(n_words)}
        self.word_vocab['<unk>'] = n_words
        self._g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx + 1)
        speaker = f'spk{idx % self.n_speakers:02d}'
        sentence_id = f'syn{idx:04d}'

        if self.level == 'phoneme':
            wave = torch.randn(self.phoneme_target_samples, generator=g) * 0.1
            label = int(torch.randint(N_PHONEMES, (1,), generator=g).item())
            return PhonemeSegment(waveform=wave, label=label,
                                  speaker=speaker, sentence_id=sentence_id)

        if self.level == 'word':
            wave = torch.randn(self.word_target_samples, generator=g) * 0.1
            label = int(torch.randint(self.n_words + 1, (1,), generator=g).item())
            return WordSegment(waveform=wave, label=label,
                               speaker=speaker, sentence_id=sentence_id)

        wave = torch.randn(self.sentence_target_samples, generator=g) * 0.1
        label = int(torch.randint(N_SENTENCE_TYPES, (1,), generator=g).item())
        return SentenceSegment(waveform=wave, label=label,
                               speaker=speaker, sentence_id=sentence_id)


def build_dataset(level: str, split: str = 'train',
                  timit_root: Optional[str] = None,
                  synthetic_n: int = 64, **kwargs):
    """Build a TIMIT dataset, falling back to synthetic if data missing."""
    if timit_root is not None and Path(timit_root).exists():
        return TIMITRealDataset(timit_root=timit_root, split=split,
                                level=level, **kwargs)
    return SyntheticTIMITDataset(level=level, n_items=synthetic_n,
                                 seed=hash((level, split)) & 0xFFFF)


# ---------------------------------------------------------------------------
# LibriSpeech + MFA alignments loader
# ---------------------------------------------------------------------------

# ARPAbet → TIMIT-39 mapping. MFA's LibriSpeech dictionary uses ARPAbet phones
# with 0/1/2 stress digits on vowels and "sil"/"sp"/"spn" for non-speech. After
# stripping stress and lowercasing, we drop any remaining labels through
# PHONEME_61_TO_39 so zh→sh, pau→h# etc. survive the TIMIT reduction.
_ARPABET_TO_TIMIT = {
    'AA': 'aa', 'AE': 'ae', 'AH': 'ah', 'AO': 'ao', 'AW': 'aw', 'AY': 'ay',
    'EH': 'eh', 'ER': 'er', 'EY': 'ey', 'IH': 'ih', 'IY': 'iy',
    'OW': 'ow', 'OY': 'oy', 'UH': 'uh', 'UW': 'uw',
    'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'dh', 'F': 'f', 'G': 'g',
    'HH': 'hh', 'JH': 'jh', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n',
    'NG': 'ng', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'sh', 'T': 't',
    'TH': 'th', 'V': 'v', 'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'zh',
}
_SILENCE_LABELS = {'sil', 'sp', 'spn', ''}


def arpabet_to_timit39(label: str) -> Optional[str]:
    """Map an MFA/ARPAbet phone label to TIMIT 39-class phoneme, or None."""
    if label is None:
        return None
    raw = label.strip()
    if raw.lower() in _SILENCE_LABELS:
        return 'h#'
    base = raw.rstrip('0123456789').upper()
    timit = _ARPABET_TO_TIMIT.get(base)
    if timit is None:
        return None
    return PHONEME_61_TO_39.get(timit, None)


def _librispeech_flac_path(audio_root: Path, utt_id: str) -> Path:
    """Standard LibriSpeech layout: <root>/<split>/<spk>/<chap>/<utt>.flac.

    `audio_root` is expected to already include the split dir. The utt id
    encodes speaker and chapter as "<spk>-<chap>-<idx>".
    """
    spk, chap, _ = utt_id.split('-', 2)
    return audio_root / spk / chap / f"{utt_id}.flac"


def _read_flac(path: Path) -> torch.Tensor:
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(path), dtype='int16')
    if sr != 16000:
        raise ValueError(f"expected 16 kHz, got {sr} in {path}")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data.astype('float32') / 32768.0)


class LibriSpeechMFADataset(Dataset):
    """LibriSpeech loader with Montreal Forced Aligner TextGrid alignments.

    Emits the same `PhonemeSegment` / `WordSegment` / `SentenceSegment` dataclasses
    as `TIMITRealDataset` so downstream code is unchanged.

    Expected layout:
        audio_root/<spk>/<chap>/<spk>-<chap>-<idx>.flac
        alignment_root/<spk>/<chap>/<spk>-<chap>-<idx>.TextGrid

    Both roots should already point at a specific split (e.g. `train-clean-100`).

    Sentence-level labels have no natural 3-way TIMIT partition in LibriSpeech.
    We deterministically hash the speaker id into {0,1,2} as a placeholder so
    the downstream N_SENTENCE_TYPES readout still sees a balanced-ish target.
    This is NOT a meaningful classification task — it only exists so the L3
    interface is exercised end-to-end on LibriSpeech utterances.
    """

    SAMPLE_RATE = 16000

    def __init__(self, audio_root: str, alignment_root: str, split: str,
                 level: str, word_vocab: Optional[dict] = None,
                 vocab_size: int = 500, phoneme_target_samples: int = 1600,
                 max_utterances: Optional[int] = None,
                 speaker_include: Optional[set] = None,
                 speaker_exclude: Optional[set] = None):
        super().__init__()
        self.audio_root = Path(audio_root)
        self.alignment_root = Path(alignment_root)
        self.split = split
        self.level = level
        self.vocab_size = vocab_size
        self.phoneme_target_samples = phoneme_target_samples
        self.speaker_include = set(speaker_include) if speaker_include else None
        self.speaker_exclude = set(speaker_exclude) if speaker_exclude else None

        if not self.alignment_root.exists():
            raise FileNotFoundError(f"alignment_root missing: {self.alignment_root}")
        if not self.audio_root.exists():
            raise FileNotFoundError(f"audio_root missing: {self.audio_root}")

        self._utterances = self._index_utterances(max_utterances)

        if level == 'phoneme':
            self._items = self._build_phoneme_index()
        elif level == 'word':
            if word_vocab is None:
                word_vocab = self._build_word_vocab()
            self.word_vocab = word_vocab
            self._items = self._build_word_index()
        elif level == 'sentence':
            self._items = self._build_sentence_index()
        else:
            raise ValueError(f"unknown level: {level}")

    def _index_utterances(self, max_utterances: Optional[int]):
        import tgt
        items = []
        for tg_path in sorted(self.alignment_root.rglob('*.TextGrid')):
            utt_id = tg_path.stem
            try:
                spk, chap, _idx = utt_id.split('-', 2)
            except ValueError:
                continue
            if self.speaker_include is not None and spk not in self.speaker_include:
                continue
            if self.speaker_exclude is not None and spk in self.speaker_exclude:
                continue
            wav_path = _librispeech_flac_path(self.audio_root, utt_id)
            if not wav_path.exists():
                continue
            try:
                tg = tgt.read_textgrid(str(tg_path))
                phones = tg.get_tier_by_name('phones')
                words = tg.get_tier_by_name('words')
            except Exception:
                continue
            items.append({
                'utt_id': utt_id,
                'wav': wav_path,
                'textgrid': tg_path,
                'speaker': spk,
                'chapter': chap,
                'sentence_id': utt_id,
                'phones': [(iv.start_time, iv.end_time, iv.text) for iv in phones.intervals],
                'words':  [(iv.start_time, iv.end_time, iv.text) for iv in words.intervals],
            })
            if max_utterances is not None and len(items) >= max_utterances:
                break
        if not items:
            raise FileNotFoundError(
                f"No LibriSpeech utterances matched between {self.audio_root} "
                f"and {self.alignment_root}"
            )
        return items

    @classmethod
    def _t2s(cls, t: float) -> int:
        return int(round(t * cls.SAMPLE_RATE))

    def _build_phoneme_index(self):
        items = []
        for u in self._utterances:
            for start_t, end_t, label in u['phones']:
                timit = arpabet_to_timit39(label)
                if timit is None or timit not in PHONEME_TO_IDX:
                    continue
                # drop h# (silence) to match how TIMIT training typically skips it
                if timit == 'h#':
                    continue
                items.append({
                    'wav': u['wav'],
                    'start': self._t2s(start_t),
                    'end': self._t2s(end_t),
                    'speaker': u['speaker'],
                    'sentence_id': u['sentence_id'],
                    'label': PHONEME_TO_IDX[timit],
                })
        return items

    def _build_word_vocab(self):
        from collections import Counter
        counter = Counter()
        for u in self._utterances:
            for _s, _e, w in u['words']:
                w = w.strip().lower()
                if not w:
                    continue
                counter[w] += 1
        most = [w for w, _ in counter.most_common(self.vocab_size)]
        vocab = {w: i for i, w in enumerate(most)}
        vocab['<unk>'] = len(vocab)
        return vocab

    def _build_word_index(self):
        items = []
        for u in self._utterances:
            for start_t, end_t, w in u['words']:
                w = w.strip().lower()
                if not w:
                    continue
                idx = self.word_vocab.get(w, self.word_vocab['<unk>'])
                items.append({
                    'wav': u['wav'],
                    'start': self._t2s(start_t),
                    'end': self._t2s(end_t),
                    'speaker': u['speaker'],
                    'sentence_id': u['sentence_id'],
                    'label': idx,
                })
        return items

    def _build_sentence_index(self):
        items = []
        for u in self._utterances:
            label = (hash(u['speaker']) & 0xFFFFFFFF) % N_SENTENCE_TYPES
            items.append({
                'wav': u['wav'],
                'start': None,
                'end': None,
                'speaker': u['speaker'],
                'sentence_id': u['sentence_id'],
                'label': label,
            })
        return items

    def __len__(self):
        return len(self._items)

    def _load_wave(self, item):
        wave = _read_flac(item['wav'])
        if item['start'] is None:
            return wave
        s, e = item['start'], item['end']
        s = max(0, min(s, wave.numel()))
        e = max(s, min(e, wave.numel()))
        return wave[s:e]

    def __getitem__(self, idx):
        item = self._items[idx]
        wave = self._load_wave(item)

        if self.level == 'phoneme':
            tgt_len = self.phoneme_target_samples
            if wave.numel() >= tgt_len:
                start = (wave.numel() - tgt_len) // 2
                wave = wave[start:start + tgt_len]
            else:
                pad = tgt_len - wave.numel()
                wave = torch.nn.functional.pad(wave, (pad // 2, pad - pad // 2))
            return PhonemeSegment(
                waveform=wave, label=item['label'],
                speaker=item['speaker'], sentence_id=item['sentence_id'],
            )

        if self.level == 'word':
            return WordSegment(
                waveform=wave, label=item['label'],
                speaker=item['speaker'], sentence_id=item['sentence_id'],
            )

        return SentenceSegment(
            waveform=wave, label=item['label'],
            speaker=item['speaker'], sentence_id=item['sentence_id'],
        )


def build_librispeech_dataset(level: str, audio_root: str, alignment_root: str,
                              split: str = 'train-clean-100', **kwargs):
    """Construct a LibriSpeechMFADataset. Thin wrapper for symmetry with build_dataset."""
    return LibriSpeechMFADataset(
        audio_root=audio_root, alignment_root=alignment_root,
        split=split, level=level, **kwargs,
    )


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_phonemes(batch):
    waves = torch.stack([b.waveform for b in batch])
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    speakers = [b.speaker for b in batch]
    return {'waveform': waves, 'label': labels, 'speaker': speakers}


def _pad_stack(waves):
    max_len = max(w.numel() for w in waves)
    out = torch.zeros(len(waves), max_len)
    for i, w in enumerate(waves):
        out[i, :w.numel()] = w
    return out


def collate_words(batch):
    waves = _pad_stack([b.waveform for b in batch])
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    speakers = [b.speaker for b in batch]
    return {'waveform': waves, 'label': labels, 'speaker': speakers}


def collate_sentences(batch):
    waves = _pad_stack([b.waveform for b in batch])
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    speakers = [b.speaker for b in batch]
    return {'waveform': waves, 'label': labels, 'speaker': speakers}


__all__ = [
    'PHONEMES_39', 'PHONEME_TO_IDX', 'N_PHONEMES',
    'SENTENCE_TYPES', 'SENTENCE_TO_IDX', 'N_SENTENCE_TYPES',
    'PhonemeSegment', 'WordSegment', 'SentenceSegment',
    'TIMITRealDataset', 'SyntheticTIMITDataset', 'build_dataset',
    'LibriSpeechMFADataset', 'build_librispeech_dataset', 'arpabet_to_timit39',
    'collate_phonemes', 'collate_words', 'collate_sentences',
]
