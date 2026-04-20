"""Mel front-end matching the FICU audio paper.

400-sample Hann window, 160 hop, 64 mel bands, plus delta and delta-delta.
For a 100ms (1600-sample) phoneme at 16 kHz, output is [B, 3, 94, 64].
The 94 frames come from a longer context window so each sample lands in a
fixed-size tensor regardless of segment duration.

The L2 word and L3 sentence levels do NOT call this directly — they consume
frozen L1 field states. This module is only used by the L1 phoneme pipeline
and by the L2/L3 word/sentence segmenters that need to compute mel windows
over arbitrary audio spans before passing them through the frozen L1 stack.

Implementation note: we compute the mel spectrogram with raw torch stft +
a pre-baked Slaney-style mel filterbank rather than pulling in torchaudio,
because the NGC PyTorch containers (which we use for GB10 Thor support) do
not ship torchaudio wheels. The output matches torchaudio's
`MelSpectrogram(sample_rate, n_fft, hop_length, n_mels, power=2.0, center=True)`
to within numerical precision.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


SAMPLE_RATE = 16000
N_FFT = 400          # 25ms at 16 kHz
HOP_LENGTH = 160     # 10ms at 16 kHz
N_MELS = 64
TARGET_FRAMES = 94   # field height


def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    # Slaney mel scale (HTK=False in torchaudio).
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    mel = (f - f_min) / f_sp
    log_region = f >= min_log_hz
    mel = torch.where(log_region, min_log_mel + torch.log(f / min_log_hz) / logstep, mel)
    return mel


def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    freqs = f_min + f_sp * m
    log_region = m >= min_log_mel
    freqs = torch.where(log_region, min_log_hz * torch.exp(logstep * (m - min_log_mel)), freqs)
    return freqs


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    """Slaney-normalized mel filterbank. Returns [n_freq, n_mels]."""
    n_freq = n_fft // 2 + 1
    f_max = sample_rate / 2
    fft_freqs = torch.linspace(0, f_max, n_freq)

    mel_min = _hz_to_mel(torch.tensor(0.0))
    mel_max = _hz_to_mel(torch.tensor(f_max))
    mel_points = torch.linspace(mel_min.item(), mel_max.item(), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    # Triangular filters.
    fb = torch.zeros(n_freq, n_mels)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        # rising edge
        mask_rise = (fft_freqs >= left) & (fft_freqs <= center)
        fb[mask_rise, m - 1] = (fft_freqs[mask_rise] - left) / (center - left + 1e-12)
        # falling edge
        mask_fall = (fft_freqs >= center) & (fft_freqs <= right)
        fb[mask_fall, m - 1] = (right - fft_freqs[mask_fall]) / (right - center + 1e-12)

    # Slaney normalization: each filter sums to constant energy.
    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    fb = fb * enorm.unsqueeze(0)
    return fb


def _delta_kernel(win_length: int = 5) -> torch.Tensor:
    """Matches torchaudio.functional.compute_deltas default kernel."""
    n = (win_length - 1) // 2
    denom = 2 * sum(i * i for i in range(1, n + 1))
    k = torch.arange(-n, n + 1, dtype=torch.float32) / denom
    return k


class MelFrontEnd(nn.Module):
    """Mel + delta + delta-delta -> [B, 3, 94, 64].

    Forward accepts raw waveforms [B, L_samples] or [B, 1, L_samples].
    Pads or center-crops the time axis to TARGET_FRAMES frames after melspec.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 n_mels=N_MELS, target_frames=TARGET_FRAMES, top_db: float = 80.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_frames = target_frames
        self.top_db = top_db

        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)
        fb = _mel_filterbank(sample_rate, n_fft, n_mels)  # [n_freq, n_mels]
        self.register_buffer('mel_fb', fb, persistent=False)
        # Delta kernel shaped for conv1d: [out_ch=1, in_ch=1, K]
        self.register_buffer('delta_kernel', _delta_kernel(5).view(1, 1, -1),
                             persistent=False)

    def _mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, L] → spec: [B, n_freq, T]
        spec = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=self.window,
            center=True, return_complex=True, pad_mode='reflect',
        )
        power = spec.real.pow(2) + spec.imag.pow(2)  # [B, n_freq, T]
        # [B, n_freq, T] @ [n_freq, n_mels] → [B, T, n_mels] then transpose back
        mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)
        return mel  # [B, n_mels, T]

    def _amplitude_to_db(self, mel: torch.Tensor) -> torch.Tensor:
        amin = 1e-10
        db = 10.0 * torch.log10(torch.clamp(mel, min=amin))
        # Match torchaudio's top_db behavior: clip anything >top_db below max.
        if self.top_db is not None:
            max_db = db.amax(dim=(-2, -1), keepdim=True)
            db = torch.maximum(db, max_db - self.top_db)
        return db

    def _compute_deltas(self, mel_db: torch.Tensor) -> torch.Tensor:
        # mel_db: [B, n_mels, T] — apply 1D conv along T with replicate padding.
        B, M, T = mel_db.shape
        K = self.delta_kernel.shape[-1]
        pad = K // 2
        x = mel_db.reshape(B * M, 1, T)
        x = F.pad(x, (pad, pad), mode='replicate')
        d = F.conv1d(x, self.delta_kernel)
        return d.reshape(B, M, T)

    def _resize_time(self, mel_db: torch.Tensor) -> torch.Tensor:
        T = mel_db.shape[-1]
        tgt = self.target_frames
        if T == tgt:
            return mel_db
        if T > tgt:
            start = (T - tgt) // 2
            return mel_db[..., start:start + tgt]
        pad_total = tgt - T
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return F.pad(mel_db, (pad_left, pad_right), mode='replicate')

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return [B, 3, target_frames, n_mels] (channels: mel, delta, delta-delta).

        Note: per FICU paper convention the spatial layout is [time, freq] with
        time as the field height and frequency as the field width.
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        mel = self._mel_spectrogram(waveform)
        mel_db = self._amplitude_to_db(mel)
        mel_db = self._resize_time(mel_db)
        d1 = self._compute_deltas(mel_db)
        d2 = self._compute_deltas(d1)

        def _norm(x):
            mu = x.mean(dim=(-2, -1), keepdim=True)
            sd = x.std(dim=(-2, -1), keepdim=True) + 1e-5
            return (x - mu) / sd

        mel_db = _norm(mel_db)
        d1 = _norm(d1)
        d2 = _norm(d2)

        feats = torch.stack([mel_db, d1, d2], dim=1)   # [B, 3, n_mels, T]
        feats = feats.transpose(-1, -2).contiguous()   # [B, 3, T, n_mels]
        return feats


__all__ = ['MelFrontEnd', 'SAMPLE_RATE', 'TARGET_FRAMES', 'N_MELS']
