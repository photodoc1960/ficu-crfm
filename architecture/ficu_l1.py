"""L1 phoneme FICU.

Adapted from FICU-audio / ficu_l2_multimodal/ficu_visual.py — same per-channel
Landau-Ginzburg field dynamics on a 94×64×3 complex field with semi-implicit
Euler integration, Mexican Hat lateral inhibition, XPM-modulated rotation,
evanescent cross-channel coupling, and EP learning via a phase-sensitive
holographic readout.

Differences from the visual L1:
  - Sensory entry is the mel front-end instead of a visual tensor pass-through.
  - exposes physics_parameters() so LevelGate can freeze the field cleanly.
  - exposes settle_with_topdown(...) accepting an optional bias field for the
    L3 → L2 → L1 prediction signal.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ficu_crfm.dataset.feature_extractor import MelFrontEnd, N_MELS, TARGET_FRAMES


class HolographicAssociativeReadout(nn.Module):
    """Linear readout trained with the delta rule on normalized phase features.

    Reused (verbatim layout) from ficu_l2_multimodal/ficu_visual.py.
    """

    def __init__(self, n_classes, field_channels, field_height, field_width):
        super().__init__()
        self.n_classes = n_classes
        self.field_channels = field_channels
        self.field_height = field_height
        self.field_width = field_width
        self.field_dim = 2 * field_channels * field_height * field_width
        self.W = nn.Parameter(torch.randn(n_classes, self.field_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.register_buffer('field_mean', torch.zeros(self.field_dim))
        self.register_buffer('field_var', torch.ones(self.field_dim))
        self.register_buffer('n_samples_seen', torch.tensor(0, dtype=torch.long))

    def _flatten_field(self, Z_r, Z_i):
        return torch.cat([Z_r.flatten(start_dim=1), Z_i.flatten(start_dim=1)], dim=1)

    def _normalize(self, field_flat):
        if self.n_samples_seen < 100:
            return field_flat / (field_flat.norm(dim=1, keepdim=True) + 1e-8)
        return (field_flat - self.field_mean) / (torch.sqrt(self.field_var) + 1e-8)

    def forward(self, Z_r, Z_i):
        flat = self._flatten_field(Z_r, Z_i)
        return self._normalize(flat) @ self.W.T + self.bias

    @torch.no_grad()
    def update(self, Z_r, Z_i, target, lr=0.01, weight_decay=0.01):
        B = Z_r.shape[0]
        flat = self._flatten_field(Z_r, Z_i)
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0)
        n = self.n_samples_seen.item()
        if n == 0:
            self.field_mean = batch_mean
            self.field_var = batch_var
        else:
            delta = batch_mean - self.field_mean
            new_n = n + B
            self.field_mean = self.field_mean + delta * (B / new_n)
            self.field_var = (self.field_var * n + batch_var * B
                              + delta**2 * n * B / new_n) / new_n
        self.n_samples_seen += B

        norm = self._normalize(flat)
        logits = norm @ self.W.T + self.bias
        pred = torch.softmax(logits, dim=1)
        target_oh = torch.zeros_like(pred)
        target_oh.scatter_(1, target.unsqueeze(1), 1.0)
        error = target_oh - pred

        delta_W = (error.T @ norm) / B - weight_decay * self.W
        delta_b = error.mean(dim=0)
        self.W.data += lr * delta_W
        self.bias.data += lr * delta_b
        return {'mean_error': error.abs().mean().item()}


class ComplexCoherentReadout(nn.Module):
    """Phase-coherent matched-filter classifier in the pooled phase-feature space.

    This is v3 — it operates on the 9×3×3 pooled features produced by
    `FICUL1Phoneme._phase_features`, NOT on the raw 94×64 field. The earlier
    raw-field variant saturated val_acc around 17% because the native field
    resolution isn't phoneme-separable without feature extraction. The
    holographic linear readout was doing the feature extraction correctly;
    its only weakness was classifying those features with a real-valued
    linear projection.

    The insight: the 162-D pooled feature space produced by
    `_phase_features` is physically complex. Each (feat_r, feat_i) pair at a
    given channel and spatial cell is a phasor:
        (Z_r_pooled,        Z_i_pooled)        — field state
        (amplitude_pooled,  sin_phase_pooled)  — amplitude + phase
        (cos_phase_pooled,  coherence_pooled)  — phase + coherence
    Treating them as a single complex vector
        feat_c = complex(feat_r_flat, feat_i_flat)    [B, 81] complex
    preserves their phase relationships. A complex matched-filter
    classifier in this space combines the holographic readout's feature
    extraction with a coherent (phase-aware) classifier.

    Scores:
        s_k = <feat_c, T_k>  :=  Σ_i conj(feat_c)_i · T_{k,i}    (complex)
        logit_k = logit_scale · |s_k|²                             (real)

    Nudge gradient (true CE derivative, Wirtinger-derived) is computed in
    complex *feature* space, then split into (Re, Im) and pushed back to
    field space via the same chain rule the holographic path uses (shared
    via `FICUL1Phoneme._feature_gradient_to_field`).

    Templates are EMA-updated per-class toward the mean complex feature of
    their class's nudged fields. No norm clipping is needed at this
    dimensionality — templates stay bounded by the field's own amplitude
    clamp (which bounds ||feat_c|| to a ~ O(1) range).
    """

    def __init__(self, n_classes, field_channels, field_height, field_width,
                 logit_scale=0.01):
        super().__init__()
        self.n_classes = n_classes
        self.field_channels = field_channels   # 9 after _phase_features
        self.field_height = field_height       # 3 (pool_size)
        self.field_width = field_width         # 3 (pool_size)
        self.feat_dim = field_channels * field_height * field_width  # 81
        # Temperature. With whitened features (unit variance per dim) the
        # typical magnitude is ||feat_c|| ≈ √feat_dim ≈ 9, and the template
        # asymptote is similar, so |<feat_c, T_k>|² ≲ (9·9)² ≈ 6500 at
        # saturation. logit_scale=0.01 keeps max logits ≤ ~65 which is
        # comfortably stable through logsumexp cross-entropy.
        self.logit_scale = logit_scale
        # Complex templates in 81-D feature space.
        self.templates = nn.Parameter(
            torch.randn(n_classes, self.feat_dim, dtype=torch.complex64) * 0.01
        )
        # Running statistics for feature whitening — matching the
        # HolographicAssociativeReadout's `_normalize` logic so both heads
        # classify on the same feature distribution. The mean is complex
        # (per-dim), the variance is the real scalar E[|z-μ|²] per-dim.
        # Buffers are prefixed `coh_` to avoid colliding with the
        # holographic head's same-named buffers when resuming a
        # holographic checkpoint under strict=False.
        self.register_buffer('coh_running_mean',
                             torch.zeros(self.feat_dim, dtype=torch.complex64))
        self.register_buffer('coh_running_var',
                             torch.ones(self.feat_dim))
        self.register_buffer('coh_n_samples_seen',
                             torch.tensor(0, dtype=torch.long))

    def _feat_c(self, feat_r, feat_i):
        """Combine [B, 9, 3, 3] real tensors into a [B, 81] complex vector."""
        return torch.complex(
            feat_r.flatten(start_dim=1),
            feat_i.flatten(start_dim=1),
        )

    # Variance floor used during whitening. Some feature dimensions
    # (e.g. spatial cells where the field is nearly constant across samples)
    # have running variance ≪ 1, and an unbounded `1/std` factor turns the
    # whitening chain rule into a numerical bomb in the gradient path. The
    # floor caps `1/std` at `1/sqrt(MIN_VAR) ≈ 10`, which is enough to put
    # all dimensions on a comparable scale without amplifying noise from
    # near-constant features.
    MIN_VAR = 0.01

    def _whiten(self, feat_c):
        """Z-score features against running mean/var (unit variance per dim).

        Before 100 samples are seen we fall back to per-sample L2 normalisation,
        mirroring the pattern used by HolographicAssociativeReadout._normalize.
        """
        if self.coh_n_samples_seen.item() < 100:
            norm = feat_c.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
            return feat_c / (norm + 1e-8)
        std = torch.sqrt(self.coh_running_var.clamp(min=self.MIN_VAR))   # [D] real
        return (feat_c - self.coh_running_mean) / std                    # [B, D] complex

    def _inner(self, feat_c_norm):
        """Complex inner products s[b,k] = Σ_i conj(feat_c_norm[b,i]) · T[k,i]."""
        return torch.einsum('bi,ki->bk', feat_c_norm.conj(), self.templates)

    def score(self, feat_r, feat_i):
        """[B, 9, 3, 3] real pair → [B, K] real logits (whitened matched filter)."""
        feat_c = self._feat_c(feat_r, feat_i)
        feat_c_norm = self._whiten(feat_c)
        s = self._inner(feat_c_norm)
        return self.logit_scale * s.abs().pow(2)

    def forward(self, feat_r, feat_i):
        return self.score(feat_r, feat_i)

    @torch.no_grad()
    def update(self, feat_r_nudge, feat_i_nudge, target, lr=0.005):
        """Per-class scatter EMA in whitened feature space.

        Two things happen here. First we update running statistics with the
        batch's unwhitened features (same Welford-like incremental update as
        HolographicAssociativeReadout). Then we whiten the batch's features
        with the updated stats and scatter the per-class mean of the
        whitened features into each template. This keeps templates living
        in the same normalised space as `score()` and `nudge_features()`.
        """
        feat_c = self._feat_c(feat_r_nudge, feat_i_nudge)  # [B, 81] complex
        B = feat_c.shape[0]

        # Running mean/var update (complex mean, real scalar variance).
        batch_mean = feat_c.mean(dim=0)                                # [D] complex
        batch_var = (feat_c - batch_mean).abs().pow(2).mean(dim=0)     # [D] real
        n = self.coh_n_samples_seen.item()
        if n == 0:
            self.coh_running_mean = batch_mean
            self.coh_running_var = batch_var
        else:
            delta = batch_mean - self.coh_running_mean
            new_n = n + B
            self.coh_running_mean = self.coh_running_mean + delta * (B / new_n)
            self.coh_running_var = (self.coh_running_var * n
                                + batch_var * B
                                + delta.abs().pow(2) * (n * B / new_n)) / new_n
        self.coh_n_samples_seen += B

        # Template EMA on whitened features.
        feat_c_norm = self._whiten(feat_c)
        for l in torch.unique(target):
            mask = target == l
            mean_c = feat_c_norm[mask].mean(dim=0)   # [81]
            idx = int(l.item())
            self.templates.data[idx] += lr * (mean_c - self.templates.data[idx])
        return {}

    @torch.no_grad()
    def nudge_features(self, feat_r, feat_i, target):
        """True cross-entropy gradient w.r.t. the complex feature vector.

        Returns (d_feat_r, d_feat_i) each shaped [B, 9, 3, 3] — the callers'
        shared chain rule will project these back to field-space via
        `FICUL1Phoneme._feature_gradient_to_field`.

        Derivation (Wirtinger, convention <feat_c, T> = Σ conj(feat_c)·T):
            s_k(feat_c) = Σ_i conj(feat_c)_i T_{k,i}
            score_k     = |s_k|²
            logits      = logit_scale · score_k
            p           = softmax(logits)
            loss        = −log p_target

            ∂logit_k / ∂feat_c = logit_scale · conj(s_k) · T_k
            ∇_{feat_c} (−loss) = logit_scale · Σ_k (1_{k=target} − p_k) ·
                                 conj(s_k) · T_k
        """
        feat_c = self._feat_c(feat_r, feat_i)              # [B, 81] complex
        feat_c_norm = self._whiten(feat_c)                 # [B, 81] complex (whitened)
        s = self._inner(feat_c_norm)                       # [B, K] complex
        scores = self.logit_scale * s.abs().pow(2)         # [B, K] real
        p = torch.softmax(scores, dim=1)                   # [B, K] real
        target_oh = F.one_hot(target, num_classes=self.n_classes).to(scores.dtype)
        err = target_oh - p                                # [B, K] real
        w = self.logit_scale * err * s.conj()              # [B, K] complex
        # Gradient in WHITENED feature space
        d_feat_c_norm = torch.einsum('bk,ki->bi', w, self.templates)  # [B, 81] complex
        # Chain rule through whitening: feat_c_norm = (feat_c - μ) / σ,
        # so ∂feat_c_norm/∂feat_c = 1/σ (elementwise real). Gradient in the
        # raw feat_c space is therefore d_feat_c = d_feat_c_norm / σ.
        if self.coh_n_samples_seen.item() < 100:
            d_feat_c = d_feat_c_norm  # per-sample-norm fallback has unit scale
        else:
            # Same MIN_VAR floor as `_whiten` so the gradient chain rule is
            # bounded — without this, low-variance dimensions explode the
            # gradient and corrupt the running stats permanently.
            std = torch.sqrt(self.coh_running_var.clamp(min=self.MIN_VAR))  # [D] real
            d_feat_c = d_feat_c_norm / std                                  # [B, 81] complex

        B = feat_r.shape[0]
        C, H, W = feat_r.shape[1], feat_r.shape[2], feat_r.shape[3]
        d_feat_r = d_feat_c.real.view(B, C, H, W)
        d_feat_i = d_feat_c.imag.view(B, C, H, W)
        return d_feat_r, d_feat_i


class TrajectoryHolographicReadout(nn.Module):
    """Spatiotemporal matched-filter readout.

    Extends the endpoint-only `HolographicAssociativeReadout` by reading the
    field at `n_taps` evenly-spaced times during the settling trajectory.
    Each tap produces a 162-D phase-feature vector (identical to the endpoint
    readout's feature extraction); the vectors are independently normalised
    against per-tap Welford running statistics and concatenated into a single
    (n_taps × 162)-D input to a linear matched-filter template matrix W.

    The field dynamics and EP physics are unmodified — the trajectory extension
    is purely readout-side. The EP nudge direction is derived from the endpoint
    tap's columns of W only, so the physics-side update rule sees the same
    gradient structure as the endpoint-only readout.
    """

    def __init__(self, n_classes, field_channels, field_height, field_width,
                 n_taps=3):
        super().__init__()
        self.n_classes = n_classes
        self.n_taps = n_taps
        self.slice_dim = 2 * field_channels * field_height * field_width  # 162
        self.total_dim = n_taps * self.slice_dim  # 486

        self.W = nn.Parameter(torch.zeros(n_classes, self.total_dim))
        self.bias = nn.Parameter(torch.zeros(n_classes))

        # Per-tap Welford running statistics: [n_taps, slice_dim].
        self.register_buffer('tap_means', torch.zeros(n_taps, self.slice_dim))
        self.register_buffer('tap_vars', torch.ones(n_taps, self.slice_dim))
        self.register_buffer('tap_n_seen', torch.zeros(n_taps, dtype=torch.long))

    @staticmethod
    def _flatten(feat_r, feat_i):
        return torch.cat([feat_r.flatten(start_dim=1),
                          feat_i.flatten(start_dim=1)], dim=1)

    def _normalize_tap(self, flat, tap_idx):
        n = self.tap_n_seen[tap_idx].item()
        if n < 100:
            return flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
        mean = self.tap_means[tap_idx]
        var = self.tap_vars[tap_idx]
        return (flat - mean) / (torch.sqrt(var) + 1e-8)

    def _build_trajectory_vector(self, feat_list):
        """feat_list: list of n_taps (feat_r, feat_i) tuples.
        Returns [B, total_dim] normalised concatenation."""
        slices = []
        for i, (fr, fi) in enumerate(feat_list):
            flat = self._flatten(fr, fi)
            slices.append(self._normalize_tap(flat, i))
        return torch.cat(slices, dim=1)

    def forward(self, feat_list):
        """feat_list: list of n_taps (feat_r, feat_i) tuples → [B, K] logits."""
        full = self._build_trajectory_vector(feat_list)
        return full @ self.W.T + self.bias

    @torch.no_grad()
    def update(self, feat_list, target, lr=0.01, weight_decay=0.01):
        B = feat_list[0][0].shape[0]
        # Update per-tap running stats.
        for i, (fr, fi) in enumerate(feat_list):
            flat = self._flatten(fr, fi)
            bm = flat.mean(dim=0)
            bv = flat.var(dim=0)
            n = self.tap_n_seen[i].item()
            if n == 0:
                self.tap_means[i] = bm
                self.tap_vars[i] = bv
            else:
                delta = bm - self.tap_means[i]
                new_n = n + B
                self.tap_means[i] = self.tap_means[i] + delta * (B / new_n)
                self.tap_vars[i] = (self.tap_vars[i] * n + bv * B
                                    + delta**2 * n * B / new_n) / new_n
            self.tap_n_seen[i] += B

        # Delta-rule template update on full trajectory vector.
        full = self._build_trajectory_vector(feat_list)
        logits = full @ self.W.T + self.bias
        pred = torch.softmax(logits, dim=1)
        target_oh = torch.zeros_like(pred)
        target_oh.scatter_(1, target.unsqueeze(1), 1.0)
        error = target_oh - pred
        delta_W = (error.T @ full) / B - weight_decay * self.W
        delta_b = error.mean(dim=0)
        self.W.data += lr * delta_W
        self.bias.data += lr * delta_b
        return {'mean_error': error.abs().mean().item()}


def _mexican_hat(size=9, sigma_exc=1.0, sigma_inh=3.0):
    center = size // 2
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32) - center,
        torch.arange(size, dtype=torch.float32) - center,
        indexing='ij',
    )
    r_sq = x**2 + y**2
    g_exc = torch.exp(-r_sq / (2 * sigma_exc**2))
    g_exc = g_exc / g_exc.sum()
    g_inh = torch.exp(-r_sq / (2 * sigma_inh**2))
    g_inh = g_inh / g_inh.sum()
    return (g_exc - g_inh).view(1, 1, size, size)


class FICUL1Phoneme(nn.Module):
    """L1 phoneme field. 94×64×3 complex field, 39 phoneme classes.

    Forward accepts a raw waveform [B, 1600] (or [B, 1, 1600]) at 16 kHz.
    The mel front-end converts to a 3-channel feature [B, 3, 94, 64] which
    becomes the field drive (real part); imag drive is zero.
    """

    HEIGHT = TARGET_FRAMES   # 94
    WIDTH = N_MELS           # 64
    CHANNELS = 3

    def __init__(self, n_classes=39, n_settle_steps=24, dt=0.07, beta=0.1,
                 readout_mode='holographic'):
        super().__init__()
        self.n_classes = n_classes
        self.n_settle_steps = n_settle_steps
        self.dt = dt
        self.beta = beta
        self.readout_mode = readout_mode

        self.mel = MelFrontEnd()

        C = self.CHANNELS
        gamma_values = [0.025, 0.050, 0.100]
        self.gamma_per_channel = nn.Parameter(torch.tensor(gamma_values[:C]))

        omega_values = torch.logspace(math.log10(0.8), math.log10(2.0), C)
        self.omega_per_channel = nn.Parameter(omega_values)

        self.diffusion_per_channel = nn.Parameter(torch.full((C,), 0.1))

        beta_values = [0.060, 0.110, 0.210]
        self.beta_per_channel = nn.Parameter(torch.tensor(beta_values[:C]))

        self.chi_spm = nn.Parameter(torch.tensor(0.15))
        self.chi_xpm = nn.Parameter(torch.tensor(0.10))
        self.raw_evanescent = nn.Parameter(torch.tensor(0.01))

        self.register_buffer('mexican_hat_kernel', _mexican_hat(9, 1.0, 3.0))

        self.pool_size = 3
        if readout_mode == 'holographic':
            self.readout = HolographicAssociativeReadout(
                n_classes=n_classes,
                field_channels=self.CHANNELS * 3,  # 9: (Zr,|Z|,cos) + (Zi,sin,coh)
                field_height=self.pool_size,
                field_width=self.pool_size,
            )
        elif readout_mode == 'coherent':
            self.readout = ComplexCoherentReadout(
                n_classes=n_classes,
                field_channels=self.CHANNELS * 3,
                field_height=self.pool_size,
                field_width=self.pool_size,
            )
        elif readout_mode == 'trajectory':
            self.n_taps = 3
            self.readout = TrajectoryHolographicReadout(
                n_classes=n_classes,
                field_channels=self.CHANNELS * 3,
                field_height=self.pool_size,
                field_width=self.pool_size,
                n_taps=self.n_taps,
            )
        else:
            raise ValueError(f"unknown readout_mode: {readout_mode}")
        self.freeze_physics = False

    # ------------------------------------------------------------------
    # Public hook for the LevelGate
    # ------------------------------------------------------------------
    def physics_parameters(self):
        return [
            self.gamma_per_channel, self.omega_per_channel,
            self.diffusion_per_channel, self.beta_per_channel,
            self.chi_spm, self.chi_xpm, self.raw_evanescent,
        ]

    # ------------------------------------------------------------------
    # Field dynamics
    # ------------------------------------------------------------------
    def field_step(self, Z_r, Z_i, drive_r, drive_i):
        B, C, H, W = Z_r.shape
        gamma = (torch.sigmoid(self.gamma_per_channel) * 0.2).view(1, C, 1, 1)
        omega = (torch.abs(self.omega_per_channel) + 0.1).view(1, C, 1, 1)
        D = (torch.sigmoid(self.diffusion_per_channel) * 0.5).view(1, C, 1, 1)
        beta = (torch.abs(self.beta_per_channel) + 0.01).view(1, C, 1, 1)

        intensity = Z_r**2 + Z_i**2
        dZ_r = -2.0 * beta * intensity * Z_r
        dZ_i = -2.0 * beta * intensity * Z_i

        chi_spm = torch.sigmoid(self.chi_spm) * 0.5
        chi_xpm = torch.sigmoid(self.chi_xpm) * 0.3
        int_mean = intensity.mean(dim=(2, 3), keepdim=True)
        total_int = int_mean.sum(dim=1, keepdim=True)
        other_int = total_int - int_mean
        omega_eff = omega + chi_spm * int_mean + chi_xpm * other_int
        dZ_r = dZ_r - omega_eff * Z_i
        dZ_i = dZ_i + omega_eff * Z_r

        pad = self.mexican_hat_kernel.shape[-1] // 2
        Z_r_pad = F.pad(Z_r, (pad, pad, pad, pad), mode='circular')
        Z_i_pad = F.pad(Z_i, (pad, pad, pad, pad), mode='circular')
        kernel = self.mexican_hat_kernel.expand(C, 1, -1, -1)
        mh_r = F.conv2d(Z_r_pad, kernel, groups=C)
        mh_i = F.conv2d(Z_i_pad, kernel, groups=C)
        dZ_r = dZ_r + D * mh_r
        dZ_i = dZ_i + D * mh_i

        epsilon = F.softplus(self.raw_evanescent) * 0.1
        int_per_ch = intensity
        total = int_per_ch.sum(dim=1, keepdim=True)
        other = (total - int_per_ch) / max(C - 1, 1)
        dZ_r = dZ_r + epsilon * other * Z_r
        dZ_i = dZ_i + epsilon * other * Z_i

        dZ_r = dZ_r + 0.1 * drive_r
        dZ_i = dZ_i + 0.1 * drive_i

        Z_r = (Z_r + self.dt * dZ_r) / (1 + self.dt * gamma)
        Z_i = (Z_i + self.dt * dZ_i) / (1 + self.dt * gamma)

        amplitude = torch.sqrt(Z_r**2 + Z_i**2 + 1e-8)
        scale = torch.clamp(10.0 / amplitude, max=1.0)
        return Z_r * scale, Z_i * scale

    # ------------------------------------------------------------------
    def _phase_features(self, Z_r, Z_i):
        ps = self.pool_size
        amplitude = torch.sqrt(Z_r**2 + Z_i**2 + 1e-8)
        cos_phase = Z_r / amplitude
        sin_phase = Z_i / amplitude
        local_cos = F.avg_pool2d(cos_phase, kernel_size=5, stride=1, padding=2)
        local_sin = F.avg_pool2d(sin_phase, kernel_size=5, stride=1, padding=2)
        coherence_R = torch.sqrt(local_cos**2 + local_sin**2 + 1e-8)

        Z_r_g = F.adaptive_avg_pool2d(Z_r, (ps, ps))
        Z_i_g = F.adaptive_avg_pool2d(Z_i, (ps, ps))
        amplitude_g = F.adaptive_avg_pool2d(amplitude, (ps, ps))
        cos_phase_g = F.adaptive_avg_pool2d(cos_phase, (ps, ps))
        sin_phase_g = F.adaptive_avg_pool2d(sin_phase, (ps, ps))
        coherence_g = F.adaptive_avg_pool2d(coherence_R, (ps, ps))

        feat_r = torch.cat([Z_r_g, amplitude_g, cos_phase_g], dim=1)
        feat_i = torch.cat([Z_i_g, sin_phase_g, coherence_g], dim=1)
        return feat_r, feat_i

    def _tap_steps(self):
        """Settle-step indices at which trajectory snapshots are taken.
        Evenly spaced through the settle: for n_settle_steps=24, n_taps=3
        this gives {8, 16, 24}. Returns a set for O(1) lookup."""
        n_taps = getattr(self, 'n_taps', 0)
        if n_taps == 0:
            return set()
        return {(i + 1) * self.n_settle_steps // n_taps
                for i in range(n_taps)}

    def settle(self, waveform, target_class=None, topdown_bias=None):
        """Run the field to settle. Returns (Z_r, Z_i).

        When `readout_mode == 'trajectory'`, intermediate field-sample
        snapshots are stored in `self._trajectory` (a list of
        `(feat_r, feat_i)` tuples) for use by the trajectory readout.

        Args:
            waveform: [B, 1600] or [B, 1, 1600]
            target_class: optional [B] tensor for EP nudge phase
            topdown_bias: optional (bias_r, bias_i) [B, 3, 94, 64] additive
                          bias from L2/L3 prediction; mixed into the drive.
        """
        feats = self.mel(waveform)               # [B, 3, 94, 64]
        drive_r = feats
        drive_i = torch.zeros_like(feats)
        if topdown_bias is not None:
            br, bi = topdown_bias
            drive_r = drive_r + br
            drive_i = drive_i + bi

        B = drive_r.shape[0]
        dev = drive_r.device
        Z_r = torch.zeros(B, self.CHANNELS, self.HEIGHT, self.WIDTH, device=dev)
        Z_i = torch.zeros_like(Z_r)

        tap_steps = self._tap_steps()
        trajectory = []

        for step in range(1, self.n_settle_steps + 1):
            Z_r, Z_i = self.field_step(Z_r, Z_i, drive_r, drive_i)

            if target_class is not None:
                Z_r, Z_i = self._apply_nudge(Z_r, Z_i, target_class)

            if step in tap_steps:
                trajectory.append(self._phase_features(Z_r, Z_i))

        self._trajectory = trajectory
        return Z_r, Z_i

    def _apply_nudge(self, Z_r, Z_i, target_class):
        """Apply the readout-specific EP nudge to the current field state.

        Both readout modes share the same phase-feature extraction step and
        the same feature-space→field-space chain rule. They differ only in
        how they compute the feature-space gradient `(d_feat_r, d_feat_i)`:
          - 'holographic': `error @ self.readout.W`, splitting the 162-D
            real output into the (feat_r, feat_i) halves.
          - 'coherent': `ComplexCoherentReadout.nudge_features`, which
            returns the true CE gradient as a [B, 81] complex vector and
            splits it into (Re, Im).
        """
        feat_r, feat_i = self._phase_features(Z_r, Z_i)

        if self.readout_mode == 'coherent':
            d_feat_r, d_feat_i = self.readout.nudge_features(
                feat_r, feat_i, target_class
            )
        elif self.readout_mode == 'trajectory':
            # Nudge uses the ENDPOINT tap's W columns only. This keeps the EP
            # physics update semantically identical to the endpoint-only case
            # while letting the readout benefit from the full trajectory.
            sd = self.readout.slice_dim  # 162
            W_endpoint = self.readout.W[:, -sd:]  # [K, 162]
            flat = TrajectoryHolographicReadout._flatten(feat_r, feat_i)
            norm = self.readout._normalize_tap(flat, self.n_taps - 1)
            logits = norm @ W_endpoint.T + self.readout.bias
            pred = torch.softmax(logits, dim=1)
            target_oh = torch.zeros_like(pred)
            target_oh.scatter_(1, target_class.unsqueeze(1), 1.0)
            error = target_oh - pred

            nudge_flat = error @ W_endpoint
            half = nudge_flat.shape[1] // 2
            C = self.CHANNELS
            total_feats = 3 * C
            ps = self.pool_size
            B = Z_r.shape[0]
            d_feat_r = nudge_flat[:, :half].view(B, total_feats, ps, ps)
            d_feat_i = nudge_flat[:, half:].view(B, total_feats, ps, ps)
        else:  # holographic
            flat = self.readout._flatten_field(feat_r, feat_i)
            norm = self.readout._normalize(flat)
            logits = norm @ self.readout.W.T + self.readout.bias
            pred = torch.softmax(logits, dim=1)
            target_oh = torch.zeros_like(pred)
            target_oh.scatter_(1, target_class.unsqueeze(1), 1.0)
            error = target_oh - pred

            nudge_flat = error @ self.readout.W
            half = nudge_flat.shape[1] // 2
            C = self.CHANNELS
            total_feats = 3 * C
            ps = self.pool_size
            B = Z_r.shape[0]
            d_feat_r = nudge_flat[:, :half].view(B, total_feats, ps, ps)
            d_feat_i = nudge_flat[:, half:].view(B, total_feats, ps, ps)

        d_Zr, d_Zi = self._feature_gradient_to_field(
            Z_r, Z_i, d_feat_r, d_feat_i,
        )
        return Z_r + self.beta * d_Zr, Z_i + self.beta * d_Zi

    def _feature_gradient_to_field(self, Z_r, Z_i, d_feat_r, d_feat_i):
        """Analytical Jacobian of _phase_features, inverted to push a
        feature-space gradient back to field space.

        Shared by both readout modes. See `_phase_features` for how
        (Z_r, Z_i) maps to the 9-channel pooled feature tensor; this is the
        corresponding chain rule. The `coherence_g` feature (indices 2C:3C
        of feat_i) is deliberately not back-propagated — matches the
        original holographic behaviour.
        """
        C = self.CHANNELS
        ps = self.pool_size
        H, W = Z_r.shape[2], Z_r.shape[3]

        d_feat_r_up = F.interpolate(d_feat_r, size=(H, W), mode='nearest')
        d_feat_i_up = F.interpolate(d_feat_i, size=(H, W), mode='nearest')

        d_Zr_g = d_feat_r_up[:, :C]
        d_amp = d_feat_r_up[:, C:2 * C]
        d_cos = d_feat_r_up[:, 2 * C:3 * C]
        d_Zi_g = d_feat_i_up[:, :C]
        d_sin = d_feat_i_up[:, C:2 * C]

        amplitude = torch.sqrt(Z_r**2 + Z_i**2 + 1e-8)
        amp3 = amplitude ** 3
        cell = (H / ps) * (W / ps)
        scale = 1.0 / cell

        d_Zr = d_Zr_g * scale
        d_Zr = d_Zr + d_amp * (Z_r / amplitude) * scale
        d_Zr = d_Zr + d_cos * (Z_i**2 / amp3) * scale
        d_Zr = d_Zr + d_sin * (-Z_r * Z_i / amp3) * scale

        d_Zi = d_Zi_g * scale
        d_Zi = d_Zi + d_amp * (Z_i / amplitude) * scale
        d_Zi = d_Zi + d_cos * (-Z_r * Z_i / amp3) * scale
        d_Zi = d_Zi + d_sin * (Z_r**2 / amp3) * scale

        return d_Zr, d_Zi

    def forward(self, waveform):
        Z_r, Z_i = self.settle(waveform)
        if self.readout_mode == 'trajectory':
            return self.readout(self._trajectory)
        feat_r, feat_i = self._phase_features(Z_r, Z_i)
        return self.readout(feat_r, feat_i)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def ep_observables(self, Z_r, Z_i):
        """Per-channel observables used by the EP physics deltas."""
        intensity = Z_r**2 + Z_i**2
        i_mean = intensity.mean(dim=(0, 2, 3))
        q_mean = (intensity ** 2).mean(dim=(0, 2, 3))
        gy = Z_r[:, :, 1:, :] - Z_r[:, :, :-1, :]
        gx = Z_r[:, :, :, 1:] - Z_r[:, :, :, :-1]
        gyi = Z_i[:, :, 1:, :] - Z_i[:, :, :-1, :]
        gxi = Z_i[:, :, :, 1:] - Z_i[:, :, :, :-1]
        grad = ((gy**2 + gyi**2).mean(dim=(0, 2, 3))
                + (gx**2 + gxi**2).mean(dim=(0, 2, 3)))
        phase = (Z_r * Z_i).mean(dim=(0, 2, 3))
        return {'i': i_mean, 'q': q_mean, 'grad': grad, 'phase': phase}

    @torch.no_grad()
    def apply_ep_update(self, obs_free, obs_nudge, lr_physics=0.001):
        if self.freeze_physics:
            return {}
        clamp = 0.01
        dg = -(1.0 / self.beta) * (obs_nudge['i'] - obs_free['i'])
        dg = dg.clamp(-clamp, clamp)
        self.gamma_per_channel.data += lr_physics * dg

        db = -(1.0 / self.beta) * (obs_nudge['q'] - obs_free['q'])
        db = db.clamp(-clamp, clamp)
        self.beta_per_channel.data += lr_physics * db

        dD = -(1.0 / self.beta) * (obs_nudge['grad'] - obs_free['grad'])
        dD = dD.clamp(-clamp, clamp)
        self.diffusion_per_channel.data += lr_physics * dD

        dw = -(1.0 / self.beta) * (obs_nudge['phase'] - obs_free['phase'])
        dw = dw.clamp(-clamp, clamp)
        self.omega_per_channel.data += lr_physics * dw

        gamma_floor = math.log(0.01 / (0.2 - 0.01))
        self.gamma_per_channel.data.clamp_(min=gamma_floor)
        D_floor = math.log(0.001 / (0.5 - 0.001))
        self.diffusion_per_channel.data.clamp_(min=D_floor)
        self.beta_per_channel.data.clamp_(min=0.0)

        return {
            'd_gamma': dg.abs().mean().item(),
            'd_beta': db.abs().mean().item(),
            'd_D': dD.abs().mean().item(),
            'd_omega': dw.abs().mean().item(),
        }


__all__ = ['FICUL1Phoneme', 'HolographicAssociativeReadout',
           'ComplexCoherentReadout', 'TrajectoryHolographicReadout']
