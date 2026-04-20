"""L2 word-binding FICU.

Reads frozen L1 field states (a temporal sequence over a word's mel frames),
pools them to 47×32, projects through a learnable coupling matrix to drive
its own 47×32×3 complex field, settles via the same per-channel LG dynamics
as L1, and reads out word identities through a phase-feature holographic head.

Top-down feedback to L1 (lambda_td_L2_L1) is wired through but kept at zero
unless L3 enables it during phase 3.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ficu_crfm.architecture.ficu_l1 import (
    FICUL1Phoneme,
    HolographicAssociativeReadout,
    _mexican_hat,
)


class FICUL2Word(nn.Module):
    HEIGHT = 47
    WIDTH = 32
    CHANNELS = 3
    L1_HEIGHT = FICUL1Phoneme.HEIGHT
    L1_WIDTH = FICUL1Phoneme.WIDTH

    def __init__(self, n_classes, n_settle_steps=24, dt=0.07, beta=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.n_settle_steps = n_settle_steps
        self.dt = dt
        self.beta = beta

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

        self.coupling_l1 = nn.Parameter(0.1 * torch.eye(C))
        self.lambda_td_L2_L1 = nn.Parameter(torch.tensor(0.0))

        self.register_buffer('mexican_hat_kernel', _mexican_hat(7, 1.5, 4.0))

        self.pool_size = 3
        self.readout = HolographicAssociativeReadout(
            n_classes=n_classes,
            field_channels=self.CHANNELS * 3,
            field_height=self.pool_size,
            field_width=self.pool_size,
        )
        self.freeze_physics = False

    def physics_parameters(self):
        return [
            self.gamma_per_channel, self.omega_per_channel,
            self.diffusion_per_channel, self.beta_per_channel,
            self.chi_spm, self.chi_xpm, self.raw_evanescent,
            self.coupling_l1,
        ]

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
        total = intensity.sum(dim=1, keepdim=True)
        other = (total - intensity) / max(C - 1, 1)
        dZ_r = dZ_r + epsilon * other * Z_r
        dZ_i = dZ_i + epsilon * other * Z_i

        dZ_r = dZ_r + 1.0 * drive_r
        dZ_i = dZ_i + 1.0 * drive_i

        Z_r = (Z_r + self.dt * dZ_r) / (1 + self.dt * gamma)
        Z_i = (Z_i + self.dt * dZ_i) / (1 + self.dt * gamma)

        amplitude = torch.sqrt(Z_r**2 + Z_i**2 + 1e-8)
        scale = torch.clamp(10.0 / amplitude, max=1.0)
        return Z_r * scale, Z_i * scale

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

    # ------------------------------------------------------------------
    def drive_from_l1(self, Z1_r, Z1_i):
        """Pool L1 → L2 spatial size and apply coupling matrix."""
        Z1p_r = F.avg_pool2d(Z1_r, kernel_size=2, stride=2)  # [B, 3, 47, 32]
        Z1p_i = F.avg_pool2d(Z1_i, kernel_size=2, stride=2)
        # Match L2 size exactly (in case L1 dims aren't exactly 2× L2)
        if Z1p_r.shape[-2:] != (self.HEIGHT, self.WIDTH):
            Z1p_r = F.adaptive_avg_pool2d(Z1p_r, (self.HEIGHT, self.WIDTH))
            Z1p_i = F.adaptive_avg_pool2d(Z1p_i, (self.HEIGHT, self.WIDTH))
        Wm = self.coupling_l1.T  # [C, C]
        drive_r = torch.einsum('bchw,cd->bdhw', Z1p_r, Wm)
        drive_i = torch.einsum('bchw,cd->bdhw', Z1p_i, Wm)
        return drive_r, drive_i

    def settle(self, Z1_r, Z1_i, target_class=None,
               predicted_init=None):
        """Settle L2 from cached L1 states.

        Args:
            Z1_r, Z1_i: [B, 3, 94, 64] frozen L1 field state
            target_class: optional [B] for EP nudge phase
            predicted_init: optional (init_r, init_i) bias from PredictiveField
        """
        drive_r, drive_i = self.drive_from_l1(Z1_r, Z1_i)
        B = drive_r.shape[0]
        dev = drive_r.device

        if predicted_init is not None:
            Z_r = predicted_init[0].clone()
            Z_i = predicted_init[1].clone()
        else:
            Z_r = torch.zeros(B, self.CHANNELS, self.HEIGHT, self.WIDTH, device=dev)
            Z_i = torch.zeros_like(Z_r)

        for _ in range(self.n_settle_steps):
            Z_r, Z_i = self.field_step(Z_r, Z_i, drive_r, drive_i)
            if target_class is not None:
                Z_r, Z_i = self._apply_nudge(Z_r, Z_i, target_class)
        return Z_r, Z_i

    def _apply_nudge(self, Z_r, Z_i, target_class):
        ps = self.pool_size
        C = self.CHANNELS
        feat_r, feat_i = self._phase_features(Z_r, Z_i)
        flat = self.readout._flatten_field(feat_r, feat_i)
        norm = self.readout._normalize(flat)
        logits = norm @ self.readout.W.T + self.readout.bias
        pred = torch.softmax(logits, dim=1)
        target_oh = torch.zeros_like(pred)
        target_oh.scatter_(1, target_class.unsqueeze(1), 1.0)
        error = target_oh - pred

        nudge_flat = error @ self.readout.W
        half = nudge_flat.shape[1] // 2
        total_feats = 3 * C
        B = Z_r.shape[0]
        d_feat_r = nudge_flat[:, :half].view(B, total_feats, ps, ps)
        d_feat_i = nudge_flat[:, half:].view(B, total_feats, ps, ps)

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

        return Z_r + self.beta * d_Zr, Z_i + self.beta * d_Zi

    def forward(self, Z1_r, Z1_i):
        Z_r, Z_i = self.settle(Z1_r, Z1_i)
        feat_r, feat_i = self._phase_features(Z_r, Z_i)
        return self.readout(feat_r, feat_i), (Z_r, Z_i)

    def l2_to_l1_topdown_bias(self, Z2_r, Z2_i):
        """Compute the L2 → L1 top-down bias from current L2 state.

        Inverts `drive_from_l1` (channel reprojection through coupling_l1
        plus spatial upsample from L2's pooled grid back to L1's). Scaled
        by `self.lambda_td_L2_L1` — when that parameter is 0 the wire
        contributes nothing, matching the original "wired but inactive"
        behaviour. The bias has the additive form L1.settle(topdown_bias=...)
        expects.

        Args:
            Z2_r, Z2_i: [B, C, H2, W2] current L2 field state.

        Returns:
            (bias_r, bias_i): each [B, C, L1_HEIGHT, L1_WIDTH], scaled by lambda.
        """
        # Channel reprojection. drive_from_l1 uses `coupling_l1.T` so the
        # back-projection is `coupling_l1` (no transpose) — i.e. the same
        # learnable matrix in the opposite direction.
        Wm = self.coupling_l1   # [C, C]
        bias_r = torch.einsum('bchw,cd->bdhw', Z2_r, Wm)
        bias_i = torch.einsum('bchw,cd->bdhw', Z2_i, Wm)
        # Spatial upsample L2 grid → L1 grid.
        bias_r = F.interpolate(
            bias_r, size=(self.L1_HEIGHT, self.L1_WIDTH),
            mode='bilinear', align_corners=False,
        )
        bias_i = F.interpolate(
            bias_i, size=(self.L1_HEIGHT, self.L1_WIDTH),
            mode='bilinear', align_corners=False,
        )
        scale = self.lambda_td_L2_L1
        return bias_r * scale, bias_i * scale

    @torch.no_grad()
    def ep_observables(self, Z_r, Z_i):
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
    def apply_ep_update(self, obs_free, obs_nudge,
                        Z1_r, Z1_i, Z2_r_free, Z2_i_free,
                        Z2_r_nudge, Z2_i_nudge,
                        lr_physics=0.001):
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

        # Coupling matrix update via free/nudge field correlation
        Z1p_r_free = F.avg_pool2d(Z1_r, kernel_size=2, stride=2)
        Z1p_i_free = F.avg_pool2d(Z1_i, kernel_size=2, stride=2)
        if Z1p_r_free.shape[-2:] != (self.HEIGHT, self.WIDTH):
            Z1p_r_free = F.adaptive_avg_pool2d(Z1p_r_free, (self.HEIGHT, self.WIDTH))
            Z1p_i_free = F.adaptive_avg_pool2d(Z1p_i_free, (self.HEIGHT, self.WIDTH))
        corr_free = (Z1p_r_free * Z2_r_free + Z1p_i_free * Z2_i_free).mean(dim=(0, 2, 3))
        corr_nudge = (Z1p_r_free * Z2_r_nudge + Z1p_i_free * Z2_i_nudge).mean(dim=(0, 2, 3))
        d_coup = -(1.0 / self.beta) * (corr_nudge - corr_free)
        d_coup = d_coup.clamp(-clamp, clamp)
        self.coupling_l1.data += lr_physics * d_coup.unsqueeze(1).expand_as(self.coupling_l1)

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
            'd_coupling': d_coup.abs().mean().item(),
        }


__all__ = ['FICUL2Word']
