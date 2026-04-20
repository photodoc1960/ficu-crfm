"""L3 sentence-prediction FICU.

Receives bottom-up drive from a temporal mean of frozen L2 field states across
all words in a sentence, runs the same per-channel LG dynamics on a 24×16×3
field, and reads out 3 sentence types (sa/si/sx).

Top-down: emits a prediction of the next-expected L2 state via a
PredictiveField (built externally and held by the training loop). The L3 layer
itself is the consumer of bottom-up L2 means and the producer of L3 field
states; the prediction operator and lambda_td_L3_L2 live separately in
predictive_field.py.

L3 never freezes — it has no LevelGate.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ficu_crfm.architecture.ficu_l1 import (
    HolographicAssociativeReadout,
    _mexican_hat,
)


class FICUL3Sentence(nn.Module):
    HEIGHT = 24
    WIDTH = 16
    CHANNELS = 3

    def __init__(self, n_classes=3, n_settle_steps=24, dt=0.07, beta=0.1):
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

        self.coupling_l2 = nn.Parameter(1.0 * torch.eye(C))

        self.register_buffer('mexican_hat_kernel', _mexican_hat(5, 1.0, 2.5))

        self.pool_size = 3
        self.readout = HolographicAssociativeReadout(
            n_classes=n_classes,
            field_channels=self.CHANNELS * 3,
            field_height=self.pool_size,
            field_width=self.pool_size,
        )
        self.freeze_physics = False  # L3 never freezes, but kept for symmetry

    def physics_parameters(self):
        return [
            self.gamma_per_channel, self.omega_per_channel,
            self.diffusion_per_channel, self.beta_per_channel,
            self.chi_spm, self.chi_xpm, self.raw_evanescent,
            self.coupling_l2,
        ]

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
        local_cos = F.avg_pool2d(cos_phase, kernel_size=3, stride=1, padding=1)
        local_sin = F.avg_pool2d(sin_phase, kernel_size=3, stride=1, padding=1)
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

    def drive_from_l2_mean(self, Z2_mean_r, Z2_mean_i):
        """Pool a temporal-mean L2 field [B, 3, 47, 32] to L3 size [B, 3, 24, 16]
        and apply coupling matrix."""
        Z2p_r = F.adaptive_avg_pool2d(Z2_mean_r, (self.HEIGHT, self.WIDTH))
        Z2p_i = F.adaptive_avg_pool2d(Z2_mean_i, (self.HEIGHT, self.WIDTH))
        Wm = self.coupling_l2.T
        drive_r = torch.einsum('bchw,cd->bdhw', Z2p_r, Wm)
        drive_i = torch.einsum('bchw,cd->bdhw', Z2p_i, Wm)
        return drive_r, drive_i

    def settle(self, Z2_mean_r, Z2_mean_i, target_class=None):
        drive_r, drive_i = self.drive_from_l2_mean(Z2_mean_r, Z2_mean_i)
        B = drive_r.shape[0]
        dev = drive_r.device
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

    def forward(self, Z2_mean_r, Z2_mean_i):
        Z_r, Z_i = self.settle(Z2_mean_r, Z2_mean_i)
        feat_r, feat_i = self._phase_features(Z_r, Z_i)
        return self.readout(feat_r, feat_i), (Z_r, Z_i)

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
    def apply_ep_update(self, obs_free, obs_nudge, lr_physics=0.001):
        clamp = 0.01
        dg = -(1.0 / self.beta) * (obs_nudge['i'] - obs_free['i'])
        self.gamma_per_channel.data += lr_physics * dg.clamp(-clamp, clamp)
        db = -(1.0 / self.beta) * (obs_nudge['q'] - obs_free['q'])
        self.beta_per_channel.data += lr_physics * db.clamp(-clamp, clamp)
        dD = -(1.0 / self.beta) * (obs_nudge['grad'] - obs_free['grad'])
        self.diffusion_per_channel.data += lr_physics * dD.clamp(-clamp, clamp)
        dw = -(1.0 / self.beta) * (obs_nudge['phase'] - obs_free['phase'])
        self.omega_per_channel.data += lr_physics * dw.clamp(-clamp, clamp)
        gamma_floor = math.log(0.01 / (0.2 - 0.01))
        self.gamma_per_channel.data.clamp_(min=gamma_floor)
        D_floor = math.log(0.001 / (0.5 - 0.001))
        self.diffusion_per_channel.data.clamp_(min=D_floor)
        self.beta_per_channel.data.clamp_(min=0.0)
        return {
            'd_gamma': dg.abs().mean().item(),
            'd_D': dD.abs().mean().item(),
        }


__all__ = ['FICUL3Sentence']
