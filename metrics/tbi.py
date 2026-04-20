"""Temporal Binding Index (TBI).

TBI = mean phase coherence between free and nudged field states, averaged
over spatial locations and channels. Range [0, 1].
  - High TBI: free and nudged phases align (prediction matches input,
    no update needed, automaticity operating).
  - Low TBI: phases diverge (prediction fails, supervision engaged).

Computed as |E[exp(i·(angle(nudged) − angle(free)))]|, where the expectation
is over space (and optionally batch). The CRFM derivation and the
ficu_l2_binding cross-modal coherence formula reduce to the same expression
when written in real-pair form.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def compute_TBI(Z_free_r: torch.Tensor, Z_free_i: torch.Tensor,
                Z_nudge_r: torch.Tensor, Z_nudge_i: torch.Tensor) -> torch.Tensor:
    """Return per-sample TBI [B].

    For each spatial location, compute z_nudge * conj(z_free), normalize by the
    product of magnitudes (giving a unit phasor), then average over space and
    channels and take the magnitude.
    """
    amp_free = torch.sqrt(Z_free_r**2 + Z_free_i**2 + 1e-8)
    amp_nudge = torch.sqrt(Z_nudge_r**2 + Z_nudge_i**2 + 1e-8)
    # nudge * conj(free) = (n_r + i n_i)(f_r - i f_i)
    corr_r = Z_nudge_r * Z_free_r + Z_nudge_i * Z_free_i
    corr_i = Z_nudge_i * Z_free_r - Z_nudge_r * Z_free_i
    norm = amp_free * amp_nudge + 1e-8
    unit_r = corr_r / norm
    unit_i = corr_i / norm
    mean_r = unit_r.mean(dim=(2, 3))   # [B, C]
    mean_i = unit_i.mean(dim=(2, 3))
    coherence = torch.sqrt(mean_r**2 + mean_i**2 + 1e-8)
    return coherence.mean(dim=1)       # [B]


@torch.no_grad()
def tbi_summary(Z_free_r, Z_free_i, Z_nudge_r, Z_nudge_i):
    tbi = compute_TBI(Z_free_r, Z_free_i, Z_nudge_r, Z_nudge_i)
    return {'mean': tbi.mean().item(), 'std': tbi.std().item()}


__all__ = ['compute_TBI', 'tbi_summary']
