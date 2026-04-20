"""Handshake success rate.

Fraction of samples in a batch for which the L3 → L2 prediction is within
threshold of the actual settled L2 field state. The PredictiveField module
exposes the per-sample errors directly; this module just thresholds them.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def handshake_success_rate(errors: torch.Tensor, threshold: float) -> float:
    return (errors < threshold).float().mean().item()


__all__ = ['handshake_success_rate']
