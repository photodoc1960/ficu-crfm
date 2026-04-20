"""Prediction-Error-Triggered Update (PETU) gating.

Standard EP applies physics updates on every sample. PETU applies them only
when the current level's prediction handshake fails — i.e. when the readout
loss exceeds a threshold or the field's phase coherence drops below a floor.

Usage in a training loop:

    err = F.cross_entropy(logits_free, target, reduction='none')   # [B]
    coh = compute_TBI(Z_free_r, Z_free_i, Z_nudge_r, Z_nudge_i)    # [B]
    mask = should_update_physics(err, coh, threshold=0.5,
                                 coherence_floor=0.3)               # [B] bool
    if mask.any():
        # Run nudge phase + EP delta on the masked subset
        ...

The fraction of samples for which `mask` is True is the per-batch update
fraction, which we log per epoch — it should decrease as the layer becomes
automatic.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def should_update_physics(prediction_error: torch.Tensor,
                          coherence: torch.Tensor,
                          threshold: Optional[float] = None,
                          coherence_floor: float = 0.3,
                          num_classes: Optional[int] = None,
                          threshold_frac: float = 0.5) -> torch.Tensor:
    """Return per-sample boolean mask of samples that should trigger EP.

    Args:
        prediction_error: [B] per-sample loss (cross-entropy or any scalar
                          measuring how badly the readout failed).
        coherence: [B] per-sample TBI / phase coherence.
        threshold: absolute error threshold. If None, derived from `num_classes`
                   as `threshold_frac * log(num_classes)` — i.e. the sample is
                   "doing OK" once its CE loss falls below half of the random-
                   chance loss. Passing an absolute `threshold` overrides this.
        coherence_floor: coherence below which we force an update even if
                         the loss is already low.
        num_classes: size of the classification head; used to derive the loss
                     threshold when `threshold` is None.
        threshold_frac: fraction of `log(num_classes)` to use as the "learned
                        enough" cutoff. Default 0.5.
    """
    if threshold is None:
        if num_classes is None:
            raise ValueError(
                "should_update_physics: must pass either an absolute "
                "`threshold` or `num_classes` to derive it."
            )
        threshold = threshold_frac * math.log(num_classes)
    return (prediction_error > threshold) | (coherence < coherence_floor)


def update_fraction(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item())


__all__ = ['should_update_physics', 'update_fraction']
