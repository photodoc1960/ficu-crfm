"""PredictiveField: L3 → L2 top-down prediction operator.

Maintains a learnable linear map from L3 field state to expected L2 field
state (real and imag pooled to L2 spatial size). Trained via the delta rule
on every sample using the actual settled L2 state as target.

The L2 layer initializes its field with the prediction (rather than zeros)
when this module is wired in — this is the operationalization of "pre-activate
L2 toward expected pattern."

Prediction error is the per-sample L2 norm of (actual − predicted), used by
PETU to decide whether to apply EP physics updates at L2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveField(nn.Module):
    """Linear L3 → L2 predictor.

    L3 field shape: [B, C3, H3, W3] = [B, 3, 24, 16]
    L2 field shape: [B, C2, H2, W2] = [B, 3, 47, 32]

    The predictor flattens the L3 field, projects through a learnable matrix
    to the L2 flat dimension, and reshapes. Real and imag parts share weights
    (the matrix is real-valued).
    """

    def __init__(self, l3_shape=(3, 24, 16), l2_shape=(3, 47, 32),
                 lr=0.01, lambda_td: float = 0.5,
                 handshake_threshold: float = 1.0):
        super().__init__()
        self.l3_shape = l3_shape
        self.l2_shape = l2_shape
        self.lr = lr
        self.lambda_td = lambda_td
        self.handshake_threshold = handshake_threshold

        l3_dim = int(torch.tensor(l3_shape).prod().item())
        l2_dim = int(torch.tensor(l2_shape).prod().item())
        # Initialize small so the predicted bias starts near zero.
        self.W = nn.Parameter(torch.randn(l2_dim, l3_dim) * 0.001)
        self.W.requires_grad = False  # delta-rule only

    def predict(self, Z3_r: torch.Tensor, Z3_i: torch.Tensor):
        """Return (predicted_l2_r, predicted_l2_i) shaped to l2_shape."""
        B = Z3_r.shape[0]
        flat_r = Z3_r.flatten(start_dim=1)
        flat_i = Z3_i.flatten(start_dim=1)
        pred_r = (flat_r @ self.W.T).view(B, *self.l2_shape) * self.lambda_td
        pred_i = (flat_i @ self.W.T).view(B, *self.l2_shape) * self.lambda_td
        return pred_r, pred_i

    @torch.no_grad()
    def update(self, Z3_r, Z3_i, Z2_actual_r, Z2_actual_i):
        """Delta-rule update of W toward (actual − predicted_unscaled)."""
        B = Z3_r.shape[0]
        l3_flat_r = Z3_r.flatten(start_dim=1)        # [B, l3_dim]
        l3_flat_i = Z3_i.flatten(start_dim=1)
        l2_flat_r = Z2_actual_r.flatten(start_dim=1) # [B, l2_dim]
        l2_flat_i = Z2_actual_i.flatten(start_dim=1)

        # Predict (without lambda scaling for the regression target).
        pred_r = l3_flat_r @ self.W.T
        pred_i = l3_flat_i @ self.W.T
        err_r = l2_flat_r - pred_r
        err_i = l2_flat_i - pred_i

        # Combine real & imag into one regression update.
        dW = (err_r.T @ l3_flat_r + err_i.T @ l3_flat_i) / max(B, 1)
        self.W.data += self.lr * dW

    @torch.no_grad()
    def prediction_error(self, Z3_r, Z3_i, Z2_actual_r, Z2_actual_i) -> torch.Tensor:
        """Per-sample L2 squared error between actual and *unscaled* predicted L2."""
        B = Z3_r.shape[0]
        l3_flat_r = Z3_r.flatten(start_dim=1)
        l3_flat_i = Z3_i.flatten(start_dim=1)
        pred_r = (l3_flat_r @ self.W.T).view(B, *self.l2_shape)
        pred_i = (l3_flat_i @ self.W.T).view(B, *self.l2_shape)
        err_r = Z2_actual_r - pred_r
        err_i = Z2_actual_i - pred_i
        return ((err_r**2 + err_i**2).mean(dim=(1, 2, 3)))

    @torch.no_grad()
    def handshake_success_rate(self, errors: torch.Tensor) -> float:
        """Fraction of samples below the handshake threshold."""
        return (errors < self.handshake_threshold).float().mean().item()


__all__ = ['PredictiveField']
