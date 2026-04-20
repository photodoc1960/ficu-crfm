"""LevelGate: monitors per-layer accuracy and freezes physics on cross."""

from __future__ import annotations


class LevelGate:
    """Freeze a FICU layer's physics once validation accuracy crosses threshold.

    Usage:
        gate = LevelGate(threshold=0.60, patience=3)
        for epoch in ...:
            val_acc = validate(layer)
            triggered = gate.update(val_acc, layer)
            if triggered:
                # signal next phase
                break
    """

    def __init__(self, threshold: float, patience: int = 3, name: str = ''):
        self.threshold = threshold
        self.patience = patience
        self.name = name
        self.epochs_above_threshold = 0
        self.frozen = False
        self.last_acc = 0.0

    def update(self, val_accuracy: float, field) -> bool:
        self.last_acc = val_accuracy
        if val_accuracy >= self.threshold:
            self.epochs_above_threshold += 1
        else:
            self.epochs_above_threshold = 0

        if self.epochs_above_threshold >= self.patience and not self.frozen:
            self.freeze(field)
            return True
        return False

    def freeze(self, field) -> None:
        for param in field.physics_parameters():
            param.requires_grad = False
        if hasattr(field, 'freeze_physics'):
            field.freeze_physics = True
        self.frozen = True
        tag = f"[{self.name}] " if self.name else ""
        print(f"{tag}Layer frozen at {self.last_acc * 100:.1f}% — automaticity achieved")


__all__ = ['LevelGate']
