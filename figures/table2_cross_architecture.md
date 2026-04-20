**Table 2.** Cross-architecture comparison on TIMIT word recognition (501 classes). Both models receive identical mel-spectrogram input features; they differ only in the classification mechanism. The MLP uses backpropagation (Adam optimiser); the FICU model uses equilibrium propagation with a holographic delta-rule readout.

| Model | Training rule | Epochs | Val Acc (%) | x Chance |
|---|---|---|---|---|
| MLP (512->256->501) | Backpropagation (Adam) | 100 | 25.3 | 127x |
| **FICU L2 (holographic)** | **EP + delta rule** | **15** | **38.85** | **195x** |
| *Improvement* | | | **+13.5 pp** | |
