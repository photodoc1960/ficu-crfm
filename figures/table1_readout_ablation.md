**Table 1.** Readout ablation at Layer 1 (phoneme recognition, 40 classes, TIMIT). All variants resume from the same trained L1 physics checkpoint; only the readout extraction and classification scheme differ.

| Readout | Feature space | Dim | Val Acc (%) |
|---|---|---|---|
| **Holographic linear** | Pooled phase features | 162 | **38.55** |
| Coherent matched filter | Raw complex field | 18,048 | 17.5 |
| Coherent matched filter | Pooled complex features | 81 (complex) | 2.6 |
| Whitened coherent | Pooled complex features | 81 (complex) | 3.0 |
| *Chance level (40 classes)* | | | 2.5 |
