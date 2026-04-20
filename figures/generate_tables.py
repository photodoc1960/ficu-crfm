"""Generate Tables 1 and 2 for the FICU-CRFM paper.

Outputs both LaTeX (.tex) and Markdown (.md) formats to figures/.
"""
from pathlib import Path

FIGS = Path(__file__).resolve().parent

# -----------------------------------------------------------------------
# Table 1: Readout ablation at L1
# -----------------------------------------------------------------------

TABLE1_LATEX = r"""
\begin{table}[ht]
\centering
\caption{Readout ablation at Layer~1 (phoneme recognition, 40 classes, TIMIT).
All variants resume from the same trained L1 physics checkpoint; only the
readout extraction and classification scheme differ.}
\label{tab:readout_ablation}
\begin{tabular}{llcc}
\toprule
\textbf{Readout} & \textbf{Feature space} & \textbf{Dim} & \textbf{Val Acc (\%)} \\
\midrule
Holographic linear       & Pooled phase features          & 162   & \textbf{38.55} \\
Coherent matched filter  & Raw complex field              & 18,048 & 17.5 \\
Coherent matched filter  & Pooled complex features        & 81 (complex) & 2.6 \\
Whitened coherent        & Pooled complex features        & 81 (complex) & 3.0 \\
\midrule
\multicolumn{3}{l}{\textit{Chance level (40 classes)}} & 2.5 \\
\bottomrule
\end{tabular}
\end{table}
""".strip()

TABLE1_MD = """
**Table 1.** Readout ablation at Layer 1 (phoneme recognition, 40 classes, TIMIT). All variants resume from the same trained L1 physics checkpoint; only the readout extraction and classification scheme differ.

| Readout | Feature space | Dim | Val Acc (%) |
|---|---|---|---|
| **Holographic linear** | Pooled phase features | 162 | **38.55** |
| Coherent matched filter | Raw complex field | 18,048 | 17.5 |
| Coherent matched filter | Pooled complex features | 81 (complex) | 2.6 |
| Whitened coherent | Pooled complex features | 81 (complex) | 3.0 |
| *Chance level (40 classes)* | | | 2.5 |
""".strip()

# -----------------------------------------------------------------------
# Table 2: Cross-architecture comparison on TIMIT word recognition
# -----------------------------------------------------------------------

TABLE2_LATEX = r"""
\begin{table}[ht]
\centering
\caption{Cross-architecture comparison on TIMIT word recognition (501 classes).
Both models receive identical mel-spectrogram input features; they differ only
in the classification mechanism. The MLP uses backpropagation (Adam optimiser);
the FICU model uses equilibrium propagation with a holographic delta-rule readout.}
\label{tab:cross_architecture}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Training rule} & \textbf{Epochs} & \textbf{Val Acc (\%)} & \textbf{$\times$ Chance} \\
\midrule
MLP (512$\to$256$\to$501) & Backpropagation (Adam) & 100 & 25.3 & 127$\times$ \\
FICU L2 (holographic)     & EP + delta rule        & 15  & \textbf{38.85} & \textbf{195$\times$} \\
\midrule
\multicolumn{3}{l}{\textit{Improvement}} & \multicolumn{2}{c}{\textbf{+13.5 pp}} \\
\bottomrule
\end{tabular}
\end{table}
""".strip()

TABLE2_MD = """
**Table 2.** Cross-architecture comparison on TIMIT word recognition (501 classes). Both models receive identical mel-spectrogram input features; they differ only in the classification mechanism. The MLP uses backpropagation (Adam optimiser); the FICU model uses equilibrium propagation with a holographic delta-rule readout.

| Model | Training rule | Epochs | Val Acc (%) | x Chance |
|---|---|---|---|---|
| MLP (512->256->501) | Backpropagation (Adam) | 100 | 25.3 | 127x |
| **FICU L2 (holographic)** | **EP + delta rule** | **15** | **38.85** | **195x** |
| *Improvement* | | | **+13.5 pp** | |
""".strip()


def main():
    # LaTeX
    with open(FIGS / 'table1_readout_ablation.tex', 'w') as f:
        f.write(TABLE1_LATEX + '\n')
    with open(FIGS / 'table2_cross_architecture.tex', 'w') as f:
        f.write(TABLE2_LATEX + '\n')

    # Markdown
    with open(FIGS / 'table1_readout_ablation.md', 'w') as f:
        f.write(TABLE1_MD + '\n')
    with open(FIGS / 'table2_cross_architecture.md', 'w') as f:
        f.write(TABLE2_MD + '\n')

    print('Tables generated:')
    for ext in ('tex', 'md'):
        for t in ('table1_readout_ablation', 'table2_cross_architecture'):
            p = FIGS / f'{t}.{ext}'
            print(f'  {p.name}')


if __name__ == '__main__':
    main()
