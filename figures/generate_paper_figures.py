"""Generate all six paper figures for the FICU-CRFM hierarchical cascade paper.

Usage:
    python figures/generate_paper_figures.py

Outputs PDF figures to figures/ directory.
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / 'logs'
FIGS = ROOT / 'figures'
FIGS.mkdir(exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'L1': '#2196F3',
    'L2': '#FF9800',
    'L3': '#4CAF50',
    'baseline': '#9E9E9E',
    'accent': '#E91E63',
}


def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


# -----------------------------------------------------------------------
# Figure 1: Architecture diagram (schematic, no data)
# -----------------------------------------------------------------------

def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Figure 1: Three-Layer FICU-CRFM Architecture', fontweight='bold', pad=15)

    # Layer boxes
    layers = [
        (1.5, 1.0, 2.0, 1.2, 'L1\n94 x 64 x 3\nPhoneme', COLORS['L1']),
        (4.0, 1.0, 2.0, 1.2, 'L2\n47 x 32 x 3\nWord', COLORS['L2']),
        (6.5, 1.0, 2.0, 1.2, 'L3\n24 x 16 x 3\nSentence', COLORS['L3']),
    ]
    for x, y, w, h, label, color in layers:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                        facecolor=color, alpha=0.3, edgecolor=color, lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, fontweight='bold')

    # Input
    ax.annotate('Mel\n3ch x 94 x 64', xy=(1.5, 1.6), xytext=(0.0, 1.6),
                fontsize=8, ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Bottom-up arrows
    for x1, x2, label in [(3.5, 4.0, 'pool 2x\n+ coupling'), (6.0, 6.5, 'pool\n+ coupling')]:
        ax.annotate('', xy=(x2, 1.6), xytext=(x1, 1.6),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text((x1+x2)/2, 2.0, label, ha='center', va='bottom', fontsize=7, color='#555')

    # Top-down arrow (PredictiveField)
    ax.annotate('', xy=(6.0, 2.5), xytext=(6.5, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2, linestyle='--'))
    ax.text(6.25, 2.9, 'PredictiveField\n(L3 -> L2)', ha='center', va='bottom',
            fontsize=7, color=COLORS['accent'], fontstyle='italic')

    # L2 -> L1 top-down
    ax.annotate('', xy=(3.5, 2.5), xytext=(4.0, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2, linestyle='--'))
    ax.text(3.75, 2.9, 'Top-down\nbias', ha='center', va='bottom',
            fontsize=7, color=COLORS['accent'], fontstyle='italic')

    # Readout boxes
    for x, y, label in [(1.8, 3.5, 'Holographic\nReadout'), (4.3, 3.5, 'Holographic\nReadout'), (6.8, 3.5, 'Holographic\nReadout')]:
        rect = mpatches.FancyBboxPatch((x, y), 1.4, 0.7, boxstyle='round,pad=0.05',
                                        facecolor='#FFF9C4', edgecolor='#FBC02D', lw=1)
        ax.add_patch(rect)
        ax.text(x + 0.7, y + 0.35, label, ha='center', va='center', fontsize=7)

    # Arrows to readout
    for x in [2.5, 5.0, 7.5]:
        ax.annotate('', xy=(x, 3.5), xytext=(x, 2.2),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1))

    # EP + PETU label
    ax.text(5.0, 0.4, 'EP physics updates gated by PETU (per-sample TBI + CE threshold)',
            ha='center', va='center', fontsize=8, fontstyle='italic', color='#777')

    fig.savefig(FIGS / 'fig1_architecture.pdf')
    fig.savefig(FIGS / 'fig1_architecture.png')
    plt.close(fig)
    print('  Fig 1: architecture diagram saved')


# -----------------------------------------------------------------------
# Figure 2: L1 TBI dynamics
# -----------------------------------------------------------------------

def fig2_l1_tbi_dynamics():
    # Two data sources: 10-epoch diagnostic + 30-epoch extended Run C
    diag = read_csv(LOGS / 'l1_timit_diagnostic.csv')
    ext = read_csv(LOGS / 'l1_runc_extended.csv')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # (a) Validation accuracy — use the extended 100-epoch readout run if available
    ext_acc = LOGS / 'exp_a_l1_trajectory.csv'
    if ext_acc.exists():
        acc_data = read_csv(ext_acc)
        ep_acc = [int(r['epoch']) for r in acc_data]
        va_acc = [float(r['val_acc'])*100 for r in acc_data]
        axes[0].plot(ep_acc, va_acc, color=COLORS['L1'], lw=1.5, label='Trajectory readout')

    # Also plot the 10-epoch diagnostic
    ep_d = [int(r['epoch']) for r in diag]
    va_d = [float(r['val_acc'])*100 for r in diag]
    axes[0].plot(ep_d, va_d, color=COLORS['L1'], lw=2, marker='o', ms=4, label='L1 (10-ep diagnostic)')
    axes[0].axhline(38.55, color=COLORS['baseline'], ls='--', lw=1, label='Endpoint baseline (38.55%)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('(a) L1 Phoneme Accuracy')
    axes[0].legend(fontsize=7, loc='lower right')

    # (b) TBI mean — extended Run C (30 epochs, beta=1.0, shows basin-hopping)
    ep_e = [int(r['epoch']) for r in ext]
    tbi_e = [float(r['TBI_mean']) for r in ext]
    axes[1].plot(ep_e, tbi_e, color=COLORS['L1'], lw=1.5)
    axes[1].axhline(0.380, color=COLORS['baseline'], ls=':', lw=1, label='Fixed point ~0.380')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('TBI Mean')
    axes[1].set_title('(b) TBI Mean (Extended Run C)')
    axes[1].legend(fontsize=7)

    # (c) TBI std — extended Run C
    std_e = [float(r['TBI_std']) for r in ext]
    axes[2].plot(ep_e, std_e, color=COLORS['L1'], lw=1.5)
    axes[2].axhline(0.114, color=COLORS['baseline'], ls=':', lw=1, label='Converged σ ~0.114')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('TBI Std')
    axes[2].set_title('(c) TBI Std Contraction')
    axes[2].legend(fontsize=7)

    fig.suptitle('Figure 2: L1 TBI Dynamics', fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig2_l1_tbi_dynamics.pdf')
    fig.savefig(FIGS / 'fig2_l1_tbi_dynamics.png')
    plt.close(fig)
    print('  Fig 2: L1 TBI dynamics saved')


# -----------------------------------------------------------------------
# Figure 3: Amplitude attenuation correction
# -----------------------------------------------------------------------

def fig3_amplitude_fix():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # (a) Field amplitudes before/after
    layers = ['L1', 'L2', 'L3']
    before = [0.0993, 0.0013, 0.0001]
    after = [0.0993, 0.0126, 0.0075]

    x = np.arange(len(layers))
    w = 0.35
    axes[0].bar(x - w/2, before, w, label='drive_gain=0.1', color=COLORS['baseline'], alpha=0.7)
    axes[0].bar(x + w/2, after, w, label='drive_gain=1.0 (L2/L3)', color=COLORS['accent'], alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layers)
    axes[0].set_ylabel('Mean Field Amplitude')
    axes[0].set_yscale('log')
    axes[0].set_title('(a) Field Amplitudes')
    axes[0].legend(fontsize=8)
    # Annotate ratios
    for i, (b, a) in enumerate(zip(before, after)):
        if a > b:
            axes[0].annotate(f'{a/b:.0f}x', xy=(i + w/2, a), xytext=(0, 5),
                           textcoords='offset points', ha='center', fontsize=8, color=COLORS['accent'])

    # (b) L2 TBI before/after
    conditions = ['Before\n(drive=0.1)', 'After\n(drive=1.0)']
    tbi_vals = [0.2331, 0.3492]
    bars = axes[1].bar(conditions, tbi_vals,
                       color=[COLORS['baseline'], COLORS['L2']], alpha=0.7, width=0.5)
    axes[1].axhline(0.524, color=COLORS['L1'], ls='--', lw=1, label='L1 TBI (frozen)')
    axes[1].set_ylabel('L2 TBI Mean')
    axes[1].set_title('(b) L2 TBI Shift (No Retraining)')
    axes[1].set_ylim(0, 0.6)
    axes[1].legend(fontsize=8)
    # Annotate delta
    axes[1].annotate(f'+{tbi_vals[1]-tbi_vals[0]:.3f}\n(amplitude\nartifact)',
                    xy=(1, tbi_vals[1]), xytext=(1.3, 0.35),
                    fontsize=8, color=COLORS['L2'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['L2']))

    fig.suptitle('Figure 3: Amplitude Attenuation Correction', fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig3_amplitude_fix.pdf')
    fig.savefig(FIGS / 'fig3_amplitude_fix.png')
    plt.close(fig)
    print('  Fig 3: amplitude fix saved')


# -----------------------------------------------------------------------
# Figure 4: Classification/cascade dissociation
# -----------------------------------------------------------------------

def fig4_dissociation():
    # Post-fix data: healthy field amplitudes, TBI 0.22-0.59
    betas_post = [0.01, 0.05, 0.1, 0.3, 1.0]
    val_post, tbi_post = [], []
    for b in betas_post:
        rows = read_csv(LOGS / f'l3_beta_cal2_{b}.csv')
        last = rows[-1]
        val_post.append(float(last['L3_val_acc']) * 100)
        tbi_post.append(float(last['L3_TBI_mean']))

    # Pre-fix data: attenuated field amplitudes, TBI 0.024-0.034
    betas_pre = [0.01, 0.05, 0.1, 0.3, 1.0]
    val_pre, tbi_pre = [], []
    for b in betas_pre:
        rows = read_csv(LOGS / f'l3_beta_cal_{b}.csv')
        last = rows[-1]
        val_pre.append(float(last['L3_val_acc']) * 100)
        tbi_pre.append(float(last['L3_TBI_mean']))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot accuracy vs TBI directly (not vs beta) — this is the dissociation plot
    ax.scatter(tbi_post, val_post, s=80, color=COLORS['L3'], zorder=5, edgecolors='white',
               lw=1.5, label='Healthy field regime (drive=1.0)')
    ax.scatter(tbi_pre, val_pre, s=80, color=COLORS['accent'], marker='D', zorder=5,
               edgecolors='white', lw=1.5, label='Attenuated field regime (drive=0.1)')

    # Horizontal band showing the accuracy plateau
    all_accs = val_post + val_pre
    acc_mean = np.mean(all_accs)
    acc_std = np.std(all_accs)
    ax.axhspan(acc_mean - acc_std, acc_mean + acc_std, alpha=0.12, color=COLORS['L3'],
               label=f'Accuracy band: {acc_mean:.1f} +/- {acc_std:.1f}%')
    ax.axhline(acc_mean, color=COLORS['L3'], ls=':', lw=1, alpha=0.5)

    ax.set_xlabel('L3 TBI Mean')
    ax.set_ylabel('L3 Validation Accuracy (%)')
    ax.set_title('Figure 4: Classification / Cascade Dissociation', fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(0.015, 0.8)
    ax.set_ylim(30, 62)
    ax.legend(fontsize=8, loc='lower left', bbox_to_anchor=(0.0, 0.12))

    # Annotate the 25x TBI span — label ABOVE the arrow
    ax.annotate('', xy=(0.024, 52.5), xytext=(0.59, 52.5),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1.5))
    ax.text(0.10, 53.5, 'TBI spans 25x (0.024 to 0.590), accuracy varies < 2pp',
            ha='center', fontsize=9, fontstyle='italic', color='#555')

    # Chance level — now visible with y-axis extended to 30
    ax.axhline(33.3, color=COLORS['baseline'], ls='--', lw=1, alpha=0.7)
    ax.text(0.02, 34.5, 'Chance (3 classes)', fontsize=8, color=COLORS['baseline'])

    fig.tight_layout()
    fig.savefig(FIGS / 'fig4_dissociation.pdf')
    fig.savefig(FIGS / 'fig4_dissociation.png')
    plt.close(fig)
    print('  Fig 4: dissociation saved (with pre-fix TBI~0.03 data)')


# -----------------------------------------------------------------------
# Figure 5: Cascade convergence
# -----------------------------------------------------------------------

def fig5_cascade():
    cascade_20 = read_csv(LOGS / 'l3_cascade_retest.csv')
    cascade_50 = read_csv(LOGS / 'extended_l3_accuracy.csv')

    fig, ax = plt.subplots(figsize=(7, 4))

    # 20-epoch cascade
    ep20 = [int(r['epoch']) for r in cascade_20]
    l2_tbi_20 = [float(r['L2_TBI_mean']) for r in cascade_20]
    l3_tbi_20 = [float(r['L3_TBI_mean']) for r in cascade_20]
    ax.plot(ep20, l2_tbi_20, 'o-', color=COLORS['L2'], lw=2, ms=5, label='L2 TBI (20-ep cascade)')
    ax.plot(ep20, l3_tbi_20, 's-', color=COLORS['L3'], lw=1.5, ms=4, alpha=0.7, label='L3 TBI (20-ep cascade)')

    # 50-epoch cascade (offset x-axis for clarity)
    ep50 = [int(r['epoch']) for r in cascade_50]
    l2_tbi_50 = [float(r['L2_TBI_mean']) for r in cascade_50]
    ax.plot(ep50, l2_tbi_50, '--', color=COLORS['L2'], lw=1.5, alpha=0.5, label='L2 TBI (50-ep extended)')

    # L1 frozen baseline
    ax.axhline(0.524, color=COLORS['L1'], ls='--', lw=2, label='L1 TBI (frozen, 0.524)')

    # L2 pre-cascade baseline
    ax.axhline(0.349, color=COLORS['baseline'], ls=':', lw=1, label='L2 pre-cascade (0.349)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('TBI Mean')
    ax.set_title('Figure 5: Hierarchical Cascade Convergence', fontweight='bold')
    ax.legend(fontsize=8, loc='center right')
    ax.set_ylim(0.2, 0.6)

    # Annotate gap closure — arrow points to L2 ep19 endpoint
    ax.annotate(f'Gap closed: 92%\n(0.349 → 0.510)',
                xy=(19, 0.510), xytext=(6, 0.28),
                fontsize=9, fontweight='bold', color=COLORS['L2'],
                arrowprops=dict(arrowstyle='->', color=COLORS['L2'], lw=1.5))

    # Annotate L3 non-monotonic dip
    ax.annotate('L3 adapts under\ncascade feedback',
                xy=(7, 0.33), xytext=(1, 0.25),
                fontsize=7, fontstyle='italic', color=COLORS['L3'], alpha=0.8,
                arrowprops=dict(arrowstyle='->', color=COLORS['L3'], lw=1, alpha=0.5))

    fig.tight_layout()
    fig.savefig(FIGS / 'fig5_cascade.pdf')
    fig.savefig(FIGS / 'fig5_cascade.png')
    plt.close(fig)
    print('  Fig 5: cascade convergence saved')


# -----------------------------------------------------------------------
# Figure 6: Three-layer TBI bar chart
# -----------------------------------------------------------------------

def fig6_tbi_bars():
    # Load per-sample distributions from the cascade-endpoint measurement
    dist_file = FIGS / 'cascade_tbi_distributions.npz'
    if dist_file.exists():
        data = np.load(dist_file)
        l1_tbi = data['l1_tbi']
        l2_tbi = data['l2_tbi']
        l3_tbi = data['l3_tbi']
        has_distributions = True
    else:
        has_distributions = False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [3, 2]})

    if has_distributions:
        # (a) Box + strip plot showing actual distributions
        ax = axes[0]
        distributions = [l1_tbi, l2_tbi, l3_tbi]
        labels = ['L1\n(frozen)', 'L2\n(cascade)', 'L3\n(cascade)']
        colors_list = [COLORS['L1'], COLORS['L2'], COLORS['L3']]

        bp = ax.boxplot(distributions, labels=labels, patch_artist=True,
                        widths=0.5, showfliers=False,
                        medianprops=dict(color='black', lw=2),
                        whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2))
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Jittered strip of individual samples (subsample for clarity)
        rng = np.random.RandomState(42)
        for i, (dist, color) in enumerate(zip(distributions, colors_list)):
            n_show = min(200, len(dist))
            idx = rng.choice(len(dist), n_show, replace=False)
            jitter = rng.uniform(-0.15, 0.15, n_show)
            ax.scatter(np.full(n_show, i + 1) + jitter, dist[idx],
                       s=6, alpha=0.25, color=color, zorder=2)

        # Median labels
        for i, dist in enumerate(distributions):
            med = np.median(dist)
            ax.text(i + 1 + 0.32, med, f'med={med:.3f}', fontsize=8,
                    va='center', fontweight='bold')

        ax.set_ylabel('TBI (per sample)')
        ax.set_title('(a) Per-sample TBI distributions\n(cascade endpoint, single run)',
                     fontweight='bold', fontsize=10)
        ax.set_ylim(-0.05, 1.05)

        # (b) Zero-coherence fraction comparison
        ax2 = axes[1]
        conditions = ['L1', 'L2\npre-cascade', 'L2\npost-cascade', 'L3']
        fracs = [
            (l1_tbi < 0.05).mean() * 100,
            np.load(dist_file)['l2_pre_tbi'].__len__() and
            (np.load(dist_file)['l2_pre_tbi'] < 0.05).mean() * 100,
            (l2_tbi < 0.05).mean() * 100,
            (l3_tbi < 0.05).mean() * 100,
        ]
        bar_colors = [COLORS['L1'], COLORS['baseline'], COLORS['L2'], COLORS['L3']]
        bars = ax2.bar(conditions, fracs, color=bar_colors, alpha=0.7, width=0.6)
        ax2.set_ylabel('% samples with TBI < 0.05')
        ax2.set_title('(b) Zero-coherence mode\n(bimodality diagnostic)',
                      fontweight='bold', fontsize=10)
        ax2.set_ylim(0, 40)

        # Value labels
        for bar, val in zip(bars, fracs):
            if val > 0.5:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
                         fontweight='bold')

        # Annotate the key finding
        ax2.annotate('Unchanged\nby cascade',
                    xy=(2.0, fracs[2]), xytext=(2.8, 35),
                    fontsize=8, fontstyle='italic', color='#555',
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1))

    else:
        # Fallback: simple bar chart with median + IQR
        ax = axes[0]
        layers = ['L1\n(frozen)', 'L2\n(cascade)', 'L3\n(cascade)']
        medians = [0.472, 0.566, 0.332]
        q25s = [0.436, 0.000, 0.274]
        q75s = [0.619, 0.609, 0.388]
        colors_list = [COLORS['L1'], COLORS['L2'], COLORS['L3']]
        x = np.arange(len(layers))
        ax.bar(x, medians, color=colors_list, alpha=0.8, width=0.5)
        for i in range(len(layers)):
            ax.vlines(i, q25s[i], q75s[i], color='black', lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_ylabel('TBI (median, IQR)')
        ax.set_title('Figure 6: Three-Layer TBI at Cascade Convergence',
                     fontweight='bold')
        axes[1].axis('off')

    fig.suptitle('Figure 6: Three-Layer TBI at Cascade Convergence',
                 fontweight='bold', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig6_tbi_bars.pdf')
    fig.savefig(FIGS / 'fig6_tbi_bars.png')
    plt.close(fig)
    print('  Fig 6: TBI distributions saved')


# -----------------------------------------------------------------------

def main():
    print('Generating paper figures...')
    fig1_architecture()
    fig2_l1_tbi_dynamics()
    fig3_amplitude_fix()
    fig4_dissociation()
    fig5_cascade()
    fig6_tbi_bars()
    print(f'\nAll figures saved to {FIGS}/')
    for f in sorted(FIGS.glob('fig*.pdf')):
        print(f'  {f.name}')


if __name__ == '__main__':
    main()
