"""Generate the four signature figures from the training CSVs.

Fig 1: TBI cascade across L1/L2/L3
Fig 2: Update fraction trajectory
Fig 3: Handshake success rate (L3)
Fig 4: Top-down benefit (L2 word accuracy with vs without top-down)

Top-down benefit requires a separate ablation eval and isn't computed by
the standard training scripts; the figure here will fall back to a
placeholder if the comparison file isn't present.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows.append(r)
    return rows


def fig_tbi_cascade(l1_rows, l2_rows, l3_rows, out):
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for ax, rows, name in zip(axes,
                              [l1_rows, l2_rows, l3_rows],
                              ['L1 (phoneme)', 'L2 (word)', 'L3 (sentence)']):
        if not rows:
            ax.text(0.5, 0.5, f'no data: {name}', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_ylabel(f'{name} TBI')
            continue
        epochs = [r['epoch'] for r in rows]
        tbi = [r['TBI_mean'] for r in rows]
        ax.plot(epochs, tbi, lw=2)
        ax.set_ylabel(f'{name} TBI')
        ax.set_ylim(0, 1)
        for r in rows:
            if r.get('physics_frozen', 0):
                ax.axvline(r['epoch'], ls='--', alpha=0.5, color='red')
                break
    axes[-1].set_xlabel('epoch')
    fig.suptitle('TBI cascade across the FICU hierarchy')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_update_fraction(l1_rows, l2_rows, l3_rows, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for rows, name, color in [(l1_rows, 'L1', 'C0'),
                              (l2_rows, 'L2', 'C1'),
                              (l3_rows, 'L3', 'C2')]:
        if not rows:
            continue
        ax.plot([r['epoch'] for r in rows],
                [r['update_fraction'] for r in rows],
                label=name, color=color, lw=2)
    ax.set_xlabel('epoch')
    ax.set_ylabel('PETU update fraction')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title('Fraction of samples triggering physics updates')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_handshake(l3_rows, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    if l3_rows:
        ax.plot([r['epoch'] for r in l3_rows],
                [r['handshake_success_rate'] for r in l3_rows],
                lw=2, color='C3')
    ax.set_xlabel('epoch')
    ax.set_ylabel('handshake success rate')
    ax.set_ylim(0, 1.05)
    ax.set_title('L3 prediction handshake success rate')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_topdown_benefit(ablation_path, out):
    fig, ax = plt.subplots(figsize=(5, 4))
    rows = read_csv(ablation_path)
    if not rows:
        ax.text(0.5, 0.5, f'no ablation file: {ablation_path}',
                ha='center', va='center', transform=ax.transAxes)
    else:
        names = [r['condition'] for r in rows]
        accs = [r['accuracy'] for r in rows]
        ax.bar(names, accs, color=['C7', 'C2'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('L2 word accuracy')
        ax.set_title('Top-down benefit')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    l1 = read_csv(os.path.join(args.log_dir, 'l1_training.csv'))
    l2 = read_csv(os.path.join(args.log_dir, 'l2_training.csv'))
    l3 = read_csv(os.path.join(args.log_dir, 'l3_training.csv'))

    fig_tbi_cascade(l1, l2, l3, os.path.join(args.out_dir, 'fig1_tbi_cascade.png'))
    fig_update_fraction(l1, l2, l3, os.path.join(args.out_dir, 'fig2_update_fraction.png'))
    fig_handshake(l3, os.path.join(args.out_dir, 'fig3_handshake.png'))
    fig_topdown_benefit(os.path.join(args.log_dir, 'topdown_ablation.csv'),
                        os.path.join(args.out_dir, 'fig4_topdown_benefit.png'))
    print(f"figures written to {args.out_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir', default='ficu_crfm/logs')
    p.add_argument('--out_dir', default='ficu_crfm/figures/out')
    main(p.parse_args())
