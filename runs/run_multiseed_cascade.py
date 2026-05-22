"""Run the 20-epoch cascade experiment with 5 seeds and aggregate.

Single-seed protocol (matching the manuscript):
    epochs=20, batch_size=32, beta_nudge=0.05, lambda_max=5.0,
    petu_coh_floor=0.35, drive_gain=1.0 at L2/L3,
    L1 frozen from checkpoints/l1_phoneme.pt,
    L2 frozen from checkpoints/l2_phase2_extended.pt,
    L3 fresh, --no_gate, PETU active.

Each seed N produces:
    runs/cascade_seed{N}/cascade.csv             - per-epoch metrics
    runs/cascade_seed{N}/cascade.pt              - final checkpoint
    runs/cascade_seed{N}/per_sample_tbi.npz      - per-sample TBI arrays
    runs/cascade_seed{N}/distribution_stats.json - distribution summary
    runs/cascade_seed{N}/stdout.log              - full stdout

After all seeds complete, this script writes:
    runs/cascade_multiseed_summary.md
    runs/figures/fig5_multiseed.png
    runs/figures/fig6_multiseed.png

Usage (inside the GPU container):
    python -u -m runs.run_multiseed_cascade
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))

RUNS = _ROOT / 'runs'
FIGS = RUNS / 'figures'
SEEDS = [0, 1, 2, 3, 4]

L1_CKPT = 'checkpoints/l1_phoneme.pt'
L2_CKPT = 'checkpoints/l2_phase2_extended.pt'
TIMIT = '/data/timit/TIMIT'


# ----------------------------------------------------------------------------
# Per-seed training run
# ----------------------------------------------------------------------------

def run_seed(seed: int) -> Path:
    seed_dir = RUNS / f'cascade_seed{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    csv_name = 'cascade.csv'
    ckpt_name = 'cascade.pt'
    stdout_log = seed_dir / 'stdout.log'

    if (seed_dir / csv_name).exists() and (seed_dir / ckpt_name).exists():
        # Idempotent: skip already-completed seeds.
        n_rows = sum(1 for _ in open(seed_dir / csv_name)) - 1
        if n_rows >= 20:
            print(f"  seed {seed}: already complete ({n_rows} epochs), skipping",
                  flush=True)
            return seed_dir

    cmd = [
        'python', '-u', '-m', 'training.train_l3',
        '--timit_root', TIMIT,
        '--epochs', '20',
        '--batch_size', '32',
        '--beta_nudge', '0.05',
        '--lambda_max', '5.0',
        '--petu_coh_floor', '0.35',
        '--l1_checkpoint', L1_CKPT,
        '--l2_checkpoint', L2_CKPT,
        '--num_workers', '4',
        '--log_dir', str(seed_dir),
        '--log_name', csv_name,
        '--checkpoint_dir', str(seed_dir),
        '--checkpoint_name', ckpt_name,
        '--seed', str(seed),
    ]
    print(f"\n=== Seed {seed} ===", flush=True)
    print(f"  Command: {' '.join(cmd)}", flush=True)

    with open(stdout_log, 'w') as fout:
        result = subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Seed {seed} training failed; see {stdout_log}")

    return seed_dir


# ----------------------------------------------------------------------------
# Per-seed distribution measurement
# ----------------------------------------------------------------------------

def measure_distributions(seed_dir: Path) -> dict:
    """Run the cascade forward at the final checkpoint and save per-sample TBI
    distributions for L1 (frozen, phonemes), L2 pre-cascade (words, λ=0),
    L2 post-cascade (words, full cascade with λ=5), and L3 (sentences)."""
    out_npz = seed_dir / 'per_sample_tbi.npz'
    out_json = seed_dir / 'distribution_stats.json'
    if out_npz.exists() and out_json.exists():
        with open(out_json) as f:
            return json.load(f)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from ficu_crfm.architecture.ficu_l1 import FICUL1Phoneme
    from ficu_crfm.architecture.ficu_l2 import FICUL2Word
    from ficu_crfm.architecture.ficu_l3 import FICUL3Sentence
    from ficu_crfm.architecture.predictive_field import PredictiveField
    from ficu_crfm.dataset.timit_loader import (
        build_dataset, collate_phonemes, collate_words,
        collate_sentences, N_PHONEMES, N_SENTENCE_TYPES,
    )
    from ficu_crfm.metrics.tbi import compute_TBI

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def word_to_l1(w, t=6400):
        B, L = w.shape
        if L >= t:
            s = (L - t) // 2
            return w[:, s:s + t]
        return F.pad(w, ((t - L) // 2, t - L - (t - L) // 2))

    def sent_to_l1(w, t=32000):
        B, L = w.shape
        if L >= t:
            s = (L - t) // 2
            return w[:, s:s + t]
        return F.pad(w, ((t - L) // 2, t - L - (t - L) // 2))

    # Models
    l1 = FICUL1Phoneme(n_classes=N_PHONEMES, n_settle_steps=24).to(device)
    l1.load_state_dict(torch.load(_ROOT / L1_CKPT, map_location=device))
    l1.eval()
    for p in l1.parameters():
        p.requires_grad = False

    cascade_ckpt = torch.load(seed_dir / 'cascade.pt', map_location=device)
    l2_state = cascade_ckpt.get('l2', None) or torch.load(_ROOT / L2_CKPT, map_location=device)
    n_l2 = l2_state['readout.W'].shape[0]
    l2 = FICUL2Word(n_classes=n_l2, n_settle_steps=24).to(device)
    l2.load_state_dict(l2_state, strict=False)
    l2.eval()

    l3 = FICUL3Sentence(n_classes=N_SENTENCE_TYPES, n_settle_steps=24, beta=0.05).to(device)
    l3.load_state_dict(cascade_ckpt.get('l3', cascade_ckpt), strict=False)
    l3.eval()

    pred = PredictiveField(
        l3_shape=(3, 24, 16), l2_shape=(3, 47, 32),
        lr=0.01, lambda_td=5.0, handshake_threshold=1.0,
    ).to(device)
    if 'pred' in cascade_ckpt:
        pred.load_state_dict(cascade_ckpt['pred'], strict=False)

    # Cascade λ at endpoint
    with torch.no_grad():
        l2.lambda_td_L2_L1.fill_(5.0)
    pred.lambda_td = 5.0

    # ---- L1 baseline (phonemes) ----
    ph_ds = build_dataset('phoneme', 'test', TIMIT)
    ph_loader = DataLoader(ph_ds, batch_size=64, collate_fn=collate_phonemes)
    l1_tbis = []
    with torch.no_grad():
        for i, batch in enumerate(ph_loader):
            if i >= 20:
                break
            wave, label = batch['waveform'].to(device), batch['label'].to(device)
            zfr, zfi = l1.settle(wave)
            znr, zni = l1.settle(wave, target_class=label)
            l1_tbis.append(compute_TBI(zfr, zfi, znr, zni).cpu())
    l1_tbi = torch.cat(l1_tbis).numpy()

    # ---- L2 pre-cascade (words, λ=0) ----
    wd_ds = build_dataset('word', 'test', TIMIT)
    wd_loader = DataLoader(wd_ds, batch_size=64, collate_fn=collate_words)
    l2_pre = []
    with torch.no_grad():
        for i, batch in enumerate(wd_loader):
            if i >= 20:
                break
            wave = word_to_l1(batch['waveform'].to(device), 6400)
            label = batch['label'].to(device)
            z1r, z1i = l1.settle(wave)
            z2fr, z2fi = l2.settle(z1r, z1i)
            z2nr, z2ni = l2.settle(z1r, z1i, target_class=label)
            l2_pre.append(compute_TBI(z2fr, z2fi, z2nr, z2ni).cpu())
    l2_pre_tbi = torch.cat(l2_pre).numpy()

    # ---- L2 post-cascade (words, full cascade with λ=5) ----
    l2_post = []
    with torch.no_grad():
        for i, batch in enumerate(wd_loader):
            if i >= 20:
                break
            wave = word_to_l1(batch['waveform'].to(device), 6400)
            label = batch['label'].to(device)
            z1r, z1i = l1.settle(wave)
            z2r_init, z2i_init = l2.settle(z1r, z1i)
            z3r, z3i = l3.settle(z2r_init, z2i_init)
            pred_l2_r, pred_l2_i = pred.predict(z3r, z3i)
            bias_l1_r, bias_l1_i = l2.l2_to_l1_topdown_bias(z2r_init, z2i_init)
            z1r_td, z1i_td = l1.settle(wave, topdown_bias=(bias_l1_r, bias_l1_i))
            z2fr, z2fi = l2.settle(z1r_td, z1i_td,
                                    predicted_init=(pred_l2_r, pred_l2_i))
            z2nr, z2ni = l2.settle(z1r_td, z1i_td, target_class=label,
                                    predicted_init=(pred_l2_r, pred_l2_i))
            l2_post.append(compute_TBI(z2fr, z2fi, z2nr, z2ni).cpu())
    l2_tbi = torch.cat(l2_post).numpy()

    # ---- L3 (sentences) ----
    st_ds = build_dataset('sentence', 'test', TIMIT)
    st_loader = DataLoader(st_ds, batch_size=32, collate_fn=collate_sentences)
    l3_tbis = []
    with torch.no_grad():
        for i, batch in enumerate(st_loader):
            if i >= 20:
                break
            wave = sent_to_l1(batch['waveform'].to(device), 32000)
            label = batch['label'].to(device)
            z1r, z1i = l1.settle(wave)
            z2r, z2i = l2.settle(z1r, z1i)
            z3fr, z3fi = l3.settle(z2r, z2i)
            z3nr, z3ni = l3.settle(z2r, z2i, target_class=label)
            l3_tbis.append(compute_TBI(z3fr, z3fi, z3nr, z3ni).cpu())
    l3_tbi = torch.cat(l3_tbis).numpy()

    # Save raw arrays
    np.savez(out_npz, l1_tbi=l1_tbi, l2_pre_tbi=l2_pre_tbi,
             l2_tbi=l2_tbi, l3_tbi=l3_tbi)

    # Summary stats per layer
    def stats(arr, name):
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        return {
            'name': name, 'n': int(len(arr)),
            'mean': float(arr.mean()), 'std': float(arr.std()),
            'median': float(q50), 'q25': float(q25), 'q75': float(q75),
            'iqr': float(q75 - q25),
            'skewness': float(((arr - arr.mean())**3).mean()
                              / (arr.std()**3 + 1e-12)),
            'frac_tbi_lt_005': float((arr < 0.05).mean()),
        }

    out = {
        'l1': stats(l1_tbi, 'L1 (frozen, phonemes)'),
        'l2_pre': stats(l2_pre_tbi, 'L2 pre-cascade (words)'),
        'l2_post': stats(l2_tbi, 'L2 post-cascade (words)'),
        'l3': stats(l3_tbi, 'L3 (cascade, sentences)'),
    }
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  L2_pre  frac<0.05={out['l2_pre']['frac_tbi_lt_005']*100:.1f}% "
          f"median={out['l2_pre']['median']:.4f}", flush=True)
    print(f"  L2_post frac<0.05={out['l2_post']['frac_tbi_lt_005']*100:.1f}% "
          f"median={out['l2_post']['median']:.4f}", flush=True)
    return out


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def aggregate_and_report(seed_dirs):
    """Compute mean ± SD across seeds for all key quantities and emit the
    summary report + figures."""
    FIGS.mkdir(parents=True, exist_ok=True)

    # ---- Load per-epoch CSVs and per-sample distributions ----
    epoch_data = {}   # seed -> list of dicts per epoch
    dist_stats = {}   # seed -> distribution_stats.json
    arrays = {}       # seed -> {layer: np.ndarray of per-sample TBI}
    for s in SEEDS:
        sd = RUNS / f'cascade_seed{s}'
        # CSV
        rows = []
        with open(sd / 'cascade.csv') as f:
            header = f.readline().strip().split(',')
            for line in f:
                vals = line.strip().split(',')
                rows.append(dict(zip(header, vals)))
        epoch_data[s] = rows
        with open(sd / 'distribution_stats.json') as f:
            dist_stats[s] = json.load(f)
        with np.load(sd / 'per_sample_tbi.npz') as d:
            arrays[s] = {k: d[k] for k in d.files}

    # ---- Final-epoch aggregates ----
    def final_field(seed, field):
        return float(epoch_data[seed][-1][field])

    def initial_field(seed, field):
        return float(epoch_data[seed][0][field])

    L1_final = np.array([final_field(s, 'L1_TBI_mean') for s in SEEDS])
    L1_init  = np.array([initial_field(s, 'L1_TBI_mean') for s in SEEDS])
    L2_final = np.array([final_field(s, 'L2_TBI_mean') for s in SEEDS])
    L2_init  = np.array([initial_field(s, 'L2_TBI_mean') for s in SEEDS])
    L3_final = np.array([final_field(s, 'L3_TBI_mean') for s in SEEDS])
    L3_init  = np.array([initial_field(s, 'L3_TBI_mean') for s in SEEDS])
    L2_val_final = np.array([final_field(s, 'L2_val_acc') for s in SEEDS])
    L3_val_final = np.array([final_field(s, 'L3_val_acc') for s in SEEDS])
    upd_final   = np.array([final_field(s, 'update_fraction') for s in SEEDS])

    # Distribution stats arrays
    def dist_array(key, stat):
        return np.array([dist_stats[s][key][stat] for s in SEEDS])

    l1_median = dist_array('l1', 'median')
    l1_iqr    = dist_array('l1', 'iqr')
    l1_skew   = dist_array('l1', 'skewness')

    l2_pre_frac = dist_array('l2_pre', 'frac_tbi_lt_005')
    l2_post_frac = dist_array('l2_post', 'frac_tbi_lt_005')
    l2_post_median = dist_array('l2_post', 'median')
    l2_post_iqr    = dist_array('l2_post', 'iqr')
    l2_post_skew   = dist_array('l2_post', 'skewness')
    l2_pre_median  = dist_array('l2_pre', 'median')

    l3_median = dist_array('l3', 'median')
    l3_iqr    = dist_array('l3', 'iqr')
    l3_skew   = dist_array('l3', 'skewness')
    l3_frac   = dist_array('l3', 'frac_tbi_lt_005')

    def ms(arr):
        return f"{arr.mean():.4f} ± {arr.std():.4f}"

    def msmm(arr):
        return f"{arr.mean():.4f} ± {arr.std():.4f}  ({arr.min():.4f}–{arr.max():.4f})"

    # ---- Per-epoch trajectory across seeds ----
    n_ep = len(epoch_data[SEEDS[0]])
    epochs = np.arange(n_ep)
    l2_traj = np.zeros((len(SEEDS), n_ep))
    l3_traj = np.zeros((len(SEEDS), n_ep))
    l1_traj = np.zeros((len(SEEDS), n_ep))
    for i, s in enumerate(SEEDS):
        for j in range(n_ep):
            l1_traj[i, j] = float(epoch_data[s][j]['L1_TBI_mean'])
            l2_traj[i, j] = float(epoch_data[s][j]['L2_TBI_mean'])
            l3_traj[i, j] = float(epoch_data[s][j]['L3_TBI_mean'])

    # ---- Figure: multi-seed L2 trajectory ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'legend.fontsize': 9, 'figure.dpi': 300,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    COL = {'L1': '#2196F3', 'L2': '#FF9800', 'L3': '#4CAF50', 'gray': '#9E9E9E'}

    fig, ax = plt.subplots(figsize=(7, 4))
    # L1 frozen reference
    l1_mean = l1_traj.mean(axis=0).mean()
    ax.axhline(l1_mean, color=COL['L1'], ls='--', lw=2,
               label=f'L1 frozen ({l1_mean:.3f})')
    # Individual L2 seed traces (light)
    for i, s in enumerate(SEEDS):
        ax.plot(epochs, l2_traj[i], color=COL['L2'], alpha=0.25, lw=1)
    # L2 mean ± SD band
    l2_mean = l2_traj.mean(axis=0)
    l2_sd   = l2_traj.std(axis=0)
    ax.plot(epochs, l2_mean, color=COL['L2'], lw=2.5,
            label=f'L2 mean across {len(SEEDS)} seeds')
    ax.fill_between(epochs, l2_mean - l2_sd, l2_mean + l2_sd,
                    color=COL['L2'], alpha=0.20, label='L2 ±1 SD')
    # L3 mean trajectory (lighter)
    l3_mean = l3_traj.mean(axis=0)
    ax.plot(epochs, l3_mean, color=COL['L3'], lw=1.5,
            alpha=0.7, label=f'L3 mean across {len(SEEDS)} seeds')

    ax.axhline(L2_init.mean(), color=COL['gray'], ls=':', lw=1,
               label=f'L2 pre-cascade ({L2_init.mean():.3f})')

    gap_closed_pct = (L2_final - L2_init) / (L1_final - L2_init) * 100.0
    ax.set_xlabel('Epoch')
    ax.set_ylabel('TBI Mean')
    ax.set_title(
        f'Multi-Seed Cascade Convergence (n={len(SEEDS)} seeds)\n'
        f'L2 TBI {L2_init.mean():.3f} → {L2_final.mean():.3f}, '
        f'gap closure {gap_closed_pct.mean():.0f}% ± {gap_closed_pct.std():.0f}%',
        fontweight='bold',
    )
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(0.2, 0.6)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig5_multiseed.pdf')
    fig.savefig(FIGS / 'fig5_multiseed.png')
    plt.close(fig)

    # ---- Figure: pooled box plots of final-epoch per-sample TBI ----
    l1_pool = np.concatenate([arrays[s]['l1_tbi'] for s in SEEDS])
    l2_pool = np.concatenate([arrays[s]['l2_tbi'] for s in SEEDS])
    l3_pool = np.concatenate([arrays[s]['l3_tbi'] for s in SEEDS])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                             gridspec_kw={'width_ratios': [3, 2]})

    ax = axes[0]
    distributions = [l1_pool, l2_pool, l3_pool]
    labels = ['L1\n(frozen)', 'L2\n(cascade)', 'L3\n(cascade)']
    color_list = [COL['L1'], COL['L2'], COL['L3']]

    bp = ax.boxplot(distributions, labels=labels, patch_artist=True,
                    widths=0.5, showfliers=False,
                    medianprops=dict(color='black', lw=2),
                    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
    for patch, c in zip(bp['boxes'], color_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    rng = np.random.RandomState(42)
    for i, (dist, c) in enumerate(zip(distributions, color_list)):
        n_show = min(300, len(dist))
        idx = rng.choice(len(dist), n_show, replace=False)
        jitter = rng.uniform(-0.15, 0.15, n_show)
        ax.scatter(np.full(n_show, i + 1) + jitter, dist[idx],
                   s=4, alpha=0.20, color=c, zorder=2)

    for i, dist in enumerate(distributions):
        med = np.median(dist)
        ax.text(i + 1 + 0.32, med, f'med={med:.3f}', fontsize=8,
                va='center', fontweight='bold')

    ax.set_ylabel('TBI (per sample, pooled across seeds)')
    ax.set_title(
        f'(a) Per-sample TBI distributions\n'
        f'({len(SEEDS)} seeds × {len(l1_pool)//len(SEEDS)} samples each)',
        fontweight='bold', fontsize=10,
    )
    ax.set_ylim(-0.05, 1.05)

    ax2 = axes[1]
    conds = ['L1', 'L2\npre', 'L2\npost', 'L3']
    fracs_mean = np.array([
        (l1_pool < 0.05).mean() * 100,
        l2_pre_frac.mean() * 100,
        l2_post_frac.mean() * 100,
        (l3_pool < 0.05).mean() * 100,
    ])
    fracs_sd = np.array([
        0.0,  # L1 frozen, frac is bit-identical across seeds — use within-seed SD
        l2_pre_frac.std() * 100,
        l2_post_frac.std() * 100,
        l3_frac.std() * 100,
    ])
    bar_colors = [COL['L1'], COL['gray'], COL['L2'], COL['L3']]
    bars = ax2.bar(conds, fracs_mean, color=bar_colors, alpha=0.7, width=0.6,
                   yerr=fracs_sd, error_kw=dict(lw=1.2, capsize=4))
    ax2.set_ylabel('% samples with TBI < 0.05')
    ax2.set_title(
        f'(b) Zero-coherence mode\n(mean ± SD across {len(SEEDS)} seeds)',
        fontweight='bold', fontsize=10,
    )
    ax2.set_ylim(0, max(fracs_mean.max() + 5, 40))
    for bar, m, s in zip(bars, fracs_mean, fracs_sd):
        if m > 0.5:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + s + 0.6,
                     f'{m:.1f}%', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

    fig.suptitle(
        f'Three-Layer TBI at Cascade Convergence — {len(SEEDS)} seeds',
        fontweight='bold', fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGS / 'fig6_multiseed.pdf')
    fig.savefig(FIGS / 'fig6_multiseed.png')
    plt.close(fig)

    # ---- Summary markdown ----
    delta_L2 = L2_final - L2_init
    gap_at_start = L1_final - L2_init
    pct_closed = delta_L2 / gap_at_start * 100.0

    lines = [
        '# Multi-Seed Cascade Replication Summary',
        '',
        f'**Protocol**: 20-epoch L3 cascade, β_nudge=0.05, λ_max=5.0, '
        f'drive_gain=1.0 at L2/L3, --no_gate, PETU active. L1 frozen '
        f'(`{L1_CKPT}`), L2 frozen (`{L2_CKPT}`), L3 fresh per seed.',
        '',
        f'**Seeds**: {SEEDS}  •  **Only RNG varies** (PredictiveField W init, '
        f'batch shuffling, PETU sample masking).',
        '',
        '---',
        '',
        '## Table 1 — Final-epoch TBI (mean ± SD across seeds, min–max)',
        '',
        '| Layer | TBI mean ± SD (min–max) |',
        '|---|---|',
        f'| L1 (frozen, sanity) | {msmm(L1_final)} |',
        f'| L2 (post-cascade)   | {msmm(L2_final)} |',
        f'| L3 (post-cascade)   | {msmm(L3_final)} |',
        '',
        '*Single-seed manuscript value: L1=0.524, L2=0.510, L3=0.492.*',
        '',
        '## Table 2 — L2 cascade trajectory across seeds',
        '',
        '| Quantity | mean ± SD | min – max |',
        '|---|---|---|',
        f'| L2 TBI initial (epoch 0)   | {ms(L2_init)} | {L2_init.min():.4f} – {L2_init.max():.4f} |',
        f'| L2 TBI final (epoch 19)    | {ms(L2_final)} | {L2_final.min():.4f} – {L2_final.max():.4f} |',
        f'| Δ L2 TBI                   | {ms(delta_L2)} | {delta_L2.min():.4f} – {delta_L2.max():.4f} |',
        f'| Gap closure (% of L1−L2)   | {pct_closed.mean():.1f}% ± {pct_closed.std():.1f}% | {pct_closed.min():.1f}% – {pct_closed.max():.1f}% |',
        '',
        '*Single-seed manuscript value: Δ = +0.161 nats, 92% gap closure.*',
        '',
        '## Table 3 — Final-epoch per-layer distribution (mean ± SD across seeds)',
        '',
        '| Layer | median | IQR | skewness | % TBI < 0.05 |',
        '|---|---|---|---|---|',
        f'| L1 (frozen)         | {ms(l1_median)} | {ms(l1_iqr)} | {ms(l1_skew)} | {(l1_pool < 0.05).mean()*100:.1f}% (pooled) |',
        f'| L2 pre-cascade      | {ms(l2_pre_median)} | — | — | {l2_pre_frac.mean()*100:.1f}% ± {l2_pre_frac.std()*100:.1f}% |',
        f'| L2 post-cascade     | {ms(l2_post_median)} | {ms(l2_post_iqr)} | {ms(l2_post_skew)} | {l2_post_frac.mean()*100:.1f}% ± {l2_post_frac.std()*100:.1f}% |',
        f'| L3 (cascade)        | {ms(l3_median)} | {ms(l3_iqr)} | {ms(l3_skew)} | {l3_frac.mean()*100:.1f}% ± {l3_frac.std()*100:.1f}% |',
        '',
        '*Single-seed manuscript values: L2 zero-coherence fraction 32.1% pre and 32.1% post.*',
        '',
        '## Table 4 — Per-seed final numbers',
        '',
        '| seed | L1_TBI | L2_TBI | L3_TBI | ΔL2 | gap closed | L2 zero-frac post | L3_val_acc | upd_frac |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for i, s in enumerate(SEEDS):
        lines.append(
            f'| {s} | {L1_final[i]:.4f} | {L2_final[i]:.4f} | {L3_final[i]:.4f} | '
            f'{delta_L2[i]:+.4f} | {pct_closed[i]:.1f}% | '
            f'{l2_post_frac[i]*100:.1f}% | {L3_val_final[i]*100:.1f}% | '
            f'{upd_final[i]:.3f} |'
        )

    lines += [
        '',
        '## Figures',
        '',
        '- `runs/figures/fig5_multiseed.png` — L2 TBI trajectory, mean ± SD over '
        f'{len(SEEDS)} seeds, with individual seed traces overlaid.',
        '- `runs/figures/fig6_multiseed.png` — three-layer per-sample TBI box plots '
        f'(pooled across {len(SEEDS)} seeds) plus zero-coherence fraction with '
        f'cross-seed error bars.',
        '',
        '## Flags',
    ]

    # Flag any seed whose L2 cascade behavior diverges from the rest.
    flagged = []
    for i, s in enumerate(SEEDS):
        if delta_L2[i] < 0.05:
            flagged.append(
                f'- Seed {s}: L2 TBI failed to rise meaningfully '
                f'(ΔL2={delta_L2[i]:+.4f} nats, gap_closed={pct_closed[i]:.1f}%).'
            )
        if pct_closed[i] < 50:
            flagged.append(
                f'- Seed {s}: cascade closed <50% of the gap '
                f'({pct_closed[i]:.1f}%).'
            )
    if not flagged:
        lines.append('- No anomalous seeds. All 5 seeds replicate the cascade '
                     'within expected variance.')
    else:
        lines += flagged

    summary_path = RUNS / 'cascade_multiseed_summary.md'
    summary_path.write_text('\n'.join(lines) + '\n')
    print(f"\nWrote {summary_path}", flush=True)
    print(f"Wrote {FIGS / 'fig5_multiseed.png'}", flush=True)
    print(f"Wrote {FIGS / 'fig6_multiseed.png'}", flush=True)
    return summary_path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    RUNS.mkdir(parents=True, exist_ok=True)

    for s in SEEDS:
        sd = run_seed(s)
        try:
            measure_distributions(sd)
        except Exception as e:
            print(f"  WARN: distribution measurement for seed {s} failed: {e}",
                  flush=True)

    seed_dirs = [RUNS / f'cascade_seed{s}' for s in SEEDS]
    summary_path = aggregate_and_report(seed_dirs)

    # Print summary to stdout
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(summary_path.read_text())


if __name__ == '__main__':
    main()
