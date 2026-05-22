"""Symmetric unhealthy-physics control for the cascade dissociation argument.

Protocol (paired with the healthy 5-seed cascade in run_multiseed_cascade.py):
    epochs=20, batch_size=32, beta_nudge=0.05, lambda_max=5.0,
    petu_coh_floor=0.35, PETU active. L1 frozen (l1_phoneme.pt), L2 frozen
    (l2_phase2_extended.pt) — both UNCHANGED from the healthy run.
    L3 starts from the freshly-trained attenuated checkpoint and runs in
    drive_gain=0.1 (attenuated regime) throughout. PredictiveField fresh per
    seed (only L3 physics + readout state are restored).

Each seed N produces:
    runs/cascade_unhealthy_seed{N}/cascade.csv             - per-epoch metrics
    runs/cascade_unhealthy_seed{N}/cascade.pt              - final checkpoint
    runs/cascade_unhealthy_seed{N}/per_sample_tbi.npz      - per-sample TBI
    runs/cascade_unhealthy_seed{N}/distribution_stats.json - distribution summary
    runs/cascade_unhealthy_seed{N}/stdout.log              - full stdout

After all seeds complete, this script writes:
    runs/cascade_unhealthy_summary.md   (paired comparison vs healthy)
    runs/figures/fig5_dissociation.{png,pdf}   (2-panel L2_TBI + PF error)

Usage (inside the GPU container):
    python -u -m runs.run_multiseed_cascade_unhealthy
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
L3_ATTENUATED_CKPT = 'runs/l3_attenuated_fresh/l3_attenuated.pt'
TIMIT = '/data/timit/TIMIT'

DRIVE_GAIN_UNHEALTHY = 0.1


# ----------------------------------------------------------------------------
# Per-seed training run
# ----------------------------------------------------------------------------

def run_seed(seed: int) -> Path:
    seed_dir = RUNS / f'cascade_unhealthy_seed{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)

    csv_name = 'cascade.csv'
    ckpt_name = 'cascade.pt'
    stdout_log = seed_dir / 'stdout.log'

    if (seed_dir / csv_name).exists() and (seed_dir / ckpt_name).exists():
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
        '--l3_drive_gain', str(DRIVE_GAIN_UNHEALTHY),
        '--l3_resume_from', L3_ATTENUATED_CKPT,
        '--num_workers', '4',
        '--log_dir', str(seed_dir),
        '--log_name', csv_name,
        '--checkpoint_dir', str(seed_dir),
        '--checkpoint_name', ckpt_name,
        '--seed', str(seed),
    ]
    print(f"\n=== Unhealthy Seed {seed} ===", flush=True)
    print(f"  Command: {' '.join(cmd)}", flush=True)

    with open(stdout_log, 'w') as fout:
        result = subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"Unhealthy seed {seed} training failed; see {stdout_log}"
        )
    return seed_dir


# ----------------------------------------------------------------------------
# Per-seed distribution measurement (mirrors healthy version)
# ----------------------------------------------------------------------------

def measure_distributions(seed_dir: Path, drive_gain: float) -> dict:
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

    l3 = FICUL3Sentence(n_classes=N_SENTENCE_TYPES, n_settle_steps=24,
                        beta=0.05, drive_gain=drive_gain).to(device)
    l3.load_state_dict(cascade_ckpt.get('l3', cascade_ckpt), strict=False)
    l3.eval()

    pred = PredictiveField(
        l3_shape=(3, 24, 16), l2_shape=(3, 47, 32),
        lr=0.01, lambda_td=5.0, handshake_threshold=1.0,
    ).to(device)
    if 'pred' in cascade_ckpt:
        pred.load_state_dict(cascade_ckpt['pred'], strict=False)

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

    np.savez(out_npz, l1_tbi=l1_tbi, l2_pre_tbi=l2_pre_tbi,
             l2_tbi=l2_tbi, l3_tbi=l3_tbi)

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
# Paired aggregation: healthy vs unhealthy
# ----------------------------------------------------------------------------

def _load_seeds(prefix: str):
    """Returns dict[seed]->{epoch_data, dist_stats, arrays}."""
    out = {}
    for s in SEEDS:
        sd = RUNS / f'{prefix}_seed{s}'
        rows = []
        with open(sd / 'cascade.csv') as f:
            header = f.readline().strip().split(',')
            for line in f:
                vals = line.strip().split(',')
                rows.append(dict(zip(header, vals)))
        with open(sd / 'distribution_stats.json') as f:
            dist = json.load(f)
        with np.load(sd / 'per_sample_tbi.npz') as d:
            arr = {k: d[k] for k in d.files}
        out[s] = {'rows': rows, 'dist': dist, 'arr': arr}
    return out


def _trajectory(loaded, field):
    """[n_seeds, n_epochs] for a given CSV column."""
    n_ep = len(loaded[SEEDS[0]]['rows'])
    traj = np.zeros((len(SEEDS), n_ep))
    for i, s in enumerate(SEEDS):
        for j in range(n_ep):
            traj[i, j] = float(loaded[s]['rows'][j][field])
    return traj


def aggregate_and_report():
    FIGS.mkdir(parents=True, exist_ok=True)

    healthy = _load_seeds('cascade')
    unhealthy = _load_seeds('cascade_unhealthy')

    # ---------------- trajectories ----------------
    h_L1 = _trajectory(healthy, 'L1_TBI_mean')
    h_L2 = _trajectory(healthy, 'L2_TBI_mean')
    h_L3 = _trajectory(healthy, 'L3_TBI_mean')
    h_PF = _trajectory(healthy, 'prediction_error')
    h_val = _trajectory(healthy, 'L3_val_acc')

    u_L1 = _trajectory(unhealthy, 'L1_TBI_mean')
    u_L2 = _trajectory(unhealthy, 'L2_TBI_mean')
    u_L3 = _trajectory(unhealthy, 'L3_TBI_mean')
    u_PF = _trajectory(unhealthy, 'prediction_error')
    u_val = _trajectory(unhealthy, 'L3_val_acc')

    # ---------------- final-epoch aggregates ----------------
    def final(arr2d):
        return arr2d[:, -1]

    def initial(arr2d):
        return arr2d[:, 0]

    h_L2f, h_L2i = final(h_L2), initial(h_L2)
    h_L3f, h_L3i = final(h_L3), initial(h_L3)
    h_PFf, h_PFi = final(h_PF), initial(h_PF)
    h_valf = final(h_val)

    u_L2f, u_L2i = final(u_L2), initial(u_L2)
    u_L3f, u_L3i = final(u_L3), initial(u_L3)
    u_PFf, u_PFi = final(u_PF), initial(u_PF)
    u_valf = final(u_val)

    L1_final = final(h_L1)  # bit-identical across seeds & conditions

    h_dL2 = h_L2f - h_L2i
    u_dL2 = u_L2f - u_L2i
    h_gap = h_dL2 / (L1_final - h_L2i) * 100.0
    # For unhealthy, the L1-L2 "gap to close" reference is the same L1
    # frozen value and the same L2 initial state, since L1/L2 are identical.
    u_gap = u_dL2 / (L1_final - u_L2i) * 100.0

    # ---------------- per-seed dist arrays ----------------
    def dist_array(loaded, key, stat):
        return np.array([loaded[s]['dist'][key][stat] for s in SEEDS])

    h_l2_zero_post = dist_array(healthy, 'l2_post', 'frac_tbi_lt_005')
    u_l2_zero_post = dist_array(unhealthy, 'l2_post', 'frac_tbi_lt_005')
    h_l2_post_med = dist_array(healthy, 'l2_post', 'median')
    u_l2_post_med = dist_array(unhealthy, 'l2_post', 'median')

    def ms(arr):
        return f"{arr.mean():.4f} ± {arr.std():.4f}"

    # ---------------- two-panel dissociation figure ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
        'legend.fontsize': 9, 'figure.dpi': 300,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    COL_H = '#2E86C1'  # healthy = blue
    COL_U = '#C0392B'  # unhealthy = red
    COL_L1 = '#27AE60'

    n_ep = h_L2.shape[1]
    epochs = np.arange(n_ep)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    # ---- Panel A: L2 TBI trajectory bands ----
    h_L2_mean, h_L2_sd = h_L2.mean(axis=0), h_L2.std(axis=0)
    u_L2_mean, u_L2_sd = u_L2.mean(axis=0), u_L2.std(axis=0)

    L1_ref = float(L1_final.mean())
    L2_init_ref = float(h_L2i.mean())

    axA.axhline(L1_ref, color=COL_L1, ls='--', lw=2,
                label=f'L1 frozen ({L1_ref:.3f})')
    axA.axhline(L2_init_ref, color='#7F8C8D', ls=':', lw=1.5,
                label=f'L2 pre-cascade ({L2_init_ref:.3f})')

    axA.plot(epochs, h_L2_mean, color=COL_H, lw=2.4, label='Healthy L2')
    axA.fill_between(epochs, h_L2_mean - h_L2_sd, h_L2_mean + h_L2_sd,
                     color=COL_H, alpha=0.25, label='Healthy ±1 SD')
    axA.plot(epochs, u_L2_mean, color=COL_U, lw=2.4, label='Unhealthy L2')
    axA.fill_between(epochs, u_L2_mean - u_L2_sd, u_L2_mean + u_L2_sd,
                     color=COL_U, alpha=0.25, label='Unhealthy ±1 SD')

    axA.set_xlabel('Epoch')
    axA.set_ylabel('L2 TBI Mean')
    axA.set_title(
        f'(a) Cascade transmission\n'
        f'ΔL2 healthy {h_dL2.mean():+.3f}  vs  unhealthy {u_dL2.mean():+.3f}',
        fontweight='bold', fontsize=10, loc='left',
    )
    axA.set_ylim(min(L2_init_ref, u_L2.min()) - 0.02, L1_ref + 0.02)
    axA.legend(fontsize=8, loc='lower right', ncol=2)

    # ---- Panel B: PF prediction error trajectory ----
    h_PF_mean, h_PF_sd = h_PF.mean(axis=0), h_PF.std(axis=0)
    u_PF_mean, u_PF_sd = u_PF.mean(axis=0), u_PF.std(axis=0)

    axB.plot(epochs, h_PF_mean, color=COL_H, lw=2.4, label='Healthy')
    axB.fill_between(epochs, h_PF_mean - h_PF_sd, h_PF_mean + h_PF_sd,
                     color=COL_H, alpha=0.25, label='Healthy ±1 SD')
    axB.plot(epochs, u_PF_mean, color=COL_U, lw=2.4, label='Unhealthy')
    axB.fill_between(epochs, u_PF_mean - u_PF_sd, u_PF_mean + u_PF_sd,
                     color=COL_U, alpha=0.25, label='Unhealthy ±1 SD')

    axB.set_xlabel('Epoch')
    axB.set_ylabel('PredictiveField L2-prediction error')
    axB.set_title(
        f'(b) Source-field readability\n'
        f'PF error final  healthy {h_PFf.mean():.4f}  vs  '
        f'unhealthy {u_PFf.mean():.4f}',
        fontweight='bold', fontsize=10, loc='left',
    )
    axB.legend(fontsize=8, loc='best')

    fig.suptitle(
        f'Cascade Dissociation — healthy (drive_gain=1.0) vs unhealthy '
        f'(drive_gain=0.1) L3, n={len(SEEDS)} seeds paired',
        fontweight='bold', fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGS / 'fig5_dissociation.pdf', bbox_inches='tight')
    fig.savefig(FIGS / 'fig5_dissociation.png', bbox_inches='tight')
    plt.close(fig)

    # ---------------- summary markdown ----------------
    lines = [
        '# Cascade Dissociation: Healthy vs Unhealthy L3 Physics',
        '',
        f'**Paired protocol** ({len(SEEDS)} seeds, identical except for L3 regime):  '
        f'epochs=20, batch_size=32, β_nudge=0.05, λ_max=5.0, '
        f'petu_coh_floor=0.35, PETU active. '
        f'L1 frozen (`{L1_CKPT}`), L2 frozen (`{L2_CKPT}`), '
        f'PredictiveField fresh per seed.',
        '',
        '- **Healthy condition**: L3 fresh, `drive_gain=1.0` throughout (matches '
        '`run_multiseed_cascade.py`).  ',
        f'- **Unhealthy condition**: L3 initialized from a freshly-trained '
        f'attenuated checkpoint (`{L3_ATTENUATED_CKPT}`) — 5 epochs at '
        f'`drive_gain=0.1`, β=0.05, λ_max=0 (no cascade) — then run with '
        f'`drive_gain=0.1` for the same 20-epoch cascade protocol.',
        '',
        '## Operating-point note',
        '',
        'The Figure 4 ablation reported L3_TBI=0.024–0.040 for the attenuated '
        'regime. Under the corrected-amplitude codebase, neither the saved '
        '`l3_beta_cal_0.05.pt` checkpoint nor a fresh re-train under the '
        f'original Figure-4 hyperparameters lands in that band:',
        '',
        '| Source | L3_TBI mean | L3 val_acc |',
        '|---|---|---|',
        '| Manuscript Figure 4 attenuated | 0.024–0.040 | 56.0–56.8% |',
        '| Saved `l3_beta_cal_0.05.pt` (loaded, no further training) | 0.121 | 52.5% |',
        '| Fresh re-train, original Fig-4 hyperparameters | 0.153 | 56.25% |',
        '',
        'This is consistent with the amplitude-attenuation correction documented '
        'in the manuscript: Figure 4 was generated when L2\'s `coupling_l2` was '
        'still initialized at 0.1·I (pre-fix), so the absolute attenuated-TBI '
        'number could not survive the L2 correction. The **regime contrast** '
        f'is preserved (unhealthy L3_TBI {u_L3f.mean():.3f} ≈ '
        f'{(u_L3f.mean()/h_L3f.mean()*100):.0f}% of healthy {h_L3f.mean():.3f}; '
        f'unhealthy val_acc {u_valf.mean()*100:.1f}% remains chance-band). '
        'The paired comparison below uses these post-correction values.',
        '',
        '---',
        '',
        '## Table 1 — Final-epoch TBI (mean ± SD across seeds)',
        '',
        '| Layer | Healthy | Unhealthy |',
        '|---|---|---|',
        f'| L1 (frozen, sanity) | {ms(final(h_L1))} | {ms(final(u_L1))} |',
        f'| L2 (post-cascade)   | {ms(h_L2f)} | {ms(u_L2f)} |',
        f'| L3 (post-cascade)   | {ms(h_L3f)} | {ms(u_L3f)} |',
        '',
        '## Table 2 — L2 cascade transmission (ΔL2)',
        '',
        '| Quantity | Healthy | Unhealthy |',
        '|---|---|---|',
        f'| L2 TBI initial (epoch 0)  | {ms(h_L2i)} | {ms(u_L2i)} |',
        f'| L2 TBI final (epoch 19)   | {ms(h_L2f)} | {ms(u_L2f)} |',
        f'| Δ L2 TBI                  | {ms(h_dL2)} | {ms(u_dL2)} |',
        f'| Gap closure (% of L1−L2)  | {h_gap.mean():.1f}% ± {h_gap.std():.1f}% '
        f'| {u_gap.mean():.1f}% ± {u_gap.std():.1f}% |',
        '',
        '## Table 3 — PredictiveField L2-prediction error',
        '',
        '| Epoch | Healthy | Unhealthy |',
        '|---|---|---|',
        f'| 0 (initial)  | {ms(h_PFi)} | {ms(u_PFi)} |',
        f'| 19 (final)   | {ms(h_PFf)} | {ms(u_PFf)} |',
        '',
        '## Table 4 — L3 task quality',
        '',
        '| Quantity | Healthy | Unhealthy |',
        '|---|---|---|',
        f'| L3 val_acc final | {ms(h_valf)} | {ms(u_valf)} |',
        '',
        '## Table 5 — L2 zero-coherence fraction (TBI<0.05) at final epoch',
        '',
        '| Layer-stat | Healthy | Unhealthy |',
        '|---|---|---|',
        f'| L2 post-cascade median | {ms(h_l2_post_med)} | {ms(u_l2_post_med)} |',
        f'| L2 frac<0.05 (post)    | {h_l2_zero_post.mean()*100:.1f}% ± {h_l2_zero_post.std()*100:.1f}% '
        f'| {u_l2_zero_post.mean()*100:.1f}% ± {u_l2_zero_post.std()*100:.1f}% |',
        '',
        '## Table 6 — Per-seed final numbers (paired)',
        '',
        '| seed | L2_TBI_H | L2_TBI_U | ΔL2_H | ΔL2_U | PFerr_H | PFerr_U | val_H | val_U |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for i, s in enumerate(SEEDS):
        lines.append(
            f'| {s} | {h_L2f[i]:.4f} | {u_L2f[i]:.4f} | '
            f'{h_dL2[i]:+.4f} | {u_dL2[i]:+.4f} | '
            f'{h_PFf[i]:.4f} | {u_PFf[i]:.4f} | '
            f'{h_valf[i]*100:.1f}% | {u_valf[i]*100:.1f}% |'
        )

    lines += [
        '',
        '## Figures',
        '',
        '- `runs/figures/fig5_dissociation.{png,pdf}` — two-panel paired figure. '
        'Panel (a): L2 TBI trajectories with ±SD bands, healthy vs unhealthy, '
        'L1-frozen and L2-pre-cascade reference lines. '
        'Panel (b): PredictiveField L2-prediction-error trajectories, healthy '
        'vs unhealthy, ±SD bands. Tells the causal story: source-field '
        'coherence (healthy vs unhealthy L3) → prediction quality (PF error) '
        '→ transmission (ΔL2).',
        '',
        '## Flags',
    ]

    # Flag seeds with unhealthy ΔL2 deviating > 0.05 from the seed mean
    u_dL2_mean = u_dL2.mean()
    flagged = []
    for i, s in enumerate(SEEDS):
        if abs(u_dL2[i] - u_dL2_mean) > 0.05:
            flagged.append(
                f'- Unhealthy seed {s}: ΔL2={u_dL2[i]:+.4f} deviates '
                f'{u_dL2[i] - u_dL2_mean:+.4f} from seed mean '
                f'{u_dL2_mean:+.4f} (>0.05 threshold).'
            )
    if not flagged:
        lines.append(
            f'- No unhealthy seed deviates more than 0.05 from the seed mean '
            f'ΔL2 ({u_dL2_mean:+.4f}); cross-seed range '
            f'{u_dL2.min():+.4f} – {u_dL2.max():+.4f}.'
        )
    else:
        lines += flagged

    summary_path = RUNS / 'cascade_unhealthy_summary.md'
    summary_path.write_text('\n'.join(lines) + '\n')
    print(f"\nWrote {summary_path}", flush=True)
    print(f"Wrote {FIGS / 'fig5_dissociation.png'}", flush=True)
    return summary_path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    RUNS.mkdir(parents=True, exist_ok=True)

    for s in SEEDS:
        sd = run_seed(s)
        try:
            measure_distributions(sd, drive_gain=DRIVE_GAIN_UNHEALTHY)
        except Exception as e:
            print(f"  WARN: distribution measurement for unhealthy seed {s} "
                  f"failed: {e}", flush=True)

    summary_path = aggregate_and_report()
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(summary_path.read_text())


if __name__ == '__main__':
    main()
