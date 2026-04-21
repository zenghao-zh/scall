#!/usr/bin/env python3
"""Overlay the calibration curves and Q distributions of two SAM files.

Typical use: compare the Viterbi decoder (our new code path) against the
beam_search baseline (koi/bonito) on the same set of reads.

Reuses the per-base collector from validate_qscores.py.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from validate_qscores import collect_per_base


def summarise(name, stats, max_q=50):
    q_flat = stats['q_flat']
    n_total = stats['n_total']
    n_error = stats['n_error']
    total = n_total[1:max_q + 1].sum()
    errors = n_error[1:max_q + 1].sum()
    global_p = errors / total if total else float('nan')
    global_q_emp = -10 * np.log10(max(global_p, 1e-7)) if total else float('nan')
    qs = np.arange(max_q + 1)
    mean_reported = ((n_total[:max_q + 1] * qs).sum()
                     / max(n_total[:max_q + 1].sum(), 1))
    return {
        'name': name,
        'reads_total': stats['total_reads'],
        'reads_mapped': stats['mapped_reads'],
        'bases': len(q_flat),
        'q_min': int(q_flat.min()) if len(q_flat) else 0,
        'q_max': int(q_flat.max()) if len(q_flat) else 0,
        'q_mean': float(q_flat.mean()) if len(q_flat) else float('nan'),
        'q_median': float(np.median(q_flat)) if len(q_flat) else float('nan'),
        'q_p25': float(np.percentile(q_flat, 25)) if len(q_flat) else float('nan'),
        'q_p75': float(np.percentile(q_flat, 75)) if len(q_flat) else float('nan'),
        'sat_low_pct': float(np.mean(q_flat <= 1) * 100) if len(q_flat) else 0.0,
        'sat_high_pct': float(np.mean(q_flat >= max_q) * 100) if len(q_flat) else 0.0,
        'global_reported_q': float(mean_reported),
        'global_empirical_q': float(global_q_emp),
        'global_err_pct': float(global_p * 100) if total else float('nan'),
    }


def print_summary(summaries):
    rows = [
        ('total reads',         'reads_total',       '{:>12,}'),
        ('mapped reads',        'reads_mapped',      '{:>12,}'),
        ('total bases (w/Q)',   'bases',             '{:>12,}'),
        ('Q min',               'q_min',             '{:>12d}'),
        ('Q max',               'q_max',             '{:>12d}'),
        ('Q mean',              'q_mean',            '{:>12.2f}'),
        ('Q median',            'q_median',          '{:>12.1f}'),
        ('Q 25/75 %ile',        None,                None),  # custom
        ('% bases at Q<=1',     'sat_low_pct',       '{:>11.2f}%'),
        ('% bases at Q>=50',    'sat_high_pct',      '{:>11.2f}%'),
        ('global reported Q',   'global_reported_q', '{:>12.2f}'),
        ('global empirical Q',  'global_empirical_q','{:>12.2f}'),
        ('global error rate',   'global_err_pct',    '{:>11.3f}%'),
    ]
    header = f"{'metric':<22}" + ''.join(f"{s['name']:>14}" for s in summaries)
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for label, key, fmt in rows:
        line = f"{label:<22}"
        if key is None:  # Q 25/75
            for s in summaries:
                line += f"  {s['q_p25']:>4.1f} / {s['q_p75']:<4.1f}"
        else:
            for s in summaries:
                line += '  ' + fmt.format(s[key])
        print(line)
    print('=' * len(header))


def print_joint_table(stats_list, names, max_q=50, min_count=200):
    print()
    header = f"{'Q_rep':>6}"
    for n in names:
        header += f" | {n + ' N_tot':>12} {n + ' Nerr':>8} {'Perr':>8} {'Qemp':>6} {'dlt':>6}"
    print(header)
    print('-' * len(header))
    for q in range(1, max_q + 1):
        counts = [st['n_total'][q] for st in stats_list]
        if max(counts) < min_count:
            continue
        line = f"{q:>6d}"
        for st in stats_list:
            nt, ne = st['n_total'][q], st['n_error'][q]
            if nt == 0:
                line += f" | {'-':>12} {'-':>8} {'-':>8} {'-':>6} {'-':>6}"
                continue
            p_err = ne / nt
            q_emp = -10 * np.log10(max(p_err, 1e-7))
            delta = q_emp - q
            line += (f" | {nt:>12,d} {ne:>8,d} {p_err:>8.4f}"
                     f" {q_emp:>6.2f} {delta:>+6.2f}")
        print(line)


def plot_overlay(stats_list, names, out_png, max_q=50, min_count=200):
    colours = ['crimson', 'royalblue', 'seagreen', 'darkorange']
    colours = colours[:len(stats_list)]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.7))

    # (a) Q distribution (log-y histograms, overlaid)
    ax = axes[0]
    bins = np.arange(max_q + 2) - 0.5
    for st, name, col in zip(stats_list, names, colours):
        q_flat = st['q_flat']
        if len(q_flat) == 0:
            continue
        ax.hist(q_flat, bins=bins, histtype='step', linewidth=1.6,
                color=col, label=f'{name}  (mean={q_flat.mean():.1f})')
    ax.set_yscale('log')
    ax.set_xlabel('Reported Q')
    ax.set_ylabel('Count (log)')
    ax.set_title('(a) Per-base Q distribution')
    ax.grid(alpha=0.3, which='both')
    ax.legend()

    # (b) Calibration curve overlay
    ax = axes[1]
    ax.plot([0, max_q], [0, max_q], 'k--', alpha=0.6, label='y = x (perfect)')
    for st, name, col in zip(stats_list, names, colours):
        n_total = st['n_total']
        n_error = st['n_error']
        qs = np.arange(len(n_total))
        valid = n_total >= min_count
        p_err = np.where(n_total > 0,
                         n_error / np.maximum(n_total, 1), np.nan)
        q_emp = -10 * np.log10(np.maximum(p_err, 1e-7))
        sizes = np.clip(np.log10(n_total + 1) * 35, 18, 320)
        ax.plot(qs[valid], q_emp[valid], '-', color=col, alpha=0.45, linewidth=1.0)
        ax.scatter(qs[valid], q_emp[valid], s=sizes[valid], color=col,
                   alpha=0.78, edgecolor='black', linewidth=0.35, label=name)
    ax.set_xlim(0, max_q + 1)
    ax.set_ylim(0, max_q + 5)
    ax.set_xlabel('Reported Q')
    ax.set_ylabel(r'Empirical Q = $-10\log_{10}(P_{err})$')
    ax.set_title('(b) Q calibration (per base)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')

    # (c) Per-read: mean Q vs. read identity (shown as Q)
    ax = axes[2]
    for st, name, col in zip(stats_list, names, colours):
        per_read = st['per_read']
        if not per_read:
            continue
        mean_qs = np.array([r['mean_q'] for r in per_read])
        ids = np.array([r['identity'] for r in per_read])
        id_q = -10 * np.log10(np.clip(1.0 - ids, 1e-4, 1.0))
        r = float(np.corrcoef(mean_qs, id_q)[0, 1]) if len(mean_qs) > 1 else float('nan')
        ax.scatter(mean_qs, id_q, s=4, alpha=0.35, color=col,
                   label=f'{name}  r={r:.3f}  n={len(mean_qs)}')
    lo = 0
    hi = max_q + 2
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Read mean reported Q')
    ax.set_ylabel(r'Read identity Q = $-10\log_{10}(1-\mathrm{identity})$')
    ax.set_title('(c) Per-read reported vs. identity')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(out_png, dpi=115)
    plt.close(fig)
    print(f"\nSaved overlay plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sam', nargs='+', required=True,
                    help='SAM files (minimap2 --eqx), 2+ for overlay')
    ap.add_argument('--label', nargs='+', default=None,
                    help='label per SAM (defaults to file basename)')
    ap.add_argument('--output_png', required=True)
    ap.add_argument('--max_q', type=int, default=50)
    ap.add_argument('--min_count', type=int, default=200)
    args = ap.parse_args()

    if args.label and len(args.label) != len(args.sam):
        raise SystemExit('--label count must match --sam count')
    labels = args.label or [
        os.path.splitext(os.path.basename(p))[0] for p in args.sam
    ]

    stats_list = []
    summaries = []
    for sam, name in zip(args.sam, labels):
        print(f"Analyzing: {sam}  (label: {name})")
        st = collect_per_base(sam, max_q=args.max_q)
        stats_list.append(st)
        summaries.append(summarise(name, st, max_q=args.max_q))

    print_summary(summaries)
    print_joint_table(stats_list, labels,
                      max_q=args.max_q, min_count=args.min_count)
    plot_overlay(stats_list, labels, args.output_png,
                 max_q=args.max_q, min_count=args.min_count)


if __name__ == '__main__':
    main()
