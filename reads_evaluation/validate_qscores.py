#!/usr/bin/env python3
"""
Validate per-base Phred quality scores produced by the Viterbi decoder.

What this script checks
-----------------------
1.  Basic sanity
    - Q distribution (min / max / mean / median / quartiles).
    - Fraction of bases saturated at Q=1 or Q=50 (both should be small;
      massive saturation means the clamp is hiding a bug).
    - Mean Q per read vs. read identity (should be positively correlated).

2.  Calibration curve (the real test)
    - For each aligned base walk the CIGAR and mark match (=) / mismatch (X)
      / insertion (I). Deletions and soft-clips do not have a quality value
      and are skipped.
    - Bin bases by reported integer Q, compute empirical error rate per bin,
      then empirical_Q = -10 * log10(empirical_error_rate).
    - A well-calibrated Q should fall on y = x (with a ~1-2 dB slack at the
      tails where counts are small).

Assumes: SAM was generated with `minimap2 --eqx` so match/mismatch live in
the CIGAR as `=` / `X` (no need to also pull MD tag + reference FASTA).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pysam


# CIGAR op codes (pysam convention)
CIGAR_M = 0    # M (generic alignment; shouldn't appear with --eqx)
CIGAR_I = 1    # I  insertion
CIGAR_D = 2    # D  deletion
CIGAR_S = 4    # S  soft clip
CIGAR_H = 5    # H  hard clip
CIGAR_EQ = 7   # =  match
CIGAR_X = 8    # X  mismatch


def collect_per_base(sam_path, max_q=50):
    """Walk SAM and accumulate per-base (reported_Q, is_error) counts.

    Returns
    -------
    n_total : (max_q+2,) int64   total bases with reported Q = i
    n_error : (max_q+2,) int64   erroneous bases (X or I) with Q = i
    q_flat  : (M,)       uint8   every reported Q (for distribution plot)
    per_read: list of dicts      mean Q + identity per aligned read
    """
    total_q_chunks = []
    error_q_chunks = []
    all_q_chunks = []
    per_read = []

    n_total_reads = 0
    n_mapped = 0

    with pysam.AlignmentFile(sam_path, 'r') as sam:
        for read in sam:
            n_total_reads += 1
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            n_mapped += 1

            quals = read.query_qualities
            if quals is None or len(quals) == 0:
                continue
            quals = np.asarray(quals, dtype=np.int16)

            # full per-read Q dump (incl. soft-clipped bases) for distribution
            all_q_chunks.append(quals)

            # walk CIGAR to classify each aligned base.
            # pysam cigartuples are (op_code, length).
            qpos = 0
            n_match_r = 0
            n_err_r = 0
            for op, length in read.cigartuples:
                if op == CIGAR_S:
                    qpos += length
                elif op == CIGAR_H:
                    pass
                elif op == CIGAR_EQ:
                    seg = quals[qpos:qpos + length]
                    total_q_chunks.append(seg)
                    n_match_r += length
                    qpos += length
                elif op == CIGAR_X:
                    seg = quals[qpos:qpos + length]
                    total_q_chunks.append(seg)
                    error_q_chunks.append(seg)
                    n_err_r += length
                    qpos += length
                elif op == CIGAR_I:
                    seg = quals[qpos:qpos + length]
                    total_q_chunks.append(seg)
                    error_q_chunks.append(seg)
                    n_err_r += length
                    qpos += length
                elif op == CIGAR_D:
                    # no read base, but still an error vs. reference
                    n_err_r += length
                elif op == CIGAR_M:
                    seg = quals[qpos:qpos + length]
                    total_q_chunks.append(seg)
                    n_match_r += length
                    qpos += length
                else:
                    raise ValueError(f"unexpected CIGAR op code: {op}")

            aligned_base_read = n_match_r + n_err_r
            if aligned_base_read > 0:
                identity = n_match_r / aligned_base_read
                per_read.append({
                    'name': read.query_name,
                    'mean_q': float(np.mean(quals)),
                    'identity': identity,
                    'aligned_bases': aligned_base_read,
                })

    nbins = max_q + 2
    if total_q_chunks:
        total_flat = np.concatenate(total_q_chunks)
        error_flat = np.concatenate(error_q_chunks) if error_q_chunks else np.array([], dtype=np.int16)
        total_flat = np.clip(total_flat, 0, max_q + 1)
        error_flat = np.clip(error_flat, 0, max_q + 1)
        n_total = np.bincount(total_flat, minlength=nbins)
        n_error = np.bincount(error_flat, minlength=nbins)
    else:
        n_total = np.zeros(nbins, dtype=np.int64)
        n_error = np.zeros(nbins, dtype=np.int64)

    q_flat = np.concatenate(all_q_chunks) if all_q_chunks else np.array([], dtype=np.int16)

    return {
        'n_total': n_total,
        'n_error': n_error,
        'q_flat': q_flat,
        'per_read': per_read,
        'total_reads': n_total_reads,
        'mapped_reads': n_mapped,
    }


def print_basic_stats(stats, max_q=50):
    q_flat = stats['q_flat']
    print("=" * 60)
    print("1) BASIC Q DISTRIBUTION (every base in the FASTQ)")
    print("=" * 60)
    print(f"total reads         : {stats['total_reads']}")
    print(f"mapped reads        : {stats['mapped_reads']}")
    print(f"total bases (with Q): {len(q_flat):,}")
    if len(q_flat) == 0:
        return
    print(f"Q  min              : {q_flat.min()}")
    print(f"Q  max              : {q_flat.max()}")
    print(f"Q  mean             : {q_flat.mean():.2f}")
    print(f"Q  median           : {float(np.median(q_flat)):.1f}")
    print(f"Q  25/75 percentile : {np.percentile(q_flat, 25):.1f} / "
          f"{np.percentile(q_flat, 75):.1f}")
    sat_low  = np.mean(q_flat <= 1) * 100
    sat_high = np.mean(q_flat >= max_q) * 100
    print(f"% bases at Q<=1     : {sat_low:.2f}%   (should be small)")
    print(f"% bases at Q>={max_q}    : {sat_high:.2f}%   (should be small)")


def print_calibration_table(n_total, n_error, max_q=50, min_count=200):
    print()
    print("=" * 60)
    print("2) CALIBRATION  (per-base match/mismatch/insertion from CIGAR)")
    print("=" * 60)
    print(f"{'Q_rep':>6} {'N_total':>12} {'N_err':>10} "
          f"{'P_err':>10} {'Q_emp':>8} {'delta':>7}")
    for q in range(1, max_q + 1):
        if n_total[q] < min_count:
            continue
        p_err = n_error[q] / n_total[q] if n_total[q] > 0 else np.nan
        q_emp = -10 * np.log10(max(p_err, 1e-7))
        delta = q_emp - q
        print(f"{q:>6d} {n_total[q]:>12,d} {n_error[q]:>10,d} "
              f"{p_err:>10.4f} {q_emp:>8.2f} {delta:>+7.2f}")
    print()
    # global empirical Q
    total = n_total[1:max_q + 1].sum()
    errors = n_error[1:max_q + 1].sum()
    if total > 0:
        global_p = errors / total
        global_q_emp = -10 * np.log10(max(global_p, 1e-7))
        # mean reported Q weighted by base counts
        qs = np.arange(max_q + 1)
        mean_reported = (n_total[:max_q + 1] * qs).sum() / n_total[:max_q + 1].sum()
        print(f"global reported  Q (weighted mean)  : {mean_reported:.2f}")
        print(f"global empirical Q (-10log10 error) : {global_q_emp:.2f}")
        print(f"global error rate                   : {global_p * 100:.3f}%")


def plot_validation(stats, out_png, max_q=50, min_count=200):
    q_flat = stats['q_flat']
    n_total = stats['n_total']
    n_error = stats['n_error']
    per_read = stats['per_read']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ---------- (a) Q distribution ----------
    ax = axes[0]
    if len(q_flat):
        bins = np.arange(max_q + 2) - 0.5
        ax.hist(q_flat, bins=bins, alpha=0.75, color='steelblue',
                edgecolor='black', linewidth=0.3)
        mean_q = q_flat.mean()
        median_q = float(np.median(q_flat))
        ax.axvline(mean_q, color='red', linestyle='--',
                   label=f'mean = {mean_q:.1f}')
        ax.axvline(median_q, color='orange', linestyle='--',
                   label=f'median = {median_q:.1f}')
    ax.set_xlabel('Reported Q')
    ax.set_ylabel('Count')
    ax.set_title('(a) Per-base Q distribution')
    ax.grid(alpha=0.3)
    ax.legend()

    # ---------- (b) Calibration curve ----------
    ax = axes[1]
    # n_total / n_error both have shape (max_q + 2,); bin i corresponds to Q = i.
    qs = np.arange(len(n_total))                    # 0 .. max_q+1
    valid = n_total >= min_count
    p_err = np.where(n_total > 0, n_error / np.maximum(n_total, 1), np.nan)
    q_emp = -10 * np.log10(np.maximum(p_err, 1e-7))

    ax.plot([0, max_q], [0, max_q], 'k--', alpha=0.6, label='y = x (perfect)')
    sizes = np.clip(np.log10(n_total + 1) * 40, 20, 400)
    ax.scatter(qs[valid], q_emp[valid], s=sizes[valid],
               c='crimson', alpha=0.7, edgecolor='black', linewidth=0.4,
               label=f'bins with ≥{min_count} bases')
    # annotate sparse-but-shown bins with their counts
    for q in qs[valid]:
        ax.annotate(f'{int(n_total[q]):,}', (q, q_emp[q]),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=7, alpha=0.6)
    ax.set_xlim(0, max_q + 1)
    ax.set_ylim(0, max_q + 5)
    ax.set_xlabel('Reported Q')
    ax.set_ylabel(r'Empirical Q  = $-10\log_{10}(P_{err})$')
    ax.set_title('(b) Q calibration (per-base)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')

    # ---------- (c) Per-read: mean Q vs identity ----------
    ax = axes[2]
    if per_read:
        mean_qs = np.array([r['mean_q'] for r in per_read])
        ids = np.array([r['identity'] for r in per_read])
        id_q = -10 * np.log10(np.clip(1.0 - ids, 1e-4, 1.0))
        ax.scatter(mean_qs, id_q, s=4, alpha=0.35, color='teal')
        lo = min(mean_qs.min(), id_q.min()) - 1
        hi = max(mean_qs.max(), id_q.max()) + 1
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.6, label='y = x')
        # Pearson r
        if len(mean_qs) > 1:
            r = float(np.corrcoef(mean_qs, id_q)[0, 1])
            ax.text(0.04, 0.96, f'Pearson r = {r:.3f}\nreads = {len(mean_qs)}',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_xlabel('Read mean reported Q')
    ax.set_ylabel(r'Read identity (as Q = $-10\log_{10}(1-\mathrm{identity})$)')
    ax.set_title('(c) Per-read: reported vs identity')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"Saved validation plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sam', required=True, help='aligned SAM (minimap2 --eqx)')
    ap.add_argument('--output_png', default=None,
                    help='output PNG (default: <sam_dir>/<basename>_qval.png)')
    ap.add_argument('--max_q', type=int, default=50)
    ap.add_argument('--min_count', type=int, default=200,
                    help='minimum bases per bin to report/plot a calibration point')
    args = ap.parse_args()

    print(f"Analyzing: {args.sam}")
    stats = collect_per_base(args.sam, max_q=args.max_q)

    print_basic_stats(stats, max_q=args.max_q)
    print_calibration_table(stats['n_total'], stats['n_error'],
                            max_q=args.max_q, min_count=args.min_count)

    if args.output_png:
        out_png = args.output_png
    else:
        sam_dir = os.path.dirname(os.path.abspath(args.sam))
        base = os.path.splitext(os.path.basename(args.sam))[0]
        out_png = os.path.join(sam_dir, f"{base}_qval.png")
    plot_validation(stats, out_png, max_q=args.max_q,
                    min_count=args.min_count)


if __name__ == '__main__':
    main()
