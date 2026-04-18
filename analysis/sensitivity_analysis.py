"""
Sensitivity analysis: epoch selection threshold.

For each threshold in [50, 100, 200], identify informative epochs and
run minP combination. Reports: N epochs selected, Bonferroni threshold,
N significant SNPs, and loci (from FUMA - must be run separately).

Usage:
    python sensitivity_analysis.py --gwas_dir /path/to/gwas_output
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from record_class import GB
from gwas.epoch_selection import identify_informative_epochs, combine_minp

OUTPUT_DIR = 'results/sensitivity'
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

THRESHOLDS = [50, 100, 200]
BASE_ALPHA = 5e-8
N_DIMS = 128


def run_sensitivity(gwas_dir, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for thresh in THRESHOLDS:
        print(f"\n{'='*40}")
        print(f"Threshold: >{thresh} unique SNPs")
        print(f"{'='*40}")

        informative, epoch_df = identify_informative_epochs(
            gwas_dir, threshold=thresh)

        n_epochs = len(informative)
        bonf_thresh = BASE_ALPHA / (N_DIMS * n_epochs) if n_epochs > 0 else np.nan

        # combine minP for these epochs
        combined_path = os.path.join(output_dir, f'combined_thresh_{thresh}.pkl')
        combined = combine_minp(gwas_dir, informative, output_path=combined_path)

        # count significant SNPs (minP across dims < bonf_thresh)
        min_p = combined['pvals'].min(axis=1)
        n_sig = (min_p < bonf_thresh).sum()

        print(f"Epochs selected: {n_epochs}")
        print(f"Bonferroni threshold: {bonf_thresh:.3e}")
        print(f"Significant SNPs: {n_sig}")
        print(f"Informative epochs: {informative}")

        results.append({
            'threshold': thresh,
            'n_epochs': n_epochs,
            'bonf_thresh': bonf_thresh,
            'n_sig_snps': n_sig,
            'epochs': informative,
        })

    df = pd.DataFrame(results)
    out_csv = os.path.join(output_dir, 'sensitivity_summary.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    return df


def plot_sensitivity(summary_df, output_dir=FIGURE_DIR):
    # add loci counts if available (fill in after FUMA)
    # summary_df['n_loci'] = [86, 90, 90]  # example

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    ax = axes[0]
    ax.plot(summary_df['threshold'], summary_df['n_epochs'],
            marker='o', color='steelblue', linewidth=1.5)
    ax.set_xlabel('Unique SNP threshold')
    ax.set_ylabel('N informative epochs')
    ax.set_xticks(THRESHOLDS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    ax.plot(summary_df['threshold'], summary_df['n_sig_snps'],
            marker='o', color='coral', linewidth=1.5)
    ax.set_xlabel('Unique SNP threshold')
    ax.set_ylabel('N significant SNPs (minP)')
    ax.set_xticks(THRESHOLDS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'sensitivity_threshold.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gwas_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    summary = run_sensitivity(args.gwas_dir, args.output_dir)
    plot_sensitivity(summary)
