"""
TRACE epoch selection and minP combination.

For each epoch's GWAS output, compute:
  1. Number of genome-wide significant SNPs per dimension (unique signal)
  2. Select informative epochs (> threshold unique SNPs)
  3. Combine selected epochs via minP across dimensions
  4. Apply Bonferroni correction: 5e-8 / (128 * N_selected_epochs)

Thresholds tested: 50, 100, 200 (see sensitivity_analysis.py)
Default threshold: 100
"""

import os
import sys
import re
import pickle as pkl
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from record_class import GB


# genome-wide significance
BASE_ALPHA = 5e-8
N_DIMS = 128

# these are the epochs that had > 100 unique significant SNPs in our analysis
# re-run identify_informative_epochs() on your data
DEFAULT_INFORMATIVE_EPOCHS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 18, 37, 38, 46]


def load_epoch_pvals(gwas_pkl_path):
    with open(gwas_pkl_path, 'rb') as f:
        d = pkl.load(f)
    return d['pvals'], d['snp_info'], d['epoch']


def get_significant_snps(pvals, snp_info, alpha):
    """Return SNP ids with minP (across 128 dims) < alpha."""
    min_p = pvals.min(axis=1)
    sig_mask = min_p < alpha
    sig_snps = set(snp_info.loc[sig_mask, 'SNP'].tolist())
    return sig_snps


def identify_informative_epochs(gwas_dir, threshold=100, alpha=None):
    """
    For each epoch, count unique significant SNPs not found in any earlier epoch.
    Returns list of epoch ids with > threshold unique SNPs.
    """
    if alpha is None:
        alpha = BASE_ALPHA / N_DIMS  # per-epoch Bonferroni

    epoch_dirs = sorted(
        [d for d in os.listdir(gwas_dir) if os.path.isdir(os.path.join(gwas_dir, d))],
        key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1))
        if re.search(r'epoch_(\d+)', x) else 0
    )

    seen_snps = set()
    informative = []
    results = []

    for epoch_dir in epoch_dirs:
        pkl_path = os.path.join(gwas_dir, epoch_dir, f'epoch_{epoch_dir.split("_")[-1]}_gwas.pkl')
        if not os.path.exists(pkl_path):
            continue

        epoch_id = int(re.search(r'epoch_(\d+)', epoch_dir).group(1))
        pvals, snp_info, _ = load_epoch_pvals(pkl_path)

        sig_snps = get_significant_snps(pvals, snp_info, alpha)
        unique_snps = sig_snps - seen_snps
        seen_snps |= sig_snps

        results.append({
            'epoch': epoch_id,
            'n_sig': len(sig_snps),
            'n_unique': len(unique_snps),
        })

        if len(unique_snps) > threshold:
            informative.append(epoch_id)

        print(f"Epoch {epoch_id}: {len(sig_snps)} sig, {len(unique_snps)} unique")

    df = pd.DataFrame(results)
    print(f"\nInformative epochs (>{threshold} unique): {informative}")
    return informative, df


def combine_minp(gwas_dir, informative_epochs, output_path=None):
    """
    Combine p-values across informative epochs via minP.
    Saves combined result as pkl.
    """
    all_pvals = []
    snp_info_ref = None

    for epoch in sorted(informative_epochs):
        pkl_path = os.path.join(gwas_dir, f'{os.path.basename(gwas_dir)}_epoch_{epoch}',
                                f'epoch_{epoch}_gwas.pkl')
        if not os.path.exists(pkl_path):
            print(f"WARNING: missing epoch {epoch} gwas output")
            continue

        pvals, snp_info, _ = load_epoch_pvals(pkl_path)
        all_pvals.append(pvals)

        if snp_info_ref is None:
            snp_info_ref = snp_info

    print(f"Combining {len(all_pvals)} epochs...")
    combined = np.stack(all_pvals, axis=0).min(axis=0)  # minP across epochs

    n_epochs = len(all_pvals)
    bonferroni_thresh = BASE_ALPHA / (N_DIMS * n_epochs)
    print(f"Bonferroni threshold (5e-8 / ({N_DIMS} dims * {n_epochs} epochs)): {bonferroni_thresh:.3e}")

    result = {
        'pvals': combined,
        'snp_info': snp_info_ref,
        'informative_epochs': informative_epochs,
        'bonferroni_thresh': bonferroni_thresh,
        'n_epochs': n_epochs,
    }

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pkl.dump(result, f)
        print(f"Combined result saved: {output_path}")

    return result


def export_for_fuma(combined_result, output_dir):
    """
    Export minP summary stats in FUMA-compatible format.
    Columns: SNP, CHR, BP, A1, A2, P
    """
    os.makedirs(output_dir, exist_ok=True)

    pvals = combined_result['pvals']
    snp_info = combined_result['snp_info']

    min_p_per_snp = pvals.min(axis=1)

    out_df = snp_info[['SNP', 'CHR', 'BP', 'A1', 'A2']].copy()
    out_df['P'] = min_p_per_snp

    out_path = os.path.join(output_dir, 'trace_minp_for_fuma.txt')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"FUMA input saved: {out_path}")
    return out_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gwas_dir', type=str, required=True)
    parser.add_argument('--threshold', type=int, default=100,
                        help='min unique SNPs to consider an epoch informative')
    parser.add_argument('--output', type=str, default=None,
                        help='path to save combined pkl result')
    parser.add_argument('--fuma_dir', type=str, default=None,
                        help='output dir for FUMA-ready summary stats')
    args = parser.parse_args()

    informative, epoch_df = identify_informative_epochs(
        args.gwas_dir, threshold=args.threshold)

    gb = GB()
    output_path = args.output or gb.combined_minp_df_path

    combined = combine_minp(args.gwas_dir, informative, output_path=output_path)

    if args.fuma_dir:
        export_for_fuma(combined, args.fuma_dir)
