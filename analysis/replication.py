"""
Replication analysis.

Two levels:
  1. Locus replication: do TRACE-unique loci replicate in the held-out cohort?
     Nominal p < 0.05 and directional consistency.

  2. Epoch-specificity replication: for key loci (MAPT, MCPH1), is the
     epoch profile of association strength concordant between discovery and replication?
     Uses Spearman correlation of -log10(P) across informative epochs.

Usage:
    python replication.py --discovery_dir /path --replication_dir /path --loci_csv ST2_loci.csv
"""

import os
import sys
import re
import pickle as pkl
import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from record_class import GB

INFORMATIVE_EPOCHS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 18, 37, 38, 46]
N_DIMS = 128
NOMINAL = 0.05


def load_gwas_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pkl.load(f)
    return d['pvals'], d['snp_info']


def get_lead_snp_pval(pvals, snp_info, snp_id):
    """Return minP across 128 dims for a specific SNP."""
    idx = snp_info[snp_info['SNP'] == snp_id].index
    if len(idx) == 0:
        return None
    return pvals[idx[0]].min()


def get_epoch_minp_profile(gwas_dir, exp_name, snp_id, epochs=INFORMATIVE_EPOCHS):
    """
    For a given SNP, return -log10(minP) at each epoch.
    Returns dict {epoch: neg_log10_p}
    """
    profile = {}
    pattern = re.compile(r'epoch_(\d+)')

    for epoch in epochs:
        epoch_dir = os.path.join(gwas_dir, f'{exp_name}_epoch_{epoch}')
        pkl_path = os.path.join(epoch_dir, f'epoch_{epoch}_gwas.pkl')
        if not os.path.exists(pkl_path):
            continue

        pvals, snp_info = load_gwas_pkl(pkl_path)
        p = get_lead_snp_pval(pvals, snp_info, snp_id)
        if p is not None and p > 0:
            profile[epoch] = -np.log10(p)
        else:
            profile[epoch] = 0.0

    return profile


def locus_replication(disc_gwas_dir, repli_gwas_dir, exp_name,
                      loci_df, output_dir):
    """
    For each locus in loci_df, check nominal replication in replication cohort.
    loci_df must have columns: locus_id, lead_snp, category (TRACE-unique/Shared/UDIP-unique)
    """
    os.makedirs(output_dir, exist_ok=True)

    disc_thresh = 5e-8 / (N_DIMS * len(INFORMATIVE_EPOCHS))
    print(f"Discovery threshold: {disc_thresh:.3e}")

    results = []
    for _, row in loci_df.iterrows():
        snp = row['lead_snp']
        category = row.get('category', 'unknown')

        # discovery minP
        disc_minp = None
        for epoch in INFORMATIVE_EPOCHS:
            pkl_path = os.path.join(disc_gwas_dir, f'{exp_name}_epoch_{epoch}',
                                    f'epoch_{epoch}_gwas.pkl')
            if not os.path.exists(pkl_path):
                continue
            pvals, snp_info = load_gwas_pkl(pkl_path)
            p = get_lead_snp_pval(pvals, snp_info, snp)
            if p is not None:
                disc_minp = p if disc_minp is None else min(disc_minp, p)

        # replication minP
        repli_minp = None
        for epoch in INFORMATIVE_EPOCHS:
            pkl_path = os.path.join(repli_gwas_dir, f'{exp_name}_epoch_{epoch}',
                                    f'epoch_{epoch}_gwas.pkl')
            if not os.path.exists(pkl_path):
                continue
            pvals, snp_info = load_gwas_pkl(pkl_path)
            p = get_lead_snp_pval(pvals, snp_info, snp)
            if p is not None:
                repli_minp = p if repli_minp is None else min(repli_minp, p)

        replicated = (repli_minp is not None) and (repli_minp < NOMINAL)

        results.append({
            'locus': row.get('locus_id', snp),
            'lead_snp': snp,
            'category': category,
            'disc_minP': disc_minp,
            'repli_minP': repli_minp,
            'replicated_nominal': replicated,
        })
        print(f"{snp}: disc={disc_minp:.2e if disc_minp else 'NA'}, "
              f"repli={repli_minp:.2e if repli_minp else 'NA'}, "
              f"replicated={replicated}")

    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, 'locus_replication.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # summary by category
    print("\nReplication rate by category:")
    print(df.groupby('category')['replicated_nominal'].agg(['sum', 'count', 'mean']))

    return df


def epoch_specificity_replication(disc_gwas_dir, repli_gwas_dir, exp_name,
                                   key_loci, output_dir):
    """
    For key loci, correlate -log10(P) epoch profile between discovery and replication.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []
    fig, axes = plt.subplots(1, len(key_loci), figsize=(4 * len(key_loci), 4))
    if len(key_loci) == 1:
        axes = [axes]

    for ax, (locus_label, snp_id) in zip(axes, key_loci.items()):
        disc_profile = get_epoch_minp_profile(disc_gwas_dir, exp_name, snp_id)
        repli_profile = get_epoch_minp_profile(repli_gwas_dir, exp_name, snp_id)

        common_epochs = sorted(set(disc_profile.keys()) & set(repli_profile.keys()))
        if len(common_epochs) < 3:
            print(f"WARNING: {locus_label} has <3 common epochs, skipping")
            continue

        disc_vals = [disc_profile[e] for e in common_epochs]
        repli_vals = [repli_profile[e] for e in common_epochs]

        rho, pval = spearmanr(disc_vals, repli_vals)

        results.append({
            'locus': locus_label,
            'snp': snp_id,
            'spearman_rho': rho,
            'spearman_p': pval,
            'n_epochs': len(common_epochs),
        })

        ax.scatter(disc_vals, repli_vals, c='steelblue', s=30, alpha=0.8)
        ax.set_xlabel('Discovery -log10(P)')
        ax.set_ylabel('Replication -log10(P)')
        ax.set_title(f'{locus_label}\nrho={rho:.2f}, p={pval:.2e}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        print(f"{locus_label}: rho={rho:.3f}, p={pval:.3e}")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_specificity_replication.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'epoch_specificity_replication.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    out_df = pd.DataFrame(results)
    out_df.to_csv(os.path.join(output_dir, 'epoch_specificity_replication.csv'), index=False)
    print(f"\nSaved epoch specificity results to {output_dir}")
    return out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discovery_dir', type=str, required=True)
    parser.add_argument('--replication_dir', type=str, required=True)
    parser.add_argument('--loci_csv', type=str, required=True,
                        help='CSV with columns: locus_id, lead_snp, category')
    parser.add_argument('--output_dir', type=str, default='results/replication')
    parser.add_argument('--step', type=str, default='all',
                        choices=['locus', 'epoch', 'all'])
    args = parser.parse_args()

    gb = GB()

    loci_df = pd.read_csv(args.loci_csv)

    if args.step in ('locus', 'all'):
        locus_replication(
            disc_gwas_dir=args.discovery_dir,
            repli_gwas_dir=args.replication_dir,
            exp_name=gb.experiment_identifier_long,
            loci_df=loci_df,
            output_dir=args.output_dir,
        )

    if args.step in ('epoch', 'all'):
        # key loci with known biology
        key_loci = {
            'MAPT': 'rs8070723',     # fill in your lead SNP
            'MCPH1': 'rs2932538',    # fill in your lead SNP
        }
        epoch_specificity_replication(
            disc_gwas_dir=args.discovery_dir,
            repli_gwas_dir=args.replication_dir,
            exp_name=gb.experiment_identifier_long,
            key_loci=key_loci,
            output_dir=args.output_dir,
        )
