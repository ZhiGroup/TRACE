"""
Estimate SNP heritability (h2) for informative epochs using GCTA --model-only.

Runs fastGWA-mlm --model-only which fits the mixed model variance components
(V(G), V(e)) WITHOUT scanning individual SNPs. Much faster than full GWAS.

Output: .log files with h2 = Vg/Vp per dimension per epoch.
~19 sec per dimension. Total: 13 epochs x 128 dims = 1,664 jobs.

Usage:
    python heritability.py --step prepare
    python heritability.py --step run --parallel 8
    python heritability.py --step summarize
    python heritability.py --step all --parallel 8
"""

import os
import sys
import pickle as pkl
import pandas as pd
import numpy as np
from multiprocessing import Pool
from glob import glob
import argparse
import re
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from record_class import GB

# ============================================================
# paths - edit these for your system
# ============================================================
informative_epochs = [0, 1, 2, 3, 4, 6, 7, 8, 9, 18, 37, 38, 46]
n_dimensions = 128
cohort_type = 'discovery'

gb = GB()
embedding_dir = gb.embedding_dir
base_output_dir = os.path.join(gb.RESULT_DIR, 'h2_model_only')

# GCTA binary - download from https://yanglab.westlake.edu.cn/software/gcta/
gcta_bin = '/path/to/gcta/gcta64'

# genetic data paths
bgen_file    = '/path/to/your/genotypes.bgen'
sample_file  = '/path/to/your/genotypes.sample'
grm_file     = '/path/to/your/grm/prefix'          # output prefix from gcta --make-grm
covar_file   = f'/path/to/your/covariates/ccovar_{cohort_type}'
qcovar_file  = f'/path/to/your/covariates/qcovar_{cohort_type}'

pheno_dir   = os.path.join(base_output_dir, 'phenotypes')
log_dir     = os.path.join(base_output_dir, 'logs')
summary_dir = os.path.join(base_output_dir, 'summary')


# ============================================================
# Step 1: Prepare phenotype files
# ============================================================
def load_embeddings(epoch):
    embedding_file = os.path.join(
        embedding_dir,
        f'discovery_WB_T1_128_{gb.experiment_identifier_long}_epoch_{epoch}.pkl'
    )
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Embedding file not found: {embedding_file}")

    with open(embedding_file, 'rb') as f:
        data = pkl.load(f)

    phenos, pids = [], []
    for p, pid in data:
        phenos.extend(p)
        pids.extend(pid)
    pids = [i.item() if hasattr(i, 'item') else i for i in pids]

    embed_dict = dict(zip(pids, phenos))
    df = pd.DataFrame.from_dict(embed_dict, orient='index')
    return df


def prepare_phenotype_files(epoch):
    df = load_embeddings(epoch)
    epoch_dir = os.path.join(pheno_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    for dim in range(n_dimensions):
        pheno_df = pd.DataFrame({
            'FID': df.index,
            'IID': df.index,
            f'QT{dim}': df.iloc[:, dim]
        })
        pheno_file = os.path.join(epoch_dir, f'QT{dim}.pheno')
        pheno_df.to_csv(pheno_file, sep='\t', index=False, header=False)

    print(f"[Epoch {epoch}] Prepared {n_dimensions} phenotype files "
          f"({df.shape[0]} samples)")
    return epoch


def step_prepare():
    os.makedirs(pheno_dir, exist_ok=True)

    for epoch in informative_epochs:
        emb_file = os.path.join(
            embedding_dir,
            f'discovery_WB_T1_128_{gb.experiment_identifier_long}_epoch_{epoch}.pkl'
        )
        if not os.path.exists(emb_file):
            print(f"WARNING: Missing embedding for epoch {epoch}: {emb_file}")

    with Pool(min(len(informative_epochs), 8)) as p:
        p.map(prepare_phenotype_files, informative_epochs)

    total = len(informative_epochs) * n_dimensions
    print(f"\nDone. Created {total} phenotype files in {pheno_dir}")


# ============================================================
# Step 2: Run GCTA --model-only
# ============================================================
def run_model_only_single(args):
    epoch, dim = args
    pheno_file = os.path.join(pheno_dir, f'epoch_{epoch}', f'QT{dim}.pheno')
    out_prefix = os.path.join(log_dir, f'epoch_{epoch}', f'QT{dim}')
    log_file = f'{out_prefix}.log'

    if os.path.exists(log_file):
        return epoch, dim, 'EXISTS'

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    cmd = (
        f"{gcta_bin} "
        f"--bgen {bgen_file} "
        f"--sample {sample_file} "
        f"--grm {grm_file} "
        f"--pheno {pheno_file} "
        f"--covar {covar_file} "
        f"--qcovar {qcovar_file} "
        f"--fastGWA-mlm --model-only "
        f"--out {out_prefix}"
    )

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"ERROR epoch={epoch} dim={dim}: {result.stderr[:200]}")
            return epoch, dim, 'ERROR'
        return epoch, dim, 'OK'
    except subprocess.TimeoutExpired:
        return epoch, dim, 'TIMEOUT'


def step_run(n_parallel=8):
    os.makedirs(log_dir, exist_ok=True)

    jobs = [(epoch, dim)
            for epoch in informative_epochs
            for dim in range(n_dimensions)]

    print(f"Running {len(jobs)} GCTA jobs with {n_parallel} parallel workers...")
    with Pool(n_parallel) as p:
        results = p.map(run_model_only_single, jobs)

    ok = sum(1 for _, _, s in results if s == 'OK')
    exists = sum(1 for _, _, s in results if s == 'EXISTS')
    errors = sum(1 for _, _, s in results if s == 'ERROR')
    print(f"OK: {ok}, EXISTS: {exists}, ERRORS: {errors}")


# ============================================================
# Step 3: Parse log files and summarize h2
# ============================================================
def parse_h2_from_log(log_file):
    if not os.path.exists(log_file):
        return None, None

    h2, se = None, None
    with open(log_file, 'r') as f:
        for line in f:
            if 'V(G)/Vp' in line:
                parts = line.strip().split()
                try:
                    h2 = float(parts[1])
                    se = float(parts[2])
                except (IndexError, ValueError):
                    pass
    return h2, se


def step_summarize():
    os.makedirs(summary_dir, exist_ok=True)

    records = []
    for epoch in informative_epochs:
        for dim in range(n_dimensions):
            log_file = os.path.join(log_dir, f'epoch_{epoch}', f'QT{dim}.log')
            h2, se = parse_h2_from_log(log_file)
            records.append({'epoch': epoch, 'dim': dim, 'h2': h2, 'h2_se': se})

    df = pd.DataFrame(records)
    df_valid = df.dropna(subset=['h2'])

    print(f"\nParsed {len(df_valid)}/{len(df)} log files successfully")
    print(f"\nMean h2 per epoch:")
    print(df_valid.groupby('epoch')['h2'].agg(['mean', 'std', 'count']))

    out_path = os.path.join(summary_dir, 'h2_summary.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, required=True,
                        choices=['prepare', 'run', 'summarize', 'all'])
    parser.add_argument('--parallel', type=int, default=8)
    args = parser.parse_args()

    if args.step in ('prepare', 'all'):
        step_prepare()
    if args.step in ('run', 'all'):
        step_run(n_parallel=args.parallel)
    if args.step in ('summarize', 'all'):
        step_summarize()
