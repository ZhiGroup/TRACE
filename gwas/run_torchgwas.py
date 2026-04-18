import re
import sys
import os
import torch
import logging
import pickle as pkl
import pandas as pd
import numpy as np
from glob import glob
from time import time
from tqdm import tqdm
import threading
import queue
from mpmath import erf, sqrt, mp
mp.dps = 50

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from record_class import GB


def calc_cdf_mpmath(beta_np, precision=50):
    mp.dps = precision
    beta_mp = mp.matrix(beta_np.tolist())
    beta_cdf_mp = mp.matrix(beta_mp.rows, beta_mp.cols,
                             lambda i, j: 0.5 * (1 + erf(beta_mp[i, j] / sqrt(2))))
    return np.array([[float(beta_cdf_mp[i, j]) for j in range(beta_cdf_mp.cols)]
                     for i in range(beta_cdf_mp.rows)])


def calc_p(pheno_normalized, geno, beta, gamma):
    N = pheno_normalized.shape[0]
    with torch.no_grad():
        geno.sub_(geno.mean(1, keepdim=True)).div_(geno.std(1, keepdim=True))
        torch.matmul(geno, pheno_normalized, out=beta)
        beta.div_(N)
        gamma.copy_(beta)
        gamma.pow_(2).sub_(1).div_(2 - N)
        torch.sqrt(gamma, out=gamma)
        beta.div_(gamma).abs_().neg_()
        beta_np = beta.cpu().numpy()
        beta_np = calc_cdf_mpmath(beta_np)
        return beta_np


class GenoLoader:
    """Loads pre-processed genotype data. See README for data preparation."""
    def __init__(self, batchsize=2000, sample_mask=None, geno_path=None):
        if geno_path is None:
            raise ValueError("geno_path must be specified - path to binary genotype file")
        self.data = open(geno_path, 'rb')
        self.batchsize = batchsize
        self.snp_info = None
        self.sample_mask = sample_mask

    def load_snp_info(self, snp_info_path):
        self.snp_info = pd.read_csv(snp_info_path)

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.data.read(self.batchsize * self._snp_bytes)
        if not chunk:
            raise StopIteration
        # parse genotype matrix from binary format
        geno = np.frombuffer(chunk, dtype=np.float32).reshape(self.batchsize, -1)
        if self.sample_mask is not None:
            geno = geno[:, self.sample_mask]
        return geno


def load_embeddings_for_gwas(embed_pkl_path, covar_path=None, qcovar_path=None,
                              sample_mask_path=None):
    """
    Load embeddings and regress out covariates if provided.
    Returns normalized phenotype matrix (n_samples x n_dims).
    """
    with open(embed_pkl_path, 'rb') as f:
        data = pkl.load(f)

    embeddings, pids = data[0][0], data[0][1]
    pids = [p.item() if hasattr(p, 'item') else p for p in pids]

    embed_df = pd.DataFrame(embeddings, index=pids,
                             columns=[f'QT_{i}' for i in range(len(embeddings[0]))])

    if sample_mask_path is not None:
        sample_mask = np.load(sample_mask_path)
        embed_df = embed_df.iloc[sample_mask]

    # z-normalize each dimension
    embed_df = (embed_df - embed_df.mean()) / embed_df.std()

    return embed_df


def run_gwas_epoch(embed_pkl_path, geno_path, snp_info_path, sample_mask_path,
                   covarQ_path, output_dir, epoch, gpu_id=0, batchsize=2000,
                   embed_dim=128):

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f'cuda:{gpu_id}')

    print(f"Loading embeddings for epoch {epoch}...")
    embed_df = load_embeddings_for_gwas(embed_pkl_path, sample_mask_path=sample_mask_path)

    # regress out covariates
    if covarQ_path is not None and os.path.exists(covarQ_path):
        covarQ = np.load(covarQ_path)
        pheno = embed_df.values
        # project out covariates
        covarQ_t = torch.tensor(covarQ, dtype=torch.float32, device=device)
        pheno_t = torch.tensor(pheno, dtype=torch.float32, device=device)
        proj = covarQ_t @ (covarQ_t.T @ pheno_t)
        pheno_t = pheno_t - proj
        pheno = pheno_t.cpu().numpy()
    else:
        pheno = embed_df.values

    n_samples, n_dims = pheno.shape
    print(f"n_samples: {n_samples}, n_dims: {n_dims}")

    # normalize
    pheno_t = torch.tensor(pheno, dtype=torch.float32, device=device)
    pheno_normalized = pheno_t / pheno_t.norm(dim=0, keepdim=True)

    snp_info = pd.read_csv(snp_info_path)
    n_snps = len(snp_info)

    beta_buf = torch.zeros(batchsize, n_dims, device=device)
    gamma_buf = torch.zeros(batchsize, n_dims, device=device)

    all_pvals = []
    loader = GenoLoader(batchsize=batchsize, geno_path=geno_path)

    print(f"Running GWAS for epoch {epoch}...")
    for batch_idx, geno_np in enumerate(tqdm(loader)):
        geno_t = torch.tensor(geno_np, dtype=torch.float32, device=device)
        actual_bs = geno_t.shape[0]
        pvals = calc_p(pheno_normalized, geno_t,
                       beta_buf[:actual_bs], gamma_buf[:actual_bs])
        all_pvals.append(pvals)

    all_pvals = np.vstack(all_pvals)[:n_snps]

    # save results
    out_path = os.path.join(output_dir, f'epoch_{epoch}_gwas.pkl')
    with open(out_path, 'wb') as f:
        pkl.dump({'pvals': all_pvals, 'snp_info': snp_info, 'epoch': epoch}, f)

    print(f"GWAS saved: {out_path}")
    return out_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--geno_path', type=str, required=True,
                        help='path to binary genotype file')
    parser.add_argument('--snp_info', type=str, required=True,
                        help='path to SNP info CSV')
    parser.add_argument('--sample_mask', type=str, default=None)
    parser.add_argument('--covarQ', type=str, default=None)
    parser.add_argument('--batchsize', type=int, default=2000)
    args = parser.parse_args()

    gb = GB()
    weight_dict = gb.get_weight_dict()

    if args.epoch not in weight_dict:
        raise ValueError(f"Epoch {args.epoch} not found in weight dict")

    embed_path = os.path.join(
        gb.embedding_dir,
        f'discovery_WB_T1_128_{gb.experiment_identifier_long}_epoch_{args.epoch}.pkl'
    )

    output_dir = os.path.join(gb.torchgwas_dir, f'{gb.experiment_identifier_long}_epoch_{args.epoch}')

    run_gwas_epoch(
        embed_pkl_path=embed_path,
        geno_path=args.geno_path,
        snp_info_path=args.snp_info,
        sample_mask_path=args.sample_mask,
        covarQ_path=args.covarQ,
        output_dir=output_dir,
        epoch=args.epoch,
        gpu_id=args.gpu,
        batchsize=args.batchsize,
    )
