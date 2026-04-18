# PyTorch

from functools import partial
import re
import argparse
import sys
import os
import nibabel
from monai import transforms
from multiprocessing import Pool
import pickle as pkl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from glob import glob
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.nn import functional as F
from joblib import Parallel, delayed
import torch
import importlib
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training.engine import engine_AE
from training.dataset import (aedataset_region_indiv, aedataset_region_indiv_border,
                               brain_seg_dict_rev, first_region_dict_rev)
from record_class import GB

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def clear_cache():
    torch.cuda.empty_cache()


def extract_embedding_region_cohort(region_name, checkpoint_path, datafile_path,
                                    gpu, cohort, batch_size, region_id,
                                    region_type, embed_dim=128):

    train_dataset = aedataset_region_indiv(
        datafile=datafile_path,
        region_id=region_id,
        region_type=region_type
    )

    T1_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )

    embedd_save_dir = config.embedding_dir

    T1_ckpt = checkpoint_path
    device = torch.device(gpu)
    model_T1 = engine_AE(0.001, normalizer='groupnorm')
    model_T1 = model_T1.to(device)
    checkpoint = torch.load(T1_ckpt, map_location=device)
    model_T1.load_state_dict(checkpoint["state_dict"])

    bottle_neck_T1 = []
    img_names_T1 = []
    model_T1 = model_T1.eval()
    with torch.no_grad():
        for data in tqdm(T1_dataloader, total=len(T1_dataloader)):
            img, mask, fg_mask, pids = data
            img = img.to(device)
            recon, lin1 = model_T1(img)
            bottle_neck_T1.extend(lin1.cpu().numpy())
            img_names_T1.extend(pids)

    save_path = f"{cohort}_{region_name}_T1_{embed_dim}.pkl"
    with open(os.path.join(embedd_save_dir, save_path), 'wb') as f:
        pkl.dump([bottle_neck_T1, img_names_T1], f)

    print(f"Done for embedding extraction for {cohort} cohort for {region_name}")
    print(f"saved in {os.path.join(embedd_save_dir, save_path)}")


def extract_embeddings_for_epoch(checkpoint_path, datafile_path, gpu, cohort,
                                 batch_size, region_id, region_type,
                                 embed_dim=128, normalizer='groupnorm',
                                 data_indices=None, exp_name=None, epoch=None):
    clear_cache()

    train_dataset = aedataset_region_indiv(
        datafile=datafile_path,
        region_id=region_id,
        region_type=region_type
    )

    if data_indices is not None:
        train_dataset = Subset(train_dataset, data_indices)

    T1_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=10, shuffle=False
    )

    embedd_save_dir = config.embedding_dir
    os.makedirs(embedd_save_dir, exist_ok=True)

    device = torch.device(gpu)
    model_T1 = engine_AE(0.001, normalizer=normalizer)
    model_T1 = model_T1.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_T1.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded {checkpoint_path}")

    bottle_neck_T1 = []
    img_names_T1 = []
    model_T1 = model_T1.eval()
    with torch.no_grad():
        print('Extraction Started')
        for data in tqdm(T1_dataloader, total=len(T1_dataloader)):
            img, mask, fg_mask, pids = data
            img = img.to(device).contiguous()
            recon, lin1 = model_T1(img)
            bottle_neck_T1.extend(lin1.cpu().numpy())
            img_names_T1.extend(pids)

    # save per-epoch per-cohort embedding
    region_name = 'WB'
    if exp_name is not None and epoch is not None:
        save_path = f"{cohort}_{region_name}_T1_{embed_dim}_{exp_name}_epoch_{epoch}.pkl"
    else:
        save_path = f"{cohort}_{region_name}_T1_{embed_dim}.pkl"

    full_path = os.path.join(embedd_save_dir, save_path)
    with open(full_path, 'wb') as f:
        pkl.dump([[bottle_neck_T1, img_names_T1]], f)

    print(f"Saved: {full_path}")
    return full_path


def run_extraction_all_epochs(args):
    gb = GB()
    weight_dict = gb.get_weight_dict()

    region_name = args.region
    region_id = first_region_dict_rev.get(region_name, 'subcort') if region_name not in [
        'CSF', 'GM', 'WM', 'WB'] else brain_seg_dict_rev.get(region_name, 'WB')
    region_type = 'subc' if region_name not in [
        'CSF', 'GM', 'WM', 'WB'] else 'wmc'

    datafile = config.disc_path if args.cohort == 'discovery' else config.rep_path

    for epoch, ckpt_path in sorted(weight_dict.items()):
        print(f"\n--- Epoch {epoch} ---")
        extract_embeddings_for_epoch(
            checkpoint_path=ckpt_path,
            datafile_path=datafile,
            gpu=f'cuda:{args.gpu}',
            cohort=args.cohort,
            batch_size=args.batch_size,
            region_id=region_id,
            region_type=region_type,
            normalizer=args.normalizer,
            exp_name=gb.experiment_identifier_long,
            epoch=epoch
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--cohort', type=str, default='discovery',
                        choices=['discovery', 'replication'])
    parser.add_argument('--region', type=str, default='WB')
    parser.add_argument('--normalizer', type=str, default='groupnorm')
    args = parser.parse_args()

    run_extraction_all_epochs(args)
