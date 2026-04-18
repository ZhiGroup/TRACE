import numpy as np
import pickle as pkl
import os
import re
import sys
import argparse
from glob import glob
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

import torch
from tqdm import tqdm
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, Subset

from nilearn import image as nimg
import nibabel as nib
from nilearn import plotting as nplot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from record_class import GB
from training.engine import engine_AE
from training.dataset import (aedataset_region_indiv, brain_seg_dict_rev,
                               first_region_dict_rev)


# --- Stage 1: Train classifier to distinguish early vs late epochs ---

def get_embed_df(embed_pkl_path):
    with open(embed_pkl_path, 'rb') as f:
        s = pkl.load(f)
    embed_dim = len(s[0][0][0])
    final_array = pd.DataFrame(s[0][0], columns=[f'QT_{i}' for i in range(embed_dim)])
    final_array.index = [x.item() if hasattr(x, 'item') else x for x in s[0][1]]
    return final_array


def get_epoch_embed_paths(embed_dir, exp_name, cohort='discovery'):
    prefix = f'{cohort}_WB_T1_128_{exp_name}_epoch_'
    files = [f for f in os.listdir(embed_dir) if prefix in f]
    files = sorted(files, key=lambda x: int(x.replace(prefix, '').replace('.pkl', '')))
    files = [os.path.join(embed_dir, f) for f in files]
    pattern = re.compile(r'epoch_(\d+)')
    return {int(pattern.search(x).group(1)): x for x in files}


def get_epoch_chkpt_paths(weight_dir, exp_name):
    pattern = re.compile(r'epoch=(\d+)')
    chkpoints = [os.path.join(weight_dir, f) for f in os.listdir(weight_dir)
                 if exp_name in f]
    chkpoints = sorted(chkpoints, key=lambda x: int(pattern.search(x).group(1)))
    return {int(pattern.search(x).group(1)): x for x in chkpoints}


def get_labels_for_locus(locus_id):
    # epoch labels: 0 = early (no signal), 1 = late (signal present)
    if locus_id == 'MAPT':
        negative = {i: 0 for i in range(37)}
        positive = {i: 1 for i in range(60, 84)}
        labels = {**negative, **positive}
    else:
        labels = {}
    return labels


def train_epoch_classifier(locus_id, embed_dir, exp_name, cohort='discovery',
                            n_samples=500, random_state=42):

    epoch_embed_paths = get_epoch_embed_paths(embed_dir, exp_name, cohort)
    labels = get_labels_for_locus(locus_id)

    rng = np.random.default_rng(seed=random_state)
    X, y = [], []
    random_samples_id = []

    for epoch_id in labels:
        label = labels[epoch_id]
        embed_pkl_path = epoch_embed_paths[epoch_id]

        final_array = get_embed_df(embed_pkl_path)

        if len(random_samples_id) == 0:
            random_samples = rng.choice(final_array.index, size=n_samples, replace=False)
            random_samples_id = random_samples

        selected_samples = final_array.loc[random_samples_id]
        X.extend(selected_samples.values.tolist())
        y.extend([label] * len(selected_samples))

    X = np.asarray(X)
    y = np.asarray(y)

    permuted_indices = rng.permutation(X.shape[0])
    X = X[permuted_indices]
    y = y[permuted_indices]

    split_seed = int(rng.integers(2**32 - 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=split_seed, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training classifier...")
    classifier = LogisticRegression(penalty='l1', solver='liblinear', random_state=split_seed)
    classifier.fit(X_train_scaled, y_train)
    print("\n--- Classifier Performance ---")
    print(classification_report(y_test, classifier.predict(X_test_scaled)))

    return classifier, scaler, random_samples_id


# --- Stage 2: Integrated Gradients attribution ---

def build_ig_pipeline(classifier, scaler, model, device):
    """
    Wrap the full pipeline (encoder -> scaler -> classifier) into a single
    differentiable function for Captum IG.
    Returns a callable that takes raw MRI tensor and returns classifier logit.
    """
    import torch.nn as nn

    class FullPipeline(nn.Module):
        def __init__(self, encoder, scaler_mean, scaler_std, clf_weights, clf_intercept):
            super().__init__()
            self.encoder = encoder
            # scaler as fixed linear layer
            self.register_buffer('scaler_mean', torch.tensor(scaler_mean, dtype=torch.float32))
            self.register_buffer('scaler_std', torch.tensor(scaler_std, dtype=torch.float32))
            self.register_buffer('clf_w', torch.tensor(clf_weights, dtype=torch.float32))
            self.register_buffer('clf_b', torch.tensor(clf_intercept, dtype=torch.float32))

        def forward(self, x):
            _, embedding = self.encoder(x)
            embedding_scaled = (embedding - self.scaler_mean) / self.scaler_std
            logit = embedding_scaled @ self.clf_w.T + self.clf_b
            return logit

    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    scaler_std  = torch.tensor(scaler.scale_, dtype=torch.float32)
    clf_weights = classifier.coef_
    clf_intercept = classifier.intercept_

    pipeline = FullPipeline(model, scaler.mean_, scaler.scale_,
                            clf_weights, clf_intercept).to(device)
    return pipeline


def compute_saliency_maps(classifier, scaler, epoch_chkpt_paths, positive_epochs,
                          datafile_path, output_dir, n_subjects=30,
                          random_state=42, batch_size=1, gpu_id=0,
                          n_steps=50, baseline_path=None):
    """
    For each subject (n_subjects total), compute IG saliency maps
    averaged across all positive epoch checkpoints.

    Note: ~30 subjects x 24 positive epoch checkpoints x ~5 min/IG = ~60 GPU hours.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f'cuda:{gpu_id}')

    rng = np.random.default_rng(seed=random_state)
    data_df = pd.read_csv(datafile_path)
    data_indices = rng.choice(len(data_df), n_subjects, replace=False)

    # build background baseline (mean over brainless images)
    print("Building baseline image...")
    baseline_imgs = []
    for idxx in data_indices:
        pid, t1_mri_path, t1_brain_path, t1_seg_path = data_df.iloc[idxx].tolist()
        img = nib.load(t1_mri_path).get_fdata()
        fg_mask = img != 0
        img = (img - img[fg_mask].mean()) / img[fg_mask].std()
        avg_bg = img[~fg_mask].mean()
        brainless = img.copy()
        brainless[fg_mask] = avg_bg
        baseline_imgs.append(brainless)

    baseline_data = np.mean(baseline_imgs, axis=0)
    baseline_tensor = torch.from_numpy(baseline_data).float().unsqueeze(0).unsqueeze(0).to(device)

    dataset = aedataset_region_indiv(
        datafile=datafile_path,
        region_id=1,  # CSF / whole-brain mask
        region_type='wmc',
    )
    subset = Subset(dataset, data_indices.tolist())
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=2)

    print(f"Computing IG for {n_subjects} subjects across {len(positive_epochs)} checkpoints...")

    subject_saliency_maps = {}

    for batch_idx, (img, mask, fg_mask, pids) in enumerate(tqdm(loader)):
        pid = pids[0].item()
        epoch_maps = []

        for epoch, ckpt_path in sorted(epoch_chkpt_paths.items()):
            if epoch not in positive_epochs:
                continue

            model = engine_AE(0.001, normalizer='batchnorm')
            model = model.to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['state_dict'])
            model.eval()

            pipeline = build_ig_pipeline(classifier, scaler, model, device)
            pipeline.eval()

            img_t = img.to(device).requires_grad_(True)

            ig = IntegratedGradients(pipeline)
            attr = ig.attribute(img_t, baselines=baseline_tensor,
                                target=0,  # positive class
                                n_steps=n_steps,
                                internal_batch_size=1)

            epoch_maps.append(attr.squeeze().cpu().detach().numpy())

        if len(epoch_maps) > 0:
            avg_map = np.mean(epoch_maps, axis=0)
            subject_saliency_maps[pid] = avg_map

            # save individual map
            ref_img = nib.load(data_df.iloc[data_indices[batch_idx]]['t1_mri_path'])
            saliency_nii = nib.Nifti1Image(avg_map, affine=ref_img.affine)
            out_path = os.path.join(output_dir, f'saliency_map_pid_{pid}.nii.gz')
            nib.save(saliency_nii, out_path)

    print(f"Saved {len(subject_saliency_maps)} saliency maps to {output_dir}")
    return subject_saliency_maps


# --- Stage 3: Group-level t-test ---

def compute_group_saliency(saliency_dir, output_dir, alpha_fdr=0.05):
    """
    Stack individual saliency maps and run voxel-wise one-sample t-test.
    FDR correction at alpha_fdr.
    """
    os.makedirs(output_dir, exist_ok=True)

    maps = []
    ref_img = None
    for f in sorted(glob(os.path.join(saliency_dir, 'saliency_map_*.nii.gz'))):
        img = nib.load(f)
        maps.append(img.get_fdata())
        if ref_img is None:
            ref_img = img

    maps = np.stack(maps, axis=0)  # (n_subjects, x, y, z)
    print(f"Stacked {maps.shape[0]} saliency maps, shape: {maps.shape[1:]}")

    shape = maps.shape[1:]
    flat = maps.reshape(maps.shape[0], -1)

    t_stats, p_vals = ttest_1samp(flat, popmean=0, axis=0)

    reject, p_fdr = fdrcorrection(p_vals, alpha=alpha_fdr)
    print(f"Significant voxels after FDR correction: {reject.sum()}")

    t_map = t_stats.reshape(shape)
    p_map = p_fdr.reshape(shape)
    sig_map = t_map.copy()
    sig_map[~reject.reshape(shape)] = 0

    for name, data in [('t_statistics', t_map), ('pvals_fdr', p_map),
                        ('t_statistics_fdr_thresholded', sig_map)]:
        out = nib.Nifti1Image(data, affine=ref_img.affine)
        nib.save(out, os.path.join(output_dir, f'MAPT_{name}.nii.gz'))

    print(f"Group saliency maps saved to {output_dir}")
    return sig_map, ref_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--locus', type=str, default='MAPT')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_subjects', type=int, default=30)
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='results/locus_mapping/MAPT')
    parser.add_argument('--step', type=str, default='all',
                        choices=['classify', 'ig', 'group', 'all'])
    args = parser.parse_args()

    gb = GB()

    # positive epochs for MAPT = epochs 60-83
    positive_epochs = set(range(60, 84))
    epoch_chkpt_paths = get_epoch_chkpt_paths(gb.weight_dir, gb.experiment_identifier_long)

    if args.step in ('classify', 'all'):
        classifier, scaler, sample_ids = train_epoch_classifier(
            locus_id=args.locus,
            embed_dir=gb.embedding_dir,
            exp_name=gb.experiment_identifier_long,
            cohort='discovery',
            random_state=42
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'{args.locus}_classifier.pkl'), 'wb') as f:
            pkl.dump([classifier, scaler, sample_ids], f)
        print(f"Classifier saved.")

    if args.step in ('ig', 'all'):
        with open(os.path.join(args.output_dir, f'{args.locus}_classifier.pkl'), 'rb') as f:
            classifier, scaler, sample_ids = pkl.load(f)

        compute_saliency_maps(
            classifier=classifier,
            scaler=scaler,
            epoch_chkpt_paths=epoch_chkpt_paths,
            positive_epochs=positive_epochs,
            datafile_path=config.disc_path,
            output_dir=args.output_dir,
            n_subjects=args.n_subjects,
            gpu_id=args.gpu,
            n_steps=args.n_steps,
        )

    if args.step in ('group', 'all'):
        compute_group_saliency(
            saliency_dir=args.output_dir,
            output_dir=args.output_dir,
        )
