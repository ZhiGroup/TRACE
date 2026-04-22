# please use affine registered MRI. See README for preprocessing instructions.
# cohort CSV columns: participant_id, t1_mri_path, t1_brain_path

# imports
import torch.nn.functional as F
from scipy.ndimage import binary_erosion
import re
from monai import transforms
import pandas as pd
import nibabel as nib
import torch
import numpy as np
import logging
from scipy.ndimage import binary_dilation, binary_erosion

import config

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

torch.set_num_threads(4)


def get_border_extended_mask_struct(n_voxel=1):

    size = (2*n_voxel) + 1
    struct = np.ones((size, size, size), dtype=bool)

    return struct


def sample_cube(img, fg_mask, cube_size, p, boundary_weight=0.7):
    """
    Sample a random cube from the image (shape: (b, 1, x, y, z)) and ensure it contains at least p% brain voxels.
    Favor boundary voxels by giving them a higher probability of being selected.

    Parameters:
    img (torch.Tensor): The T1-MRI image data with shape (b, 1, x, y, z).
    fg_mask (torch.Tensor): The mask of the same shape as `img` indicating the voxels in the target region.
    cube_size (tuple): The size of the cube to sample (dx, dy, dz).
    p (float): The minimum percentage of brain voxels required in the cube.
    boundary_weight (float): Weight for prioritizing boundary voxels (0 < boundary_weight < 1).

    Returns:
    cube_fg_mask_ret (torch.Tensor): A mask of shape (x, y, z) with 1s in the cube region.
    img_cube (torch.Tensor): The corresponding cube patch from `img` of shape (1, dx, dy, dz).
    start_x, end_x, start_y, end_y, start_z, end_z: The coordinates of the cube.
    """
    if len(img.shape) == 4:
        c, x_dim, y_dim, z_dim = img.shape
    else:
        b, c, x_dim, y_dim, z_dim = img.shape

    if all([x == y for x, y in zip((x_dim, y_dim, z_dim), cube_size)]):
        return 1
    fg_mask = fg_mask.squeeze()

    cube_volume = torch.prod(torch.tensor(cube_size)).item()
    min_brain_voxels = int(p * cube_volume)

    foreground_voxels = torch.nonzero(fg_mask)

    if foreground_voxels.shape[0] == 0:
        raise ValueError("No foreground voxels found in the mask.")

    # Define boundary by dilating the foreground mask and subtracting the original
    kernel = torch.ones((3, 3, 3)).to(fg_mask.device)
    fg_mask_dilated = F.conv3d(fg_mask.unsqueeze(0).unsqueeze(
        0).float(), kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
    boundary_mask = (fg_mask_dilated > 0) & (fg_mask == 0)
    boundary_voxels = torch.nonzero(boundary_mask)

    all_voxels = torch.cat([foreground_voxels, boundary_voxels], dim=0)
    weights = torch.cat([torch.ones(foreground_voxels.shape[0]),
                         torch.ones(boundary_voxels.shape[0]) * boundary_weight])
    weights /= weights.sum()

    while True:
        center_voxel_idx = torch.multinomial(weights, 1).item()
        center_voxel = all_voxels[center_voxel_idx]

        start_x = max(0, center_voxel[0] - cube_size[0] // 2)
        start_y = max(0, center_voxel[1] - cube_size[1] // 2)
        start_z = max(0, center_voxel[2] - cube_size[2] // 2)

        end_x = min(x_dim, start_x + cube_size[0])
        end_y = min(y_dim, start_y + cube_size[1])
        end_z = min(z_dim, start_z + cube_size[2])

        cube_fg_mask = fg_mask[start_x:end_x, start_y:end_y, start_z:end_z]

        brain_voxel_count = torch.sum(cube_fg_mask).item()

        if brain_voxel_count >= min_brain_voxels:
            cube_fg_mask_ret = torch.zeros_like(fg_mask)
            cube_fg_mask_ret[start_x:end_x, start_y:end_y, start_z:end_z] = 1

            img_cube = img[:, start_x:end_x, start_y:end_y, start_z:end_z]
            img_cube = img_cube.type(torch.float)
            return cube_fg_mask_ret, img_cube, start_x, end_x, start_y, end_y, start_z, end_z


first_region_dict = {
    10: 'Left_Thalamus_Proper',
    11: 'Left_Caudate',
    12: 'Left_Putamen',
    13: 'Left_Pallidum',
    16: 'Brain_Stem_or_4th_Ventricle',
    17: 'Left_Hippocampus',
    18: 'Left_Amygdala',
    26: 'Left_Accumbens-area',
    49: 'Right_Thalamus-Proper',
    50: 'Right_Caudate',
    51: 'Right_Putamen',
    52: 'Right_Pallidum',
    53: 'Right_Hippocampus',
    54: 'Right_Amygdala',
    58: 'Right_Accumbens-area'
}

brain_seg_dict = {
    1: 'CSF',
    2: 'GM',
    3: 'WM'
}

brain_seg_dict_rev = {v: k for k, v in brain_seg_dict.items()}
first_region_dict_rev = {v: k for k, v in first_region_dict.items()}


def normalize_non_zero(query_x):
    non_zero_mask = query_x[0, 0] != 0

    mean_non_zero = torch.zeros(
        query_x.shape[0], query_x.shape[1], device=query_x.device)
    std_non_zero = torch.zeros(
        query_x.shape[0], query_x.shape[1], device=query_x.device)

    for b in range(query_x.shape[0]):
        for c in range(query_x.shape[1]):
            non_zero_elements = query_x[b, c][non_zero_mask]
            if non_zero_elements.numel() > 0:
                mean_non_zero[b, c] = non_zero_elements.mean()
                std_non_zero[b, c] = non_zero_elements.std()

    std_non_zero[std_non_zero == 0] = 1

    mean_non_zero = mean_non_zero.view(
        query_x.shape[0], query_x.shape[1], 1, 1, 1)
    std_non_zero = std_non_zero.view(
        query_x.shape[0], query_x.shape[1], 1, 1, 1)

    normalized_query_x = (query_x - mean_non_zero) / std_non_zero
    return normalized_query_x


def remove_border_function(mask, n, reg_id=1):
    csf_mask = mask == reg_id
    csf_boundary = binary_dilation(csf_mask, iterations=n) & ~csf_mask
    return csf_boundary


class aedataset_region_indiv(torch.utils.data.Dataset):
    def __init__(self, datafile, region_id, region_type='wmc', n_neighbours=1,
                 use_dilation=False, normalization='z_norm', only_tg_region=False,
                 ):
        self.datafile = pd.read_csv(datafile)
        self.participant_ids = self.datafile['participant_id']
        self.region_id = region_id
        self.region_type = region_type
        self.n_neighbours = n_neighbours
        self.use_dilation = use_dilation
        self.normalization = normalization
        self.only_tg_region = only_tg_region
        print(f"self.only_tg_region: {self.only_tg_region}")
        print(f"self.region_type == 'wmc': {self.region_type == 'wmc'}")

    def __len__(self):
        return len(self.participant_ids)

    def __getitem__(self, idxx=int):
        row = self.datafile.iloc[idxx].tolist()
        pid, t1_mri_path, t1_brain_path = row[0], row[1], row[2]

        mask = nib.load(t1_brain_path).get_fdata()
        if isinstance(self.region_id, int):
            mask = (mask == self.region_id).astype('int')
        else:
            mask = mask.astype('int')

        img = nib.load(t1_mri_path)
        img = img.get_fdata()

        fg_mask = img != 0

        n_voxels = 2
        structuring_element = np.ones((3, 3, 3), dtype=bool)
        expanded_fg = fg_mask.copy()
        for _ in range(n_voxels):
            expanded_fg = binary_dilation(
                expanded_fg, structure=structuring_element)
        fg_mask = expanded_fg

        if self.only_tg_region is True:
            boolean_mask = mask != 0
            img[~boolean_mask] = 0
            if self.normalization == 'z_norm':
                img = (img - img[img != 0].mean()) / img[img != 0].std()
            else:
                img = (img - img[img != 0].min()) / \
                    (img[img != 0].max() - img[img != 0].min())
        else:
            if self.normalization == 'z_norm':
                img = (img - img[img != 0].mean()) / img[img != 0].std()
            else:
                img = (img - img[img != 0].min()) / \
                    (img[img != 0].max() - img[img != 0].min())

        img = torch.from_numpy(img[np.newaxis, :])
        img = img.type(torch.float)

        if self.use_dilation is True:
            print('dilation used in the sample preparation')
            struct = get_border_extended_mask_struct(n_voxel=self.n_neighbours)
            mask = binary_dilation(mask, structure=struct)
        mask = torch.tensor(mask)

        return img, mask, fg_mask, torch.tensor([pid])


class aedataset_region_indiv_border(torch.utils.data.Dataset):
    def __init__(self, datafile, region_id, region_type='wmc', n_neighbours=1,
                 use_dilation=False, normalization='z_norm', only_tg_region=False,
                 remove_border_function=None, remove_border_by_n_voxel=1
                 ):
        self.datafile = pd.read_csv(datafile)
        self.participant_ids = self.datafile['participant_id']
        self.region_id = region_id
        self.region_type = region_type
        self.n_neighbours = n_neighbours
        self.use_dilation = use_dilation
        self.normalization = normalization
        self.only_tg_region = only_tg_region
        self.remove_border_function = remove_border_function
        self.remove_border_by_n_voxel = remove_border_by_n_voxel
        print(f"self.remove_border_function: {self.remove_border_function}")
        print(f"self.remove_border_by_n_voxel: {self.remove_border_by_n_voxel}")

    def __len__(self):
        return len(self.participant_ids)

    def __getitem__(self, idxx=int):
        row = self.datafile.iloc[idxx].tolist()
        pid, t1_mri_path, t1_brain_path = row[0], row[1], row[2]

        mask = nib.load(t1_brain_path).get_fdata()
        if self.remove_border_function is not None:
            m = self.remove_border_function(
                mask, self.remove_border_by_n_voxel)
        if isinstance(self.region_id, int):
            mask = (mask == self.region_id).astype('int')
        else:
            mask = mask.astype('int')

        img = nib.load(t1_mri_path)
        img = img.get_fdata()
        fg_mask = img != 0

        if self.only_tg_region is True:
            boolean_mask = mask != 0
            img[~boolean_mask] = 0
            if self.normalization == 'z_norm':
                img = (img - img[img != 0].mean()) / img[img != 0].std()
            else:
                img = (img - img[img != 0].min()) / \
                    (img[img != 0].max() - img[img != 0].min())
        else:
            if self.normalization == 'z_norm':
                img = (img - img[img != 0].mean()) / img[img != 0].std()
            else:
                img = (img - img[img != 0].min()) / \
                    (img[img != 0].max() - img[img != 0].min())

        if self.remove_border_function is not None:
            img[m] = np.unique(img[~fg_mask])[0]

        img = torch.from_numpy(img[np.newaxis, :])
        img = img.type(torch.float)

        if self.use_dilation is True:
            struct = get_border_extended_mask_struct(n_voxel=self.n_neighbours)
            mask = binary_dilation(mask, structure=struct)
        mask = torch.tensor(mask)

        return img, mask, fg_mask, torch.tensor([pid])
