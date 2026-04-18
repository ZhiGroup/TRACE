"""
IDP prediction analysis.

For each epoch embedding, predict 139 regional brain volumes (IDPs)
from the 128-dim latent space using ridge regression.
Reports mean R2 across IDPs per epoch.

The IDP phenotype CSV should have columns:
  participant_id, <idp_field_1>, <idp_field_2>, ...

IDP field IDs from UK Biobank imaging phenotypes:
  subcortical volumes: 25011-25025
  regional volumes: depends on your atlas
"""

import os
import sys
import re
import pickle as pkl
import numpy as np
import pandas as pd
from glob import glob
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from record_class import GB


def get_last_visit_columns(data_field_ids: List[int], column_names: List[str]) -> Dict[int, str]:
    """
    Selects the column name corresponding to the last visit for each data field ID.
    Useful for UK Biobank data where columns are named <field_id>-<visit>.<instance>
    """
    selected_columns = {}
    pattern = re.compile(
        r'^(?P<field_id>\d+)-(?P<visit_number>\d+)\.\d+$'
    )

    for field_id in data_field_ids:
        matching_columns = [col for col in column_names if re.match(
            r'^' + str(field_id) + r'-.*', col)]

        if not matching_columns:
            if str(field_id) in column_names:
                selected_columns[field_id] = str(field_id)
            else:
                print(f"Warning: No column found for data field ID {field_id}")
            continue

        last_visit_col = None
        last_visit_num = -1

        for col in matching_columns:
            match = pattern.match(col)
            if match:
                visit_num = int(match.group('visit_number'))
                if visit_num > last_visit_num:
                    last_visit_num = visit_num
                    last_visit_col = col

        if last_visit_col:
            selected_columns[field_id] = last_visit_col
        else:
            print(f"Warning: Could not determine last visit for data field ID {field_id}.")
            if str(field_id) in column_names:
                selected_columns[field_id] = str(field_id)

    return selected_columns


def load_embeddings(embed_pkl_path):
    with open(embed_pkl_path, 'rb') as f:
        data = pkl.load(f)
    embeddings, pids = data[0][0], data[0][1]
    pids = [p.item() if hasattr(p, 'item') else p for p in pids]
    embed_df = pd.DataFrame(embeddings, index=pids,
                             columns=[f'QT_{i}' for i in range(len(embeddings[0]))])
    return embed_df


def load_idp_phenotypes(idp_csv_path, idp_field_ids=None):
    """
    Load IDP phenotypes. If idp_field_ids is provided, selects the last-visit
    column for each field (UKB column naming convention).
    Otherwise expects a simple CSV with participant_id as index.
    """
    df = pd.read_csv(idp_csv_path, index_col=0)

    if idp_field_ids is not None:
        col_map = get_last_visit_columns(idp_field_ids, df.columns.tolist())
        df = df[list(col_map.values())]
        df.columns = [f'field_{k}' for k in col_map.keys()]

    return df


def predict_idps_for_epoch(embed_df, idp_df, cv=5):
    """
    For one epoch embedding, predict all IDPs with ridge regression.
    Returns dict of {idp_col: mean_cv_r2}.
    """
    # align samples
    common_ids = embed_df.index.intersection(idp_df.index)
    X = embed_df.loc[common_ids].values
    Y = idp_df.loc[common_ids]

    # drop IDPs with too many NaN
    Y = Y.dropna(axis=1, thresh=int(0.8 * len(common_ids)))
    Y = Y.fillna(Y.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    for col in Y.columns:
        y = Y[col].values
        # skip constant IDPs
        if y.std() < 1e-6:
            continue
        model = RidgeCV(alphas=alphas, cv=cv)
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        results[col] = scores.mean()

    return results


def run_idp_prediction_all_epochs(embed_dir, exp_name, idp_csv_path,
                                   output_dir, cohort='discovery',
                                   idp_field_ids=None):
    os.makedirs(output_dir, exist_ok=True)

    prefix = f'{cohort}_WB_T1_128_{exp_name}_epoch_'
    files = [f for f in os.listdir(embed_dir) if prefix in f]
    files = sorted(files, key=lambda x: int(x.replace(prefix, '').replace('.pkl', '')))

    pattern = re.compile(r'epoch_(\d+)')
    epoch_embed_paths = {int(pattern.search(f).group(1)): os.path.join(embed_dir, f)
                         for f in files}

    idp_df = load_idp_phenotypes(idp_csv_path, idp_field_ids)
    print(f"Loaded {idp_df.shape[1]} IDP phenotypes, {idp_df.shape[0]} samples")

    all_results = {}
    for epoch, path in sorted(epoch_embed_paths.items()):
        print(f"\nEpoch {epoch}...")
        embed_df = load_embeddings(path)
        r2_dict = predict_idps_for_epoch(embed_df, idp_df)
        all_results[epoch] = r2_dict
        mean_r2 = np.mean(list(r2_dict.values()))
        print(f"  mean R2: {mean_r2:.4f} across {len(r2_dict)} IDPs")

    # summary
    summary = []
    for epoch, r2_dict in all_results.items():
        vals = list(r2_dict.values())
        summary.append({
            'epoch': epoch,
            'mean_r2': np.mean(vals),
            'median_r2': np.median(vals),
            'n_idps': len(vals),
        })
    summary_df = pd.DataFrame(summary).sort_values('epoch')

    out_csv = os.path.join(output_dir, 'idp_prediction_by_epoch.csv')
    summary_df.to_csv(out_csv, index=False)

    # save full r2 dict
    with open(os.path.join(output_dir, 'idp_r2_full.pkl'), 'wb') as f:
        pkl.dump(all_results, f)

    print(f"\nSaved: {out_csv}")
    return summary_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--idp_csv', type=str, required=True,
                        help='CSV with IDP phenotypes, participant_id as first column')
    parser.add_argument('--output_dir', type=str, default='results/idp_prediction')
    parser.add_argument('--cohort', type=str, default='discovery')
    args = parser.parse_args()

    gb = GB()

    # subcortical IDP field IDs - adjust to your phenotype CSV format
    idp_field_ids = [
        25023, 25024, 25021, 25022, 25013, 25014,
        25019, 25020, 25017, 25018, 25015, 25016,
        25011, 25012, 25003, 25025
    ]

    run_idp_prediction_all_epochs(
        embed_dir=gb.embedding_dir,
        exp_name=gb.experiment_identifier_long,
        idp_csv_path=args.idp_csv,
        output_dir=args.output_dir,
        cohort=args.cohort,
        idp_field_ids=idp_field_ids,
    )
