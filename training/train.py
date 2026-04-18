import sys
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training.engine import engine_AE
from training.dataset import aedataset_region_indiv, brain_seg_dict_rev, first_region_dict_rev


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--normalizer', type=str, default='groupnorm',
                        choices=['batchnorm', 'groupnorm'])
    parser.add_argument('--region', type=str, default='WB',
                        help='brain region to use: WB, CSF, GM, WM, or subcortical region name')
    parser.add_argument('--target_wt', type=float, nargs=2, default=[0.5, 0.5],
                        help='initial and final target region weights')
    parser.add_argument('--exp_name', type=str,
                        default='basemodel-CSF_wd_1e-5_norma_z_norm_lr_0.0001_10_GB_scan')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--wandb', action='store_true',
                        help='use wandb logger (set WANDB_API_KEY env var)')
    return parser.parse_args()


def main():
    args = get_args()

    gpu_ids = [int(g) for g in args.gpus.split(',')]

    region_name = args.region
    region_id = first_region_dict_rev.get(region_name, 'subcort') if region_name not in [
        'CSF', 'GM', 'WM', 'WB'] else brain_seg_dict_rev.get(region_name, 'WB')
    region_type = 'subc' if region_name not in [
        'CSF', 'GM', 'WM', 'WB'] else 'wmc'

    print(f"region_name: {region_name}, region_id: {region_id}, region_type: {region_type}")

    train_dataset = aedataset_region_indiv(
        datafile=config.train_datafile,
        region_id=region_id,
        region_type=region_type,
        normalization='z_norm',
    )

    val_dataset = aedataset_region_indiv(
        datafile=config.val_datafile,
        region_id=region_id,
        region_type=region_type,
        normalization='z_norm',
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    tot_steps = len(train_loader)
    print(f"Steps per epoch: {tot_steps}")

    model = engine_AE(
        lr=args.lr,
        weight_decay=args.weight_decay,
        normalizer=args.normalizer,
        target_region_weigths=args.target_wt,
        optimizer='adamw',
        use_mse=True,
        tot_steps=tot_steps,
    )

    save_dir = args.save_dir or config.model_log_dir
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename=f'{args.exp_name}' + 'epoch={epoch}',
        save_top_k=-1,  # save all checkpoints
        every_n_epochs=1,
        monitor='val_loss',
        mode='min',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    loggers = [CSVLogger(save_dir=save_dir, name='csv_logs')]
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(project='TRACE', name=args.exp_name)
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=gpu_ids,
        callbacks=[checkpoint_cb, lr_monitor],
        logger=loggers,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
