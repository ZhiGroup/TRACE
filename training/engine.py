# imports
import sys
import os
import re
import argparse
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
)
from dataset import *
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
# PyTorch Lightning
# use ealystopping

# Custom imports

# Model architecture and forward pass to Pytorch lightning module.


class engine_AE(pl.LightningModule):
    def __init__(self, lr,
                 weight_decay=1e-5,
                 grdient_log_step=25,
                 normalizer='batchnorm',
                 target_region_weigths=[0.5, 0.5],
                 transition_epochs=10,
                 train_base_model=False,
                 only_tg_region=False,
                 optimizer='adam',
                 use_mse=True,
                 frac_ae_loss=0.2,
                 tot_steps=200,
                 fixed_point=False
                 ):
        super().__init__()
        print(target_region_weigths)
        self.current_epoch_running = 0
        self.weight_decay = weight_decay
        self.grdient_log_step = grdient_log_step
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.tot_steps = tot_steps
        self.frac_ae_loss = frac_ae_loss
        self.fixed_point = fixed_point
        print(f"optimizer : {self.optimizer}")
        if only_tg_region is not True:
            self.initial_wts = target_region_weigths[0]
            self.final_wts = target_region_weigths[1]
        else:
            print('Only target region selected')
            self.initial_wts = [0, 1]
            self.final_wts = [0, 1]
        self.transition_epochs = transition_epochs

        print(
            f"AE received: initial_wts: {self.initial_wts}, final_wts: {self.final_wts}")
        if self.normalizer == 'groupnorm':
            self.num_group = 8
        self.train_base_model = train_base_model
        self.use_mse = use_mse
        self.save_hyperparameters()

        self.is_first_loss_computed = False
        self.hidden_dim = 128  # hidden dimension for latent space used as endophenotype

        # defining layers
        # first CNN block
        self.first_cnn = self.first_CNN_block(1, 16)

        # encoders
        self.first_max_poold = self.max_poold((1, 1, 1))
        self.first_encoder = self.encoder_block(16, 32)
        self.second_max_poold = self.max_poold((0, 1, 0))
        self.second_encoder = self.encoder_block(32, 64)
        self.third_max_poold = self.max_poold((1, 0, 1))
        self.third_encoder = self.encoder_block(64, 128)
        self.fourth_max_poold = self.max_poold((0, 0, 0))
        self.fourth_encoder = self.encoder_block(128, 256)

        # latent space
        self.encoding_mlp = torch.nn.Linear(
            256 * 12 * 14 * 12, self.hidden_dim)
        self.decoding_mlp = torch.nn.Linear(
            self.hidden_dim, 256 * 12 * 14 * 12)

        # decoders
        self.first_decoder = self.decoder_block(256, 128)
        self.first_transconv = self.conv_transpose(
            128, input_padding=(0, 0, 0))
        self.second_decoder = self.decoder_block(128, 64)
        self.second_transconv = self.conv_transpose(
            64, input_padding=(1, 0, 1))
        self.third_decoder = self.decoder_block(64, 32)
        self.third_transconv = self.conv_transpose(32, input_padding=(0, 1, 0))
        self.fourth_decoder = self.decoder_block(32, 16)
        self.fourth_transconv = self.conv_transpose(
            16, input_padding=(1, 1, 1))

        # last CNN block
        self.last_cnn = self.last_CNN_block(16, 1)

        # loss function
        if self.use_mse is True:
            print('MSE loss is selected')
            self.train_loss_function1 = torch.nn.MSELoss(
                size_average=None, reduce=None, reduction="none"
            )
            self.valid_loss_function = torch.nn.MSELoss(
                size_average=None, reduce=None, reduction="none"
            )
        else:
            print('L1 loss is selected')
            self.train_loss_function1 = torch.nn.L1Loss(
                size_average=None, reduce=None, reduction="none"
            )
            self.valid_loss_function = torch.nn.L1Loss(
                size_average=None, reduce=None, reduction="none"
            )

    def on_epoch_end(self):
        self.current_epoch_running += 1

    def max_poold(self, max_padding):
        max_pd = nn.MaxPool3d(kernel_size=2, padding=max_padding)
        return max_pd

    def encoder_block(self, input_channels, output_channels, padding=1):
        encoder = nn.Sequential(
            nn.Conv3d(input_channels, output_channels,
                      kernel_size=3, padding=padding,),
            nn.BatchNorm3d(output_channels) if self.normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, output_channels),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(
                output_channels, output_channels, kernel_size=3, padding=padding,
            ),
            nn.BatchNorm3d(output_channels) if self.normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, output_channels),
            nn.LeakyReLU(inplace=False),
        )
        return encoder

    def conv_transpose(self, output_channels, input_padding):
        conv_t = nn.ConvTranspose3d(
            output_channels,
            output_channels,
            kernel_size=2,
            stride=2,
            padding=input_padding,
        )
        return conv_t

    def decoder_block(self, input_channels, output_channels, input_padding=(0, 0, 0)):
        decoder = nn.Sequential(
            nn.Conv3d(input_channels, output_channels,
                      kernel_size=3, padding=1,),
            nn.BatchNorm3d(output_channels) if self.normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(output_channels, output_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels) if self.normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, output_channels),
            nn.LeakyReLU(inplace=True),
        )
        return decoder

    def first_CNN_block(self, input_channels, output_channels, padding=1):
        cnn_block = nn.Sequential(
            nn.Conv3d(input_channels, output_channels,
                      kernel_size=3, padding=padding,),
            nn.BatchNorm3d(output_channels) if self.normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                output_channels, output_channels, kernel_size=3, padding=padding,
            ),
            nn.BatchNorm3d(output_channels) if self.normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, output_channels),
            nn.LeakyReLU(inplace=True),
        )
        return cnn_block

    def last_CNN_block(self, input_channels, output_channels, padding=1):
        cnn_block = nn.Sequential(
            nn.Conv3d(input_channels, input_channels,
                      kernel_size=3, padding=padding),
            nn.BatchNorm3d(input_channels) if self.normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, input_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(input_channels, input_channels,
                      kernel_size=3, padding=padding),
            nn.BatchNorm3d(input_channels) if self.normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, input_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(input_channels, output_channels, kernel_size=1),
        )
        return cnn_block

    # Forward function
    def forward(self, x):
        x = self.first_cnn(x)        # 1,16,182,218,182
        x = self.first_max_poold(x)  # 1,16,92,110,92
        x = self.first_encoder(x)    # 1,32,92,110,92
        x = self.second_max_poold(x) # 1,32,46,56,46
        x = self.second_encoder(x)   # 1,64,46,56,46
        x = self.third_max_poold(x)  # 1,64,24,28,24
        x = self.third_encoder(x)    # 1,128,24,28,24
        x = self.fourth_max_poold(x) # 1,128,12,14,12
        x = self.fourth_encoder(x)   # 1,256,12,14,12
        shape = x.size()

        # flattening encoder output
        enc_features = torch.flatten(x, start_dim=1, end_dim=-1)
        lin1 = self.encoding_mlp(enc_features)  # 1,128

        # Going from hidden dimension to original image recon
        dec = self.decoding_mlp(lin1)  # 1,516096
        dec = dec.view(shape)          # 1,256,12,14,12

        dec = self.first_decoder(dec)    # 1,128,12,14,12
        dec = self.first_transconv(dec)  # 1,128,24,28,24
        dec = self.second_decoder(dec)   # 1,64,24,28,24
        dec = self.second_transconv(dec) # 1,64,46,56,46
        dec = self.third_decoder(dec)    # 1,32,46,56,46
        dec = self.third_transconv(dec)  # 1,32,92,110,92
        dec = self.fourth_decoder(dec)   # 1,16,92,110,92
        dec = self.fourth_transconv(dec) # 1,16,182,218,182
        recon = self.last_cnn(dec)       # 1, 182, 218, 182

        return recon, lin1

    def calculate_loss(self, loss, mask, fg_mask, epoch, is_train=True):
        threshold_epoch = 0
        fg_wt, tg_wt = self.compute_target_region_weight(epoch)

        if self.train_base_model is True:
            print(f"GB model is being trained")
            fg_mask[fg_mask > 0] = 1
            loss_fg = loss.squeeze(1) * fg_mask
            loss_fg = loss_fg.sum()
            loss_fg = loss_fg / (fg_mask > 0).sum()
            loss_type = 'train_loss_tg' if is_train else 'val_loss_tg'
            self.log(loss_type, loss_fg, prog_bar=True)
            return loss_fg
        else:
            fg_mask[mask > 0] = 0
            loss_fg = loss.squeeze(1) * fg_mask
            loss_fg = loss_fg.sum()
            loss_fg = loss_fg / (fg_mask > 0).sum()
            loss_type = 'train_loss_fg' if is_train else 'val_loss_fg'
            self.log(loss_type, loss_fg, prog_bar=True)

            loss_tr = loss.squeeze(1) * mask
            loss_tr = loss_tr.sum()
            loss_tr = loss_tr / (mask > 0).sum()
            loss_type = 'train_loss_tg' if is_train else 'val_loss_tg'
            self.log(loss_type, loss_tr, prog_bar=True)

            if not self.is_first_loss_computed:
                print(
                    f"self.fg_region_weight: {self.initial_wts[0]}, self.target_region_weigth: {self.initial_wts[1]}")
                print(
                    f"Original AE : foreground loss: {loss_fg}, target_region loss: {loss_tr} ")
                self.is_first_loss_computed = True

            return fg_wt * loss_fg + tg_wt * loss_tr

    def compute_target_region_weight(self, epoch, fixed_point=False):

        if self.fixed_point is True:
            cur_fg_wt = self.final_wts[0]
            cur_tg_wt = self.final_wts[1]
        else:
            transition_epochs = 100
            if epoch < transition_epochs:
                fg_change_rate = (
                    self.final_wts[0] - self.initial_wts[0]) / transition_epochs
                tg_change_rate = (
                    self.final_wts[1] - self.initial_wts[1]) / transition_epochs
                cur_fg_wt = self.initial_wts[0] + fg_change_rate * epoch
                cur_tg_wt = self.initial_wts[1] + tg_change_rate * epoch
            else:
                cur_fg_wt = self.final_wts[0]
                cur_tg_wt = self.final_wts[1]

        self.log("cur_fg_wt", cur_fg_wt, prog_bar=True, sync_dist=False)
        self.log("cur_tg_wt", cur_tg_wt, prog_bar=True, sync_dist=False)

        return cur_fg_wt, cur_tg_wt

    def set_individual_voxel_weight(self, fg_mask, mask, epoch):
        fg_wt, tg_wt = self.compute_target_region_weight(epoch)
        fg_mask = fg_mask.int()
        fg_mask = fg_mask * fg_wt
        mask = mask * tg_wt
        return fg_mask, mask

    # pytorch lightning training step
    def training_step(self, batch, batch_idx):
        x, mask, fg_mask, _ = batch
        recon, _ = self(x)
        loss1 = self.train_loss_function1(x, recon)
        train_loss = self.calculate_loss(
            loss1, mask, fg_mask, self.current_epoch_running, is_train=True)
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss

    # pytorch lightning validation step
    def validation_step(self, batch, batch_idx):
        x, mask, fg_mask, _ = batch
        recon, _ = self(x)
        loss1 = self.valid_loss_function(x, recon)
        val_loss = self.calculate_loss(
            loss1, mask, fg_mask, self.current_epoch_running, is_train=False)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=False)
        return val_loss

    def configure_optimizers(self):

        encoder_params = [
            {"params": self.first_cnn.parameters()},
            {"params": self.first_encoder.parameters()},
            {"params": self.second_encoder.parameters()},
            {"params": self.third_encoder.parameters()},
            {"params": self.fourth_encoder.parameters()},
            {"params": self.encoding_mlp.parameters()},
        ]
        fac = 1
        decoder_params = [
            {"params": self.first_decoder.parameters(),
             "weight_decay": self.weight_decay * fac},
            {"params": self.second_decoder.parameters(),
             "weight_decay": self.weight_decay * fac},
            {"params": self.third_decoder.parameters(),
             "weight_decay": self.weight_decay * fac},
            {"params": self.fourth_decoder.parameters(),
             "weight_decay": self.weight_decay * fac},
            {"params": self.decoding_mlp.parameters(),
             "weight_decay": self.weight_decay * fac},
            {"params": self.first_transconv.parameters(),
             "weight_decay": self.weight_decay * fac},
            {"params": self.second_transconv.parameters(),
             "weight_decay": self.weight_decay * fac},
            {"params": self.third_transconv.parameters(),
             "weight_decay": self.weight_decay * fac},
            {"params": self.fourth_transconv.parameters(),
             "weight_decay": self.weight_decay * fac},
            {"params": self.last_cnn.parameters(),
             "weight_decay": self.weight_decay * fac},
        ]

        if self.optimizer == 'adamw':
            print('using adamw')
            optimizer = torch.optim.AdamW(
                encoder_params + decoder_params, lr=self.hparams["lr"], weight_decay=self.weight_decay)
        else:
            print('using adam')
            optimizer = torch.optim.AdamW(
                encoder_params + decoder_params, lr=self.hparams["lr"], weight_decay=self.weight_decay)

        print(f"total epochs: {self.trainer.max_epochs}")
        print(f"current lr: {self.hparams['lr']}")
        lr_scheduler_config = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.hparams['lr'],
                steps_per_epoch=self.tot_steps,
                epochs=self.trainer.max_epochs,
                pct_start=0.5,
                div_factor=80,
                final_div_factor=1e3,
            ),
            "interval": "step",
            "name": None,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def test_step(self, batch, batch_idx):
        x, mask, fg_mask, _ = batch
        recon, _ = self(x)
        losst = self.train_loss_function1(x, recon)
        fg_mask[mask == 1] = 1
        losst = losst.squeeze(1) * fg_mask
        losst = losst.sum()
        losst = losst / (fg_mask > 0).sum()
        self.log("test_loss", losst, prog_bar=True)
        return losst
