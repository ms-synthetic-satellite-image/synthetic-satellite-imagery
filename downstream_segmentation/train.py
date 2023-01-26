import os
import argparse
import logging

import torchgeo
from torchgeo.datasets import ChesapeakeICLR
from torchgeo.datamodules import ChesapeakeICLRDataModule, ChesapeakeCVPRDataModule
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.samplers import RandomGeoSampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import matplotlib.pyplot as plt
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--min_epochs', type=int, default=10, help='Min number of epochs')
    parser.add_argument('--max_epochs', type=int, default=200, help='Max number of epochs')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')

    parser.add_argument('--label_nc', '-nc', metavar='NC', type=int, default=6, help='Number of classes, 6 for lc')
    parser.add_argument('--nThreads', default=12, type=int, help='# workers')
    parser.add_argument('--mix_rate', type=float, default=0.0, help='Portion of synthetic images in the training dataset.')
    parser.add_argument('--lambda_diverse', type=int, default=6, help='Set diversity lambda')
    parser.add_argument('--name', type=str, default='test', help = 'Set experiment name')
    parser.add_argument('--patience', type=int, default=10, help = 'Set patience of scheduler')
    parser.add_argument('--gpu_id', type=int, default=0, help = 'Set gpu id') 
    parser.add_argument('--data_root', type= str, default='../data_chesapeakeiclr', help = 'Set dataset root')
    parser.add_argument('--patches_per_tile', type = int, default=200, help = 'Set number of patches we get per tile')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # define some experiment parameters

    experiment_dir = "results/"+args.name
    experiment_name = args.name

    # Set semantic_nc based on the option.
    # This will be convenient in many places
    args.semantic_nc = args.label_nc+1

    datamodule = ChesapeakeICLRDataModule(
        root=args.data_root,
        num_workers=args.nThreads,
        batch_size=args.batch_size,
        class_set=args.semantic_nc,
        train_splits=["md-train"],
        val_splits=["md-val"],
        test_splits=["md-test"],
        patches_per_tile = args.patches_per_tile,
        # layers=["naip-new", "lc"],
        mix_rate=args.mix_rate,
        diversity=args.lambda_diverse
    )
    
    task = SemanticSegmentationTask(
        model="unet",
        backbone="resnet50",
        weights="imagenet",
        in_channels=3,
        num_classes=args.semantic_nc,
        loss="jaccard",
        ignore_index=None,
        learning_rate=args.learning_rate,
        learning_rate_schedule_patience=args.patience
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=experiment_dir,
        save_top_k=1,
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
    )

    tb_logger = TensorBoardLogger(
        save_dir="logs/",
        name=experiment_name
    )


    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=[tb_logger],
        default_root_dir=experiment_dir,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=[args.gpu_id]
    )
    _ = trainer.fit(model=task, datamodule=datamodule)