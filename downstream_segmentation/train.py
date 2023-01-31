"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Script for training a downstream segmentation model using the chesapeake dataset mixed with the synthetic imagery.
Example:
1. For training models with various sizes of synthetic datasets:
python train.py --name <model name> --mix_rate <mix rate choosing from 1.0, 2.0 or 3.0> --gpu_id 0
(Note that the lambda diversity can only be set to 6 by default)

2. For training models with various lambda diversities:
python train.py --name <model name> --mix_rate 1.0 --lambda_diverse <lambda choosing from integers from 0 to 10> --gpu_id 3
(Note that the mix_rate should be fixed to 1.0)

3. For training models with real dataset mixed with various portions of synthetic imagery:
python train.py --name <model name> --mix_rate <mix rate between 0.0 and 1.0> --gpu_id 0
(Note that the lambda diversity can only be set to 6 by default)

Please refer to get_args() in this script to modify other training settings.
"""
import argparse
import logging
import sys

sys.path.append('../torchgeo')
from datamodules import ChesapeakeICLRDataModule
from torchgeo.trainers import SemanticSegmentationTask

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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
    parser.add_argument('--n_channels', type=int, default=4, help='The number of channels of the input image')
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
        # We use layers=["naip-new", "lc"], which is also the default layers of  ChesapeakeCVPRataModule
        mix_rate=args.mix_rate,
        diversity=args.lambda_diverse,
        channels = args.n_channels
    )
    
    task = SemanticSegmentationTask(
        segmentation_model="unet",
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=args.n_channels,
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