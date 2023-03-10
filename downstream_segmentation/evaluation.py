"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Script for running an evaluation script for a downstream segmentation model trained using the training script.
Example:
python evaluation.py --name <name of csv file to save the evaluation metrics, which will be saved in ./evaluation> 
                     --model_path <path to the a trained segmentation model using train.py>

Please refer to get_args() in this script to modify other evaluation settings.
"""
import argparse
import csv
import os
import sys
sys.path.append('../torchgeo')

import pytorch_lightning as pl
import torch
from torchmetrics.classification import Accuracy, JaccardIndex
from torchmetrics import MetricCollection
from torchgeo.trainers import SemanticSegmentationTask
from typing import Any, Dict, Union, cast

from datamodules import ChesapeakeICLRDataModule

def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the IoU of given model and test set')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch size')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Load model from a .ckpt file')
    parser.add_argument('--label_nc', type=int, default=6,
                        help='Number of classes, 6 for lc')
    parser.add_argument('--name', type=str, default='test',
                        help='Set name for saved csv')
    parser.add_argument('--gpu_id', type=int, default=0, help='Set gpu id')
    parser.add_argument('--data_root', type=str,
                        default='../data_chesapeakeiclr', help='Set dataset root')
    parser.add_argument('--nThreads', default=12, type=int, help='# workers')
    parser.add_argument('--seed', default=0, type=int,
                        help="random seed for reproducibility")
    parser.add_argument('--ignore_index', default=0, type=int,
                        help="the paddding index to be ignored")
    parser.add_argument('--n_channels', type=int, default=4,
                        help='The number of channels of the input image')
    return parser.parse_args()

def run_eval_loop(
    model: pl.LightningModule,
    dataloader: Any,
    device: torch.device,
    metrics: MetricCollection,
) -> Any:
    """Runs a standard test loop over a dataloader and records metrics.

    Args:
        model: the model used for inference
        dataloader: the dataloader to get samples from
        device: the device to put data on
        metrics: a torchmetrics compatible metric collection to score the output
            from the model

    Returns:
        the result of ``metrics.compute()``
    """
    for batch in dataloader:
        x = batch["image"].to(device)
        if "mask" in batch:
            y = batch["mask"].to(device)
        elif "label" in batch:
            y = batch["label"].to(device)
        elif "boxes" in batch:
            y = [
                {
                    "boxes": batch["boxes"][i].to(device),
                    "labels": batch["labels"][i].to(device),
                }
                for i in range(len(batch["image"]))
            ]
        with torch.inference_mode():
            y_pred = model(x)
        metrics(y_pred, y)
    results = metrics.compute()
    metrics.reset()
    return results


def main(args: argparse.Namespace) -> None:
    """High-level pipeline.

    Runs a model checkpoint on a test set and saves results to file.

    Args:
        args: command-line arguments
    """
    args.semantic_nc = args.label_nc+1

    dm = ChesapeakeICLRDataModule(
        root=args.data_root,
        num_workers=args.nThreads,
        batch_size=args.batch_size,
        class_set=args.semantic_nc,
        train_splits=["md-train"],
        val_splits=["md-val"],
        test_splits=["md-test"],
        channels=args.n_channels,
    )
    dm.setup("test")

    TASK = SemanticSegmentationTask

    model = TASK.load_from_checkpoint(args.model_path)
    model = cast(pl.LightningModule, model)
    model.freeze()
    model.eval()

    # Record model hyperparameters
    hparams = cast(Dict[str, Union[str, float]], model.hparams)
    # Note we use torchgeo 0.4.0 as the framework for model training, but for torchgeo 0.3.1 the parameter names are different.
    # Therefore we use different names from `hparams`.
    # Here segmentation_model is equivalant to model, encoder_name is equivalant to backbone and encoder_weights is equivalant to weights.
    test_row = {
        "split": "test",
        "segmentation_model": hparams["segmentation_model"],
        "encoder_name": hparams["encoder_name"],
        "encoder_weights": hparams["encoder_weights"],
        "learning_rate": hparams["learning_rate"],
        "loss": hparams["loss"],
    }

    # Compute metrics
    device = torch.device("cuda:%d" % (args.gpu_id))
    model = model.to(device)
    metrics = MetricCollection({"Accuracy": Accuracy(
        num_classes=hparams["num_classes"],
        ignore_index=0,
        mdmc_average="global",
    ),
        # Calculate mean IoU
        # The original code is wrong - modifications made when storing
        "JaccardIndex": JaccardIndex(
        num_classes=hparams["num_classes"],
        ignore_index=args.ignore_index,
    ),
        # labelwise IoU
        "Labelwise_JaccardIndex": JaccardIndex(
        num_classes=hparams["num_classes"],
        ignore_index=args.ignore_index,
        average=None)
    }).to(device)
    test_results = run_eval_loop(model, dm.test_dataloader(), device, metrics)
    test_row.update(
        {
            "overall_accuracy": test_results["Accuracy"].item(),
            "jaccard_index_corrected": test_results["JaccardIndex"].item(),
            "labelwise_jaccard_index": test_results["Labelwise_JaccardIndex"].tolist()
        })

    fieldnames = list(test_row.keys())

    print(test_row)
    # Write to file
    if not os.path.exists('evaluation'):
        os.makedirs('evaluation')
    output_path = "evaluation/"+args.name+".csv"
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    with open(output_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(test_row)


if __name__ == '__main__':
    args = get_args()
    print(args)
    pl.seed_everything(args.seed)
    main(args)
