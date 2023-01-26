import argparse
import csv
import os
from typing import Any, Dict, Union, cast

import torchgeo
from torchgeo.datamodules import ChesapeakeCVPRDataModule
from torchgeo.trainers import SemanticSegmentationTask
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex, MulticlassAccuracy, MulticlassJaccardIndex

import pytorch_lightning as pl
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the IoU of given model and test set')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    parser.add_argument('--model_path', type=str, default=False, help='Load model from a .ckpt file')
    parser.add_argument('--label_nc', '-nc', metavar='NC', type=int, default=6, help='Number of classes, 6 for lc')
    parser.add_argument('--name', type=str, default='test', help = 'Set experiment name')
    parser.add_argument('--gpu_id', type=int, default=0, help = 'Set gpu id') 
    parser.add_argument('--data_root', type=str, default='../data_chesapeakecvpr', help = 'Set dataset root')
    parser.add_argument('--task', type=str, default='segmentation', help = 'Choose either segmentation (6-label landcover) or classification (building non-building)')
    parser.add_argument('--nThreads', default=12, type=int, help='# workers')
    parser.add_argument('--seed', default=0, type=int, help="random seed for reproducibility")
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
            print(x.shape)
            print(y_pred.shape)
            return "debug"
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

    dm = ChesapeakeCVPRDataModule(
        root=args.data_root,
        num_workers=args.nThreads,
        batch_size=args.batch_size,
        class_set=args.semantic_nc,
        train_splits=["md-train"],
        val_splits=["md-val"],
        test_splits=["md-test"]
        # patches_per_tile = args.patches_per_tile
    )
    dm.setup("test")

    TASK = SemanticSegmentationTask

    model = TASK.load_from_checkpoint(args.model_path)
    model = cast(pl.LightningModule, model)
    model.freeze()
    model.eval()

    # Record model hyperparameters
    hparams = cast(Dict[str, Union[str, float]], model.hparams)
    # if issubclass(TASK, ClassificationTask):
    #     # val_row = {
    #     #     "split": "val",
    #     #     "model": hparams["model"],
    #     #     "learning_rate": hparams["learning_rate"],
    #     #     "weights": hparams["weights"],
    #     #     "loss": hparams["loss"],
    #     # }
    #     test_row = {
    #         "split": "test",
    #         "model": hparams["model"],
    #         "learning_rate": hparams["learning_rate"],
    #         "weights": hparams["weights"],
    #         "loss": hparams["loss"],
    #     }
    if issubclass(TASK, SemanticSegmentationTask):
        # val_row = {
        #     "split": "val",
        #     "model": hparams["model"],
        #     "backbone": hparams["backbone"],
        #     "weights": hparams["weights"],
        #     "learning_rate": hparams["learning_rate"],
        #     "loss": hparams["loss"],
        # }
        test_row = {
            "split": "test",
            "model": hparams["model"],
            "backbone": hparams["backbone"],
            "weights": hparams["weights"],
            "learning_rate": hparams["learning_rate"],
            "loss": hparams["loss"],
        }

    # Compute metrics
    device = torch.device("cuda:%d" % (args.gpu_id))
    model = model.to(device)
    if issubclass(TASK, SemanticSegmentationTask):
        metrics = MetricCollection({"Accuracy": MulticlassAccuracy(
                                                num_classes=hparams["num_classes"],
                                                ignore_index=None,
                                                mdmc_average="global",
                                                ),
                                    # mean IoU
                                    "JaccardIndex": MulticlassJaccardIndex(
                                                    num_classes=hparams["num_classes"],
                                                    ignore_index=None,
                                                    ),
                                    # labelwise IoU
                                    "Labelwise_JaccardIndex": MulticlassJaccardIndex(
                                                              num_classes=hparams["num_classes"],
                                                              ignore_index=None,
                                                              average=None)
                                   }).to(device)
        #val_results = run_eval_loop(model, dm.val_dataloader(), device, metrics)
        test_results = run_eval_loop(model, dm.test_dataloader(), device, metrics)
        if test_results=='debug':
            return
        test_row.update(
                {
                    "overall_accuracy": test_results["Accuracy"].item(),
                    "jaccard_index": test_results["JaccardIndex"].item(),
                    "labelwise_jaccard_index": test_results["Labelwise_JaccardIndex"].tolist()
                })
    elif issubclass(TASK, ClassificationTask):
        pass  # TODO: building classification


    #assert set(val_row.keys()) == set(test_row.keys())
    fieldnames = list(test_row.keys())

    print(test_results)
    # Write to file
    output_path = "evaluation/"+args.name+".csv"
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    with open(output_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        #writer.writerow(val_row)
        writer.writerow(test_row)


if __name__ == '__main__':
    args = get_args()
    print(args)

    pl.seed_everything(args.seed)
    main(args)