import os
import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import dice_loss, evaluate, labelwise_dice
from unet import MixDataset, SyntheticDataset, UNet
from baseline.models.networks.generator import SPADEGenerator

dir_img = Path('/home/group-segmentation/SPADE/results/lc_crop256_bs16_ncopy50_epochs10_placeholder/train_1000imgs/train_latest/images/real_image_tif')
dir_syn = Path('/home/group-segmentation/SPADE/results/lc_crop256_bs16_ncopy50_epochs10_placeholder/train_1000imgs/train_latest/images/synthesized_image_tif')
dir_mask = Path('/home/group-segmentation/SPADE/results/lc_crop256_bs16_ncopy50_epochs10_placeholder/train_1000imgs/train_latest/images/input_label_uncolor')
dir_mask_syn = Path('/home/group-segmentation/SPADE/results/lc_crop256_bs16_ncopy50_epochs10_placeholder/train_1000imgs/train_latest/images/input_label_uncolor')
dir_checkpoint = Path('./checkpoints/')
# dir_checkpoint = Path('./results/mengyuan_test')

def train_unet(net,
              device,
              opt):
    # 1. Create dataset
    output = "results/train"
    dataset = MixDataset(dir_img, dir_mask, dir_syn, dir_mask_syn, args.mix, output, args.crop_size)
        

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * opt.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=opt.batchSize, num_workers=1)
    train_loader = DataLoader(train_set, shuffle=not opt.serial_batches, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=opt.epochs, batch_size=opt.batchSize, learning_rate=opt.learning_rate,
                                  val_percent=opt.val_percent, save_checkpoint=opt.save_checkpoint, crop_size=opt.crop_size,
                                  amp=opt.amp))

    logging.info(f'''Starting training:
        Epochs:          {opt.epochs}
        Batch size:      {opt.batchSize}
        Learning rate:   {opt.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {opt.save_checkpoint}
        Device:          {device.type}
        Crop size:  {opt.crop_size}
        Mixed Precision: {opt.amp}
        Is Synthetic: {opt.is_synthetic}
        Dontcare label: {opt.dontcare}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=opt.learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor = 0.5, min_lr = min(opt.learning_rate/100,1e-7), patience=opt.patience)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, opt.epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{opt.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=opt.amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * opt.batchSize))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device, output)[0]
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if opt.save_checkpoint:
            
            dircheckpoint = dir_checkpoint / opt.model_name
            Path(dircheckpoint).mkdir(parents=True, exist_ok=True)
            finalcheckpoint = str(dircheckpoint / f'checkpoint_epoch{epoch}_mix_{int(opt.mix*100)}.pth')
            print("==========")
            print(finalcheckpoint)
            torch.save(net.state_dict(), finalcheckpoint)
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batchSize', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--label_nc', '-nc', metavar='NC', type=int, default=6, help='Number of classes, 7 for lc_bd, 6 for lc')
    parser.add_argument('--contain_dontcare_label', type=bool, default=True, help='if the label map contains dontcare label (dontcare=opt.dontcare)')
    parser.add_argument('--dontcare', type=int, default=15, help='dontcare label value)')
    parser.add_argument('--is_synthetic', '-syn', action='store_true', default=False, help='Using synthetic images if specified')
    parser.add_argument('--val_percent', type=float, default=0.1, help='Validation percentage')
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='Whether to save checkpoints')
    parser.add_argument('--nThreads', default=1, type=int, help='# workers')
    parser.add_argument('--isTrain', type=bool, default=False)
    parser.add_argument('--mix', type=float, default=0.0, help='Portion of synthetic images in the training dataset.')
    parser.add_argument('--model_name', type=str, default='test', help = 'Set model name')
    parser.add_argument('--patience', type=int, default=5, help = 'Set patience of scheduler')
    
    SyntheticDataset.modify_commandline_options(parser=parser)
    SPADEGenerator.modify_commandline_options(parser=parser, is_train=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Set semantic_nc based on the option.
    # This will be convenient in many places
    args.semantic_nc = args.label_nc+1

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=4, n_classes=args.semantic_nc, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_unet(net=net,
                  device=device,
                  opt=args)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise