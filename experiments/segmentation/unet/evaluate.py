from baseline.util.util import save_image, Colorize
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from unet.dice_score import multiclass_dice_coeff, dice_coeff, labelwise_dice_coeff

def mask_to_image(mask: np.ndarray, out_filename, save_mask = False):
    if mask.ndim == 2:
        mask = Image.fromarray(np.uint8(mask), 'L')
        if save_mask:
            mask.save(out_filename)
        return mask
    elif mask.ndim == 3:
        return save_image(mask, out_filename)

class DashBoard:
    def __init__(self, evaluate):
        self.evaluate = evaluate
    
    def __call__(self, metric=None):

        def get_score(net, test_loader, device, webpage, args, model, save_mask=False):
            if not metric is None:
                score, ims, txts, links = self.evaluate(net, test_loader, device, model, train=False, metric=metric, save_mask=save_mask, uncolor=args.uncolor, output=args.output, is_synthetic=args.is_synthetic)
            else:
                score, ims, txts, links = self.evaluate(net, test_loader, device, model, train=False, save_mask=save_mask, uncolor=args.uncolor, output=args.output, is_synthetic=args.is_synthetic)
            if save_mask:
                webpage.add_images_downstream(ims, txts, links, width=args.display_winsize)
            return score
        return get_score

def evaluate(net, dataloader, device, model, train=True, metric=multiclass_dice_coeff, save_mask=False, uncolor=False, output="", is_synthetic=False):
    net.eval()
    num_val_batches = len(dataloader)
    total_score = 0
    masks_pred, txts, links = [], [], []
    dice_n, dice_d = [], []
    # iterate over the validation/test set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round' if train else f'Testing with metric {metric.__name__}', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        label_path = batch['path'][0].replace('.tif', '')
        if not train and not os.path.exists(label_path):
            os.makedirs(label_path)
        mode = '_syn' if is_synthetic else '_real'
        label_path += '/' + model.replace('/','_').replace('.pth', '') + mode + '.png'
    

        # move images and labels to correct device and type 
        image = image.to(device=device, dtype=torch.float)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            label_tensor = torch.sigmoid(mask_pred)[0].cpu().float()
            if label_tensor.size()[0] > 1:
                label_tensor = label_tensor.max(0, keepdim=True)[1]
            
            # save data for visualization
            if uncolor:
                label_tensor = label_tensor.numpy()
                mask_to_image(label_tensor[0], label_path, save_mask)
            else:
                label_tensor = Colorize(net.n_classes)(label_tensor)
                full_mask = np.transpose(label_tensor.numpy(), (1, 2, 0))
                if save_mask:

                    mask_to_image(full_mask, label_path, save_mask=True)
            masks_pred.append(label_path.replace(output+"/", ""))             
            links.append(label_path.replace(output+"/", ""))

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the metric score
                total_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                label_tensor = torch.sigmoid(mask_pred)[0].cpu().float()
                if label_tensor.size()[0] > 1:
                    label_tensor = label_tensor.max(0, keepdim=True)[1]
                
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the metric score, ignoring background
                if not train:
                    score = metric(mask_pred[:, 1:, ...], mask_true[:, 1:, ...]) 
                    if isinstance(score, list):
                        if not isinstance(total_score, list):
                            total_score = [s.item() for s in score]
                        else:
                            total_score = [total_score[i]+score[i].item() for i in range(len(score))]
                    else:
                        txts.append(f"{batch['name'][0]} // Dice: {score.item():.4f}")
                        total_score += score.item()
                else:
                    n, d = dice_nd(mask_pred[:, 1:, ...], mask_true[:, 1:, ...]) 
                    if len(dice_n) == 0:
                        for i in range(len(n)):
                            dice_n.append(n[i])
                            dice_d.append(d[i])
                    else:
                        for i in range(len(n)):
                            dice_n[i] = dice_n[i] + n[i]
                            dice_d[i] = dice_d[i] + d[i]

    if train:
        net.train()
    
        
        num_labels = len(dice_n)
        print("")
        for i in range(num_labels): 
            label_score = (dice_n[i]+1e-6)/(dice_d[i]+1e-6)
            print(f"label {i+1}: {label_score:.4f}", end="  ")
            total_score += label_score
        print("")
        return total_score / num_labels, masks_pred, txts, links
   
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return total_score
    if isinstance(total_score, list):
        return [score/num_val_batches for score in total_score], masks_pred, txts, links
    return total_score / num_val_batches, masks_pred, txts, links

def dice_nd(input, target):
    assert input.size() == target.size()
    dice_n = []
    dice_d = []
    for channel in range(input.shape[1]): # number of classes
        n, d = dice_coeff_nd(input[:, channel, ...], target[:, channel, ...])
        dice_n.append(n)
        dice_d.append(d)
    return dice_n, dice_d

def dice_coeff_nd(input: Tensor, target: Tensor):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    union = torch.dot(input.reshape(-1),input.reshape(-1))+torch.dot(target.reshape(-1), target.reshape(-1))
    return 2 * inter, union

@DashBoard(evaluate)
def dice(input, target):
    return multiclass_dice_coeff(input, target, reduce_batch_first=True)

@DashBoard(evaluate)
def labelwise_dice(input, target):
    return labelwise_dice_coeff(input, target, reduce_batch_first=True)

def iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of IOU coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'IOU: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        union = torch.dot(input.reshape(-1),input.reshape(-1))+torch.dot(target.reshape(-1), target.reshape(-1))

        return (inter+epsilon) / (union - inter + epsilon)
    else:
        # compute and average metric for each batch element
        iou = 0
        for i in range(input.shape[0]):
            iou += iou_coeff(input[i, ...], target[i, ...])
        return iou / input.shape[0]

@DashBoard(evaluate)
def iou(input: Tensor, target: Tensor, reduce_batch_first: bool = True, epsilon=1e-6):
    # Average of iou coefficient for all classes
    assert input.size() == target.size()
    iou = 0
    for channel in range(input.shape[1]): # number of classes
        score = iou_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        iou += score
    return iou / input.shape[1]

@DashBoard(evaluate)
def labelwise_iou(input: Tensor, target: Tensor, reduce_batch_first: bool = True, epsilon=1e-6):
    # Average of iou coefficient for all classes
    assert input.size() == target.size()
    ious = []
    for channel in range(input.shape[1]): # number of classes
        score = iou_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        ious.append(score)
    return ious
