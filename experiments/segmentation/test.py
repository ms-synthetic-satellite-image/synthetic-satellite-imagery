import argparse
from collections import OrderedDict
import logging
import os
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

from unet import BasicDataset, SyntheticDataset
from unet import UNet
from unet import evaluate, mask_to_image, dice, labelwise_dice, iou, labelwise_iou
from utils import plot_img_and_mask
from baseline.util.util import Colorize, save_image, extract_rgb
from baseline.util.visualizer import Visualizer
from baseline.util import html

dir_img_test = Path('/home/group-segmentation/SPADE/results/lc_crop256_bs16_ncopy50_epochs10_placeholder/test_200imgs/test_latest/images/real_image_tif')
dir_syn_test = Path('/home/group-segmentation/SPADE/results/lc_crop256_bs16_ncopy50_epochs10_placeholder/test_200imgs/test_latest/images/synthesized_image_tif')
dir_mask = Path('/home/group-segmentation/SPADE/results/lc_crop256_bs16_ncopy50_epochs10_placeholder/test_200imgs/test_latest/images/input_label_uncolor')
dir_mask_syn_test = Path('/home/group-segmentation/SPADE/results/lc_crop256_bs16_ncopy50_epochs10_placeholder/test_200imgs/test_latest/images/input_label_uncolor')

# dir_syn_test = Path('/home/group-segmentation/SPADE/results/experiments/real_image')
# dir_mask_syn_test = Path('/home/group-segmentation/SPADE/results/experiments/input_label_uncolor')



def predict_img(net,
                full_img,
                device,
                args,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, crop_size=args.crop_size, is_mask=False, x_crop=0, y_crop=0))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        label_tensor = probs.cpu().float()
        if label_tensor.size()[0] > 1:
            label_tensor = label_tensor.max(0, keepdim=True)[1]

        if not args.uncolor:
            label_tensor = Colorize(net.n_classes)(label_tensor)
            full_mask = np.transpose(label_tensor.numpy(), (1, 2, 0))
        else:
            full_mask = label_tensor.numpy()[0]
    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return full_mask


def get_args():
    # If --batch:
    # should specify --output, which is a direcoty for holding the html
    # If --batch not specified:
    # should specify --input, --output, [--viz]
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--models', '-m', nargs='+', metavar='FILE', default = ['checkpoints/final_real/checkpoint_epoch5.pth', 'checkpoints/final_1000/checkpoint_epoch5.pth', 'checkpoints/final_2000/checkpoint_epoch5.pth'],
                        help='Specify the files in which the models are stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Directory of output images')
    parser.add_argument('--viz', '-v', action='store_true', default=False,
                        help='Visualize the images as they are processed if specified')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--batch', '-b', action='store_true', default=False, help='Predict test data in batch mode if specified')
    parser.add_argument('--uncolor', '-uc', action='store_true', default=False, help='If specified, will generate uncolored labels, which has one channel')
    parser.add_argument('--is_synthetic', '-syn', action='store_true', default=False, help='Using synthetic images if specified')
    parser.add_argument('--dataset', '-d', default='both', help='Generate masks use synthetic/real/both dataset, can choose: syn/real/both')
    parser.add_argument('--isTrain', type=bool, default=False)
    parser.add_argument('--label_nc', '-nc', type=int, default=6,
                        help='Number of labels: 7 for lc_bd, 6 for lc', required=True)
    
    SyntheticDataset.modify_commandline_options(parser)
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))

def test_net(args, net, device, model):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not args.batch:
        in_files = args.input
        for _, filename in enumerate(in_files):
            print(f'Predicting image {filename} ...')
            img = Image.open(filename)
            mask = predict_img(net=net,
                            full_img=img,
                            args=args,
                            out_threshold=args.mask_threshold,
                            device=device)
            if not args.no_save:
                out_filename = args.output+filename[filename.rfind('/'):filename.rfind('.')]
                if not os.path.exists(out_filename):
                    os.makedirs(out_filename)

                out_filename+='/'+model.replace('/','_')+'.png'
                mask_to_image(mask, out_filename, save_mask=True)
                print(f'Mask saved to {out_filename}\n')

            if args.viz:
                print(f'Visualizing results for image {filename}, close to continue...')
                plot_img_and_mask(img, mask)
    else:
        if not args.is_synthetic:
            dataset = BasicDataset(dir_img_test, dir_mask, args.output, args.crop_size, model)
        else:
            dataset = BasicDataset(dir_syn_test, dir_mask_syn_test, args.output, args.crop_size, model)
        loader_args = dict(batch_size=200, num_workers=1)
        test_loader = DataLoader(dataset, shuffle=False, **loader_args)

        metrics = {'DICE': dice(net, test_loader, device, webpage, args, model, save_mask=True),
                   'Labelwise_DICE': labelwise_dice(net, test_loader, device, webpage, args, model, save_mask=False),
                   'IOU': iou(net, test_loader, device, webpage, args, model, save_mask=False),
                   'Labelwise_IOU': labelwise_iou(net, test_loader, device, webpage, args, model, save_mask=False),
                   }
        # TODO: add more metric scores

        for metric_name, metric_value in metrics.items():
            print(f'test {metric_name} score: {metric_value}')

        js = json.dumps(metrics)
        mode = "_syn" if args.is_synthetic else "_real"
        f = open(web_dir + "/" + model.split("/")[1] + mode + "_scores.json", "w+")
        f.write(js)
        f.close()

if __name__ == '__main__':
    args = get_args()

    # instantiate unet
    net = UNet(n_channels=4, n_classes=args.label_nc+1, bilinear=args.bilinear)
    # move net to device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'Using device {device}')
    net.to(device=device)

    # create a webpage that summarizes the all results
    if args.batch:
        web_dir = args.output
        print("HTML saved to: ", web_dir)
        webpage = html.HTML(web_dir,
                            'Experiment = %s, Phase = %s' %
                            (args.dataset, 'test'))

        # add original image and ground truth mask to webpage
        paths = []
        labels = []
        if args.dataset == 'real' or args.dataset == 'both':
            paths.append(dir_img_test)
            labels.append('original_images_real')
        if args.dataset == 'syn' or args.dataset == 'both':
            paths.append(dir_syn_test )
            labels.append('original_images_syn')
        dir_mask_color = str(dir_mask).replace("_uncolor", "")
        paths.append(dir_mask_color)
        labels.append('ground_truth_masks')

        ims = [[] for i in range(len(paths))]
        txts = [[] for i in range(len(paths))]
        links = [[] for i in range(len(paths))]

        for images in os.listdir(dir_img_test):
            for i in range(len(paths)):
                image_path = os.path.join(paths[i], images)
                if i  == len(paths) - 1:
                    image_path = image_path.replace('.tif', '.png')
                    image_numpy = np.array(Image.open(image_path))
                else:
                    image_path = image_path.replace('.png', '.tif')
                    image_numpy = extract_rgb(image_path)
                save_path = os.path.join(args.output, labels[i], images).replace('.tif', '.png')
                save_image(image_numpy, save_path, create_dir=True)
                ims[i].append(save_path.replace(args.output+"/", ""))
                links[i].append(save_path.replace(args.output+"/", ""))
                txts[i].append(f"{images}")

        for i in range(len(paths)):
            webpage.add_header(labels[i])
            webpage.add_images_downstream(ims[i], txts[i], links[i], width=args.display_winsize)


    # load model and test model
    for model in args.models:
        print(f'Loading model {model}')
        net.load_state_dict(torch.load(model, map_location=device))
        print('Model loaded!')

        if not args.batch:
            test_net(args, net, device, model)
        else:
            if args.dataset == 'real' or args.dataset == 'both':
                args.is_synthetic = False
                webpage.add_header(f"predicted_masks on real: {model}")
                test_net(args, net, device, model)
                
            if args.dataset == 'syn' or args.dataset == 'both':
                args.is_synthetic = True
                webpage.add_header(f"predicted_masks on synthetic: {model}")
                test_net(args, net, device, model)
    
    webpage.save()
    
