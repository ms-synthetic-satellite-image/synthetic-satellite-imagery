import logging
from os import listdir,path
from os.path import splitext
from pathlib import Path
import sys

import baseline.data
import numpy as np
from baseline.models.pix2pix_model import Pix2PixModel
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, output, crop_size, model_name = 'cur_no_use', mask_suffix: str = '_naip-new', dontcare: int=13):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.crop_size = crop_size
        self.dontcare = dontcare
        if output[-1] == '/':
            self.output = output[:-1]
        else:
            self.output = output
        # self.output+='/'+model_name[:model_name.rfind('.pth')].replace('/','_')+'/images'
        self.ids = [splitext(file)[0].replace('_naip-new', '') for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, crop_size, is_mask, x_crop, y_crop):
        # crop
        tw = th = crop_size
        pil_img = pil_img.crop((x_crop, y_crop, x_crop + tw, y_crop + th))
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255
        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        if name.find('_cp') != -1:
            img_file = list(self.images_dir.glob(name[:name.find('_cp')] + '_naip-new' + name[name.find('_cp'):]+ '.*'))
            mask_file = list(self.masks_dir.glob(name[:name.find('_cp')] + '_naip-new' + name[name.find('_cp'):] + '.*'))
        else:
            img_file = list(self.images_dir.glob(name + '_naip-new.' + '*'))
            mask_file = list(self.masks_dir.glob(name + '_naip-new.' + '*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name[:name.find("_cp")] + "_naip-new" + name[name.find("_cp"): ]}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name[:name.find("_cp")] + "_naip-new" + name[name.find("_cp"): ]}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        
        x_crop = random.randint(0, np.maximum(0, img.size[0] - self.crop_size))
        y_crop = random.randint(0, np.maximum(0, img.size[1] - self.crop_size))

        img = self.preprocess(img, self.crop_size, is_mask=False, x_crop=x_crop, y_crop=y_crop)
        mask = self.preprocess(mask, self.crop_size, is_mask=True, x_crop=x_crop, y_crop=y_crop)
        mask = torch.as_tensor(mask.copy()).long().contiguous()

        mask[mask == self.dontcare] = 0
        label_pred_path = str(img_file[0])
        label_pred_path = self.output+label_pred_path[label_pred_path.rfind('/'):]
        return {
            'name': name,
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': mask,
            'path': label_pred_path
        }

class MixDataset(BasicDataset):
    def __init__(self, images_dir: str, masks_dir: str, images_syn_dir: str, masks_syn_dir: str, mix_rate: float, output: str, crop_size, mask_suffix: str = '_naip-new', dontcare: int=13):
        super(BasicDataset, self).__init__()
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.crop_size = crop_size
        self.dontcare = dontcare
        if output[-1] == '/':
            self.output = output[:-1]
        else:
            self.output = output
        # self.output+='/'+model_name[:model_name.rfind('.pth')].replace('/','_')+'/images'
        self.ids = [splitext(file)[0].replace('_naip-new', '') for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        self.images_syn_dir = Path(images_syn_dir)
        self.masks_syn_dir = Path(masks_syn_dir)
        self.mix_threshold = mix_rate

    def __getitem__(self, idx):
        is_synthetic = np.random.uniform() <= self.mix_threshold
        images_dir = ""
        masks_dir = ""
        if is_synthetic:
            images_dir, masks_dir = self.images_syn_dir, self.masks_syn_dir
        else:
            images_dir, masks_dir = self.images_dir, self.masks_dir
    
        name = self.ids[idx]
        if name.find('_cp') != -1:
            img_file = list(images_dir.glob(name[:name.find('_cp')] + '_naip-new' + name[name.find('_cp'):]+ '.*'))
            mask_file = list(masks_dir.glob(name[:name.find('_cp')] + '_naip-new' + name[name.find('_cp'):] + '.*'))
        else:
            img_file = list(images_dir.glob(name + '_naip-new.' + '*'))
            mask_file = list(masks_dir.glob(name + '_naip-new.' + '*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name[:name.find("_cp")] + "_naip-new" + name[name.find("_cp"): ]}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name[:name.find("_cp")] + "_naip-new" + name[name.find("_cp"): ]}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        
        x_crop = random.randint(0, np.maximum(0, img.size[0] - self.crop_size))
        y_crop = random.randint(0, np.maximum(0, img.size[1] - self.crop_size))

        img = self.preprocess(img, self.crop_size, is_mask=False, x_crop=x_crop, y_crop=y_crop)
        mask = self.preprocess(mask, self.crop_size, is_mask=True, x_crop=x_crop, y_crop=y_crop)
        mask = torch.as_tensor(mask.copy()).long().contiguous()

        mask[mask == self.dontcare] = 0
        label_pred_path = str(img_file[0])
        label_pred_path = self.output+label_pred_path[label_pred_path.rfind('/'):]
        return {
            'name': name,
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': mask,
            'path': label_pred_path,
            'is_synthetic': is_synthetic
        }


class SyntheticDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--name', type=str, default='label2coco', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--checkpoints_dir', type=str, default='/home/group-segmentation/SPADE/checkpoints', help='models are saved here')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        parser.add_argument('--z_dim', type=int, default=256,
                            help="dimension of the latent z vector")
        parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--dataset_mode', type=str, default='naip')
        parser.add_argument('--dataroot', type=str, default='/home/group-segmentation/main/data')
        parser.add_argument('--crop_size', '-cs', metavar='CS', type=int, default=256, help='Size to crop to')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--label', type=str, default='lc', help='label type: land_cover or building')
        parser.add_argument('--ncopy', type=int, default=1, help='Number of copies in total created for dataset')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')
        parser.add_argument('--output_nc', type=int, default=4, help='# of output image channels')

        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(no_instance=True)

    def __init__(self, opt):
        super(SyntheticDataset, self).__init__()
        self.dataloader = baseline.data.create_dataloader(opt)
        self.model = Pix2PixModel(opt)
        self.model.eval()
        self.images = []
        self.class_number = opt.semantic_nc
        self.dontcare = opt.dontcare
        self.mask_suffix = ''
        self.crop_size = opt.crop_size

        for i, data_i in enumerate(self.dataloader):
            generated = self.model(data_i, mode='inference')

            for b in range(generated.shape[0]):
                mask = data_i['label'][b]
                mask[mask == self.dontcare] = 0
                self.images.append({'image':generated[b], 'mask': mask})
        self.dataset_size = len(self.images)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return self.dataset_size