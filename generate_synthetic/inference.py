import sys
import os
import argparse

sys.path.append('../SPADE')
import numpy as np
import matplotlib.pyplot as plt
import imageio

import torch
from torch.utils.data import Dataset, DataLoader
assert torch.cuda.is_available()
from tqdm import tqdm
import random

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from data import create_dataloader

import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioIOError

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='data folder path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--spade_model', type=str, default='lambda0', help='spade model')
    parser.add_argument('--no_output', type=int, default=1, help='no. of synthetic image generated')
    parser.add_argument('--gpu_id', type=str, default="0", help='gpu id')

    return parser.parse_args()

def get_filenames(folder):
    files = []
    for i, filename in enumerate(os.listdir(folder)):
        if "naip-new" in filename:
            prefix = filename.rpartition("_")[0]
            files.append(prefix)
    
    return files

class TileInferenceDataset(Dataset):
    
    def __init__(self, fn, mask_lc_fn, mask_building_fn, chip_size, stride, transform=None, windowed_sampling=False, verbose=False):
        self.fn = fn
        self.mask_fn = mask_lc_fn
        self.mask_building_fn = mask_building_fn
        self.chip_size = chip_size
        
        self.transform = transform
        self.windowed_sampling = windowed_sampling
        self.verbose = verbose
        
        # read image and masks
        with rasterio.open(self.fn) as f:
            height, width = f.height, f.width
            self.num_channels = 3
            self.dtype = f.profile["dtype"]
            if not windowed_sampling: # if we aren't using windowed sampling, then go ahead and read in all of the data
                self.data = f.read()
                
        with rasterio.open(self.mask_fn) as f:
            assert f.shape[0] == height and f.shape[1] == width
            if not windowed_sampling: # if we aren't using windowed sampling, then go ahead and read in all of the data
                self.mask_data = f.read()
        
        with rasterio.open(self.mask_building_fn) as f:
            assert f.shape[0] == height and f.shape[1] == width
            if not windowed_sampling:
                self.mask_building_data = f.read()
        
        # combine building and landcover mask
        building_mask = (self.mask_building_data == 1)
        self.mask_data[building_mask] = 7
            
        self.chip_coordinates = [] # upper left coordinate (y,x), of each chip that this Dataset will return
        for y in list(range(0, height - self.chip_size, stride)) + [height - self.chip_size]:
            for x in list(range(0, width - self.chip_size, stride)) + [width - self.chip_size]:
                self.chip_coordinates.append((y,x))
        self.num_chips = len(self.chip_coordinates)

        if self.verbose:
            print("Constructed TileInferenceDataset -- we have %d by %d file with %d channels with a dtype of %s. We are sampling %d chips from it." % (
                height, width, self.num_channels, self.dtype, self.num_chips
            ))
            
    def __getitem__(self, idx):
        '''
        Returns:
            A tuple (chip, (y,x)): `chip` is the chip that we sampled from the larger tile. (y,x) are the indices of the upper left corner of the chip.
        '''
        y, x = self.chip_coordinates[idx]
        
        sample = {
            'label': None,
            'instance': 0,
            'image': None,
            'path': "dummy.png",
            'location': (y,x)
        }
        
        if self.windowed_sampling:
            try:
                with rasterio.Env():
                    with rasterio.open(self.fn) as f:
                        sample["image"] = f.read(window=rasterio.windows.Window(x, y, self.chip_size, self.chip_size))[:3]
                    with rasterio.open(self.mask_fn) as f:
                        sample["label"] = f.read(window=rasterio.windows.Window(x, y, self.chip_size, self.chip_size))
            except RasterioIOError as e: # NOTE(caleb): I put this here to catch weird errors that I was seeing occasionally when trying to read from COGS - I don't remember the details though
                print("Reading %d failed, returning 0's" % (idx))
                sample["image"] = np.zeros((self.num_channels, self.chip_size, self.chip_size), dtype=np.uint8)
                sample["label"] = np.zeros((self.num_channels, self.chip_size, self.chip_size), dtype=np.uint8)
        else:
            sample["image"] = self.data[:3, y:y+self.chip_size, x:x+self.chip_size]
            sample["label"] = self.mask_data[:, y:y+self.chip_size, x:x+self.chip_size]


        if self.transform is not None:
            sample = self.transform(sample)

        return sample
        
    def __len__(self):
        return self.num_chips

def transform(sample):
    sample["image"] = torch.from_numpy(sample["image"]).float()
    sample["label"] = torch.from_numpy(sample["label"]).float()
    return sample

def custom_forward(model, batch, zs):
    assert "label" in batch
    assert "image" in batch
    assert batch["label"].shape[0] == zs.shape[0]
    
    input_semantics, real_image = model.preprocess_input(batch)
    
    with torch.no_grad():
        fake_image, _ = model.netG(input_semantics, z=zs)
    return fake_image

def inverse_normalize(generated):
    generated = np.rollaxis(generated.cpu().numpy(), 1, 4)
    generated = ((generated * 0.5) + 0.5) * 255
    generated = generated.astype(np.uint8)
    return generated

def get_model(checkpoints_dir, model_name, gpu_id):
    sys.argv = [
        "baseline/test.py",
        "--name", model_name,
        "--dataset_mode", "custom",
        "--label_dir", "dummy",
        "--image_dir", "dummy",
        "--results_dir",  "dummy",
        "--checkpoints_dir", checkpoints_dir,
        "--label_nc", "9",
        "--contain_dontcare_label",
        "--output_nc", "3",
        "--use_vae",
        "--no_instance",
        "--gpu_ids", str(gpu_id)
        ]

    opt = TestOptions().parse()
    model = Pix2PixModel(opt)

    return model

def generate_synthetic(input_image_fn, input_mask, input_building_mask, output_image_fn, model):
    with rasterio.open(input_image_fn) as f:
        profile = f.profile
        big_image = np.rollaxis(f.read(), 0, 3)
        height, width, _ = big_image.shape 

    CHIP_SIZE = 256
    PADDING = 128
    assert PADDING % 2 == 0
    HALF_PADDING = PADDING//2
    CHIP_STRIDE = CHIP_SIZE - PADDING

    ds = TileInferenceDataset(
        fn = input_image_fn,
        mask_lc_fn = input_mask,
        mask_building_fn = input_building_mask,
        chip_size = CHIP_SIZE,
        stride = CHIP_STRIDE,
        transform = transform
    )
    dl = DataLoader(ds, batch_size=32, shuffle=False) 

    model.eval()

    output = np.zeros((height, width, 3), dtype=np.float32)
    kernel = np.ones((CHIP_SIZE, CHIP_SIZE, 1), dtype=np.float32)
    kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
    counts = np.zeros((height, width, 1), dtype=np.float32)

    for data_i in tqdm(dl):
        batch_size = data_i["label"].shape[0]

        generated = model(data_i, mode='inference')
        generated = inverse_normalize(generated)
        for i in range(batch_size):
            y = data_i["location"][0][i]
            x = data_i["location"][1][i]
            
            output[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += generated[i] * kernel
            counts[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += kernel

    output = (output / counts).astype(np.uint8)

    profile["count"] = 3
    profile["compress"] = "lzw"
    profile["predictor"] = 2
    with rasterio.open(output_image_fn, "w", **profile) as f:
        f.write(output[:,:,0], 1)
        f.write(output[:,:,1], 2)
        f.write(output[:,:,2], 3)

    return

if __name__ == '__main__':
    args = get_args()
    file_list = get_filenames(args.data_folder)
    random.shuffle(file_list)
    
    model = get_model(args.checkpoint_dir, args.spade_model, args.gpu_id)

    for i, f in enumerate(file_list):
        print(f"Processing tile {str(i+1)}/{len(file_list)} tiles")
        input_image = os.path.join(args.data_folder, f + "_naip-new.tif")
        input_mask = os.path.join(args.data_folder, f + "_lc.tif")
        input_building_mask = os.path.join(args.data_folder, f + "_buildings.tif")

        for n in range(args.no_output):
            output_image = os.path.join(args.data_folder, f + "_syn_" + args.spade_model + ".tif")
            
            if args.no_output > 1:
                output_image = os.path.join(args.data_folder, f + "_syn_" + args.spade_model + "_" + str(n) + ".tif")
            
            
            # check if file already exists or generated:
            if os.path.exists(output_image):
                print(f"{output_image} already exists.")
                continue
            
            mode = 'r' if os.path.exists('files.txt') else 'w'
            with open('files.txt', mode) as textfile:
                if output_image in textfile.read():
                    print(f"{output_image} already being generated.")
                    continue

            # write to txt file to keep track of files generates
            with open('files.txt', 'a') as textfile:
                textfile.write("%s\n" % output_image )
    
            # generate synthetic images
            generate_synthetic(input_image, input_mask, input_building_mask, output_image, model)


