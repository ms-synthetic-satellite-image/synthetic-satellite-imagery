import sys
import os
import argparse

sys.path.append('../SPADE')
import numpy as np
import csv
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
assert torch.cuda.is_available()
from tqdm import tqdm
import random

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from fid.fid_score import calculate_activation_statistics, save_statistics, load_statistics, calculate_frechet_distance, _compute_statistics_of_path
from fid.inception import InceptionV3

import rasterio
from rasterio.errors import RasterioIOError

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, help='data folder path')
    parser.add_argument('--data_type', type=str, default='train', help='indicate train/test/val dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='path to checkpoint directory')
    parser.add_argument('--spade_model', type=str, default='lambda_0', help='spade model within checkpoint directory')
    parser.add_argument('--no_output', type=int, default=1, help='no. of synthetic image generated')
    parser.add_argument('--save_patch', type=int, default=0, help='1 - save patches from tiles as .png to calculate FID post-inference')
    parser.add_argument('--compute_fid', type=int, default=0, help='1 - calculate FID from saved patches')
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
        
        # combine building and landcover masks
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
            except RasterioIOError as e: 
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
        "SPADE/test.py",
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

def generate_synthetic(input_image_fn, input_mask, input_building_mask, output_image_fn, model,
                       data_type, patch_folders=None, file_name=None, save_patch=False):
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

    idx = 0
    for data_i in tqdm(dl):
        batch_size = data_i["label"].shape[0]

        # Use trained encoder if generating synthetic images for the train set, 
        # Use randomly sampled z when generating synthetic images and evaluate FID over test set
        if data_type == 'train':
            generated = model(data_i, mode='inference', encoder=True)
        else:
            generated = model(data_i, mode='inference', encoder=False)

        generated = inverse_normalize(generated)
        for i in range(batch_size):
            y = data_i["location"][0][i]
            x = data_i["location"][1][i]
            
            output[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += generated[i] * kernel
            counts[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += kernel

            if save_patch:
                patch_folder_real, patch_folder_syn = patch_folders[0], patch_folders[1]
                # save patch of real image
                real = Image.fromarray(np.transpose(data_i['image'][i].cpu().float().numpy(), (1,2,0)).astype(np.uint8))
                real_save = os.path.join(patch_folder_real, file_name + "_" + str(idx+i) + ".png")
                if not os.path.exists(patch_folder_real):
                    os.mkdir(patch_folder_real)
                if not os.path.exists(real_save):
                    real.save(real_save)
                
                # save patch of generated image
                syn = Image.fromarray(generated[i])
                generated_save = os.path.join(patch_folder_syn, file_name + "_" + str(idx+i) + ".png")
                if not os.path.exists(patch_folder_syn):
                    os.mkdir(patch_folder_syn)

                if not os.path.exists(generated_save):
                    syn.save(generated_save)

        idx = idx + batch_size

    output = (output / counts).astype(np.uint8)

    profile["count"] = 3
    profile["compress"] = "lzw"
    profile["predictor"] = 2
    with rasterio.open(output_image_fn, "w", **profile) as f:
        f.write(output[:,:,0], 1)
        f.write(output[:,:,1], 2)
        f.write(output[:,:,2], 3)

    return

def get_activation_statistics(dl, stats_output, model, dims, device, path=False):
    # check if file already exists or generated:
    if os.path.exists(stats_output):
        print(f"{stats_output} already exists.")
        return load_statistics(stats_output)
    if path:
        m, s = _compute_statistics_of_path(dl, model, batch_size=32, dims = dims, device = device)
    else:
        m, s = calculate_activation_statistics(dl, model, batch_size=32, dims = dims, device = device)
    save_statistics(stats_output, m, s)

    return m, s

if __name__ == '__main__':
    args = get_args()
    file_list = get_filenames(args.data_folder)
    data_type = args.data_type
    random.shuffle(file_list)
    
    model = get_model(args.checkpoint_dir, args.spade_model, args.gpu_id)

    # generate synthetic images
    for i, f in enumerate(file_list):
        print(f"Processing tile {str(i+1)}/{len(file_list)} tiles")
        input_image = os.path.join(args.data_folder, f + "_naip-new.tif")
        input_mask = os.path.join(args.data_folder, f + "_lc.tif")
        input_building_mask = os.path.join(args.data_folder, f + "_buildings.tif")
        
        if args.save_patch == 1:
            patch_folder_real = os.path.join(args.data_folder.rpartition("/")[0], args.data_type + "_patch_real")
            patch_folder_syn = os.path.join(args.data_folder.rpartition("/")[0], args.data_type + "_patch_" + args.spade_model)
       
        for n in range(args.no_output):
            output_image = os.path.join(args.data_folder, f + "_syn_" + args.spade_model + ".tif")
            
            if args.no_output > 1:
                output_image = os.path.join(args.data_folder, f + "_syn_" + args.spade_model + "_" + str(n) + ".tif")
            
            # check if file already exists or generated:
            if os.path.exists(output_image):
                print(f"{output_image} already exists.")
                continue
            
            if os.path.exists(f'{args.data_type}_tiles.txt'):
                with open(f'{args.data_type}_tiles.txt', 'r') as textfile:
                    if output_image in textfile.read():
                        print(f"{output_image} already being generated.")
                        continue
            
            mode = 'a' if os.path.exists(f'{args.data_type}_tiles.txt') else 'w'
            # write to txt file to keep track of files generates
            with open(f'{args.data_type}_tiles.txt', 'a') as textfile:
                textfile.write("%s\n" % output_image )
    
            # generate synthetic images
            if args.save_patch == 1:
                generate_synthetic(input_image, input_mask, input_building_mask, output_image, model,
                                   data_type, patch_folders=[patch_folder_real, patch_folder_syn], file_name = f, 
                                   save_patch=True)
            else:
                generate_synthetic(input_image, input_mask, input_building_mask, output_image, model, data_type)
    
    # compute fid from saved patches
    if args.compute_fid:
        # inception model
        dims = 2048
        device = 'cuda:' + args.gpu_id
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)

        # calculate and save activation statistics
        fid_dir = "fid_stats"
        if not os.path.exists(fid_dir):
            os.mkdir(fid_dir)

        stats_output_real = os.path.join(fid_dir, args.data_type + "_activation_stats_real.npz")
        stats_output_syn = os.path.join(fid_dir, args.data_type + "_activation_stats_" + args.spade_model + ".npz")
        real_patch = os.path.join(args.data_folder.rpartition("/")[0], args.data_type + "_patch_real")
        syn_patch = os.path.join(args.data_folder.rpartition("/")[0], args.data_type + "_patch_" + args.spade_model)
        
        m_real, s_real = get_activation_statistics(real_patch, stats_output_real, model, dims, device, path=True)
        m_syn, s_syn = get_activation_statistics(syn_patch, stats_output_syn, model, dims, device, path=True)
        fid_value = calculate_frechet_distance(m_real, s_real, m_syn, s_syn)

        result_row = {'spade_model': args.spade_model,
                      'FID':fid_value}

        # save results in csv file:
        output_path = "fid_" + args.data_type + ".csv"
        fieldnames = ['spade_model', 'FID']
        if not os.path.exists(output_path):
            with open(output_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        
        # write to csv file
        with open(output_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(result_row)


