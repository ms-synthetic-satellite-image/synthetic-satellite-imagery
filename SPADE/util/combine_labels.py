
import os
import argparse
import numpy as np
from PIL import Image

def combine_building_landcover_maps(img_bd, img_lc):
    '''
    Input:
        img_bd - building label map in array, values in {0,1}
        img_lc - landcover label map in array, values in {1,2,3,4,5,6,15}
    Output:
        combined label map in array, values in {1,2,3,4,5,6,7,15}
    '''
    img_comb = img_bd * 7  # building 7, no building 0
    for i in range(img_comb.shape[0]):
        for j in range(img_comb.shape[1]):
            if img_comb[i][j]==0:
                img_comb[i][j] = img_lc[i][j]
    return img_comb

def combine_labels(dir_building, dir_lc, output_dir):
    '''
    Input:
        dir_building - directory storing building label maps
        dir_lc - directory storing landcover label maps
        output_dir - directory to save the combined label maps
    Function:
        read in label maps from input directories, combine them, and save to output directory
    '''
    # create folder at output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # hard-coded based on current file name pattern
    file_prefixes = set([f[:17] for f in os.listdir(dir_building)])
    
    for i, pf in enumerate(file_prefixes):
        # read in two label maps
        building_path = os.path.join(dir_building, pf+"_buildings.tif")
        img_bd = np.array(Image.open(building_path))
        lc_path = os.path.join(dir_lc, pf+"_lc.tif")
        img_lc = np.array(Image.open(lc_path))
        # get combiend label map
        img_comb = combine_building_landcover_maps(img_bd, img_lc)
        Image.fromarray(img_comb).save(os.path.join(output_dir, pf+"_bdlc.png"))
        # tiff too large with limited space on device
        #Image.fromarray(img_comb).save(os.path.join(output_dir, pf+"_bdlc.tif"))
        print(f"Finished processing label map {i+1} ...")



# add argparse for command line calls
parser = argparse.ArgumentParser()
parser.add_argument("input_dir_building", help="type in building map directory here")
parser.add_argument("input_dir_landcover", help="type in landcover map directory here")
parser.add_argument("output_dir", help="type in output directory here")

# Note: try
#   "/home/group-segmentation/main/data/md_1m_2013_extended-debuffered-{train,val,test}_tiles/lb_building"
#   "/home/group-segmentation/main/data/md_1m_2013_extended-debuffered-{train,val,test}_tiles/lb_land_cover"
#   "/home/group-segmentation/main/data/md_1m_2013_extended-debuffered-{train,val,test}_tiles/lb_building_landcover"
args = parser.parse_args()
dir_building, dir_lc, output_dir = args.input_dir_building, args.input_dir_landcover, args.output_dir
print(f"Start combining labels and save results to {output_dir} ...")
combine_labels(dir_building, dir_lc, output_dir)
print("Done!")

#python3 combine_labels.py /home/group-segmentation/main/data/md_1m_2013_extended-debuffered-val_tiles/lb_building /home/group-segmentation/main/data/md_1m_2013_extended-debuffered-val_tiles/lb_land_cover /home/group-segmentation/main/data/md_1m_2013_extended-debuffered-val_tiles/lb_building_landcover

