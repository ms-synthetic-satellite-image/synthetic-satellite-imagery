import os
import sys
import argparse
import shutil
import numpy as np  
from PIL import Image


'''
    input_folder:   folder, that contains all label & image subfolders ('input_extra_label',...), 
                    e.g. train_1000/train_latest/images
                    
    output_folder:  folder location for the filtered images and folders; folder will be automatically created
    
    building_ratio: defines how many of the labels for a specific label mask should be of type 'building'
    
    max_no_images: defines the maximum number of the filtered building label masks & images should be copied 
    
'''

def building_selector(input_folder, output_folder, building_ratio, max_no_images):
    
    # open label mask folder
    all_uncolored_names = os.listdir(input_folder + '/input_label_uncolor')
    
    # list of names
    imgs_buildings_names  = []
    
    # percent_counter
    building_counter_max = building_ratio*256*256
    
    
    # # # # # # # 
    #  FILTER   #
    # # # # # # #
    
    # iterate over all label masks
    for i, filename in enumerate(all_uncolored_names):
        
        # check condition for max imgs
        if not len(imgs_buildings_names) >= max_no_images:
            
            # reset building counter
            building_counter = 0
                    
            # open image, flatten
            label_matrix = np.array(Image.open(input_folder + '/input_label_uncolor/' + filename), dtype=np.uint8) 
            label_matrix_flat = label_matrix.reshape(65536,)

            # loop over all pixels, and compare
            for i, label in enumerate(label_matrix_flat):
                if label == 5:
                    building_counter+=1
                    
                    if building_counter >= building_counter_max:
                        imgs_buildings_names.append(filename)                       
                        break

         
    # # # # # # # # #  
    #  COPY-PASTE   #
    # # # # # # # # #
    
    # grab all subfolders
    all_subfolders = os.listdir(input_folder)

    # loop over all filtered building images:
    for image_name in imgs_buildings_names:
           
        # loop over all subfolders, and copy-paste respective image/label mask into new folder
        for subfolder in all_subfolders:
                try:
                    shutil.copy(input_folder + '/' + subfolder + '/' + image_name, output_folder + '/' + subfolder + '/' + image_name)
                except IOError as io_err:
                    try:
                        os.makedirs(os.path.dirname(output_folder + '/' + subfolder + '/'))
                        shutil.copy(input_folder + '/' + subfolder + '/' + image_name, output_folder + '/' + subfolder + '/' + image_name)
                    except:
                        print(f"WARNING: {image_name} doesn't exist in input subfolder '{subfolder}'")

                    
    # DONE             
    print(f"finished filtering images by building_ratio: {building_ratio}, i.e. at least {building_ratio*256*256}px.\n" + 
          f"Selected a total of {len(imgs_buildings_names)} and created copies of labels and images at '{output_folder}'")
                    
               
parser = argparse.ArgumentParser()
parser.add_argument("input", help="")
parser.add_argument("output", help="")
parser.add_argument("ratio", help="")
parser.add_argument("no_imgs", help="")

args = parser.parse_args()
input, output, ratio, no_imgs = args.input, args.output, args.ratio, args.no_imgs

building_selector(input, output, float(ratio), int(no_imgs))

# usage
# python building_selector.py '../results/lc_crop256_bs16_ncopy50_epochs10_extra1000/train_latest/images/' '../results/rob_filter_ex_10percent' 0.1 5