"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from data.base_dataset import BaseDataset, get_params, get_transform
import os.path
import re
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from PIL import Image


class FIDDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=64)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=6)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(no_instance=True)
        return parser

    # def initialize(self, opt):
    #     self.opt = opt

    #     label_paths, image_paths, instance_paths = self.get_paths(opt)

    #     util.natural_sort(label_paths)
    #     util.natural_sort(image_paths)
    #     if not opt.no_instance:
    #         util.natural_sort(instance_paths)

    #     label_paths = label_paths[:opt.max_dataset_size]
    #     image_paths = image_paths[:opt.max_dataset_size]
    #     instance_paths = instance_paths[:opt.max_dataset_size]
        
    #     if not opt.no_pairing_check:
    #         for path1, path2 in zip(label_paths, image_paths):
    #             assert self.paths_match(path1, path2), \
    #                 "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

    #     self.label_paths = label_paths
    #     self.image_paths = image_paths
    #     self.instance_paths = instance_paths

    #     size = len(self.label_paths)
    #     self.dataset_size = size

    def get_paths(self, opt):
        root = opt.dataroot
        phase = re.compile('.*' + opt.phase + '.*')
        for _,dirs,_ in os.walk(root):
            for d in dirs:
                if phase.match(d):
                    dir = d

        label_dir = os.path.join(root, dir, 'lb_land_cover' if opt.label == 'lc' else 'lb_building')
        label_paths = []
        for _ in range(opt.ncopy):
            for each in make_dataset(label_dir, recursive=False, read_cache=True):
                label_paths.append(each)

        image_dir = os.path.join(root, dir, 'dataset')
        image_paths = []
        for _ in range(opt.ncopy):
            for each in make_dataset(image_dir, recursive=False, read_cache=True):
                image_paths.append(each)

        instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"
        return label_paths, image_paths, instance_paths

    # def paths_match(self, path1, path2):
    #     filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
    #     filename1_sub = filename1_without_ext[0:filename1_without_ext.rfind("_")]
    #     filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
    #     filename2_sub = filename1_without_ext[0:filename2_without_ext.rfind("_")]
    #     return filename1_sub == filename2_sub

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        # transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        # label_tensor = transform_label(label) * 255.0
        # label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        # input image (real images)
        image_path = self.image_paths[index]
        # assert self.paths_match(label_path, image_path), \
        #     "The label_path %s and image_path %s don't match." % \
        #     (label_path, image_path)
        image = Image.open(image_path)
        # image = image.convert('RGB')

        # assert label.size == image.size
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # # if using instance maps
        # if self.opt.no_instance:
        #     instance_tensor = 0
        # else:
        #     instance_path = self.instance_paths[index]
        #     instance = Image.open(instance_path)
        #     if instance.mode == 'L':
        #         instance_tensor = transform_label(instance) * 255
        #         instance_tensor = instance_tensor.long()
        #     else:
        #         instance_tensor = transform_label(instance)

        # input_dict = {'label': label_tensor,
        #               'instance': instance_tensor,
        #               'image': image_tensor,
        #               'path': image_path,
        #               }

        # Give subclasses a chance to modify the final output
        # self.postprocess(input_dict)

        return image_tensor

    # def postprocess(self, input_dict):
    #     return input_dict

    # def __len__(self):
    #     return self.dataset_size
