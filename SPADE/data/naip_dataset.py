import os.path
import re
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class NAIPDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        # parser.set_defaults(preprocess_mode='crop')
    
        # parser.set_defaults(load_size=256)
        # parser.set_defaults(crop_size=64)
        # parser.set_defaults(display_winsize=256)
        
        # parser.set_defaults(label_nc=6)
        # parser.set_defaults(contain_dontcare_label=True)
        if parser.parse_args().label == 'lc': 
            parser.set_defaults(label_nc=6)
            parser.set_defaults(contain_dontcare_label=True)
            parser.set_defaults(dontcare=13)
        elif parser.parse_args().label == 'bd':
            parser.set_defaults(label_nc=2)
            parser.set_defaults(contain_dontcare_label=False)
        elif parser.parse_args().label == 'lc_bd':
            parser.set_defaults(label_nc=7)
            parser.set_defaults(contain_dontcare_label=True)
            parser.set_defaults(dontcare=13) 
        parser.set_defaults(no_instance=True)
        # parser.set_defaults(dontcare=13)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = re.compile('.*' + opt.phase + '.*')
        for _,dirs,_ in os.walk(root):
            for d in dirs:
                if phase.match(d):
                    dir = d
        
        if opt.label == 'lc':
            label_type = 'lb_land_cover'
        elif opt.label == 'bd':
            label_type = 'lb_building'
        elif opt.label == "lc_bd":
            label_type = 'lb_building_landcover'

        label_dir = os.path.join(root, dir, label_type)
        label_paths = []

        # for monitoring and cropping extra type of label mask though using another type of label mask for generation
        if opt.extra_label == 'bd':
            extra_label_type = 'lb_building'
            extra_label_dir = os.path.join(root, dir, extra_label_type)
            extra_label_paths = []
        
        for _ in range(opt.ncopy):
            for each in make_dataset(label_dir, recursive=False, read_cache=True):
                label_paths.append(each)
            # for monitoring and cropping extra type of label mask
            if opt.extra_label == 'bd':
                for each in make_dataset(extra_label_dir, recursive=False, read_cache=True):
                    extra_label_paths.append(each)

        image_dir = os.path.join(root, dir, 'dataset')
        image_paths = []
        for _ in range(opt.ncopy):
            for each in make_dataset(image_dir, recursive=False, read_cache=True):
                image_paths.append(each)

        instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"
        if opt.extra_label == 'bd':
            return label_paths, image_paths, instance_paths, extra_label_paths
        return label_paths, image_paths, instance_paths