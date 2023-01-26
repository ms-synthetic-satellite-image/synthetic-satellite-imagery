"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--ncopy', type=int, default=1, help='Number of copies in total created for dataset')
        parser.add_argument('--extra_label', type=str, default='bd', help='Also cropping a type of label mask while focusing on another label mask type')
        parser.add_argument('--ncopy_start_idx', type=int, default=0, help='Index to start with for saving names of different copies')
        parser.add_argument('--bd_minpct', type = float, default = 0, help = 'Control the ratio of building label when generate dataloader')

        parser.set_defaults(preprocess_mode='crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        
        self.isTrain = False
        return parser
