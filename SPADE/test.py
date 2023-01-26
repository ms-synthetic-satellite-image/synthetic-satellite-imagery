"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from util import util
import numpy as np

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test

for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        if opt.extra_label == 'bd':
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                    ('input_label_uncolor', data_i['label'][b]),
                                    ('real_image', data_i['image'][b]),
                                    ('synthesized_image', generated[b]),
                                    ('input_extra_label', data_i['extra_label'][b]),
                                    ('input_extra_label_uncolor', data_i['extra_label'][b])])
        else:
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                    ('input_label_uncolor', data_i['label'][b]),
                                    ('real_image', data_i['image'][b]),
                                    ('synthesized_image', generated[b])])
        copy_index = util.get_copy_index(opt.ncopy, i)
        visualizer.save_images(webpage, visuals, img_path[b:b + 1], opt.ncopy_start_idx+copy_index)

webpage.save()
