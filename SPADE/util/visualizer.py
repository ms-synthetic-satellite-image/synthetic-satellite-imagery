"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)
                
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]
                scipy.misc.toimage(image_numpy).save(s, format="tif")
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s_%d.tif' % (epoch, step, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s.tif' % (epoch, step, label))
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]                    
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=5)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_iter%.3d_%s_%d.tif' % (n, step, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_iter%.3d_%s.tif' % (n, step, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean().float()
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    
    def print_current_fid(self, epoch, i, fid, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        message += 'FID: %.3f ' % (fid)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message) 

    def print_current_earlystop_state(self, epoch, i, counter, patience, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        message += f'Early stopping counts {counter} out of {patience}  '

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message) 

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key or 'input_extra_label' == key:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile, uncolor_label=False)
            elif 'input_label_uncolor' == key or 'input_extra_label_uncolor' == key:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile, uncolor_label=True)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, copy_index=None):        
        visuals = self.convert_visuals_to_numpy(visuals)        
        
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        # add copy index to the end of image names: e.g. xxx_cp.png
        if copy_index is not None:
            name = name + "_cp" + str(copy_index)

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            # save label as png for displaying in html
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)
            # display input_label in html
            if "input_label" == label:
                ims.append(image_name)
                txts.append(label)
                links.append(image_name)

            if "real_image" == label or "synthesized_image" == label:
                # save synthesized_images and real_images in tif format separately
                image_tif_name = os.path.join("%s_tif" % (label), '%s.tif' % (name))
                save_tif_path = os.path.join(image_dir, image_tif_name)
                util.save_image(image_numpy, save_tif_path, create_dir=True)

                # extract rgb channel and save for displaying in html
                image_rgb_name = os.path.join("%s_rgb" % (label), '%s_rgb.png' % (name))
                image_rgb_path = os.path.join(image_dir, image_rgb_name)
                rgb_numpy = util.extract_rgb(save_path)
                util.save_image(rgb_numpy, image_rgb_path, create_dir=True)
                # display in html
                ims.append(image_rgb_name)
                txts.append("%s_rgb" % (label))
                links.append(image_rgb_name)

                # extract single nir channel and save for displaying in html
                image_nir_name = os.path.join("%s_nir" % (label), '%s_nir.png' % (name))
                image_nir_path = os.path.join(image_dir, image_nir_name)
                nir_numpy = util.extract_nir(save_path)
                util.save_image(nir_numpy, image_nir_path, create_dir=True)
                # display in html
                ims.append(image_nir_name)
                txts.append("%s_nir" % (label))
                links.append(image_nir_name)
                
        webpage.add_images(ims, txts, links, width=self.win_size)
