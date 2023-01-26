"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.early_stopping import EarlyStopping
from util.visualizer import Visualizer
from util.plt_loss_log import plot_loss

from trainers.pix2pix_trainer import Pix2PixTrainer


# parse options
opt = TrainOptions().parse()


# print options to help debugging
print(' '.join(sys.argv))

# load the dataset for training
train_dataloader = data.create_dataloader(opt)


cur = opt.batchSize
# load the dataset for validation
if opt.add_validation:
    opt.phase = "val"
    opt.batchSize = 5
    valid_dataloader = data.create_dataloader(opt)
    
opt.batchSize = cur

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(train_dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

# create tool for early stopping
early_stopping = EarlyStopping(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)

    # training process
    for i, data_i in enumerate(train_dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        # train discriminator every iteration, but train generator every opt.D_steps_per_G iterations. 
        if i % opt.D_steps_per_G == 0: # number of discriminator iterations per generator iterations.
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)

    # validation process
    if opt.add_validation:
        
        print("Validation:")
        opt_copy = TrainOptions().parse()
        opt_copy.batchSize = opt.batchSize
        for i, data_i in enumerate(valid_dataloader):
            # iter_counter.record_one_iteration()
            # validation
            
            trainer.validation(data_i, opt_copy)

            # need to add FID!

            # Visualizations
            # if iter_counter.needs_printing():
        # losses = trainer.get_latest_losses()
        # print(losses)
        fid = trainer.get_fid()
        count_nobetter, patience = early_stopping.get_earlystop_state()
        early_stopping(fid)
        # print(fid)
        visualizer.print_current_fid(epoch, iter_counter.epoch_iter,
                                        fid, iter_counter.time_per_iter)
        visualizer.print_current_earlystop_state(epoch, iter_counter.epoch_iter,
                                        count_nobetter, patience, iter_counter.time_per_iter)
        # visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
        #                                         losses, iter_counter.time_per_iter)
        # visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
        # check whether to early stop
        if early_stopping.early_stop:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)
            print('Training early stopped.')
            break


    iter_counter.record_epoch_end()


    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')

# once training has finished, create loss_log visualization
plot_loss(opt.name)
