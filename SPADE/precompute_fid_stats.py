# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import os
from options.train_options import TrainOptions
import data
from fid.fid_score import calculate_activation_statistics, save_statistics, compute_statistics_of_generator, load_statistics, calculate_frechet_distance
from fid.inception import InceptionV3
# from util.datasets import get_loaders_evalimport data
from itertools import chain


def main(opt):
    device = 'cuda:'+str(opt.gpu_ids[0])
    # device = 'cuda'
    # device = 'cpu'
    dims = 2048
    # for binary datasets including MNIST and OMNIGLOT, we don't apply binarization for FID computation
    # load the dataset for training
    train_dataloader = data.create_dataloader(opt)
    
    # train_queue, valid_queue, _ = get_loaders_eval(args.dataset, args.data, args.distributed,
    #                                                args.batch_size, augment=False, drop_last_train=False,
    #                                                shuffle_train=True, binarize_binary_datasets=False)
    # print('len train queue', len(train_queue), 'len val queue', len(valid_queue), 'batch size', args.batch_size)
    # if args.dataset in {'celeba_256', 'omniglot'}:
    #     train_queue = chain(train_queue, valid_queue)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    fid_dir = os.path.join(opt.checkpoints_dir, 'fid')
    os.mkdir(fid_dir)
    model = InceptionV3([block_idx]).to(device)
    m_t, s_t = calculate_activation_statistics(train_dataloader, model, batch_size=opt.batchSize, dims = dims, device = device)
    # m, s = compute_statistics_of_generator(train_queue, model, args.batch_size, dims, device, args.max_samples)
    file_path = os.path.join(fid_dir, opt.name + '.npz')
    save_statistics(file_path, m_t, s_t)


if __name__ == '__main__':
    # python precompute_fid_statistics.py --dataset cifar10
    # parser = argparse.ArgumentParser('')
    # parser.add_argument('--dataset_mode', type=str, default='naip', help='which dataset to use')
    # parser.add_argument('--dataroot', type=str, default='/tmp/nvae-diff/data',
    #                     help='location of the data corpus')
    # parser.add_argument('--batch_size', type=int, default=64,
    #                     help='batch size per GPU')
    # parser.add_argument('--max_samples', type=int, default=50000,
    #                     help='batch size per GPU')
    # parser.add_argument('--fid_dir', type=str, default='/tmp/nvae-diff/fid-stats',
    #                     help='A dir to store fid related files')

    # args = parser.parse_args()
    # args.distributed = False
    opt = TrainOptions().parse()

    main(opt)