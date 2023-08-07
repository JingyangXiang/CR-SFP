import argparse

import torch

from utils.net_utils import time_file_str


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--save_dir', type=str, default='./', help='Folder to save checkpoints and log.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=1000, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-schedule', default='step', choices=['step', 'cos'], type=str, help='lr scheduler')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--use-pretrain', dest='use_pretrain', action='store_true',
                        help='use pre-trained model or not')

    # compress rate
    parser.add_argument('--prune-rate', type=float, default=0.3, help='compress rate of model')
    parser.add_argument('--epoch-prune', type=int, default=1, help='compress layer of model')
    parser.add_argument('--lr-adjust', type=int, default=30, help='number of epochs that change learning rate')

    # CR-SFP
    parser.add_argument('--alpha', default=1., type=float, help='KL importance')
    parser.add_argument('--two-crop', dest='two_crop', action='store_true', help='use two corp')
    parser.add_argument('--loss-type', default='ce+kl', choices=['ce+kl', 'ce+cos', 'ce'], help='loss type')
    parser.add_argument('--prune-criterion', default='l2', choices=['l2', 'fpgm', 'taylor'], help='prune criterion')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    assert args.use_cuda, "torch.cuds is not available!"
    args.prefix = time_file_str()

    return args
