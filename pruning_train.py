# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import os
import sys
import time
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models

import models
from args import parse_arguments
from data.imagenet_dali import ImageNetDali
from utils.engine import train_engine, validate
from utils.net_utils import init_mask, do_weight_mask, get_lr_schedule
from utils.net_utils import save_checkpoint, print_log, time_string, AverageMeter, convert_secs2time


def main():
    best_prec1 = 0
    args = parse_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    log = open(os.path.join(args.save_dir, '{}.{}.log'.format(args.arch, args.prefix)), 'w')

    # version information
    print_log("PyThon  version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch version : {}".format(torch.__version__), log)
    print_log("CUDNN   version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Vision  version : {}".format(torchvision.__version__), log)

    # create model
    print_log("Creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](pretrained=args.use_pretrain, num_classes=1000)
    print_log("Model : {}".format(model), log)
    print_log("Parameters: {}".format(args), log)
    print_log("Compress Rate: {}".format(args.prune_rate), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("Workers         : {}".format(args.workers), log)
    print_log("Learning-Rate   : {}".format(args.lr), log)
    print_log("Use Pre-Trained : {}".format(args.use_pretrain), log)
    print_log("lr adjust : {}".format(args.lr_adjust), log)

    # accelerate
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # init model
    model = model.cuda()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    # define lr schedule
    lr_schedule = get_lr_schedule(args)

    # init dataloader
    data_loader = ImageNetDali(args)
    train_loader = data_loader.train_loader
    val_loader = data_loader.val_loader

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)

    if args.evaluate:
        validate(val_loader, model, criterion, log, print_log, args)
        return

    # init path
    filename = os.path.join(args.save_dir, 'checkpoint.{:}.{:}.pth.tar'.format(args.arch, args.prefix))
    bestname = os.path.join(args.save_dir, 'best.{:}.{:}.pth.tar'.format(args.arch, args.prefix))

    # val acc before mask
    print_log(">>>>> accu before is: {:}".format(validate(val_loader, model, criterion, log, print_log, args)), log)

    # init mask -> do weight mask
    model.apply(partial(init_mask, prune_rate=args.prune_rate, prune_criterion=args.prune_criterion))
    model.apply(do_weight_mask)

    # val acc after mask
    print_log(">>>>> accu after is: {:}".format(validate(val_loader, model, criterion, log, print_log, args)), log)

    # start train
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        lr_schedule(optimizer, epoch)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.val * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' [{:s}] :: {:3d}/{:3d} ----- [{:s}] {:s}'.format(
            args.arch, epoch, args.epochs, time_string(), need_time), log)

        # train for one epoch
        train_engine(train_loader, model, criterion, optimizer, epoch, log, print_log, args)

        # evaluate on validation set before mask
        validate(val_loader, model, criterion, log, print_log, args)

        # do mask
        if (epoch % args.epoch_prune == 0 or epoch == args.epochs - 1):
            model.apply(partial(init_mask, prune_rate=args.prune_rate, prune_criterion=args.prune_criterion))
            model.apply(do_weight_mask)

        # evaluate on validation set before mask
        val_acc = validate(val_loader, model, criterion, log, print_log, args)
        print_log(f'=> Epoch: {epoch}, Acc: {val_acc:.2f}%', log)
        torch.cuda.empty_cache()

        # remember best prec@1 and save checkpoint
        is_best = val_acc > best_prec1
        best_prec1 = max(val_acc, best_prec1)
        save_checkpoint({'epoch': epoch + 1, 'arch': args.arch,
                         'state_dict': model.state_dict(), 'best_prec1': best_prec1,
                         'optimizer': optimizer.state_dict(), }, is_best, filename, bestname)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()


if __name__ == '__main__':
    main()
