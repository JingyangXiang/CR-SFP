from __future__ import division

import random
import shutil
import time
from functools import partial

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.nn import functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr_schedule(args):
    if args.lr_schedule == 'step':
        lr_schedule = partial(step_lr, args=args)
    elif args.lr_schedule == 'cos':
        lr_schedule = partial(cos_lr, args=args)
    else:
        raise NotImplementedError
    return lr_schedule


def step_lr(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_adjust))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cos_lr(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def print_log(print_string, log):
    print("{:} {:}".format(time_string(), print_string))
    log.write('{:} {:}\n'.format(time_string(), print_string))
    log.flush()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string + '-{}'.format(random.randint(1, 10000))


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def bikl(mask_output, no_mask_output):
    loss1 = F.kl_div(F.log_softmax(mask_output, dim=-1), F.softmax(no_mask_output.detach(), dim=-1),
                     reduction='batchmean')
    loss2 = F.kl_div(F.log_softmax(no_mask_output, dim=-1), F.softmax(mask_output.detach(), dim=-1),
                     reduction='batchmean')
    return 0.5 * (loss1 + loss2)
    # return loss1


def mutual_kl(model, input, label, loss_func, alpha=1.):
    assert isinstance(input, list) and len(input) == 2
    # input: no_mask_forward_type
    model.apply(set_mask_forward_true)
    mask_output = model(input[0])
    model.apply(set_mask_forward_false)
    no_mask_output = model(input[1], no_mask=True)
    # output: no_mask_forward_type
    loss1 = (loss_func(mask_output, label) + loss_func(no_mask_output, label))
    # 交叉熵非常容易很靠近
    loss2 = bikl(mask_output, no_mask_output)
    loss = loss1 + alpha * loss2
    return loss, no_mask_output


def bicos(mask_output, no_mask_output):
    mask_output = F.normalize(mask_output, dim=-1)
    no_mask_output = F.normalize(no_mask_output, dim=-1)
    loss1 = F.cosine_similarity(mask_output, no_mask_output.detach()).mean()
    loss2 = F.cosine_similarity(no_mask_output, mask_output.detach()).mean()
    return -0.5 * (loss1 + loss2)


def mutual_cos(model, input, label, loss_func, alpha=1.):
    assert isinstance(input, list) and len(input) == 2
    # input: no_mask_forward_type
    model.apply(set_mask_forward_true)
    mask_output = model(input[0])
    model.apply(set_mask_forward_false)
    no_mask_output = model(input[1], no_mask=True)
    # output: no_mask_forward_type
    loss1 = (loss_func(mask_output, label) + loss_func(no_mask_output, label))
    # 交叉熵非常容易很靠近
    loss2 = bicos(mask_output, no_mask_output)
    loss = loss1 + alpha * loss2
    return loss, no_mask_output


def set_mask_forward_true(module):
    if hasattr(module, "mask"):
        module.set_mask_forward_true()


def set_mask_forward_false(module):
    if hasattr(module, "mask"):
        module.set_mask_forward_false()


def init_mask(module, prune_rate, prune_criterion):
    if hasattr(module, "init_mask"):
        module.init_mask(prune_rate, prune_criterion)


def do_weight_mask(module):
    if hasattr(module, "do_weight_mask"):
        module.do_weight_mask()


def do_grad_mask(module):
    if hasattr(module, "do_grad_mask"):
        module.do_grad_mask()
