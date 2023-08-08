import time

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from utils.net_utils import AverageMeter, accuracy, mutual_kl, mutual_cos
from utils.net_utils import do_grad_mask

scaler = GradScaler()


def train_engine(train_loader, model, criterion, optimizer, epoch, log, print_log, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target_var = target.cuda(non_blocking=True)

        if args.two_crop:
            assert isinstance(input, list)
            input_var = [input[0].cuda(non_blocking=True), input[1].cuda(non_blocking=True)]
        else:
            input_var = [input.cuda(non_blocking=True), ] * 2

        # compute output
        with autocast():
            if args.loss_type == 'ce+kl':
                loss, output = mutual_kl(model, input_var, target_var, criterion, alpha=args.alpha)
                if args.symmetric:
                    loss2, output = mutual_kl(model, input_var[::-1], target_var, criterion, alpha=args.alpha)
                    loss = (loss + loss2) * 0.5
            elif args.loss_type == 'ce+cos':
                loss, output = mutual_cos(model, input_var, target_var, criterion, alpha=args.alpha)
                if args.symmetric:
                    raise NotImplementedError
            elif args.loss_type == 'ce':
                assert args.alpha == 0
                loss, output = mutual_kl(model, input_var, target_var, criterion, alpha=args.alpha)
                if args.symmetric:
                    raise NotImplementedError

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), output.size(0))
        top1.update(prec1.item(), output.size(0))
        top5.update(prec5.item(), output.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # same to fpgm paper
        if args.prune_criterion == 'fpgm':
            # Mask grad for iteration
            model.apply(do_grad_mask)

        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5), log)


def validate(val_loader, model, criterion, log, print_log, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5), log)

    print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
        top1=top1, top5=top5, error1=100 - top1.avg), log)

    return top1.avg
