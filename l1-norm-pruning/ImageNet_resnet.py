from fastai.vision.all import *
import argparse
import os
import numpy as np
import shutil
import time
import gc
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from prettytable import PrettyTable
from torch.optim.lr_scheduler import StepLR
import torchvision
import resnet
import resnet_group_convolutions
from compute_flops import count_model_param_flops
import resnet_separable_conv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import tensorflow as tf

print(tf.__version__)
# Commenting out of saving
print(tf.config.list_physical_devices('GPU'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("THIS IS THE CURRENT DEVIE : ", device)
# Main local imports :S

# source = untar_data(URLs.IMAGENETTE_160)


# python main_B.py --arch resnet34 --scratch [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# C:\Users\user\.fastai\data\imagenette2-160 path that the data are saved
# imagenette_path = 'C:/Users/user/.fastai/data/imagenette2-160/'
# export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/nas/home/ssandikci/.conda/envs/model-zoo-release/lib
parser.add_argument('--data', type=str, default='/nas/datasets/IMAGENET2012/',
                    help='path to dataset')  # deleted metavar=DIR
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1000, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--s', type=float, default=0.0001, help='scale sparse rate (default: 0.0001)')
parser.add_argument('--save',
                    default='/nas/home/tgiannilias/ABCPruner_Experiments/ABCPruner_Experiments/New_Experiments/rethinking-network-pruning/Group_Convolutions/Saved_Models/',
                    type=str, metavar='PATH', help='path to save prune model (default: current directory)')
parser.add_argument('--scratch', default='', type=str, metavar='PATH', help='the PATH to the pruned model')


# /nas/home/tgiannilias/ABCPruner_Experiments/ABCPruner_Experiments/New_Experiments/rethinking-network-pruning/Group_Convolutions/Saved_Models/resnet18_groups2_IMAGENET.pth.tar
# /nas/home/tgiannilias/ABCPruner_Experiments/ABCPruner_Experiments/New_Experiments/rethinking-network-pruning/Group_Convolutions/Saved_Models/checkpoint_groups2_resnet_IMAGENET.pth.tar

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")


# /nas/home/tgiannilias/ABCPruner_Experiments/ABCPruner_Experiments/New_Experiments/rethinking-network-pruning/Group_Convolutions/Saved_Models/checkpoint_groups2_resnet_IMAGENET.pth.tar
best_prec1 = 0


def main():
    global args, best_prec1
    args, known = parser.parse_known_args()
    args.distributed = args.world_size > 1

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size)

    if args.scratch:
        checkpoint = torch.load(args.scratch)
        print("Scratch configuration of model loaded")
        # model = models.resnet34(args.cfg)
        # model = resnet34(cfg=checkpoint['cfg'])
        model = resnet.resnet34(cfg=checkpoint['cfg'])
    else:
        model = resnet_separable_conv.resnet34()
    # Define model and set optimizers/criterion/loss etc.

    print("PASSED CHECKPOINT CONFIGURATION")
    count_parameters(model)
    count_model_param_flops(model)
    print("Current learning rate : ", args.lr)

    # model_ref = torchvision.models.resnet34()
    # Uncomment if you want to train with formula of rethinking-network-pruning
    """
    flops_std = count_model_param_flops(model_ref)
    flops_small = count_model_param_flops(model)
    ratio = flops_std / flops_small
    if ratio >= 2:
        args.epochs = 180
        stp_sz = 60
    else:
        args.epochs = int(90 * ratio)
        stp_sz = int(args.epochs / 3)
"""
    args.epochs = int(100)
    stp_sz = int(args.epochs / 3)
    model = torch.nn.DataParallel(model).cuda()  # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    step_size = StepLR(optimizer, step_size=stp_sz, gamma=0.1)
    cudnn.benchmark = True

    print("EPOCHS :", args.epochs)
    print("STEP SIZE:", stp_sz)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("Mpainei edw")
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            print("BEST PRECISION", best_prec1)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # srun --gpus 1 --mem 10G -c 6 --pty /bin/bash
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    print("Passes this scope \n")
    print("DATASETS LOADED IMAGENETTE")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    print("Loading the loaders of the DATASET")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    #    for i, (input, target) in enumerate(train_loader):
    #        print("Input:", input)
    #        print("Target:", target)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    history_score = np.zeros((args.epochs + 1, 1))
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        step_size.step()

        history_score[epoch] = prec1.cpu()
        np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    history_score[-1] = best_prec1.cpu()
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        images = torch.autograd.Variable(images)
        target = torch.autograd.Variable(target)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            images = torch.autograd.Variable(images, volatile=True)
            target = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best):
    torch.save(state, os.path.join(args.save, 'ResNet34_Separable_Groups4.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(args.save, 'ResNet34_Separable_Groups4.pth.tar'),
                        os.path.join(args.save, 'Best_ResNet34_Separable_Groups4.pth.tar'))


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()







