from PIL import Image
import argparse
import os
import shutil
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet
from utils import reparameterize


# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, dest='batch_size',
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--lr-step', '--lr_step', default=1000, type=int,
                    help='The step size of learning rate changes')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=101, type=int,
                    help='total number of layers (default: 101)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum (default: 0.9)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='ResNet_101_visDA', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')

parser.add_argument('--src-root', default=None, type=str, dest='src_root',
                    help='address of source data root folder')
parser.add_argument('--src-gt-list', default=None, type=str, dest='src_gt_list',
                    help='the source image_label list for evaluation, which are not changed')

parser.add_argument('--tgt-root', default=None, type=str, dest='tgt_root',
                    help='address of target data root folder')
parser.add_argument('--tgt-train-list', default=None, type=str, dest='tgt_train_list',
                    help='the target image_label list in training/self-training process, which may be updated dynamically')
parser.add_argument('--tgt-gt-list', default=None, type=str, dest='tgt_gt_list',
                    help='the target image_label list for evaluation, which are not changed')

parser.add_argument('--mr-weight-l2', default=0., type=float, dest='mr_weight_l2',
                    help='weight of l2 model regularization')
parser.add_argument('--mr-weight-negent', default=0., type=float, dest='mr_weight_negent',
                    help='weight of negative entropy model regularization')
parser.add_argument('--mr-weight-kld', default=0., type=float, dest='mr_weight_kld',
                    help='weight of kld model regularization')

parser.add_argument('--ls-weight-l2', default=0., type=float, dest='ls_weight_l2',
                    help='weight of l2 label smoothing')
parser.add_argument('--ls-weight-negent', default=0., type=float, dest='ls_weight_negent',
                    help='weight of negative entropy label smoothing')
parser.add_argument('--ls-weight-kld', default=0., type=float, dest='ls_weight_kld',
                    help='weight of kld label smoothing')

parser.add_argument('--num-class', default=None, type=int, dest='num_class',
                    help='the number of classes')
parser.add_argument('--gpus', default=0, type=int,
                    help='the number of classes')

parser.add_argument('--T', type=int, default=10)
parser.add_argument('--T-train', type=int, default=10)
parser.add_argument('--bayesian', action='store_true')
parser.add_argument('--bb', action='store_true')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.name))

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224,scale=(0.7,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    kwargs = {'num_workers': 1, 'pin_memory': True}
    trainset = ImageClassdata(txt_file=args.src_gt_list, root_dir=args.src_root, transform=data_transforms['train'])
    valset = ImageClassdata(txt_file=args.tgt_gt_list, root_dir=args.tgt_root, transform=data_transforms['val'])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    # model = models.resnet50(pretrained=True)
    # # two fc layers
    # # num_ftrs = model.fc.in_features
    # # fc_layers = nn.Sequential(
    # #     nn.Linear(num_ftrs, 2048),
    # #     nn.ReLU(inplace=True),
    # #     nn.Linear(2048, args.num_class),
    # # )
    # # one fc layer
    # num_ftrs = model.fc.in_features  # in_features_dimension = 2048
    # fc_layers = nn.Sequential(
    #     nn.Linear(num_ftrs, args.num_class),
    # )
    # model.fc = fc_layers

    if args.bayesian:
        model = resnet.bresnet50(bb=args.bb, pretrained=True)
        num_ftrs = model.fc.in_features
        logvar_layers = nn.Sequential(
            nn.Linear(num_ftrs, args.num_class),
        )
        model.logvar = logvar_layers
    else:
        model = resnet.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
    fc_layers = nn.Sequential(
        nn.Linear(num_ftrs, args.num_class),
    )
    model.fc = fc_layers

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to sdef RegCrossEntropyLoss(outputs, labels, weight_l2, weight_negent, weight_kld,num_class):

    # specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    # for training on multiple GPUs.
    device = torch.device("cuda:"+str(args.gpus))
    model = model.to(device)

    # define loss function (criterion) and pptimizer

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=ImageClassdata> no checkpoint found at '{}'".format(args.resume))

    # setting cudnn manualy?
    # cudnn.benchmark = True

    # all parameters are being optimized
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(args.momentum, 0.999)) # beta 1 is the momentum
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    if args.bayesian:
        train_fn = train_bayes
        validate_fn = validate_bayes
    else:
        train_fn = train
        validate_fn = validate

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_fn(train_loader, model, optimizer, scheduler, epoch, device)

        # evaluate on validation set
        prec1 = validate_fn(val_loader, model, epoch, device)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, optimizer, scheduler, epoch, device):
    """Train for one epoch on the training set"""
    scheduler.step()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, label) in enumerate(train_loader):
        label = label.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        loss = RegCrossEntropyLoss(output, label, args.mr_weight_l2, args.mr_weight_negent, args.mr_weight_kld, args.num_class)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, label, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)


def train_bayes(train_loader, model, optimizer, scheduler, epoch, device):
    """Train for one epoch on the training set"""
    scheduler.step()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, label) in enumerate(train_loader):
        label = label.to(device)
        input = input.to(device)

        # compute output
        y_softmax = 0.
        for t in range(args.T_train):
            y_pred, s_pred = model(input)
            y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
        loss = RegCrossEntropyLoss_bayes(y_softmax, label, args.mr_weight_l2, args.mr_weight_negent, args.mr_weight_kld, args.num_class)

        # measure accuracy and record loss
        prec1 = accuracy(y_softmax.data, label, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}], '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)


def validate(val_loader, model, epoch, device):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, label) in enumerate(val_loader):
            label = label.to(device)
            input = input.to(device)

            # compute output
            output = model(input)
            loss = RegCrossEntropyLoss(output, label, args.mr_weight_l2, args.mr_weight_negent, args.mr_weight_kld, args.num_class)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, label, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}], '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                      .format(i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


def validate_bayes(val_loader, model, epoch, device):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, label) in enumerate(val_loader):
            label = label.to(device)
            input = input.to(device)

            # compute output
            y_softmax = 0.
            for t in range(args.T):
                y_pred, s_pred = model(input)
                y_softmax += 1. / args.T * F.softmax(reparameterize(y_pred, s_pred), dim=1)
            loss = RegCrossEntropyLoss_bayes(y_softmax, label, args.mr_weight_l2, args.mr_weight_negent, args.mr_weight_kld, args.num_class)

            # measure accuracy and record loss
            prec1 = accuracy(y_softmax.data, label, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}], '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                      .format(i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


def RegCrossEntropyLoss(outputs, labels, mr_weight_l2, mr_weight_negent, mr_weight_kld,num_class):
    batch_size = labels.size(0)            # batch_size

    softmax = F.softmax(outputs, dim=1)   # compute the log of softmax values
    logsoftmax = F.log_softmax(outputs, dim=1)   # compute the log of softmax values

    ce = torch.sum( -logsoftmax*labels ) # cross-entropy loss with vector-form softmax
    l2 = torch.sum( softmax*softmax )
    negent = torch.sum( softmax*logsoftmax )
    kld = torch.sum( -logsoftmax/float(num_class) )

    reg_ce = ce + mr_weight_l2*l2 + mr_weight_negent*negent + mr_weight_kld*kld

    return reg_ce/batch_size


def RegCrossEntropyLoss_bayes(softmax, labels, mr_weight_l2, mr_weight_negent, mr_weight_kld,num_class):
    batch_size = labels.size(0)            # batch_size

    logsoftmax = F.log_softmax(softmax, dim=1)   # compute the log of softmax values

    ce = torch.sum( -logsoftmax*labels ) # cross-entropy loss with vector-form softmax
    l2 = torch.sum( softmax*softmax )
    negent = torch.sum( softmax*logsoftmax )
    kld = torch.sum( -logsoftmax/float(num_class) )

    reg_ce = ce + mr_weight_l2*l2 + mr_weight_negent*negent + mr_weight_kld*kld

    return reg_ce / batch_size


class ImageClassdata(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, txt_file, root_dir, transform=transforms.ToTensor()):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_frame = open(txt_file, 'r').readlines()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        ImgName_Label = str.split(self.images_frame[idx])
        img_name = os.path.join(self.root_dir,ImgName_Label[0])
        img = Image.open(img_name)
        image = img.convert('RGB')
        lbl = np.asarray(ImgName_Label[1:],dtype=np.float32)
        label = torch.from_numpy(lbl)

        if self.transform:
            image = self.transform(image)

        return image,label


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'epoch_'+str(state['epoch']) + '_' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')


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
    _, gt = target.topk(maxk, 1, True, True)
    pred = pred.t()
    gt = gt.t()
    correct = pred.eq(gt)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
