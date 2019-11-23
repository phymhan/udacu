from PIL import Image
from operator import itemgetter
import argparse
import os
import shutil
import time
import numpy as np
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import network_architectures
from utils import print_options, makedir, save_checkpoint, compute_grad2, forward, get_param_list, get_minibatch_size
import utils
from copy import deepcopy
from torch import autograd
import random
import pdb

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, dest='start_epoch', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=100, type=int, dest='batch_size', help='mini-batch size (default: 64) for training')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--layers', default=101, type=int, help='total number of layers (default: 101), only 152,101,50,18 are supported')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='ResNet_101_visDA', type=str, help='name of experiment')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')

# model training
parser.add_argument('--adapt-setting', default='mnist_usps', type=str, dest='adapt_setting', help='The domain adaptation setting')
parser.add_argument('--src-root', default=None, type=str, dest='src_root', help='address of source data root folder')
parser.add_argument('--src-train-list', default='src_train_list.txt', type=str, dest='src_train_list', help='the source image_label list for training, which can be changed in terms of the item labels (not the labels)')
parser.add_argument('--src-gt-list', default=None, type=str, dest='src_gt_list', help='the source image_label list for evaluation, which are not changed')
parser.add_argument('--tgt-root', default=None, type=str, dest='tgt_root', help='address of target data root folder')
parser.add_argument('--tgt-train-list', default='tar_train_list.txt', type=str, dest='tgt_train_list', help='the target image_label list in training/self-training process, which may be updated dynamically')
parser.add_argument('--tgt-gt-list', default=None, type=str, dest='tgt_gt_list', help='the target image_label list for evaluation, which are not changed')
parser.add_argument('--mr-weight-l2', default=0., type=float, dest='mr_weight_l2', help='weight of l2 model regularization')
parser.add_argument('--mr-weight-negent', default=0., type=float, dest='mr_weight_negent', help='weight of negative entropy model regularization')
parser.add_argument('--mr-weight-kld', default=0., type=float, dest='mr_weight_kld', help='weight of kld model regularization')
parser.add_argument('--ls-weight-l2', default=0., type=float, dest='ls_weight_l2', help='weight of l2 label smoothing')
parser.add_argument('--ls-weight-negent', default=0., type=float, dest='ls_weight_negent', help='weight of negative entropy label smoothing')
parser.add_argument('--num-class', default=None, type=int, dest='num_class', help='the number of classes')
parser.add_argument('--num-gpus', type=int, default=1, help='the number of gpus, 0 for cpu')

# self-trained network
parser.add_argument('--kc-policy', default='global', type=str, dest='kc_policy', help='The policy to determine kc. Valid values: "global" for global threshold, "cb" for class-balanced threshold, "rcb" for reweighted class-balanced threshold')
parser.add_argument('--init-tgt-port', default=0.3, type=float, dest='init_tgt_port', help='The initial portion of target to determine kc')
parser.add_argument('--max-tgt-port', default=0.6, type=float, dest='max_tgt_port', help='The max portion of target to determine kc')
parser.add_argument('--tgt-port-step', default=0.05, type=float, dest='tgt_port_step', help='The portion step in target domain in every round of self-paced self-trained neural network')
parser.add_argument('--init-src-port', default=0.5, type=float, dest='init_src_port', help='The initial portion of source portion for self-trained neural network')
parser.add_argument('--max-src-port', default=0.8, type=float, dest='max_src_port', help='The max portion of source portion for self-trained neural network')
parser.add_argument('--src-port-step', default=0.05, type=float, dest='src_port_step', help='The portion step in source domain in every round of self-paced self-trained neural network')
parser.add_argument('--init-randseed', default=0, type=int, dest='init_randseed', help='The initial random seed for source selection')
parser.add_argument('--lr-stepsize', default=7, type=int, dest='lr_stepsize', help='The step size of lr_stepScheduler')
parser.add_argument('--lr-stepfactor', default=0.1, type=float, dest='lr_stepfactor', help='The step factor of lr_stepScheduler')
parser.add_argument('--img-type', default='grayscale', type=str, dest='img_type', help='The type of images: grayscale or RGB')
parser.add_argument('--bayesian', action='store_true')
parser.add_argument('--entropy', type=str, default='min', help='Shannon, collision, min')
parser.add_argument('--T', type=int, default=10)
parser.add_argument('--T-train', type=int, default=10)
parser.add_argument('--arch', type=str, default='DTN')
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--checkpoints-dir', type=str, default='runs')
parser.add_argument('--save-list-to-txt', action='store_true')
parser.add_argument('--save-epoch-freq', type=int, default=1)
parser.add_argument('--mirror-print', action='store_true')
parser.add_argument('--minibatch-size', type=int, default=10)
parser.add_argument('--num-minibatch', type=int, default=10)
parser.add_argument('--inner-iters', type=int, default=1)
parser.add_argument('--inner-stepsize', type=float, default=0.02)
parser.add_argument('--shuffle-off', action='store_true')
parser.add_argument('--lambda-minibatch-var', type=float, default=-1.0)
parser.add_argument('--lambda-grad-penalty', type=float, default=1.0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--reg-type', type=str, default='none')
parser.add_argument('--reg-domain', type=str, nargs='+', default=['tar'])  # FIXME: temp solution
parser.add_argument('--no-aleatoric', action='store_true')
parser.add_argument('--mix', action='store_true')
parser.add_argument('--optimizer', type=str, default='adam', help='adam: orig adam; adam2: adam in wgan-gp; rmsprop')
parser.add_argument('--rmsprop-alpha', type=float, default=0.99)
parser.add_argument('--norm-layer', type=str, default='batch')
parser.add_argument('--lambda-tar-ce', type=float, default=1.0)
parser.add_argument('--gp-type', type=str, default='two-side')
parser.add_argument('--gp-center', type=float, default=0)
parser.add_argument('--var-param', type=str, default='all')
parser.add_argument('--gp-param', type=str, default='0')
parser.add_argument('--no-trick', type=str, default='', help='random or reverse')
parser.add_argument('--use-entropy-loss', action='store_true', help='use entropy loss when computing MBV')
parser.add_argument('--filter-out-invalid', action='store_true')
parser.add_argument('--use-val-transforms', action='store_true')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)
parser.set_defaults(save_list_to_txt=True)

global MAGIC_EPS
best_prec1 = 0.
MAGIC_EPS = 1e-32


def main():
    global args, best_prec1, reparameterize
    args = parser.parse_args()

    # torch.backends.cudnn.deterministic = True
    np.random.seed(args.init_randseed)
    random.seed(args.init_randseed)
    torch.manual_seed(args.init_randseed)

    args.expr_dir = os.path.join(args.checkpoints_dir, args.name)
    makedir(args.expr_dir)
    print_options(parser, args)
    if args.tensorboard:
        configure(args.expr_dir)
    if args.no_aleatoric:
        reparameterize = utils.reparameterize_straight_through
    else:
        reparameterize = utils.reparameterize

    logger = set_logger(args.expr_dir, args.name+'.log')
    logger.info('start with arguments %s', args)

    if args.adapt_setting == 'mnist2usps' or args.adapt_setting == 'usps2mnist':
        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'val': transforms.Compose([
                # transforms.Resize(28),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'val4mix': transforms.Compose([
                # transforms.Resize(28),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
        }
    elif args.adapt_setting == 'svhn2mnist':
        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'val': transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'val4mix': transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
        }
    elif args.adapt_setting == 'mnist2svhn':
        data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'val': transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'val4mix': transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
        }
    elif args.adapt_setting == 'semi-svhn':
        data_transforms = {
            'train': transforms.Compose([
                # transforms.Resize(32),
                transforms.RandomAffine(degrees=5, scale=(0.95, 1.05), resample=Image.BICUBIC, fillcolor=0),
                transforms.RandomCrop(32),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'val': transforms.Compose([
                # transforms.Resize(32),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'val4mix': transforms.Compose([
                # transforms.Resize(32),
                transforms.RandomAffine(degrees=5, scale=(0.95, 1.05), resample=Image.BICUBIC, fillcolor=0),
                transforms.RandomCrop(32),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
        }
    elif args.adapt_setting == 'semi-cifar10':
        data_transforms = {
            'train': transforms.Compose([
                # transforms.Resize(32),
                transforms.RandomAffine(degrees=5, scale=(0.95, 1.05), resample=Image.BICUBIC, fillcolor=0),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]),
            'val': transforms.Compose([
                # transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]),
            'val4mix': transforms.Compose([
                # transforms.Resize(32),
                transforms.RandomAffine(degrees=5, scale=(0.95, 1.05), resample=Image.BICUBIC, fillcolor=0),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]),
        }
    else:
        raise NotImplementedError

    kwargs = {'num_workers': args.num_workers, 'pin_memory': False, 'drop_last': True}

    args.src_train_list = os.path.join(args.expr_dir, os.path.basename(args.src_train_list))
    args.tgt_train_list = os.path.join(args.expr_dir, os.path.basename(args.tgt_train_list))
    args.minibatch_size = get_minibatch_size(args.batch_size, args.minibatch_size, args.num_minibatch)

    digit_valset = ImageClassdata(txt_file=args.tgt_gt_list, root_dir=args.tgt_root, img_type=args.img_type, transform=data_transforms['val'], domain=1)
    val_loader = torch.utils.data.DataLoader(digit_valset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    # if args.adapt_setting == 'mnist2usps':
    #     arch = 'mnist-LeNet'
    # elif args.adapt_setting == 'svhn2mnist':
    #     arch = 'DTN'
    # if args.arch:
    #     arch = args.arch
    # if args.bayesian and args.arch == '':
    #     arch = 'B' + arch
    net_class, expected_shape = network_architectures.get_net_and_shape_for_architecture(args.arch)
    model = net_class(args.num_class, args.norm_layer)
    # if args.adapt_setting == 'mnist_usps':
    #     arch = 'mnist-bn-32-64-256'
    # net_class, expected_shape = network_architectures.get_net_and_shape_for_architecture(arch)
    # model = net_class(args.num_class)

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to sdef RegCrossEntropyLoss(outputs, labels, weight_l2, weight_negent, weight_kld, num_class):

    # specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    # for training on multiple GPUs.
    #device = torch.device("cuda:"+args.gpus)
    device = torch.device("cuda")
    # device = torch.device("cpu")

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
            raise RuntimeError("=ImageClassdata> no checkpoint found at '{}'".format(args.resume))

    model = model.to(device)

    # setting cudnn manualy?
    # cudnn.benchmark = True

    param_list = get_param_list(model, args.var_param)
    param_list_gp = get_param_list(model, args.gp_param)

    # all parameters are being optimized
    # optimizer = torch.optim.SGD(model.parameters(), lr = args.lr,
    #                         momentum=args.momentum, nesterov=False,
    #                         weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    elif args.optimizer == 'adam2':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.9))
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.rmsprop_alpha)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise NotImplementedError
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize, gamma=args.lr_stepfactor)
    tgt_portion = args.init_tgt_port
    src_portion = args.init_src_port
    randseed = args.init_randseed

    if args.bayesian:
        train_fn = train_bayes
        validate_fn = validate_bayes
    else:
        train_fn = train
        validate_fn = validate

    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):

        # evaluate on validation set
        num_class = args.num_class
        # confidence vectors for target
        prec1, ImgName_list, pred_logit_list = validate_fn(val_loader, model, epoch, device, logger, param_list, param_list_gp)
        # generate kct
        pred_softlabel_list = soft_pseudo_label(pred_logit_list, num_class, args, device)
        kct_list = kc_parameters(tgt_portion, args.kc_policy, pred_logit_list, pred_softlabel_list, num_class, args, device)
        # next round's tgt portion
        tgt_portion = min(tgt_portion + args.tgt_port_step, args.max_tgt_port)
        # generate soft pseudo-labels
        # select good pseudo-labels for model retraining
        label_selection(pred_logit_list, pred_softlabel_list, kct_list, ImgName_list, args)
        # select part of source data for model retraining
        saveSRCtxt(src_portion, randseed, args)
        randseed = randseed + 1
        src_portion = min(src_portion + args.src_port_step, args.max_src_port)

        # train for one epoch
        # domain label: 0 for source, 1 for target
        digit_trainset = ImageClassdata(txt_file=args.src_train_list, root_dir=args.src_root, img_type=args.img_type, transform=data_transforms['train'], domain=0)
        digit_valset_pseudo = ImageClassdata(txt_file=args.tgt_train_list, root_dir=args.tgt_root, img_type=args.img_type,
                                             transform=data_transforms['val'] if args.use_val_transforms else data_transforms['val4mix'], domain=1)

        if args.mix:
            mix_trainset = torch.utils.data.ConcatDataset([digit_trainset, digit_valset_pseudo])
            mix_train_loader = torch.utils.data.DataLoader(mix_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
            train_fn(mix_train_loader, model, optimizer, scheduler, epoch, device, logger, 'mix-', param_list, param_list_gp)
        else:
            # train on src then train on tar
            src_train_loader = torch.utils.data.DataLoader(digit_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
            tar_train_loader = torch.utils.data.DataLoader(digit_valset_pseudo, batch_size=args.batch_size, shuffle=not args.shuffle_off, **kwargs)
            train_fn(src_train_loader, model, optimizer, scheduler, epoch, device, logger, 'src-', param_list, param_list_gp)
            logger.info('\n---')
            print('---')
            train_fn(tar_train_loader, model, optimizer, scheduler, epoch, device, logger, 'tar-', param_list, param_list_gp)
        logger.info('\n---')
        print('---')

        # remember best prec@1 and save checkpoint
        # prec1 = prec1.to(device)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best or epoch % args.save_epoch_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.cpu().state_dict(),
                'best_prec1': best_prec1,
            }, is_best, args.expr_dir)
            # model = model.to(device)
            if device.type.startswith('cuda'):
                model.cuda()
    print(f'Best accuracy: {best_prec1:.2f}%')


# def MiniBatchLogitLoss(model, param_list, data, args):
#     """
#     autograd.grad code borrowed from
#     https://github.com/dragen1860/MAML-Pytorch/blob/master/meta.py
#     :param model:
#     :param data:
#     :param args:
#     :return:
#     """
#     x, y = data
#     logits = []
#     for start in range(0, x.size(0), args.minibatch_size):
#         # for each minibatch, perform one gradient step
#         x_mb = x[start:start + args.minibatch_size, ...]
#         y_mb = y[start:start + args.minibatch_size, ...]
#         y_pred = model(x_mb)
#         loss = CrossEntropyLoss(y_pred, y_mb)
#         grad = autograd.grad(loss, model.parameters())
#         fast_weights = list(map(lambda p: p[1] - args.inner_stepsize * p[0], zip(grad, model.parameters())))
#         logits.append(forward(model, x, fast_weights, False).view(-1))
#     logits_matrix = torch.stack(logits)
#     return torch.mean(torch.var(logits_matrix, dim=0))


def MiniBatchVATVarLoss(model, param_list, data, args):
    """
    autograd.grad code borrowed from
    https://github.com/dragen1860/MAML-Pytorch/blob/master/meta.py
    :param model:
    :param data:
    :param args:
    :return:
    """
    x, y, z = data
    weights_flattened = []
    for start in range(0, x.size(0), args.minibatch_size):
        # for each minibatch, perform one gradient step
        x_mb = x[start:start + args.minibatch_size, ...]
        y_mb = y[start:start + args.minibatch_size, ...]
        y_pred = model(x_mb)
        if args.use_entropy_loss:
            z_mb = None if z is None else z[start:start + args.minibatch_size, ...]
            loss = CrossEntropyLoss(y_pred, y_mb, z_mb, args)
        else:
            loss = CrossEntropyLoss(y_pred, y_mb)
        grad = autograd.grad(loss, param_list)



        # fast_weights = list(map(lambda p: p[1] - args.inner_stepsize * p[0], zip(grad, model.parameters())))
        # weights_flattened.append(torch.cat([f.view(-1) for f in fast_weights]))
        weights_flattened.append(torch.cat([f.view(-1) for f in grad]))
    weights_matrix = torch.stack(weights_flattened) * args.inner_stepsize
    return torch.sum(torch.var(weights_matrix, dim=0))


def MiniBatchVarLoss(model, param_list, data, args):
    """
    autograd.grad code borrowed from
    https://github.com/dragen1860/MAML-Pytorch/blob/master/meta.py
    :param model:
    :param data:
    :param args:
    :return:
    """
    x, y, z = data
    weights_flattened = []
    for start in range(0, x.size(0), args.minibatch_size):
        # for each minibatch, perform one gradient step
        x_mb = x[start:start + args.minibatch_size, ...]
        y_mb = y[start:start + args.minibatch_size, ...]
        y_pred = model(x_mb)
        if args.use_entropy_loss:
            z_mb = None if z is None else z[start:start + args.minibatch_size, ...]
            loss = CrossEntropyLoss(y_pred, y_mb, z_mb, args)
        else:
            loss = CrossEntropyLoss(y_pred, y_mb)
        grad = autograd.grad(loss, param_list)
        # fast_weights = list(map(lambda p: p[1] - args.inner_stepsize * p[0], zip(grad, model.parameters())))
        # weights_flattened.append(torch.cat([f.view(-1) for f in fast_weights]))
        weights_flattened.append(torch.cat([f.view(-1) for f in grad]))
    weights_matrix = torch.stack(weights_flattened) * args.inner_stepsize
    return torch.sum(torch.var(weights_matrix, dim=0))


def MiniBatchVarLoss_bayes(model, param_list, data, args):
    """
    autograd.grad code borrowed from
    https://github.com/dragen1860/MAML-Pytorch/blob/master/meta.py
    :param model:
    :param data:
    :param args:
    :return:
    """
    x, y, z = data
    weights_flattened = []
    for start in range(0, x.size(0), args.minibatch_size):
        # for each minibatch, perform one gradient step
        x_mb = x[start:start + args.minibatch_size, ...]
        y_mb = y[start:start + args.minibatch_size, ...]
        y_softmax = 0.
        for t in range(args.T_train):
            y_pred, s_pred = model(x_mb)
            y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
        if args.use_entropy_loss:
            z_mb = None if z is None else z[start:start + args.minibatch_size, ...]
            loss = NLLLoss(y_softmax, y_mb, z_mb, args)
        else:
            loss = NLLLoss(y_softmax, y_mb)
        grad = autograd.grad(loss, param_list)
        # fast_weights = list(map(lambda p: p[1] - args.inner_stepsize * p[0], zip(grad, model.parameters())))
        # weights_flattened.append(torch.cat([f.view(-1) for f in fast_weights]))
        weights_flattened.append(torch.cat([f.view(-1) for f in grad]))
    weights_matrix = torch.stack(weights_flattened) * args.inner_stepsize
    return torch.sum(torch.var(weights_matrix, dim=0))


def GradPenaltyLoss(model, param_list, loss, args):
    """
    zero centered, part of the code borrowed from
    https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py
    :param model:
    :param loss:
    :param args:
    :return:
    """
    grad = autograd.grad(loss, param_list, create_graph=True, retain_graph=True, only_inputs=True)
    weights_flattened = torch.cat([f.view(-1) for f in grad])
    if args.gp_type == 'two-side':
        loss_gp = (weights_flattened.pow(2).mean().sqrt() - args.gp_center).pow(2)  # for historical reasons
    elif args.gp_type == 'one-side':
        loss_gp = weights_flattened.pow(2).mean().sqrt() - args.gp_center
        loss_gp = max(loss_gp, 0. * loss_gp)
    else:
        raise NotImplementedError
    # print('---')
    # print(loss_gp)
    return loss_gp


def eval_subbatch_variance(model, optimizer, data, args):
    x, y = data
    weights_list = []
    weights_before = deepcopy(model.state_dict())
    for start in range(0, x.size(0), args.minibatch_size):
        # for each minibatch, perform one gradient step
        x_mb = x[start:start + args.minibatch_size, ...]
        y_mb = y[start:start + args.minibatch_size, ...]
        model.load_state_dict({name: weights_before[name] for name in weights_before})
        model.zero_grad()
        y_pred = model(x_mb)
        loss = CrossEntropyLoss(y_pred, y_mb)
        loss.backward()
        for param in model.parameters():
            param.data -= args.inner_stepsize * param.grad.data
        weights_list.append(deepcopy(model.cpu().state_dict()))
        model.cuda()
    model.load_state_dict({name: weights_before[name] for name in weights_before})
    weights_flattened = []
    for w in weights_list:
        w_list = []
        for name in w:
            if name.endswith('weight') or name.endswith('bias'):
                w_list.append(w[name].data.view(-1))
        w_flattened = torch.cat(w_list)
        weights_flattened.append(w_flattened)
    weights_matrix = np.stack(weights_flattened)
    return np.sum(np.var(weights_matrix, axis=0))


def eval_subbatch_variance_bayes(model, optimizer, data, args):
    x, y = data
    weights_list = []
    weights_before = deepcopy(model.state_dict())
    for start in range(0, x.size(0), args.minibatch_size):
        # for each minibatch, perform one gradient step
        x_mb = x[start:start + args.minibatch_size, ...]
        y_mb = y[start:start + args.minibatch_size, ...]
        model.load_state_dict({name: weights_before[name] for name in weights_before})
        model.zero_grad()

        # use only one sample to train, thanks to ELBO
        # y_pred, s_pred = model(x_mb)
        # loss = CrossEntropyLoss(reparameterize(y_pred, s_pred), y_mb)

        # use multiple samples
        y_softmax = 0.
        for t in range(args.T_train):
            y_pred, s_pred = model(x_mb)
            y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
        loss = NLLLoss(y_softmax, y_mb)

        loss.backward()
        for param in model.parameters():
            param.data -= args.inner_stepsize * param.grad.data
        weights_list.append(deepcopy(model.cpu().state_dict()))
        model.cuda()
    model.load_state_dict({name: weights_before[name] for name in weights_before})
    weights_flattened = []
    for w in weights_list:
        w_list = []
        for name in w:
            if name.endswith('weight') or name.endswith('bias'):
                w_list.append(w[name].data.view(-1))
        w_flattened = torch.cat(w_list)
        weights_flattened.append(w_flattened)
    weights_matrix = np.stack(weights_flattened)
    return np.sum(np.var(weights_matrix, axis=0))


def train(train_loader, model, optimizer, scheduler, epoch, device, logger, prompt='', param_list=[], param_list_gp=[]):
    """Train for one epoch on the typetraining set"""
    scheduler.step()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    wvar = AverageMeter()
    # yvar = AverageMeter()

    lambda_ce = 1.0 if prompt.startswith('src') else args.lambda_tar_ce

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, label, input_name, domain_label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        domain_label = None if prompt.startswith('src') else domain_label.float().to(device)

        # wvar.update(eval_subbatch_variance(model, None, (input, label), args), input.size(0))

        # compute output
        output = model(input)
        # loss = RegCrossEntropyLoss(output, label, args)

        if not (prompt[0:3] in args.reg_domain) or args.reg_type.lower() == 'none':
            loss = RegCrossEntropyLoss(output, label, domain_label, args)
            loss_var_item = eval_subbatch_variance(model, None, (input, label), args)
        elif args.reg_type.lower() == 'gp':
            # reversal GP
            loss_ce = RegCrossEntropyLoss(output, label, domain_label, args)
            loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
            loss = lambda_ce * loss_ce + args.lambda_grad_penalty * loss_gp
            loss_var_item = eval_subbatch_variance(model, None, (input, label), args)
        elif args.reg_type.lower() == 'var':
            # Minibatch Variance
            loss_ce = RegCrossEntropyLoss(output, label, domain_label, args)
            loss_var = MiniBatchVarLoss(model, param_list, (input, label, domain_label), args)
            loss = lambda_ce * loss_ce + args.lambda_minibatch_var * loss_var
            loss_var_item = loss_var.item()
        # elif args.reg_type.lower() == 'yvar':
        #     # Minibatch Variance
        #     loss_ce = RegCrossEntropyLoss(output, label, domain_label, args)
        #     loss_var = MiniBatchLogitLoss(model, param_list, (input, label), args)
        #     loss = lambda_ce * loss_ce + args.lambda_minibatch_var * loss_var
        #     loss_var_item = loss_var.item()
        elif args.reg_type.lower() == 'var-gp':
            # Minibatch Variance + GradPanelty
            loss_ce = RegCrossEntropyLoss(output, label, domain_label, args)
            loss_var = MiniBatchVarLoss(model, param_list, (input, label, domain_label), args)
            loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
            loss = lambda_ce * loss_ce + args.lambda_minibatch_var * loss_var + args.lambda_grad_penalty * loss_gp
            loss_var_item = loss_var.item()
        else:
            raise NotImplementedError

        # measure accuracy and record loss
        prec1 = accuracy(output.data, label, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        wvar.update(loss_var_item, input.size(0))
        # yvar.update(MiniBatchLogitLoss(model, param_list, (input, label), args).item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            # print(MiniBatchLogitLoss(model, param_list, (input, label), args))
            logger.info(prompt+'Epoch: [{0}][{1}/{2}], '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                  'WVar {wvar.val:.3f} ({wvar.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1, wvar=wvar))
            if args.mirror_print:
                print(prompt+'Epoch: [{0}][{1}/{2}], '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                      'WVar {wvar.val:.3f} ({wvar.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1, wvar=wvar))
    # log to TensorBoard
    if args.tensorboard:
        prompt = prompt.replace(' ', '_').replace('-', '_')
        log_value(f'{prompt}train_loss', losses.avg, epoch)
        log_value(f'{prompt}train_acc', top1.avg, epoch)
        log_value(f'{prompt}train_wvar', wvar.avg, epoch)
        # log_value(f'{prompt}train_yvar', yvar.avg, epoch)


def train_bayes(train_loader, model, optimizer, scheduler, epoch, device, logger, prompt='', param_list=[], param_list_gp=[]):
    """Train for one epoch on the typetraining set"""
    scheduler.step()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    wvar = AverageMeter()

    lambda_ce = 1.0 if prompt.startswith('src') else args.lambda_tar_ce

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, label, input_name, domain_label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        domain_label = None if prompt.startswith('src') else domain_label.float().to(device)

        # wvar.update(eval_subbatch_variance_bayes(model, None, (input, label), args), input.size(0))

        # compute output
        if not (prompt[0:3] in args.reg_domain) or args.reg_type.lower() == 'none':
            y_softmax = 0.
            for t in range(args.T_train):
                y_pred, s_pred = model(input)
                y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
            loss = RegCrossEntropyLoss_bayes(y_softmax, label, domain_label, args)
            loss_var_item = eval_subbatch_variance_bayes(model, None, (input, label), args)
        elif args.reg_type.lower() == 'gp':
            # reversal GP
            y_softmax = 0.
            for t in range(args.T_train):
                y_pred, s_pred = model(input)
                y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
            loss_ce = NLLLoss(y_softmax, label, domain_label, args)
            loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
            loss = lambda_ce * loss_ce + args.lambda_grad_penalty * loss_gp
            loss_var_item = eval_subbatch_variance_bayes(model, None, (input, label), args)
        elif args.reg_type.lower() == 'var':
            # Minibatch Variance
            y_softmax = 0.
            for t in range(args.T_train):
                y_pred, s_pred = model(input)
                y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
            loss_ce = NLLLoss(y_softmax, label, domain_label, args)
            loss_var = MiniBatchVarLoss_bayes(model, param_list, (input, label, domain_label), args)
            loss = lambda_ce * loss_ce + args.lambda_minibatch_var * loss_var
            loss_var_item = loss_var.item()
        elif args.reg_type.lower() == 'var-gp':
            # Minibatch Variance + GradPanelty
            y_softmax = 0.
            for t in range(args.T_train):
                y_pred, s_pred = model(input)
                y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
            loss_ce = NLLLoss(y_softmax, label, domain_label, args)
            loss_var = MiniBatchVarLoss_bayes(model, param_list, (input, label, domain_label), args)
            loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
            loss = lambda_ce * loss_ce + args.lambda_minibatch_var * loss_var + args.lambda_grad_penalty * loss_gp
            loss_var_item = loss_var.item()
        else:
            raise NotImplementedError

        # measure accuracy and record loss
        prec1 = accuracy(y_softmax.data, label, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        wvar.update(loss_var_item, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(prompt+'Epoch: [{0}][{1}/{2}], '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                  'WVar {wvar.val:.3f} ({wvar.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1, wvar=wvar))
            if args.mirror_print:
                print(prompt+'Epoch: [{0}][{1}/{2}], '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                      'WVar {wvar.val:.3f} ({wvar.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1, wvar=wvar))
    # log to TensorBoard
    if args.tensorboard:
        prompt = prompt.replace(' ', '_').replace('-', '_')
        log_value(f'{prompt}train_loss', losses.avg, epoch)
        log_value(f'{prompt}train_acc', top1.avg, epoch)
        log_value(f'{prompt}train_wvar', wvar.avg, epoch)


def validate(val_loader, model, epoch, device, logger, param_list=[], param_list_gp=[]):
    """Perform validation on the validation set and save all the confidence vectors"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    wvar = AverageMeter()
    # yvar = AverageMeter()

    # switch to evaluate mode
    model.eval()
    ImgName_list = []
    pred_logit_list = []

    end = time.time()
    #with torch.no_grad():
    if True:
        for i, (input, label, input_name, domain_label) in enumerate(val_loader):
            input, label = input.to(device), label.to(device)

            wvar.update(eval_subbatch_variance(model, None, (input, label), args), input.size(0))

            # compute output
            output = model(input)

            if not args.debug:
                loss = RegCrossEntropyLoss(output, label, None, args)
            else:
                if args.reg_type.lower() == 'none':
                    print('none')
                    loss = RegCrossEntropyLoss(output, label, domain_label.float().to(device), args)
                elif args.reg_type.lower() == 'gp':
                    print('gp')
                    # reversal GP
                    loss_ce = RegCrossEntropyLoss(output, label, domain_label.float().to(device), args)
                    loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
                    print(loss_gp * args.lambda_grad_penalty)
                    loss = loss_ce + args.lambda_grad_penalty * loss_gp
                elif args.reg_type.lower() == 'var':
                    print('var')
                    # Minibatch Variance
                    loss_ce = CrossEntropyLoss(output, label, domain_label.float().to(device), args)
                    loss_var = MiniBatchVarLoss(model, param_list, (input, label, domain_label.float().to(device)), args)
                    print(loss_var * args.lambda_minibatch_var)
                    loss = loss_ce + args.lambda_minibatch_var * loss_var
                # elif args.reg_type.lower() == 'yvar':
                #     print('yvar')
                #     # Minibatch Variance
                #     loss_ce = CrossEntropyLoss(output, label, domain_label.float().to(device), args)
                #     loss_var = MiniBatchLogitLoss(model, param_list, (input, label), args)
                #     print(loss_var * args.lambda_minibatch_var)
                #     loss = loss_ce + args.lambda_minibatch_var * loss_var
                elif args.reg_type.lower() == 'var-gp':
                    # Minibatch Variance + GradPanelty
                    loss_ce = RegCrossEntropyLoss(output, label, domain_label.float().to(device), args)
                    loss_var = MiniBatchVarLoss(model, param_list, (input, label, domain_label.float().to(device)), args)
                    loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
                    loss = loss_ce + args.lambda_minibatch_var * loss_var + args.lambda_grad_penalty * loss_gp
                else:
                    raise NotImplementedError
                loss.backward()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, label, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            # yvar.update(MiniBatchLogitLoss(model, param_list, (input, label), args).item(), input.size(0))

            # measure elapsed timeImgName_Label
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                # print(MiniBatchLogitLoss(model, param_list, (input, label), args))
                logger.info('Test: [{0}/{1}], '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                      'WVar {wvar.val:.3f} ({wvar.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, wvar=wvar))
                if args.mirror_print:
                    print('Test: [{0}/{1}], '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                          'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                          'WVar {wvar.val:.3f} ({wvar.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, wvar=wvar))
            # save prediction confidence vectors
            for idx_batch in range(output.data.size(0)):
                # image_name_list and pred_conf_list have the same index
                ImgName_list.append(input_name[idx_batch])
                pred_logit_list.append(output.data[idx_batch, :])

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
        log_value('val_wvar', wvar.avg, epoch)
        # log_value('val_yvar', yvar.avg, epoch)
    return top1.avg, ImgName_list, pred_logit_list


def validate_bayes(val_loader, model, epoch, device, logger, param_list=[], param_list_gp=[]):
    """Perform validation on the validation set and save all the confidence vectors"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    wvar = AverageMeter()

    # switch to evaluate mode
    model.train()
    ImgName_list = []
    pred_softmax_list = []

    end = time.time()
    #with torch.no_grad():
    if True:
        for i, (input, label, input_name, domain_label) in enumerate(val_loader):
            input, label = input.to(device), label.to(device)

            wvar.update(eval_subbatch_variance_bayes(model, None, (input, label), args), input.size(0))

            # compute output
            if not args.debug:
                y_softmax = 0.
                for t in range(args.T):
                    y_pred, s_pred = model(input)
                    y_softmax += 1. / args.T * F.softmax(reparameterize(y_pred, s_pred), dim=1)
                loss = RegCrossEntropyLoss_bayes(y_softmax, label, None, args)
            else:
                if args.reg_type.lower() == 'none':
                    y_softmax = 0.
                    for t in range(args.T_train):
                        y_pred, s_pred = model(input)
                        y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
                    loss = RegCrossEntropyLoss_bayes(y_softmax, label, domain_label.float().to(device), args)
                    # loss_gp = torch.tensor(0)
                    loss_gp = eval_subbatch_variance_bayes(model, None, (input, label), args)
                    print(loss_gp)
                elif args.reg_type.lower() == 'gp':
                    # reversal GP
                    y_softmax = 0.
                    for t in range(args.T_train):
                        y_pred, s_pred = model(input)
                        y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
                    loss_ce = NLLLoss(y_softmax, label, domain_label.float().to(device), args)
                    loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
                    loss = loss_ce + args.lambda_grad_penalty * loss_gp
                    print(loss_gp)
                elif args.reg_type.lower() == 'var':
                    # Minibatch Variance
                    y_softmax = 0.
                    for t in range(args.T_train):
                        y_pred, s_pred = model(input)
                        y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
                    loss_ce = NLLLoss(y_softmax, label, domain_label.float().to(device), args)
                    loss_var = MiniBatchVarLoss_bayes(model, param_list, (input, label, domain_label.float().to(device)), args)
                    loss = loss_ce + args.lambda_minibatch_var * loss_var
                    print(loss_var)
                elif args.reg_type.lower() == 'var-gp':
                    # Minibatch Variance + GradPanelty
                    y_softmax = 0.
                    for t in range(args.T_train):
                        y_pred, s_pred = model(input)
                        y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
                    loss_ce = NLLLoss(y_softmax, label, domain_label.float().to(device), args)
                    loss_var = MiniBatchVarLoss_bayes(model, param_list, (input, label, domain_label.float().to(device)), args)
                    loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
                    loss = loss_ce + args.lambda_minibatch_var * loss_var + args.lambda_grad_penalty * loss_gp
                    print(loss_var, loss_gp)
                else:
                    raise NotImplementedError
                loss.backward()

            # measure accuracy and record loss
            prec1 = accuracy(y_softmax.data, label, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed timeImgName_Label
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}], '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                      'WVar {wvar.val:.3f} ({wvar.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, wvar=wvar))
                if args.mirror_print:
                    print('Test: [{0}/{1}], '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                          'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                          'WVar {wvar.val:.3f} ({wvar.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, wvar=wvar))
            # save prediction confidence vectors
            for idx_batch in range(y_softmax.data.size(0)):
                # image_name_list and pred_conf_list have the same index
                ImgName_list.append(input_name[idx_batch])
                pred_softmax_list.append(y_softmax.data[idx_batch, :])

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
        log_value('val_wvar', wvar.avg, epoch)
    return top1.avg, ImgName_list, pred_softmax_list


def soft_pseudo_label(pred_logit_list, num_class, args, device):
    """Soft pseudo-label generation"""
    ls_weight_l2 = args.ls_weight_l2
    ls_weight_negent = args.ls_weight_negent

    pred_logit_tensor = torch.stack(pred_logit_list)
    if args.bayesian:
        pred_conf_tensor = pred_logit_tensor
    else:
        pred_conf_tensor = F.softmax(pred_logit_tensor, 1)
    pred_label_conf, _ = pred_conf_tensor.topk(1, 1, True, True)
    # no regularization
    if ls_weight_l2 + ls_weight_negent == 0.0:
        pred_softlabel_matrix = pred_conf_tensor >= pred_label_conf  # one-hot
        pred_softlabel_matrix = pred_softlabel_matrix.type(torch.float)
    # L2 regularation label smoothing
    elif ls_weight_l2 > 0.0:
        # descending order of reweighted confidence
        pred_conf_tensor, rerank_idx = pred_conf_tensor.topk(pred_conf_tensor.size(1), 1, True, True)
        # initial yhat_k with L2 label smoothing
        log_softlabel_matrix = torch.log(pred_conf_tensor)
        log_softlabel_cusummatrix = log_softlabel_matrix.cumsum(1)
        Ckq = torch.range(1, num_class).to(device)
        log_softlabel_mulmatrix = Ckq * log_softlabel_matrix
        init_yhat_matrix = 1 + 0.5/ls_weight_l2*(-log_softlabel_cusummatrix+log_softlabel_mulmatrix)
        # the first zero yhat_k (descending order) with L2 label smoothing, and Ckq
        init_yhat_matrix_posbool = init_yhat_matrix <= 0
        init_yhat_matrix_posbool_cumsum = init_yhat_matrix_posbool.cumsum(1)
        init_yhat_matrix_first_zero_bool = init_yhat_matrix_posbool_cumsum == 1
        _, init_yhat_Ckq = init_yhat_matrix_first_zero_bool.topk(1, 1, True, True)
        init_yhat_Ckq = init_yhat_Ckq.view(-1)
        # yhat_k with L2 label smoothing
        pred_softlabel_matrix_rerank = log_softlabel_matrix
        pred_softlabel_matrix = torch.zeros(pred_softlabel_matrix_rerank.size()).to(device)
        for idx_yhat in range(pred_conf_tensor.size(0)):
            curr_pred_softlabel = pred_softlabel_matrix_rerank[idx_yhat, :]
            curr_Ckq = init_yhat_Ckq[idx_yhat]
            curr_pred_softlabel[curr_Ckq:] = 0
            curr_Ckq_float = curr_Ckq.type(torch.float)
            curr_pred_softlabel_sum = -curr_pred_softlabel.sum()
            curr_pred_softlabel = curr_pred_softlabel * curr_Ckq_float
            curr_pred_softlabel = ((curr_pred_softlabel + curr_pred_softlabel_sum)/2/ls_weight_l2 + 1)/curr_Ckq_float
            # set negative initial yhat_k to 0
            curr_pred_softlabel[curr_Ckq:] = 0
            # normalize
            curr_pred_softlabel = torch.max(curr_pred_softlabel, torch.zeros(curr_pred_softlabel.size()).to(device))
            curr_pred_softlabel = curr_pred_softlabel/curr_pred_softlabel.sum()
            # back to original rank
            curr_rerank_idx = rerank_idx[idx_yhat, :]
            curr_pred_softlabel_temp = torch.zeros(curr_pred_softlabel.size()).to(device)
            curr_pred_softlabel_temp[curr_rerank_idx] = curr_pred_softlabel
            pred_softlabel_matrix[idx_yhat, :] = curr_pred_softlabel_temp
    # negative entropy regularation label smoothing
    elif ls_weight_negent > 0.0:
        pred_conf_power_tensor = pred_conf_tensor.pow(1.0/ls_weight_negent)
        pred_conf_power_sum = pred_conf_power_tensor.sum(1).view(-1,1)
        pred_softlabel_matrix = pred_conf_power_tensor/pred_conf_power_sum

    pred_softlabel_list = [pred_softlabel_matrix[t_idx, :] for t_idx in range(pred_softlabel_matrix.size(0))]
    return pred_softlabel_list


def kc_parameters(tgt_portion, kc_policy, pred_logit_list, soft_pseudo_label, num_class, args, device):
    """determine kc_parameters, policy: global threshold; class-balance threshold; reweighted class-balance threshold"""
    ls_weight_l2 = args.ls_weight_l2
    ls_weight_negent = args.ls_weight_negent
    pred_logit_tensor = torch.stack(pred_logit_list)
    soft_pseudo_label_matrix = torch.stack(soft_pseudo_label)
    log_soft_pseudo_label_matrix = torch.log(soft_pseudo_label_matrix+MAGIC_EPS)  # stablize
    if args.bayesian:
        pred_conf_tensor = pred_logit_tensor
    else:
        pred_conf_tensor = F.softmax(pred_logit_tensor, 1)
    pred_log_conf_tensor = torch.log(pred_conf_tensor+MAGIC_EPS)
    pred_label_conf, pred_label = pred_conf_tensor.topk(1, 1, True, True)

    ce = torch.sum(-pred_log_conf_tensor * soft_pseudo_label_matrix, 1)  # cross-entropy loss with vector-form softmax
    l2_ls = torch.sum(soft_pseudo_label_matrix * soft_pseudo_label_matrix, 1)
    negent_ls = torch.sum(soft_pseudo_label_matrix * log_soft_pseudo_label_matrix, 1)
    #
    reg_ce = ce + ls_weight_l2 * l2_ls + ls_weight_negent * negent_ls

    kct_matrix = torch.ones(pred_conf_tensor.size()).to(device)
    if kc_policy == 'global':
        num_label_conf = len(reg_ce)
        reg_ce_sort, _ = reg_ce.view(-1).sort(descending=False)
        reg_ce_global = reg_ce_sort[int(np.floor((num_label_conf-1)*tgt_portion))]
        kc_global = -reg_ce_global
        kct_matrix = kc_global*kct_matrix
    elif kc_policy == 'cb':
        kc_values = torch.ones(num_class)
        reg_ce_values = torch.zeros(num_class)
        for idx_class in range(num_class):
            pred_class_idx = pred_label == idx_class
            reg_ce = reg_ce.view(-1, 1)
            pred_class_reg_ce = reg_ce[pred_class_idx]
            num_class_reg_ce = len(pred_class_reg_ce)
            if num_class_reg_ce != 0:
                pred_class_reg_ce_sort, _ = pred_class_reg_ce.view(-1).sort(descending=False)
                reg_ce_values[idx_class] = pred_class_reg_ce_sort[int(np.floor((num_class_reg_ce - 1) * tgt_portion))]
                kc_values[idx_class] = -reg_ce_values[idx_class]
                kct_matrix[pred_class_idx.view(-1), :] = kc_values[idx_class].to(device)
    kct_list = [kct_matrix[t_idx, :] for t_idx in range(kct_matrix.size(0))]
    return kct_list


def label_selection(logits_list, pred_softlabel_list, kct_list, ImgName_list, args):
    """Pseudo-label selection"""
    # parameters
    ls_weight_l2 = args.ls_weight_l2
    ls_weight_negent = args.ls_weight_negent
    num_class = args.num_class
    logits_matrix = torch.stack(logits_list)
    pred_softlabel_matrix = torch.stack(pred_softlabel_list)
    log_pred_softlabel_matrix = torch.log(pred_softlabel_matrix+MAGIC_EPS)
    kct_matrix = torch.stack(kct_list)
    # regularized loss with self-paced thresholding
    if args.bayesian:
        softmax = logits_matrix
    else:
        softmax = F.softmax(logits_matrix, 1)
    logsoftmax = torch.log(softmax+MAGIC_EPS)   # compute the log of softmax values

    ce = torch.sum(-logsoftmax*pred_softlabel_matrix, 1)  # cross-entropy loss with vector-form softmax
    l2_ls = torch.sum(pred_softlabel_matrix*pred_softlabel_matrix, 1)
    negent_ls = torch.sum(pred_softlabel_matrix*log_pred_softlabel_matrix, 1)

    sp_threshold = torch.sum(-pred_softlabel_matrix*kct_matrix, 1)

    reg_ce = ce + ls_weight_l2*l2_ls + ls_weight_negent*negent_ls

    # sort
    if args.shuffle_off:
        reg_ce, argsort = torch.sort(reg_ce)
        ImgName_list = [ImgName_list[i] for i in argsort]
        pred_softlabel_list = [pred_softlabel_list[i] for i in argsort]

    threshold_idx = reg_ce < sp_threshold

    if args.no_trick == 'random':
        # pdb.set_trace()
        threshold_idx = threshold_idx[torch.randperm(threshold_idx.size(0))]

    # save soft label into txt file
    with open(args.tgt_train_list, 'w') as fo:
        for idx_label in range(len(reg_ce.view(-1))):
            curr_threshold_idx = threshold_idx[idx_label]
            if curr_threshold_idx == 1:
                curr_soft_label = pred_softlabel_matrix[idx_label]
            else:
                curr_soft_label = torch.zeros(num_class)
            if curr_threshold_idx == 1 or not args.filter_out_invalid:
                curr_soft_label = curr_soft_label.cpu().numpy()
                fo.write(ImgName_list[idx_label] + ' ' + ' '.join(map(str, curr_soft_label)) + '\n')


def CrossEntropyLoss(outputs, labels, domain_label=None, args=None):
    batch_size = labels.size(0)  # batch_size
    logsoftmax = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
    ce = torch.sum(-logsoftmax*labels, dim=1)  # cross-entropy loss with vector-form softmax

    if domain_label is None or args.entropy.lower() == 'min':
        # min entropy is pseudo-label cross-entropy
        ce = torch.sum(ce)
    elif args.entropy.lower() == 'collision':
        # Renyi entropy, alpha=2
        softmax = F.softmax(outputs, dim=1)  # compute the log of softmax values
        ent = -torch.log(torch.sum(softmax.pow(2), dim=1) + MAGIC_EPS)
        ce = torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label))
    elif args.entropy.lower() == 'shannon':
        # Shannon entropy is 'entropy regularization'
        softmax = F.softmax(outputs, dim=1)  # compute the log of softmax values
        ent = -torch.sum(softmax * logsoftmax, dim=1)
        ce = torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label))
    else:
        raise NotImplementedError
    return ce / batch_size


def NLLLoss(softmax, labels, domain_label=None, args=None):
    logsoftmax = torch.log(softmax + MAGIC_EPS)
    ce = torch.sum(-logsoftmax * labels, dim=1)

    if domain_label is None or args.entropy.lower() == 'min':
        # min entropy is pseudo-label cross-entropy
        ce = torch.sum(ce)
    elif args.entropy.lower() == 'collision':
        # Renyi entropy, alpha=2
        ent = -torch.log(torch.sum(softmax.pow(2), dim=1) + MAGIC_EPS)
        ce = torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label))
    elif args.entropy.lower() == 'shannon':
        # Shannon entropy is 'entropy regularization'
        ent = -torch.sum(softmax * logsoftmax, dim=1)
        ce = torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label))
    else:
        raise NotImplementedError

    return ce / labels.size(0)


def RegCrossEntropyLoss(outputs, labels, domain_label=None, args=None):
    mr_weight_l2 = args.mr_weight_l2
    mr_weight_negent = args.mr_weight_negent
    mr_weight_kld = args.mr_weight_kld
    num_class = args.num_class
    batch_size = labels.size(0)  # batch_size
    # batch_size_valid_bool = torch.sum(labels,dim=1) > 0
    # batch_size_valid = batch_size_valid_bool.sum().type(torch.float)

    softmax = F.softmax(outputs, dim=1)   # compute the log of softmax values
    logsoftmax = F.log_softmax(outputs, dim=1)   # compute the log of softmax values

    ce = torch.sum(-logsoftmax*labels, dim=1)  # cross-entropy loss with vector-form softmax
    l2 = torch.sum(softmax*softmax)
    negent = torch.sum(softmax*logsoftmax)
    kld = torch.sum(-logsoftmax/float(num_class))

    if domain_label is None or args.entropy.lower() == 'min':
        # min entropy is pseudo-label cross-entropy
        reg_ce = torch.sum(ce) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld
    elif args.entropy.lower() == 'collision':
        # Renyi entropy, alpha=2
        ent = -torch.log(torch.sum(softmax.pow(2), dim=1) + MAGIC_EPS)
        reg_ce = torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label)) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld
    elif args.entropy.lower() == 'shannon':
        # Shannon entropy is 'entropy regularization'
        ent = -torch.sum(softmax * logsoftmax, dim=1)
        reg_ce = torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label)) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld
    else:
        raise NotImplementedError

    return reg_ce / batch_size


def RegCrossEntropyLoss_bayes(softmax, labels, domain_label=None, args=None):
    mr_weight_l2 = args.mr_weight_l2
    mr_weight_negent = args.mr_weight_negent
    mr_weight_kld = args.mr_weight_kld
    num_class = args.num_class
    batch_size = labels.size(0)  # batch_size
    # batch_size_valid_bool = torch.sum(labels,dim=1) > 0
    # batch_size_valid = batch_size_valid_bool.sum().type(torch.float)

    logsoftmax = torch.log(softmax + MAGIC_EPS)  # compute the log of softmax values

    ce = torch.sum(-logsoftmax * labels, dim=1)  # cross-entropy loss with vector-form softmax
    l2 = torch.sum(softmax * softmax)
    negent = torch.sum(softmax * logsoftmax)
    kld = torch.sum(-logsoftmax / float(num_class))

    if domain_label is None or args.entropy.lower() == 'min':
        # min entropy is pseudo-label cross-entropy
        reg_ce = torch.sum(ce) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld
    elif args.entropy.lower() == 'collision':
        # Renyi entropy, alpha=2
        ent = -torch.log(torch.sum(softmax.pow(2), dim=1) + MAGIC_EPS)
        reg_ce = torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label)) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld
    elif args.entropy.lower() == 'shannon':
        # Shannon entropy is 'entropy regularization'
        ent = -torch.sum(softmax * logsoftmax, dim=1)
        reg_ce = torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label)) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld
    else:
        raise NotImplementedError

    # if args.entropy and domain_label is not None:
    #     ent = -torch.sum(softmax * logsoftmax, dim=1)
    #     # reg_ce = ent + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld
    #     reg_ce = torch.sum(ent * domain_label) + torch.sum(ce * (1.-domain_label)) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld
    # else:
    #     reg_ce = torch.sum(ce) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld

    return reg_ce / batch_size


def saveSRCtxt(src_portion, randseed, args):
    src_gt_txt = args.src_gt_list
    src_train_txt = args.src_train_list
    item_list = []
    count0 = 0
    with open(src_gt_txt) as f:
        for item in f.readlines():
            fields = item.strip()
            item_list.append(fields)
            count0 = count0 + 1

    num_source = count0
    num_sel_source = int(np.floor(num_source*src_portion))
    np.random.seed(randseed)
    print(f'random number: {np.random.random()}')
    print(f'selected {num_sel_source}/{num_source}')
    sel_idx = list(np.random.choice(num_source, num_sel_source, replace=False))
    item_list = list(itemgetter(*sel_idx)(item_list))

    # write src_train_txt
    with open(src_train_txt, 'w') as f:
        for item in item_list:
            f.write("%s\n" % item)


class ImageClassdata(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, txt_file, root_dir, img_type, transform=transforms.ToTensor(), domain=0):
        """
        Args:
            txt_fpred_conf_tensorile (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_frame = open(txt_file, 'r').readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.img_type = img_type
        self._domain = domain

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # print(f'index: {idx}')
        ImgName_Label = str.split(self.images_frame[idx])
        img_name = os.path.join(self.root_dir, ImgName_Label[0])
        if self.img_type == 'grayscale':
            image = Image.open(img_name)
        elif self.img_type == 'RGB':
            img = Image.open(img_name)
            image = img.convert('RGB')
        else:
            raise NotImplementedError(f'img_type {self.img_type} not implemented.')
        lbl = np.asarray(ImgName_Label[1:], dtype=np.float32)
        label = torch.from_numpy(lbl)

        if self.transform:
            image = self.transform(image)

        return image, label, ImgName_Label[0], self._domain


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
    correct = pred.eq(gt.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_logger(output_dir=None, log_file=None):
    head = '%(asctime)-15s Host %(message)s'
    logger_level = logging.INFO
    if all((output_dir, log_file)) and len(log_file) > 0:
        logger = logging.getLogger()
        log_path = os.path.join(output_dir, log_file)
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logger_level)
    else:
        logging.basicConfig(level=logger_level, format=head)
        logger = logging.getLogger()
    return logger


if __name__ == '__main__':
    main()
