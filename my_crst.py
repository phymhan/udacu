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
from utils import makedir, print_options, save_checkpoint, saveMat, get_param_list, get_optimizer, get_minibatch_size, assert_args
import resnet
import utils
from copy import deepcopy
from torch import autograd
import random
import glob
import pdb


# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, dest='start_epoch', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, dest='batch_size', help='mini-batch size (default: 64) for training')
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
parser.add_argument('--src-root', default=None, type=str, dest='src_root', help='address of source data root folder')
parser.add_argument('--src-train-list', default='src_train_list.txt', type=str, dest='src_train_list', help='the source image_label list for training, which can be changed in terms of the item labels (not the labels)')
parser.add_argument('--src-gt-list', default=None, type=str, dest='src_gt_list', help='the source image_label list for evaluation, which are not changed')
parser.add_argument('--tgt-root', default=None, type=str, dest='tgt_root', help='address of target data root folder')
parser.add_argument('--tgt-train-list', default='tar_train_list.txt', type=str, dest='tgt_train_list', help='the target image_label list in training/self-training process, which may be updated dynamically')
parser.add_argument('--tgt-gt-list', default=None, type=str, dest='tgt_gt_list', help='the target image_label list for evaluation, which are not changed')
parser.add_argument('--mr-weight-l2', default=0., type=float, dest='mr_weight_l2', help='weight of l2 model regularization')
parser.add_argument('--mr-weight-negent', default=0., type=float, dest='mr_weight_negent', help='weight of negative entropy model regularization')
parser.add_argument('--mr-weight-kld', default=0., type=float, dest='mr_weight_kld', help='weight of kld model regularization')
parser.add_argument('--mr-weight-src', default=0., type=float, dest='mr_weight_src', help='weight of source model regularization')
parser.add_argument('--ls-weight-l2', default=0., type=float, dest='ls_weight_l2', help='weight of l2 label smoothing')
parser.add_argument('--ls-weight-negent', default=0., type=float, dest='ls_weight_negent', help='weight of negative entropy label smoothing')
parser.add_argument('--num-class', default=None, type=int, dest='num_class', help='the number of classes')
parser.add_argument('--num-gpus', type=int, default=1, help='the number of gpus, 0 for cpu')

# self-trained network
parser.add_argument('--kc-policy', default='global', type=str, dest='kc_policy', help='The policy to determine kc. Valid values: "global" for global threshold, "cb" for class-balanced threshold, "rcb" for reweighted class-balanced threshold')
parser.add_argument('--kc-value', default='conf', type=str, help='The way to determine kc values, either "conf", or "prob".')
parser.add_argument('--init-tgt-port', default=0.3, type=float, dest='init_tgt_port', help='The initial portion of target to determine kc')
parser.add_argument('--max-tgt-port', default=0.6, type=float, dest='max_tgt_port', help='The max portion of target to determine kc')
parser.add_argument('--tgt-port-step', default=0.05, type=float, dest='tgt_port_step', help='The portion step in target domain in every round of self-paced self-trained neural network')
parser.add_argument('--init-src-port', default=0.5, type=float, dest='init_src_port', help='The initial portion of source portion for self-trained neural network')
parser.add_argument('--max-src-port', default=0.8, type=float, dest='max_src_port', help='The max portion of source portion for self-trained neural network')
parser.add_argument('--src-port-step', default=0.05, type=float, dest='src_port_step', help='The portion step in source domain in every round of self-paced self-trai152ned neural network')
parser.add_argument('--init-randseed', default=0, type=int, dest='init_randseed', help='The initial random seed for source selection')
parser.add_argument('--lr-stepsize', default=7, type=int, dest='lr_stepsize', help='The step size of lr_stepScheduler')
parser.add_argument('--lr-stepfactor', default=0.1, type=float, dest='lr_stepfactor', help='The step factor of lr_stepScheduler')

# added
parser.add_argument('--bayesian', action='store_true')
parser.add_argument('--entropy', type=str, default='min', help='Shannon, collision, min')
parser.add_argument('--T', type=int, default=10)
parser.add_argument('--T-train', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--checkpoints-dir', type=str, default='runs')
parser.add_argument('--bb', action='store_true')
parser.add_argument('--optimizer', type=str, default='sgd', help='adam: orig adam; adam2: adam in wgan-gp; rmsprop')
parser.add_argument('--rmsprop-alpha', type=float, default=0.99)
parser.add_argument('--minibatch-size', type=int, default=8)
parser.add_argument('--num-minibatch', type=int, default=4)
parser.add_argument('--inner-iters', type=int, default=1)
parser.add_argument('--inner-stepsize', type=float, default=0.02)
parser.add_argument('--shuffle-off', action='store_true')
parser.add_argument('--no-aleatoric', action='store_true')
parser.add_argument('--save-memory', action='store_true')
parser.add_argument('--mix', action='store_true')
parser.add_argument('--var-param', type=str, default='all')
parser.add_argument('--gp-param', type=str, default='all')
parser.add_argument('--reg-type', type=str, default='none')
parser.add_argument('--reg-domain', type=str, nargs='+', default=['tar'])  # FIXME: temp solution
parser.add_argument('--gp-type', type=str, default='one-side')
parser.add_argument('--gp-center', type=float, default=1)
parser.add_argument('--lambda-minibatch-var', type=float, default=-1.0)
parser.add_argument('--lambda-grad-penalty', type=float, default=1.0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--filter-out-invalid', action='store_true')
parser.add_argument('--save-epoch-freq', type=int, default=5)
parser.add_argument('--delete-previous-event-files', action='store_true')
parser.add_argument('--use-entropy-loss', action='store_true', help='use entropy loss when computing MBV')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

global MAGIC_EPS
best_prec1 = 0.
MAGIC_EPS = 1e-32


def main():
    global args, best_prec1, reparameterize
    args = parser.parse_args()
    args.expr_dir = os.path.join(args.checkpoints_dir, args.name)
    makedir(args.expr_dir)
    if args.delete_previous_event_files:
        for eventfile in glob.glob(os.path.join(args.expr_dir, 'events.*')):
            os.remove(eventfile)
    print_options(parser, args)
    if args.tensorboard:
        configure(args.expr_dir)

    # torch.backends.cudnn.deterministic = True
    np.random.seed(args.init_randseed)
    random.seed(args.init_randseed)
    torch.manual_seed(args.init_randseed)

    logger = set_logger(args.expr_dir, args.name+'.log')
    logger.info('start with arguments %s', args)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val4mix': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    kwargs = {'num_workers': 1, 'pin_memory': False}

    args.src_train_list = os.path.join(args.expr_dir, os.path.basename(args.src_train_list))
    args.tgt_train_list = os.path.join(args.expr_dir, os.path.basename(args.tgt_train_list))
    args.minibatch_size = get_minibatch_size(args.batch_size, args.minibatch_size, args.num_minibatch)
    assert_args(args)

    visDA17_valset = ImageClassdata(txt_file=args.tgt_gt_list, root_dir=args.tgt_root, reg_weight=0.0, transform=data_transforms['val'], domain=1)
    val_loader = torch.utils.data.DataLoader(visDA17_valset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    # if args.layers == 152:
    #     model = models.resnet152(pretrained=True)
    # elif args.layers == 101:
    #     model = models.resnet101(pretrained=True)
    # elif args.layers == 50:
    #     model = models.resnet50(pretrained=True)
    # elif args.layers == 34:
    #     model = models.resnet34(pretrained=True)
    # elif args.layers == 18:
    #     model = models.resnet18(pretrained=True)
    if args.bayesian:
        if args.layers == 50:
            model = resnet.bresnet50(bb=args.bb, pretrained=True)
        elif args.layers == 101:
            model = resnet.bresnet101(bb=args.bb, pretrained=True)
        else:
            raise NotImplementedError
    else:
        if args.layers == 152:
            model = resnet.resnet152(pretrained=True)
        elif args.layers == 101:
            model = resnet.resnet101(pretrained=True)
        elif args.layers == 50:
            model = resnet.resnet50(pretrained=True)
        elif args.layers == 34:
            model = resnet.resnet34(pretrained=True)
        elif args.layers == 18:
            model = resnet.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    fc_layers = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, args.num_class),
    )
    model.fc = fc_layers
    if args.bayesian:
        logvar_layers = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, args.num_class),
        )
        model.logvar = logvar_layers

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to sdef RegCrossEntropyLoss(outputs, labels, weight_l2, weight_negent, weight_kld,num_class):

    # specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    # for training on multiple GPUs.
    #device = torch.device("cuda:"+args.gpus)
    gpus = ','.join([str(i) for i in range(args.num_gpus)])
    device = torch.device(f'cuda:{gpus}')
    # device = torch.device("cpu")
    if args.num_gpus > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda(device)
    else:
        model = model.to(device)

    # define loss function (criterion) and pptimizer

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            if args.num_gpus > 1 and torch.cuda.is_available():
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError("=ImageClassdata> no checkpoint found at '{}'".format(args.resume))

    # setting cudnn manualy?
    # cudnn.benchmark = True

    param_list = get_param_list(model, args.var_param)
    param_list_gp = get_param_list(model, args.gp_param)

    # all parameters are being optimized
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    optimizer = get_optimizer(model, args.optimizer, args)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize, gamma=args.lr_stepfactor)
    tgt_portion = args.init_tgt_port
    src_portion = args.init_src_port
    randseed = args.init_randseed

    if args.bayesian:
        train_fn = train_bayes
        validate_fn = validate_bayes_mem if args.save_memory else validate_bayes
    else:
        train_fn = train
        validate_fn = validate_mem if args.save_memory else validate
    if args.no_aleatoric:
        reparameterize = utils.reparameterize_straight_through
    else:
        reparameterize = utils.reparameterize

    # first_epoch = True

    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):

        # evaluate on validation set
        num_class = args.num_class
        # confidence vectors for target
        prec1, ImgName_list, pred_logit_list, _, label_list = validate_fn(val_loader, model, epoch, device, logger, 'tar-', param_list, param_list_gp)
        # if args.debug and first_epoch:
        #     print('loading...')
        #     tmp = '_tmp' if args.use_tmp_lists else ''
        #     prec1 = 0.
        #     ImgName_list = load_str_list(os.path.join('runs/0507_visda_debug', f'init_imgName{tmp}.txt'))
        #     pred_logit_list = load_arr_list(os.path.join('runs/0507_visda_debug', f'init_pred_logit{tmp}.txt'))
        #     # label_list = load_arr_list(os.path.join(args.expr_dir, f'init_pred_logit{tmp}.txt'))
        #     # feat_list = []
        #     print('done.')
        #     first_epoch = False
        # else:
        #     prec1, ImgName_list, pred_logit_list, _, label_list = validate_fn(val_loader, model, epoch, device, logger, param_list, param_list_gp)

        # save feat and label as mat file
        # feat_tensor = torch.stack(feat_list).cpu().numpy()
        # label_tensor = torch.stack(label_list).cpu().numpy()
        # saveMat(feat_tensor, label_tensor, epoch, args.expr_dir, args)
        # generate kct
        kct_matrix = kc_parameters(tgt_portion, args.kc_policy, pred_logit_list, num_class, device, args)
        pred_softlabel_matrix = soft_pseudo_label(pred_logit_list, kct_matrix, args)
        # next round's tgt portion
        # generate soft pseudo-labels
        # select good pseudo-labels for model retraining
        label_selection(pred_logit_list, pred_softlabel_matrix, kct_matrix, ImgName_list, args, epoch)
        # select part of source data for model retraining
        saveSRCtxt(src_portion, randseed, args)
        randseed = randseed + 1
        src_portion = min(src_portion + args.src_port_step, args.max_src_port)
        tgt_portion = min(tgt_portion + args.tgt_port_step, args.max_tgt_port)

        # train for one epoch
        if args.mr_weight_kld + args.mr_weight_l2 + args.mr_weight_negent == 0:
            reg_weight_tgt = 0.0
        else:
            # currently only one kind of model regularizer is supported
            reg_weight_tgt = args.mr_weight_kld + args.mr_weight_l2 + args.mr_weight_negent
        reg_weight_src = args.mr_weight_src
        visDA17_trainset = ImageClassdata(txt_file=args.src_train_list, root_dir=args.src_root, reg_weight=reg_weight_src, transform=data_transforms['train'], domain=0)
        visDA17_valset_pseudo = ImageClassdata(txt_file=args.tgt_train_list, root_dir=args.tgt_root, reg_weight=reg_weight_tgt, transform=data_transforms['val4mix'], domain=1)
        if args.mix:
            mix_trainset = torch.utils.data.ConcatDataset([visDA17_trainset, visDA17_valset_pseudo])
            mix_train_loader = torch.utils.data.DataLoader(mix_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
            train_fn(mix_train_loader, model, optimizer, scheduler, epoch, device, logger, 'mix-', param_list, param_list_gp)
        else:
            # train on src then train on tar
            src_train_loader = torch.utils.data.DataLoader(visDA17_trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
            tar_train_loader = torch.utils.data.DataLoader(visDA17_valset_pseudo, batch_size=args.batch_size, shuffle=not args.shuffle_off, **kwargs)
            train_fn(src_train_loader, model, optimizer, scheduler, epoch, device, logger, 'src-', param_list, param_list_gp)
            logger.info('\n---')
            print('---')
            train_fn(tar_train_loader, model, optimizer, scheduler, epoch, device, logger, 'tar-', param_list, param_list_gp)
        logger.info('\n---')
        print('---')

        # remember best prec@1 and save checkpoint
        # prec1 = prec1.to(device)
        is_best = prec1.cpu() > best_prec1.cpu()
        best_prec1 = max(prec1.cpu(), best_prec1.cpu())
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.expr_dir)
        # model.cuda(device)
    logger.info('Best accuracy: ' + str(best_prec1.numpy()))


def train(train_loader, model, optimizer, scheduler, epoch, device, logger, prompt='', param_list=[], param_list_gp=[]):
    """Train for one epoch on the typetraining set"""
    scheduler.step()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    wvar = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, label, input_name, reg_weight, domain_label) in enumerate(train_loader):
        label = label.to(device)
        input = input.to(device)
        reg_weight = reg_weight.to(device)
        domain_label = None if prompt.startswith('src') else domain_label.float().to(device)

        # compute output
        output = model(input)
        if not (prompt[0:3] in args.reg_domain) or args.reg_type.lower() == 'none':
            loss = RegCrossEntropyLoss(output, label, reg_weight, domain_label, args)
            loss_var_item = eval_subbatch_variance(model, None, (input, label), args)
        elif args.reg_type.lower() == 'gp':
            # GP wrt theta
            loss_ce = RegCrossEntropyLoss(output, label, reg_weight, domain_label, args)
            loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
            loss = loss_ce + args.lambda_grad_penalty * loss_gp
            loss_var_item = eval_subbatch_variance(model, None, (input, label), args)
        elif args.reg_type.lower() == 'var':
            # Minibatch Variance
            loss_ce = RegCrossEntropyLoss(output, label, reg_weight, domain_label, args)
            loss_var = MiniBatchVarLoss(model, param_list, (input, label, domain_label), args)
            loss = loss_ce + args.lambda_minibatch_var * loss_var
            loss_var_item = loss_var.item()
        elif args.reg_type.lower() == 'var-gp':
            # Minibatch Variance + GradPanelty
            loss_ce = RegCrossEntropyLoss(output, label, reg_weight, domain_label, args)
            loss_var = MiniBatchVarLoss(model, param_list, (input, label, domain_label), args)
            loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
            loss = loss_ce + args.lambda_minibatch_var * loss_var + args.lambda_grad_penalty * loss_gp
            loss_var_item = loss_var.item()
        else:
            raise NotImplementedError

        # debug
        if torch.isnan(loss):
            print('breakpoint 1')
            pdb.set_trace()

        # measure accuracy and record loss
        prec1, gt_num = accuracy(output.data, label, args.num_class, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], gt_num[0])
        wvar.update(loss_var_item, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # debug
        for p in model.named_parameters():
            if torch.isnan(p[1].grad).any():
                print('breakpoint 2')
                print(p[0])
                pdb.set_trace()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(prompt + 'Train Epoch: [{0}][{1}/{2}], '
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                           'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                           'WVar {wvar.val:.3f} ({wvar.avg:.3f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, wvar=wvar))
            logger.info(prompt + 'Train Epoch: [{0}][{1}/{2}], '
                                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                                 'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                                 'WVar {wvar.val:.3f} ({wvar.avg:.3f})'
                        .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, wvar=wvar))
    # log to TensorBoard
    if args.tensorboard:
        for param_group in optimizer.param_groups:
            clr = param_group['lr']
        log_value(f'{prompt}train_loss', losses.avg, epoch)
        log_value(f'{prompt}train_acc', top1.vec2sca_avg, epoch)
        log_value(f'{prompt}lr', clr, epoch)
        log_value(f'{prompt}train_wvar', wvar.avg, epoch)


def train_bayes(train_loader, model, optimizer, scheduler, epoch, device, logger, prompt='', param_list=[], param_list_gp=[]):
    """Train for one epoch on the typetraining set"""
    scheduler.step()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    wvar = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, label, input_name, reg_weight, domain_label) in enumerate(train_loader):
        label = label.to(device)
        input = input.to(device)
        reg_weight = reg_weight.to(device)
        domain_label = None if prompt.startswith('src') else domain_label.float().to(device)

        # compute output
        if not (prompt[0:3] in args.reg_domain) or args.reg_type.lower() == 'none':
            y_softmax = 0.
            for t in range(args.T_train):
                y_pred, s_pred = model(input)
                y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
            loss = RegCrossEntropyLoss_bayes(y_softmax, label, reg_weight, domain_label, args)
            loss_var_item = eval_subbatch_variance_bayes(model, None, (input, label), args)
        elif args.reg_type.lower() == 'gp':
            # GP wrt theta
            y_softmax = 0.
            for t in range(args.T_train):
                y_pred, s_pred = model(input)
                y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
            loss_ce = NLLLoss(y_softmax, label, domain_label, args)
            # loss_ce = RegCrossEntropyLoss_bayes(y_softmax, label, reg_weight, domain_label, args)
            loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
            loss = loss_ce + args.lambda_grad_penalty * loss_gp
            loss_var_item = eval_subbatch_variance_bayes(model, None, (input, label), args)
        elif args.reg_type.lower() == 'var':
            # Minibatch Variance
            y_softmax = 0.
            for t in range(args.T_train):
                y_pred, s_pred = model(input)
                y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
            loss_ce = NLLLoss(y_softmax, label, domain_label, args)
            # loss_ce = RegCrossEntropyLoss_bayes(y_softmax, label, reg_weight, domain_label, args)
            loss_var = MiniBatchVarLoss_bayes(model, param_list, (input, label, domain_label), args)
            loss = loss_ce + args.lambda_minibatch_var * loss_var
            loss_var_item = loss_var.item()
        elif args.reg_type.lower() == 'var-gp':
            # Minibatch Variance + GradPanelty
            y_softmax = 0.
            for t in range(args.T_train):
                y_pred, s_pred = model(input)
                y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
            loss_ce = NLLLoss(y_softmax, label, domain_label, args)
            # loss_ce = RegCrossEntropyLoss_bayes(y_softmax, label, reg_weight, domain_label, args)
            loss_var = MiniBatchVarLoss_bayes(model, param_list, (input, label, domain_label), args)
            loss_gp = GradPenaltyLoss(model, param_list_gp, loss_ce, args)
            loss = loss_ce + args.lambda_minibatch_var * loss_var + args.lambda_grad_penalty * loss_gp
            loss_var_item = loss_var.item()
        else:
            raise NotImplementedError

        if torch.isnan(loss):
            print('breakpoint 1')
            pdb.set_trace()

        # measure accuracy and record loss
        prec1, gt_num = accuracy(y_softmax.data, label, args.num_class, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], gt_num[0])
        wvar.update(loss_var_item, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        for p in model.named_parameters():
            if torch.isnan(p[1].grad).any():
                print('breakpoint 2')
                print(p[0])
                pdb.set_trace()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(prompt + 'Train Epoch: [{0}][{1}/{2}], '
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                           'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                           'WVar {wvar.val:.3f} ({wvar.avg:.3f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, wvar=wvar))
            logger.info(prompt + 'Train Epoch: [{0}][{1}/{2}], '
                                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                                 'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                                 'WVar {wvar.val:.3f} ({wvar.avg:.3f})'
                        .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, wvar=wvar))
    # log to TensorBoard
    if args.tensorboard:
        for param_group in optimizer.param_groups:
            clr = param_group['lr']
        log_value(f'{prompt}train_loss', losses.avg, epoch)
        log_value(f'{prompt}train_acc', top1.vec2sca_avg, epoch)
        log_value(f'{prompt}lr', clr, epoch)
        log_value(f'{prompt}train_wvar', wvar.avg, epoch)


def validate(val_loader, model, epoch, device, logger, prompt='', param_list=[], param_list_gp=[]):
    """Perform validation on the validation set and save all the confidence vectors"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    wvar = AverageMeter()

    # switch to evaluate mode
    model.eval()
    ImgName_list = []
    pred_logit_list = []
    # feat_list = []
    label_list = []

    end = time.time()
    #with torch.no_grad():
    if True:
        for i, (input, label, input_name, reg_weight, domain_label) in enumerate(val_loader):
            label = label.to(device)
            input = input.to(device)
            reg_weight = reg_weight.to(device)

            wvar.update(eval_subbatch_variance(model, None, (input, label), args), input.size(0))

            # compute output
            # if isinstance(model, nn.DataParallel):
            #     model_fc = model.module.fc
            #     model.module.fc = model.module.fc[0:2]
            #     feat = model(input)
            #     model.module.fc = model_fc
            # else:
            #     model_fc = model.fc
            #     model.fc = model.fc[0:2]
            #     feat = model(input)
            #     model.fc = model_fc
            output = model(input)
            if not args.debug:
                loss = RegCrossEntropyLoss(output, label, reg_weight, None, args)
            else:
                if args.reg_type.lower() == 'none':
                    print('none')
                    loss = RegCrossEntropyLoss(output, label, reg_weight, domain_label.float().to(device), args)
                elif args.reg_type.lower() == 'gp':
                    print('gp')
                    # GP wrt theta
                    loss_ce = RegCrossEntropyLoss(output, label, reg_weight, domain_label.float().to(device), args)
                    loss_gp = GradPenaltyLoss(model, param_list, loss_ce, args)
                    print(loss_gp * args.lambda_grad_penalty)
                    loss = loss_ce + args.lambda_grad_penalty * loss_gp
                elif args.reg_type.lower() == 'var':
                    print('var')
                    # Minibatch Variance
                    loss_ce = RegCrossEntropyLoss(output, label, reg_weight, domain_label.float().to(device), args)
                    loss_var = MiniBatchVarLoss(model, param_list, (input, label, domain_label.float().to(device)), args)
                    print(loss_var * args.lambda_minibatch_var)
                    loss = loss_ce + args.lambda_minibatch_var * loss_var
                elif args.reg_type.lower() == 'var-gp':
                    # Minibatch Variance + GradPanelty
                    loss_ce = RegCrossEntropyLoss(output, label, reg_weight, domain_label.float().to(device), args)
                    loss_var = MiniBatchVarLoss(model, param_list, (input, label, domain_label.float().to(device)), args)
                    loss_gp = GradPenaltyLoss(model, param_list, loss_ce, args)
                    loss = loss_ce + args.lambda_minibatch_var * loss_var + args.lambda_grad_penalty * loss_gp
                else:
                    raise NotImplementedError
                loss.backward()
            # measure accuracy and record loss
            prec1, gt_num = accuracy(output.data, label, args.num_class, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], gt_num[0])

            # measure elapsed timeImgName_Label
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test epoch: [{0}][{1}/{2}], '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Prec@1-pcls {top1.val} ({top1.avg}), '
                      'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f}), '
                      'WVar {wvar.val:.3f} ({wvar.avg:.3f})'
                      .format(epoch, i, len(val_loader), batch_time=batch_time, top1=top1, wvar=wvar))
                logger.info('Test epoch: [{0}][{1}/{2}], '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                            'Prec@1-pcls {top1.val} ({top1.avg}), '
                            'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f}), '
                            'WVar {wvar.val:.3f} ({wvar.avg:.3f})'
                            .format(epoch, i, len(val_loader), batch_time=batch_time, top1=top1, wvar=wvar))
            # save prediction confidence vectors
            for idx_batch in range(output.data.size(0)):
                # image_name_list and pred_conf_list have the same index
                ImgName_list.append(input_name[idx_batch])
                pred_logit_list.append(output.data[idx_batch, :])
                # feat_list.append(feat.data[idx_batch, :])
                label_list.append(label.data[idx_batch, :])

    print(' * Prec@1 {top1.vec2sca_avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.vec2sca_avg, epoch)
        log_value('val_wvar', wvar.avg, epoch)
        for j, v in enumerate(top1.avg):
            log_value(f'val_acc_{j}', v.item(), epoch)
    return top1.vec2sca_avg, ImgName_list, pred_logit_list, None, label_list


def validate_bayes(val_loader, model, epoch, device, logger, prompt='', param_list=[], param_list_gp=[]):
    """Perform validation on the validation set and save all the confidence vectors"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    wvar = AverageMeter()

    # switch to evaluate mode
    model.eval()
    ImgName_list = []
    pred_prob_list = []
    # feat_list = []
    label_list = []

    end = time.time()
    #with torch.no_grad():
    if True:
        for i, (input, label, input_name, reg_weight, domain_label) in enumerate(val_loader):
            label = label.to(device)
            input = input.to(device)
            reg_weight = reg_weight.to(device)

            wvar.update(eval_subbatch_variance_bayes(model, None, (input, label), args), input.size(0))

            # compute output
            # if isinstance(model, nn.DataParallel):
            #     model_fc = model.module.fc
            #     model.module.fc = model.module.fc[0:2]
            #     feat, feat_logvar = model(input)
            #     model.module.fc = model_fc
            # else:
            #     model_fc = model.fc
            #     model.fc = model.fc[0:2]
            #     feat, feat_logvar = model(input)
            #     model.fc = model_fc

            if not args.debug:
                y_softmax = 0.
                for t in range(args.T):
                    y_pred, s_pred = model(input)
                    y_softmax += 1. / args.T * F.softmax(reparameterize(y_pred, s_pred), dim=1)
                loss = RegCrossEntropyLoss_bayes(y_softmax, label, reg_weight, None, args)
            else:
                if args.reg_type.lower() == 'none':
                    y_softmax = 0.
                    for t in range(args.T_train):
                        y_pred, s_pred = model(input)
                        y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
                    loss = RegCrossEntropyLoss_bayes(y_softmax, label, reg_weight, domain_label.float().to(device), args)
                    # loss_gp = torch.tensor(0)
                    loss_gp = eval_subbatch_variance_bayes(model, None, (input, label), args)
                    print(loss_gp)
                elif args.reg_type.lower() == 'gp':
                    # GP wrt theta
                    y_softmax = 0.
                    for t in range(args.T_train):
                        y_pred, s_pred = model(input)
                        y_softmax += 1. / args.T_train * F.softmax(reparameterize(y_pred, s_pred), dim=1)
                    loss_ce = NLLLoss(y_softmax, label, domain_label.float().to(device), args)
                    loss_gp = GradPenaltyLoss(model, param_list, loss_ce, args)
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
                    loss_gp = GradPenaltyLoss(model, param_list, loss_ce, args)
                    loss = loss_ce + args.lambda_minibatch_var * loss_var + args.lambda_grad_penalty * loss_gp
                    print(loss_var, loss_gp)
                else:
                    raise NotImplementedError
                loss.backward()

            # measure accuracy and record loss
            prec1, gt_num = accuracy(y_softmax.data, label, args.num_class, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], gt_num[0])

            # measure elapsed timeImgName_Label
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test epoch: [{0}][{1}/{2}], '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Prec@1-pcls {top1.val} ({top1.avg}), '
                      'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f}), '
                      'WVar {wvar.val:.3f} ({wvar.avg:.3f})'
                      .format(epoch, i, len(val_loader), batch_time=batch_time, top1=top1, wvar=wvar))
                logger.info('Test epoch: [{0}][{1}/{2}], '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                            'Prec@1-pcls {top1.val} ({top1.avg}), '
                            'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f}), '
                            'WVar {wvar.val:.3f} ({wvar.avg:.3f})'
                            .format(epoch, i, len(val_loader), batch_time=batch_time, top1=top1, wvar=wvar))
            # save prediction confidence vectors
            for idx_batch in range(y_softmax.data.size(0)):
                # image_name_list and pred_conf_list have the same index
                ImgName_list.append(input_name[idx_batch])
                pred_prob_list.append(y_softmax.data[idx_batch, :])
                # feat_list.append(feat.data[idx_batch, :])
                label_list.append(label.data[idx_batch, :])

    print(' * Prec@1 {top1.vec2sca_avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.vec2sca_avg, epoch)
        log_value('val_wvar', wvar.avg, epoch)
        for j, v in enumerate(top1.avg):
            log_value(f'val_acc_{j}', v.item(), epoch)
    return top1.vec2sca_avg, ImgName_list, pred_prob_list, None, label_list


def kc_parameters(tgt_portion, kc_policy, pred_logit_list, num_class, device, args):
    """determine kc_parameters, policy: global threshold; class-balance threshold; reweighted class-balance threshold"""
    pred_logit_tensor = torch.stack(pred_logit_list)
    if args.bayesian:
        pred_prob_tensor = pred_logit_tensor
    else:
        pred_prob_tensor = F.softmax(pred_logit_tensor, 1)
    pred_conf, pred_label = pred_prob_tensor.topk(1, 1, True, True)

    # soft_pseudo_label_matrix = torch.stack(soft_pseudo_label)
    # log_soft_pseudo_label_matrix = torch.log(soft_pseudo_label_matrix+1e-20)
    # pred_conf_tensor = F.softmax(pred_logit_tensor,1)
    # pred_log_conf_tensor = torch.log(pred_conf_tensor+1e-20)
    # pred_label_conf,pred_label = pred_conf_tensor.topk(1, 1, True, True)

    # ce = torch.sum(-pred_log_conf_tensor * soft_pseudo_label_matrix, 1)  # cross-entropy loss with vector-form softmax
    # l2_ls = torch.sum(soft_pseudo_label_matrix * soft_pseudo_label_matrix, 1)
    # negent_ls = torch.sum(soft_pseudo_label_matrix * log_soft_pseudo_label_matrix, 1)
    #
    # reg_ce = ce + ls_weight_l2 * l2_ls + ls_weight_negent * negent_ls

    # if kc_policy == 'global':
    #     num_label_conf = len(reg_ce)
    #     reg_ce_sort,_ = reg_ce.view(-1).sort(descending=False)
    #     reg_ce_global = reg_ce_sort[int(np.floor((num_label_conf-1)*tgt_portion))]
    #     kc_global = -reg_ce_global
    #     kct_matrix = kc_global*kct_matrix
    kc_values = torch.ones(num_class)
    if args.kc_value == 'conf':
        for idx_class in range(num_class):
            pred_class_idx = pred_label == idx_class
            pred_conf = pred_conf.view(-1,1)
            pred_class_conf = pred_conf[pred_class_idx]
            num_class_conf = len(pred_class_conf)
            if num_class_conf != 0:
                pred_class_conf,_ = pred_class_conf.view(-1).sort(descending=True)
                kc_values[idx_class] = pred_class_conf[int(np.floor((num_class_conf - 1) * tgt_portion))]
    elif args.kc_value == 'prob':
        for idx_class in range(num_class):
            num_pred_class = torch.sum(pred_label == idx_class) # number of predications for each class
            num_pred_class = num_pred_class.cpu().numpy()
            if num_pred_class != 0:
                prob_cls = pred_prob_tensor[:,idx_class]
                prob_cls,_ = prob_cls.view(-1).sort(descending=True)
                kc_values[idx_class] = prob_cls[int(np.floor((num_pred_class - 1) * tgt_portion))] # whether to use the predication portion.
    kc_values = kc_values.to(device)
    kct_matrix = torch.ones(pred_prob_tensor.size()).to(device)
    if args.kc_policy == 'ucb':
        for idx_class in range(num_class):
            pred_class_idx = pred_label == idx_class
            num_pred_cls = pred_class_idx.sum()
            if num_pred_cls != 0:
                kct_matrix[pred_class_idx.view(-1), :] = kc_values[idx_class].to(device)
    elif args.kc_policy == 'cb':
        kct_matrix = kct_matrix*kc_values.to(device)
    return kct_matrix


def soft_pseudo_label(pred_logit_list, kct_matrix, args):
    """Soft pseudo-label generation"""
    ls_weight_negent = args.ls_weight_negent
    pred_logit_tensor = torch.stack(pred_logit_list)
    if args.bayesian:
        pred_prob_tensor = pred_logit_tensor
    else:
        pred_prob_tensor = F.softmax(pred_logit_tensor, 1)
    wpred_prob_tensor = pred_prob_tensor/kct_matrix
    # no regularization
    if ls_weight_negent == 0.0:
        pred_label_conf, _ = wpred_prob_tensor.topk(1, 1, True, True)
        pred_softlabel_matrix = wpred_prob_tensor >= pred_label_conf
        pred_softlabel_matrix = pred_softlabel_matrix.type(torch.float)
    # negative entropy regulared label smoothing
    elif ls_weight_negent > 0.0:
        wpred_prob_power_tensor = wpred_prob_tensor.pow(1.0/ls_weight_negent)
        wpred_prob_power_sum = wpred_prob_power_tensor.sum(1).view(-1, 1)
        pred_softlabel_matrix = wpred_prob_power_tensor/wpred_prob_power_sum
    else:
        raise RuntimeError
    # pred_softlabel_list = [pred_softlabel_matrix[t_idx, :] for t_idx in range(pred_softlabel_matrix.size(0))]
    return pred_softlabel_matrix


def label_selection(logits_list, pred_softlabel_matrix, kct_matrix, ImgName_list, args, epoch):
    """Pseudo-label selection"""
    # parameters
    num_class = args.num_class
    ls_weight_negent = args.ls_weight_negent
    logits_matrix = torch.stack(logits_list)
    log_pred_softlabel_matrix = torch.log(pred_softlabel_matrix+1e-20)

    # regularized loss with self-paced thresholding
    if args.bayesian:
        softmax = logits_matrix
    else:
        softmax = F.softmax(logits_matrix, 1)
    logsoftmax = torch.log(softmax + 1e-20)  # compute the log of softmax values
    ce = torch.sum(-logsoftmax * pred_softlabel_matrix, 1)  # cross-entropy loss with vector-form softmax
    if ls_weight_negent > 0.0:
        negent_ls = torch.sum(pred_softlabel_matrix * log_pred_softlabel_matrix, 1)
        reg_ce = ce + ls_weight_negent * negent_ls
    elif ls_weight_negent == 0.0:
        reg_ce = ce
    sp_threshold = torch.sum(-pred_softlabel_matrix * torch.log(kct_matrix+1e-20), 1)
    threshold_idx = reg_ce < sp_threshold

    # save soft label into txt file
    fout = args.tgt_train_list
    fo = open(fout, "w")
    for idx_label in range(len(reg_ce.view(-1))):
        curr_threshold_idx = threshold_idx[idx_label]
        curr_soft_label = torch.zeros(num_class)
        if curr_threshold_idx == 1:
            curr_soft_label = pred_softlabel_matrix[idx_label]
        curr_soft_label = curr_soft_label.cpu().numpy()
        fo.write(ImgName_list[idx_label] + ' ' + ' '.join(map(str, curr_soft_label)) + '\n')
    fo.close()

    # save softmax into txt file
    fout_softmax = args.tgt_train_list[:-4]+'_softmax.txt'
    fo = open(fout_softmax, "w")
    for idx_label in range(len(reg_ce.view(-1))):
        curr_soft_label = softmax[idx_label,:]
        curr_soft_label = curr_soft_label.cpu().numpy()
        fo.write(ImgName_list[idx_label] + ' ' + ' '.join(map(str, curr_soft_label)) + '\n')
    fo.close()
    
    # save softlabel into txt file
    # fout_softlabel = 'runs/'+args.name + '/soft_pseudo_label_epoch' + str(epoch) + '.txt'
    fout_softlabel = os.path.join(args.expr_dir, 'soft_pseudo_label_epoch' + str(epoch) + '.txt')
    fo = open(fout_softlabel, "w")
    for idx_label in range(len(reg_ce.view(-1))):
        curr_soft_pseudo_label = pred_softlabel_matrix[idx_label, :]
        curr_soft_pseudo_label = curr_soft_pseudo_label.cpu().numpy()
        fo.write(ImgName_list[idx_label] + ' ' + ' '.join(map(str, curr_soft_pseudo_label)) + '\n')
    fo.close()


def RegCrossEntropyLoss(outputs, labels, reg_weight, domain_label=None, args=None):
    mr_weight_l2 = args.mr_weight_l2
    mr_weight_negent = args.mr_weight_negent
    mr_weight_kld = args.mr_weight_kld
    num_class = args.num_class
    batch_size = labels.size(0)            # batch_size
    # batch_size_valid_bool = torch.sum(labels,dim=1) > 0
    # batch_size_valid = batch_size_valid_bool.sum().type(torch.float)

    softmax = F.softmax(outputs, dim=1)   # compute the log of softmax values
    logsoftmax = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
    ce = torch.sum(-logsoftmax*labels)  # cross-entropy loss with vector-form softmax
    valid_labels = torch.round(torch.sum(labels, dim=1)).view(batch_size, -1)
    num_valid_labels = valid_labels.sum()
    if num_valid_labels > 0:
        reg_weight = reg_weight.view(batch_size, -1)
        l2 = torch.sum(softmax * softmax * reg_weight * valid_labels)
        negent = torch.sum(softmax * logsoftmax * reg_weight * valid_labels)
        kld = torch.sum(-logsoftmax / float(num_class) * reg_weight * valid_labels)
        if domain_label is None or args.entropy.lower() == 'min':
            # min entropy is pseudo-label cross-entropy
            reg_ce = (torch.sum(ce) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld) / num_valid_labels
        elif args.entropy.lower() == 'collision':
            # Renyi entropy, alpha=2
            ent = -torch.log(torch.sum(softmax.pow(2), dim=1) + MAGIC_EPS)
            reg_ce = (torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label)) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld) / num_valid_labels
        elif args.entropy.lower() == 'shannon':
            # Shannon entropy is 'entropy regularization'
            ent = -torch.sum(softmax * logsoftmax, dim=1)
            reg_ce = (torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label)) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld) / num_valid_labels
        else:
            raise NotImplementedError
    else:  # num_valid_labels == 0:
        reg_ce = torch.sum(ce) * 0.
    return reg_ce


def RegCrossEntropyLoss_bayes(softmax, labels, reg_weight, domain_label=None, args=None):
    mr_weight_l2 = args.mr_weight_l2
    mr_weight_negent = args.mr_weight_negent
    mr_weight_kld = args.mr_weight_kld
    num_class = args.num_class
    batch_size = labels.size(0)  # batch_size
    # batch_size_valid_bool = torch.sum(labels,dim=1) > 0
    # batch_size_valid = batch_size_valid_bool.sum().type(torch.float)

    logsoftmax = torch.log(softmax + MAGIC_EPS)  # compute the log of softmax values
    ce = torch.sum(-logsoftmax*labels, dim=1)  # cross-entropy loss with vector-form softmax
    valid_labels = torch.round(torch.sum(labels, dim=1)).view(batch_size, -1)
    num_valid_labels = valid_labels.sum()
    if num_valid_labels > 0:
        reg_weight = reg_weight.view(batch_size, -1)
        l2 = torch.sum(softmax * softmax * reg_weight * valid_labels)
        negent = torch.sum(softmax * logsoftmax * reg_weight * valid_labels)
        kld = torch.sum(-logsoftmax / float(num_class) * reg_weight * valid_labels)
        if domain_label is None or args.entropy.lower() == 'min':
            # min entropy is pseudo-label cross-entropy
            reg_ce = (torch.sum(ce) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld) / num_valid_labels
        elif args.entropy.lower() == 'collision':
            # Renyi entropy, alpha=2
            ent = -torch.log(torch.sum(softmax.pow(2), dim=1) + MAGIC_EPS)
            reg_ce = (torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label)) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld) / num_valid_labels
        elif args.entropy.lower() == 'shannon':
            # Shannon entropy is 'entropy regularization'
            ent = -torch.sum(softmax * logsoftmax, dim=1)
            reg_ce = (torch.sum(ent * domain_label) + torch.sum(ce * (1. - domain_label)) + mr_weight_l2 * l2 + mr_weight_negent * negent + mr_weight_kld * kld) / num_valid_labels
        else:
            raise NotImplementedError
    else:  # num_valid_labels == 0:
        reg_ce = torch.sum(ce) * 0.
    return reg_ce


def saveSRCtxt(src_portion, randseed, args):
    src_gt_txt = args.src_gt_list
    src_train_txt = args.src_train_list
    item_list = []
    count0 = 0
    with open(src_gt_txt, "rt") as f:
        for item in f.readlines():
            fields = item.strip()
            item_list.append(fields)
            count0 = count0 + 1

    num_source = count0
    num_sel_source = int(np.floor(num_source*src_portion))
    np.random.seed(randseed)
    print(num_source)
    print(num_sel_source)
    sel_idx = list(np.random.choice(num_source, num_sel_source, replace=False))
    item_list = list(itemgetter(*sel_idx)(item_list))

    with open(src_train_txt, 'w') as f:
        for item in item_list:
            f.write("%s\n" % item)


class ImageClassdata(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, txt_file, root_dir, reg_weight, transform=transforms.ToTensor(), domain=0):
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
        self.reg_weight = reg_weight
        self._domain = domain

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        ImgName_Label = str.split(self.images_frame[idx])
        img_name = os.path.join(self.root_dir, ImgName_Label[0])
        img = Image.open(img_name)
        image = img.convert('RGB')
        lbl = np.asarray(ImgName_Label[1:], dtype=np.float32)
        label = torch.from_numpy(lbl)
        reg_weight = torch.from_numpy(np.asarray(self.reg_weight, dtype=np.float32))

        if self.transform:
            image = self.transform(image)

        return image, label, ImgName_Label[0], reg_weight, self._domain


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vec2sca_avg = 0
        self.vec2sca_val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if torch.is_tensor(self.val) and torch.numel(self.val) != 1:
            self.avg[self.count == 0] = 0
            self.vec2sca_avg = self.avg.sum() / len(self.avg)
            self.vec2sca_val = self.val.sum() / len(self.val)


def accuracy(output, label, num_class, topk=(1,)):
    """Computes the precision@k for the specified values of k, currently only k=1 is supported"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    _, gt = label.topk(maxk, 1, True, True)
    pred = pred.t()
    pred_class_idx_list = [pred == class_idx for class_idx in range(num_class)]
    gt = gt.t()
    gt_class_number_list = [(gt == class_idx).sum() for class_idx in range(num_class)]
    correct = pred.eq(gt)

    res = []
    gt_num = []
    for k in topk:
        correct_k = correct[:k].float()
        per_class_correct_list = [correct_k[pred_class_idx].sum(0) for pred_class_idx in pred_class_idx_list]
        per_class_correct_array = torch.tensor(per_class_correct_list)
        gt_class_number_tensor = torch.tensor(gt_class_number_list).float()
        gt_class_zeronumber_tensor = gt_class_number_tensor == 0
        gt_class_number_matrix = torch.tensor(gt_class_number_list).float()
        gt_class_acc = per_class_correct_array.mul_(100.0 / gt_class_number_matrix)
        gt_class_acc[gt_class_zeronumber_tensor] = 0
        res.append(gt_class_acc)
        gt_num.append(gt_class_number_matrix)
    return res, gt_num


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


def validate_mem(val_loader, model, epoch, device, logger, prompt='', param_list=[], param_list_gp=[]):
    """Perform validation on the validation set and save all the confidence vectors"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    ImgName_list = []
    pred_logit_list = []
    # feat_list = []
    label_list = []

    end = time.time()
    with torch.no_grad():
        for i, (input, label, input_name, reg_weight, domain_label) in enumerate(val_loader):
            label = label.to(device)
            input = input.to(device)
            reg_weight = reg_weight.to(device)

            # compute output
            # if isinstance(model, nn.DataParallel):
            #     model_fc = model.module.fc
            #     model.module.fc = model.module.fc[0:2]
            #     feat = model(input)
            #     model.module.fc = model_fc
            # else:
            #     model_fc = model.fc
            #     model.fc = model.fc[0:2]
            #     feat = model(input)
            #     model.fc = model_fc
            output = model(input)
            loss = RegCrossEntropyLoss(output, label, reg_weight, None, args)

            # measure accuracy and record loss
            prec1, gt_num = accuracy(output.data, label, args.num_class, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], gt_num[0])

            # measure elapsed timeImgName_Label
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test epoch: [{0}][{1}/{2}], '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Prec@1-pcls {top1.val} ({top1.avg}), '
                      'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f})'
                      .format(epoch, i, len(val_loader), batch_time=batch_time, top1=top1))
                logger.info('Test epoch: [{0}][{1}/{2}], '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                            'Prec@1-pcls {top1.val} ({top1.avg}), '
                            'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f})'
                            .format(epoch, i, len(val_loader), batch_time=batch_time, top1=top1))
            # save prediction confidence vectors
            for idx_batch in range(output.data.size(0)):
                # image_name_list and pred_conf_list have the same index
                ImgName_list.append(input_name[idx_batch])
                pred_logit_list.append(output.data[idx_batch, :])
                # feat_list.append(feat.data[idx_batch, :])
                label_list.append(label.data[idx_batch, :])

    print(' * Prec@1 {top1.vec2sca_avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.vec2sca_avg, epoch)
        for j, v in enumerate(top1.avg):
            log_value(f'val_acc_{j}', v.item(), epoch)
    return top1.vec2sca_avg, ImgName_list, pred_logit_list, None, label_list


def validate_bayes_mem(val_loader, model, epoch, device, logger, prompt='', param_list=[], param_list_gp=[]):
    """Perform validation on the validation set and save all the confidence vectors"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    ImgName_list = []
    pred_prob_list = []
    # feat_list = []
    label_list = []

    end = time.time()
    with torch.no_grad():
        for i, (input, label, input_name, reg_weight, domain_label) in enumerate(val_loader):
            label = label.to(device)
            input = input.to(device)
            reg_weight = reg_weight.to(device)

            # compute output
            # if isinstance(model, nn.DataParallel):
            #     model_fc = model.module.fc
            #     model.module.fc = model.module.fc[0:2]
            #     feat, feat_logvar = model(input)
            #     model.module.fc = model_fc
            # else:
            #     model_fc = model.fc
            #     model.fc = model.fc[0:2]
            #     feat, feat_logvar = model(input)
            #     model.fc = model_fc
            y_softmax = 0.
            for t in range(args.T):
                y_pred, s_pred = model(input)
                y_softmax += 1. / args.T * F.softmax(reparameterize(y_pred, s_pred), dim=1)
            loss = RegCrossEntropyLoss_bayes(y_softmax, label, reg_weight, None, args)

            # measure accuracy and record loss
            prec1, gt_num = accuracy(y_softmax.data, label, args.num_class, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], gt_num[0])

            # measure elapsed timeImgName_Label
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test epoch: [{0}][{1}/{2}], '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Prec@1-pcls {top1.val} ({top1.avg}), '
                      'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f})'
                      .format(epoch, i, len(val_loader), batch_time=batch_time, top1=top1))
                logger.info('Test epoch: [{0}][{1}/{2}], '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                            'Prec@1-pcls {top1.val} ({top1.avg}), '
                            'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f})'
                            .format(epoch, i, len(val_loader), batch_time=batch_time, top1=top1))
            # save prediction confidence vectors
            for idx_batch in range(y_softmax.data.size(0)):
                # image_name_list and pred_conf_list have the same index
                ImgName_list.append(input_name[idx_batch])
                pred_prob_list.append(y_softmax.data[idx_batch, :])
                # feat_list.append(feat.data[idx_batch, :])
                label_list.append(label.data[idx_batch, :])

    print(' * Prec@1 {top1.vec2sca_avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.vec2sca_avg, epoch)
        for j, v in enumerate(top1.avg):
            log_value(f'val_acc_{j}', v.item(), epoch)
    return top1.vec2sca_avg, ImgName_list, pred_prob_list, None, label_list


def save_str_list(l, name):
    with open(name, 'w') as f:
        for s in l:
            f.write(f'{s}\n')


def save_arr_list(l, name):
    with open(name, 'w') as f:
        for a in l:
            f.write(' '.join(map(str, a.cpu().numpy())) + '\n')


def load_str_list(name):
    with open(name, 'r') as f:
        l = f.readlines()
    return [s.rstrip('\n') for s in l]


def load_arr_list(name, gpu=True):
    with open(name, 'r') as f:
        l = f.readlines()
    if gpu:
        return [torch.from_numpy(np.asarray(a.split(), dtype=np.float32)).cuda() for a in l]
    else:
        return [torch.from_numpy(np.asarray(a.split(), dtype=np.float32)) for a in l]


if __name__ == '__main__':
    main()
