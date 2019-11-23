import os
import shutil
import torch
from torch import autograd
import scipy.io
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
import math


def print_options(parser, opt):
    message = ''
    message += '--------------- Options -----------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)
    file_name = os.path.join(opt.expr_dir, 'args.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def reparameterize_straight_through(mu, logvar):
    eps = torch.zeros_like(logvar)
    return mu + eps * logvar


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_checkpoint(state, is_best, expr_dir='runs/exp', filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    # state_copy = deepcopy(state)
    # state_copy = state_copy.cpu()
    state_copy = state
    filename = os.path.join(expr_dir, 'epoch_'+str(state_copy['epoch']) + '_' + filename)
    torch.save(state_copy, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(expr_dir, 'model_best.pth.tar'))


def saveMat(feat_tensor, label_tensor, epoch, expr_dir, args):
    feat_filename = os.path.join(expr_dir, 'epoch_'+str(epoch) + '_feat.mat')
    label_filename = os.path.join(expr_dir, 'epoch_' + str(epoch) + '_label.mat')
    scipy.io.savemat(feat_filename, {'feat': feat_tensor})
    scipy.io.savemat(label_filename, {'label': label_tensor})


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
        )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def get_param_list(model, param_type, args=None):
    if param_type == 'all':
        param_list = list(model.parameters())
    elif param_type == '0':
        param_list = list(model.parameters())[0]
    elif param_type == 'no_bn':
        if isinstance(model, nn.DataParallel):
            param_list = model.module.param_without_bn()
        else:
            param_list = model.param_without_bn()
    elif param_type == 'conv':
        if isinstance(model, nn.DataParallel):
            param_list = model.module.param_conv()
        else:
            param_list = model.param_conv()
    elif param_type == 'layerx':
        if isinstance(model, nn.DataParallel):
            param_list = model.module.param_layerx(args.resnet_layer_up_to)
        else:
            param_list = model.param_layerx(args.resnet_layer_up_to)
    else:
        raise NotImplementedError
    return param_list


def get_optimizer(model, optim_type, args):
    if optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    elif optim_type == 'adam2':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.9))
    elif optim_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.rmsprop_alpha)
    elif optim_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.momentum > 0)
    else:
        raise NotImplementedError
    return optimizer


def get_minibatch_size(batch_size, minibatch_size, num_minibatch):
    if batch_size != minibatch_size * num_minibatch:
        print(f'Warning: minibatch-size ({minibatch_size}) and num-minibatch ({num_minibatch}) do not match with batch-size ({batch_size})')
        if num_minibatch <= batch_size:
            if batch_size % num_minibatch != 0:
                print(f'Warning: remainder of batch-size/num-minibatch ({batch_size}/{num_minibatch}) is not zero')
            minibatch_size = math.ceil(batch_size / num_minibatch)
        elif minibatch_size >= batch_size:
            print(f'Warning: minibatch-size ({minibatch_size}) is larger than batch-size ({batch_size})')
            minibatch_size = math.floor(batch_size / 2)
    num_minibatch = math.floor(batch_size / minibatch_size)
    print(f'-> batch-size: {batch_size}, minibatch-size: {minibatch_size}, num-minibatch: {num_minibatch}')
    return minibatch_size


def assert_args(args):
    if 'tar' in args.reg_domain and args.reg_type != 'none' and not args.filter_out_invalid:
        print('Warning: filter-out-invalid is False, but mini-batch-variance-loss is used!!!')
        return False


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


# borrowed from https://github.com/dragen1860/MAML-Pytorch/blob/master/learner.py
def forward(net, x, vars=None, bn_training=True):
    """
    This function can be called by finetunning, however, in finetunning, we dont wish to update
    running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
    Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
    but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
    :param x: [b, 1, 28, 28]
    :param vars:
    :param bn_training: set False to not update
    :return: x, loss, likelihood, kld
    """

    idx = 0
    bn_idx = 0
    net.get_bn_vars()

    for name, param in net.config:
        if name is 'conv2d':
            w, b = vars[idx], vars[idx + 1]
            # remember to keep synchrozied of forward_encoder and forward_decoder!
            x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
            idx += 2
            # print(name, param, '\tout:', x.shape)
        elif name is 'convt2d':
            w, b = vars[idx], vars[idx + 1]
            # remember to keep synchrozied of forward_encoder and forward_decoder!
            x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
            idx += 2
            # print(name, param, '\tout:', x.shape)
        elif name is 'linear':
            w, b = vars[idx], vars[idx + 1]
            x = F.linear(x, w, b)
            idx += 2
            # print('forward:', idx, x.norm().item())
        elif name is 'bn':
            w, b = vars[idx], vars[idx + 1]
            running_mean, running_var = net.vars_bn[bn_idx], net.vars_bn[bn_idx + 1]
            x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
            idx += 2
            bn_idx += 2
        elif name is 'dropout2d':
            x = F.dropout2d(x, p=param[0])
        elif name is 'dropout':
            x = F.dropout(x, p=param[0])
        elif name is 'flatten':
            # print(x.shape)
            x = x.view(x.size(0), -1)
        elif name is 'reshape':
            # [b, 8] => [b, 2, 2, 2]
            x = x.view(x.size(0), *param)
        elif name is 'relu':
            x = F.relu(x, inplace=param[0])
        elif name is 'leakyrelu':
            x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
        elif name is 'tanh':
            x = F.tanh(x)
        elif name is 'sigmoid':
            x = torch.sigmoid(x)
        elif name is 'upsample':
            x = F.upsample_nearest(x, scale_factor=param[0])
        elif name is 'max_pool2d':
            x = F.max_pool2d(x, param[0], param[1], param[2])
        elif name is 'avg_pool2d':
            x = F.avg_pool2d(x, param[0], param[1], param[2])

        else:
            raise NotImplementedError

    # make sure variable is used properly
    assert idx == len(vars)
    assert bn_idx == len(net.vars_bn)

    return x
