"""Optimizer."""
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
from configs.config import cfg
from core.net import SoftCrossEntropyLoss, unwrap_model, WeightedMSELoss, FocalLoss


# Supported loss functions
_loss_functions = {
    'cross_entropy': nn.CrossEntropyLoss,
    'soft_cross_entropy': SoftCrossEntropyLoss,
    'mse': nn.MSELoss,
    'weighted_mse': WeightedMSELoss,
    'smooth_l1': nn.SmoothL1Loss,
    'focal_loss': FocalLoss,
}


def build_loss_fun():
    loss_name = cfg.MODEL.LOSS_FUN
    err_message = "Loss function {} not supported.".format(loss_name)
    assert loss_name in _loss_functions.keys(), err_message

    return _loss_functions[loss_name]()


def construct_optimizer(model, is_tune=False):
    """Constructs the optimizer.
    Note that the momentum update in PyTorch differs from the one in Caffe2.
    In particular,
        Caffe2:
            V := mu * V + lr * g
            p := p - V
        PyTorch:
            V := mu * V + g
            p := p - lr * V
    where V is the velocity, mu is the momentum factor, lr is the learning rate,
    g is the gradient and p are the parameters.
    Since V is defined independently of the learning rate in PyTorch,
    when the learning rate is changed there is no need to perform the
    momentum correction by scaling V (unlike in the Caffe2 case).
    """

    if model is None:
        return

    optim, wd = cfg.OPTIM, cfg.OPTIM.WEIGHT_DECAY
    if cfg.MODEL.TYPE == 'backbone':
        # Split parameters into types and get weight decay for each type
        params = [[], [], [], []]
        for n, p in model.named_parameters():
            ks = [k for (k, x) in enumerate(["bn", "ln", "bias", ""]) if x in n]
            params[ks[0]].append(p)

        wds = [
            cfg.BN.CUSTOM_WEIGHT_DECAY if cfg.BN.USE_CUSTOM_WEIGHT_DECAY else wd,
            cfg.LN.CUSTOM_WEIGHT_DECAY if cfg.LN.USE_CUSTOM_WEIGHT_DECAY else wd,
            optim.BIAS_CUSTOM_WEIGHT_DECAY if optim.BIAS_USE_CUSTOM_WEIGHT_DECAY else wd,
            wd,
        ]
        param_list = [{"params": p, "weight_decay": w} for (p, w) in zip(params, wds) if p]
    else:
        # Construct optimizer for training the box predictor.
        # param_list = [{"params": unwrap_model(model).feature.parameters(), "lr": optim.BASE_LR},
        #               {"params": unwrap_model(model).box_head.parameters(), "lr": optim.FEATURE_HEAD_LR_RATIO * optim.BASE_LR}]
        param_list = model.parameters()
    # Set up optimizer
    if is_tune:
        lr = optim.TUNE_LR
    else:
        lr = optim.BASE_LR
    if optim.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(
            param_list,
            lr=lr,
            momentum=optim.MOMENTUM,
            weight_decay=wd,
            dampening=optim.DAMPENING,
            nesterov=optim.NESTEROV,
        )
    elif optim.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            param_list,
            lr=lr,
            betas=(optim.BETA1, optim.BETA2),
            weight_decay=wd,
        )
    elif optim.OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(
            param_list,
            lr=lr,
            betas=(optim.BETA1, optim.BETA2),
            weight_decay=wd,
        )
    else:
        raise NotImplementedError
    return optimizer


def lr_fun_steps(cur_epoch):
    """Steps schedule (cfg.OPTIM.LR_POLICY = 'steps')."""
    ind = [i for i, s in enumerate(cfg.OPTIM.STEPS) if cur_epoch >= s][-1]
    return cfg.OPTIM.LR_MULT ** ind


def lr_fun_exp(cur_epoch):
    """Exponential schedule (cfg.OPTIM.LR_POLICY = 'exp')."""
    return cfg.OPTIM.MIN_LR ** (cur_epoch / cfg.OPTIM.MAX_EPOCH)


def lr_fun_cos(cur_epoch):
    """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
    lr = 0.5 * (1.0 + np.cos(np.pi * cur_epoch / cfg.OPTIM.MAX_EPOCH))
    return (1.0 - cfg.OPTIM.MIN_LR) * lr + cfg.OPTIM.MIN_LR


def lr_fun_lin(cur_epoch):
    """Linear schedule (cfg.OPTIM.LR_POLICY = 'lin')."""
    lr = 1.0 - cur_epoch / cfg.OPTIM.MAX_EPOCH
    return (1.0 - cfg.OPTIM.MIN_LR) * lr + cfg.OPTIM.MIN_LR


def get_lr_fun():
    """Retrieves the specified lr policy function"""
    lr_fun = "lr_fun_" + cfg.OPTIM.LR_POLICY
    assert lr_fun in globals(), "Unknown LR policy: " + cfg.OPTIM.LR_POLICY
    err_str = "exp lr policy requires OPTIM.MIN_LR to be greater than 0."
    assert cfg.OPTIM.LR_POLICY != "exp" or cfg.OPTIM.MIN_LR > 0, err_str
    return globals()[lr_fun]


def get_epoch_lr(cur_epoch, base_lr):
    """Retrieves the lr for the given epoch according to the policy."""
    # Get lr and scale by by BASE_LR
    lr = get_lr_fun()(cur_epoch) * base_lr
    # Linear warmup
    if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS:
        alpha = cur_epoch / cfg.OPTIM.WARMUP_EPOCHS
        warmup_factor = cfg.OPTIM.WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr *= warmup_factor
    return lr


def set_lr(optimizer, new_lr):
    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def plot_lr_fun():
    """Visualizes lr function."""
    epochs = list(range(cfg.OPTIM.MAX_EPOCH))
    lrs = [get_epoch_lr(epoch) for epoch in epochs]
    plt.plot(epochs, lrs, ".-")
    plt.title("lr_policy: {}".format(cfg.OPTIM.LR_POLICY))
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    plt.ylim(bottom=0)
    plt.show()