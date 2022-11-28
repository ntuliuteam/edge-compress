"""Functions for manipulating networks."""

import torch
import torch.nn.functional as F

import itertools
import numpy as np
from configs.config import cfg


class SoftCrossEntropyLoss(torch.nn.Module):
    """SoftCrossEntropyLoss (useful for label smoothing and mixup).
    Identical to torch.nn.CrossEntropyLoss if used with one-hot labels."""

    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        loss = -y * torch.nn.functional.log_softmax(x, -1)
        return torch.sum(loss) / x.shape[0]


class WeightedMSELoss(torch.nn.Module):
    """Weight the loss of each edge for better box regression."""

    def __init__(self, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        reductions = ['none', 'mean', 'sum']
        err_message = f"Reduction ({reduction}) is not supported for WeightedMSELoss"
        assert reduction in reductions, err_message
        self.reduction = reduction

    @staticmethod
    def check_compatability(weights, preds, targets):
        # Predictions and targets should be in the form of (N, E),
        # where N is the batch size and E is the number of elements in a single sample.
        # weights should be a 1-D tensor with a size of E. The calculation of weighted mse loss for one sample
        # can be represented as: loss = self.reduction(weight[j] * x[i][j] * y[i][j])
        err_message = "The shape of predictions ({}) should be the same as the targets ({})".format(preds.shape,
                                                                                                    targets.shape)
        assert preds.shape == targets.shape, err_message
        err_message = "The size of the weight array should be {}. Got {}.".format(preds.size(-1), weights.shape)
        assert weights.dim() == 1 and weights.size(-1) == preds.size(-1), err_message

    def forward(self, preds, targets, weights=None):
        if weights is None:
            weights = torch.tensor([1, 1, 1.25, 1.25]).cuda()
        self.check_compatability(weights, preds, targets)
        mse_matrix = F.mse_loss(preds, targets, reduction='none')
        weighted_mse_matrix = mse_matrix * weights

        if self.reduction == 'mean':
            return torch.mean(weighted_mse_matrix)
        elif self.reduction == 'sum':
            return torch.sum(weighted_mse_matrix)
        else:
            return weighted_mse_matrix


def sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _neg_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(torch.nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def smooth_one_hot_labels(labels):
    """Convert each label to a smoothed one-hot vector.

    :param labels: The original labels.
    """
    n_classes, label_smooth = cfg.MODEL.NUM_CLASSES, cfg.TRAIN.LABEL_SMOOTHING
    err_str = "Invalid input to one_hot_vector()"
    assert labels.ndim == 1 and labels.max() < n_classes, err_str

    shape = (labels.shape[0], n_classes)
    neg_val = label_smooth / n_classes
    pos_val = 1.0 - label_smooth + neg_val
    labels_one_hot = torch.full(shape, neg_val, dtype=torch.float, device=labels.device)
    labels_one_hot.scatter_(1, labels.long().view(-1, 1), pos_val)

    return labels_one_hot


def one_hot_labels(labels):
    """Convert each label to a one-hot vector

    :param labels: The original labels.
    """
    shape = (labels.shape[0], cfg.MODEL.NUM_CLASSES)
    labels_one_hot = torch.full(shape, 0, dtype=torch.float, device=labels.device)
    labels_one_hot.scatter_(1, labels.long().view(-1, 1), 1)

    return labels_one_hot


def mixup(inputs, labels):
    """Apply mixup to minibatch (https://arxiv.org/abs/1710.09412).

    :param inputs: The inputs of network.
    :param labels: One-hot labels.
    """
    alpha = cfg.TRAIN.MIXUP_ALPHA
    assert labels.shape[1] == cfg.MODEL.NUM_CLASSES, "mixup labels must be one-hot."

    if alpha > 0:
        m = np.random.beta(alpha, alpha)
        permutation = torch.randperm(labels.shape[0])
        inputs = m * inputs + (1.0 - m) * inputs[permutation, :]
        labels = m * labels + (1.0 - m) * labels[permutation, :]

    return inputs, labels, labels.argmax(1)


def crop_mixup(inputs, cropped_inputs):
    """Apply mixup to the original images and the cropped images.

    :param inputs: The original images.
    :param cropped_inputs: The cropped images.
    """
    alpha = cfg.TRAIN.MIXUP_ALPHA
    return inputs * alpha + cropped_inputs * (1 - alpha)


def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present.

    :param model: The model to unwrap.
    """
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


def update_model_ema(model, model_ema, cur_epoch, cur_iter):
    """Update exponential moving average (ema) of model weights.

    :param model: The main model.
    :param model_ema: The model copy.
    :param cur_epoch: Current training epoch.
    :param cur_iter: Current training batch.
    """
    update_period = cfg.OPTIM.EMA_UPDATE_PERIOD
    if update_period == 0 or cur_iter % update_period != 0:
        return

    # Adjust alpha to be fairly independent of other parameters
    adjust = cfg.TRAIN.BATCH_SIZE / cfg.OPTIM.MAX_EPOCH * update_period
    alpha = min(1.0, cfg.OPTIM.EMA_ALPHA * adjust)
    # During warmup simply copy over weights instead of using ema
    alpha = 1.0 if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS else alpha
    # Take ema of all parameters (not just named parameters)
    params = unwrap_model(model).state_dict()
    for name, param in unwrap_model(model_ema).state_dict().items():
        param.copy_(param * (1.0 - alpha) + params[name] * alpha)
