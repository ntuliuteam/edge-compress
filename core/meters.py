"""Meters."""

from collections import deque
import torchvision.ops as ops

import numpy as np
import time
import core.logging as logging
import torch
from configs.config import cfg


logger = logging.get_logger(__name__)


def topk_accuracy(output, targets, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k.
    """
    err_str = "Batch dim of predictions and labels must match"
    assert output.size(0) == targets.size(0), err_str

    with torch.no_grad():
        max_k = max(topk)
        batch_size = output.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def topk_confidence(output, targets, topk=1, use_diff=True):
    """Computes the confidence score of the k top predictions for the specified values of k.
    """
    err_str = "Batch dim of predictions and labels must match"
    assert output.size(0) == targets.size(0), err_str

    preds = torch.softmax(output, dim=-1)
    if use_diff:
        preds = torch.topk(preds, 2, dim=-1).values
        scores = preds[:, 0] - preds[:, 1]
    else:
        scores = torch.topk(preds, topk, dim=-1).values.sum(dim=-1)
    idxs = torch.max(output, dim=-1).indices
    correct = idxs.eq(targets)

    return scores.cpu().numpy(), correct.cpu().numpy()


def loc_accuracy(output, targets):
    pass
    # """Compute the top1 localization accuracy."""
    # err_str = "Batch dim of predictions and labels must match"
    # assert output.size(0) == output.size(0), err_str
    #
    # threshold = 0.5
    # ious = torch.diag(ops.box_iou(output, targets))
    # correct = ious > threshold
    # loc_acc = torch.div(torch.sum(correct), output.size(0))
    #
    # return loc_acc


def intersect(box_a, box_b):
    """ We compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding box, Shape: [4].
      box_b: (tensor) bounding box, Shape: [4].
    Return:
      (tensor) intersection area
    """
    width = torch.min(box_a[2], box_b[2]) - torch.max(box_a[0], box_b[0])
    height = torch.min(box_a[3], box_b[3]) - torch.max(box_a[1], box_b[1])
    inter = torch.clamp(torch.tensor([width, height]), min=0)

    return inter[0] * inter[1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding box, Shape: [4]
        box_b: (tensor) Predicted bounding box, Shape: [4]
    Return:
        jaccard overlap: (tensor)
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union


def combine_boxes(boxset_a, boxset_b):
    """Combine the two predicted boxes into the final box.

    :param boxset_a: The boxes outputted by one of the predicted heads.
    :param boxset_b: The boxes outputted by the other predicted head.
    """
    err_str = "The number of boxes should be equal. Got {} and {}".format(boxset_a.size(0), boxset_b.size(0))
    assert boxset_a.size(0) == boxset_b.size(0), err_str
    new_boxes = torch.empty(boxset_a.size()).cuda()

    for box_idx, (box_a, box_b) in enumerate(zip(boxset_a, boxset_b)):
        new_boxes[box_idx][0] = min(box_a[0], box_b[0])
        new_boxes[box_idx][1] = min(box_a[1], box_b[1])
        new_boxes[box_idx][2] = max(box_a[2], box_b[2])
        new_boxes[box_idx][3] = max(box_a[3], box_b[3])

    return new_boxes


def get_miou(preds, boxes):
    """Compute the average IOU of of a batch.
    """
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == boxes.size(0), err_str

    ious = ops.box_iou(preds, boxes)

    return torch.mean(torch.diag(ious))
    # pass


def get_point_dists(preds, points):
    """Compute the average distance from the predicted points to the true points
    """
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == points.size(0), err_str

    preds = preds.mul_(cfg.TRAIN.IM_SIZE)
    points = points.mul_(cfg.TRAIN.IM_SIZE)

    deltas = preds - points
    edge_sum = torch.sum(deltas.pow_(2), dim=-1)
    dists = edge_sum.sqrt_()

    return torch.mean(dists)


def hp_topk_accuracy(preds, gt, topk=(1, 5)):
    """Compute the accuracy of center prediction.

    :param preds: The predicted heatmap in the size of (N x C x H x W).
    :param gt: The ground truth heatmap in the size of (N x C x H x W).
    :param topk: Topk accuracy is truth when the predicted center is
    within k pixels of the ground truth center.
    """
    err_str = "The shape of predictions and ground truth must be equal. Got {} and {}".format(preds.shape, gt.shape)
    assert preds.shape == gt.shape, err_str
    # For one channel output
    # new_preds, new_gt = torch.squeeze(preds), torch.squeeze(gt)

    batch_size = preds.size(0)
    # For multiple channel output
    h, w = preds.size(-2), preds.size(-1)

    # Sum up the ground truth along the channel dimension.
    # The shape of new_gt: (N x H x W).
    new_gt = torch.sum(gt, dim=1)

    # The shape of new_preds: (N x H x W).
    new_preds = torch.empty(batch_size, h, w).cuda()
    # Choose the most salient channel for each sample
    channel_inds = torch.argmax(preds.sum(-1).sum(-1), dim=-1)
    for idx, sample in enumerate(preds):
        new_preds[idx] = sample[channel_inds[idx]]

    # The index of the center point
    preds_c, gt_c = torch.empty(batch_size, 2).cuda(), torch.empty(batch_size, 2).cuda()

    preds_col_vals, preds_col_inds = torch.max(new_preds, dim=-1)
    preds_row_vals, preds_row_inds = torch.max(preds_col_vals, dim=-1)
    gt_col_vals, gt_col_inds = torch.max(new_gt, dim=-1)
    gt_row_vals, gt_row_inds = torch.max(gt_col_vals, dim=-1)

    for idx in range(batch_size):
        pred_row_ind = preds_row_inds[idx]
        pred_col_ind = preds_col_inds[idx][pred_row_ind]
        preds_c[idx][0], preds_c[idx][1] = pred_row_ind, pred_col_ind
        gt_row_ind = gt_row_inds[idx]
        gt_col_ind = gt_col_inds[idx][gt_row_ind]
        gt_c[idx][0], gt_c[idx][1] = gt_row_ind, gt_col_ind

    center_diff = (preds_c - gt_c).abs_()
    res = []
    for k in topk:
        k_cor = torch.lt(center_diff, k).float().prod(dim=-1).sum()
        res.append(k_cor.mul_(100.0 / batch_size))

    # gt_sim = torch.cosine_similarity(gt, gt, dim=-1).sum()
    # tensor_sim = torch.cosine_similarity(preds, gt, dim=-1).sum()
    # print(center_diff)

    return res


def get_acc_meter():
    acc_meters = {
        'backbone': topk_accuracy,
        'cropper': get_miou,
        'hp': hp_topk_accuracy,
    }
    model_type = cfg.MODEL.TYPE
    err_message = f"Accuracy meter for {model_type} not supported."
    assert model_type in acc_meters.keys(), err_message

    return acc_meters[model_type]


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class Timer(object):
    """A simple timer (adapted from Detectron)."""

    def __init__(self):
        self.total_time = None
        self.calls = None
        self.start_time = None
        self.diff = None
        self.average_time = None
        self.reset()

    def tic(self):
        # using time.time as time.clock does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0


class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, epoch_iters, phase="train"):
        self.epoch_iters = epoch_iters
        self.max_iter = cfg.OPTIM.MAX_EPOCH * epoch_iters
        self.phase = phase
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch accuracy (smoothed over a window)
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Number of correctly classified examples
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_samples = 0

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_acc, top5_acc, loss, lr, mb_size):
        # Current minibatch stats
        self.mb_top1_acc.add_value(top1_acc)
        self.mb_top5_acc.add_value(top5_acc)
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.num_top1_cor += top1_acc * mb_size
        self.num_top5_cor += top5_acc * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = cur_epoch * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "eta": readable_time(eta_sec),
            "top1_acc": self.mb_top1_acc.get_win_avg(),
            "top5_acc": self.mb_top5_acc.get_win_avg(),
            "loss": self.loss.get_win_avg(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD == 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            logger.info(logging.dump_log_data(stats, self.phase + "_iter"))

    def get_epoch_stats(self, cur_epoch):
        cur_iter_total = (cur_epoch + 1) * self.epoch_iters
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "time_epoch": self.iter_timer.average_time * self.epoch_iters,
            "eta": readable_time(eta_sec),
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logger.info(logging.dump_log_data(stats, self.phase + "_epoch"))


class TestMeter(object):
    """Measures testing stats."""

    def __init__(self, epoch_iters, phase="test"):
        self.epoch_iters = epoch_iters
        self.phase = phase
        self.iter_timer = Timer()
        # Current minibatch accuracy (smoothed over a window)
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Max accuracy (over the full test set)
        self.max_top1_acc = 0.0
        self.max_top5_acc = 0.0
        # Number of correctly classified examples
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_samples = 0

    def reset(self, max_acc=False):
        if max_acc:
            self.max_top1_acc = 0.0
            self.max_top5_acc = 0.0
        self.iter_timer.reset()
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_acc, top5_acc, mb_size):
        self.mb_top1_acc.add_value(top1_acc)
        self.mb_top5_acc.add_value(top5_acc)
        self.num_top1_cor += top1_acc * mb_size
        self.num_top5_cor += top5_acc * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = gpu_mem_usage()
        iter_stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "top1_acc": self.mb_top1_acc.get_win_avg(),
            "top5_acc": self.mb_top5_acc.get_win_avg(),
            "mem": int(np.ceil(mem_usage)),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD == 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            logger.info(logging.dump_log_data(stats, self.phase + "_iter"))

    def get_epoch_stats(self, cur_epoch):
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        self.max_top1_acc = max(self.max_top1_acc, top1_acc)
        self.max_top5_acc = max(self.max_top5_acc, top5_acc)
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "time_epoch": self.iter_timer.average_time * self.epoch_iters,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "max_top1_acc": self.max_top1_acc,
            "max_top5_acc": self.max_top5_acc,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logger.info(logging.dump_log_data(stats, self.phase + "_epoch"))


class CropMeter(object):
    """Measures cropping stats."""

    def __init__(self, epoch_iters, phase="crop"):
        self.epoch_iters = epoch_iters
        self.phase = phase
        self.iter_timer = Timer()
        # Current minibatch accuracy (smoothed over a window)
        self.num_samples = 0

    def reset(self):
        self.iter_timer.reset()
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, mb_size):
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = gpu_mem_usage()
        iter_stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "mem": int(np.ceil(mem_usage)),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD == 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            logger.info(logging.dump_log_data(stats, self.phase + "_iter"))

    def get_epoch_stats(self, cur_epoch):
        mem_usage = gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "time_epoch": self.iter_timer.average_time * self.epoch_iters,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logger.info(logging.dump_log_data(stats, self.phase + "_epoch"))


def readable_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    mins = seconds % 3600 // 60
    secs = seconds % 60

    return '{}H-{}M-{}S'.format(hours, mins, secs)
