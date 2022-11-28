import os

import torch.nn as nn

import time
import argparse
import xml.etree.ElementTree as et


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v: The original number
    :param divisor: It can be 4 or 8 for higher execution efficiency.
    :param min_value:
    :return: The new number that is divisible by the divisor.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def str2bool(v):
    # from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class AverageMeter(object):
    """From https://github.com/NVlabs/Taylor_pruning/blob/master/utils/utils.py
    Computes and stores the average and current value
    """
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


# def init_weights(modules):
#     """Initialize the weight of newly constructed networks.
#
#     :param modules: The top module of a network.
#     :return: None
#     """
#     for m in modules:
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1.0)
#             nn.init.constant_(m.bias, 0.0)
#         elif isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, 0.0, 0.01)
#             nn.init.constant_(m.bias, 0.0)


def transform_imagenet_bboxes(root, bbox_dir, split='val'):
    """Transform the ImageNet-style box annotation (.xml) into a class-wise array."""
    split_dir = os.path.join(root, split)
    class_names = os.listdir(split_dir)
    class_names.sort()

    class_img_box = {}
    box_fnames = os.listdir(bbox_dir)
    box_fpaths = [os.path.join(bbox_dir, name) for name in box_fnames]
    for box_path in box_fpaths:
        tree = et.parse(box_path)
        root = tree.getroot()

    # TODO: Finish this
