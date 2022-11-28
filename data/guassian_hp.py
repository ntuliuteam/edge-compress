import os
from argparse import ArgumentParser
import numpy as np
import math

from configs.config import cfg
from core.utils import make_divisible


def build_args():
    parser = ArgumentParser()
    parser.add_argument('-s', '--shape', type=str,
                        help="The shape of boxes.")
    parser.add_argument('-sp', '--split', type=str,
                        help="The dataset split.")
    parser.add_argument('-r', '--root', type=str,
                        help="The directory which stores the dataset.")

    return parser.parse_args()


def get_center(box):
    cx = int((box[0] + box[2]) / 2)
    cy = int((box[1] + box[3]) / 2)
    center = [cx, cy]

    return center


def get_size(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    size = [height, width]

    return size


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def get_gaussian_hp(box):

    orig_size = make_divisible(cfg.TRAIN.IM_SIZE * cfg.SCALING.RES_MULT_A, cfg.SCALING.ROUND)
    hp_size = math.ceil(orig_size / cfg.SCALING.DOWN_RATIO)
    hp = np.zeros((hp_size, hp_size))

    if box[3] <= box[1] or box[2] <= box[0]:
        return hp

    box = [math.ceil(ss / cfg.SCALING.DOWN_RATIO) for ss in box]
    center = get_center(box)
    box_size = get_size(box)
    radius = max(0, int(gaussian_radius(box_size)))
    draw_umich_gaussian(hp, center, radius)

    return hp


def get_gaussian_hps(boxes):
    hps = []

    assert len(boxes) != 0, "The size of boxes cannot be 0."
    for class_idx, class_boxes in enumerate(boxes):
        print("Processing class: {}/{}".format(class_idx + 1, len(boxes)))
        class_hps = []
        for img_idx, img_box in enumerate(class_boxes):
            hp = get_gaussian_hp(img_box)
            class_hps.append(hp)
        hps.append(class_hps)

    return hps


def main():
    args = build_args()
    box_path = os.path.join(args.root, args.shape + '_' + args.split + '.npy')
    boxes = np.load(box_path, allow_pickle=True)
    gaussian_hps = get_gaussian_hps(boxes)

    save_path = os.path.join(args.root, 'gaussian_' + args.shape + '_' + args.split + '.npy')
    np.save(save_path, gaussian_hps)


if __name__ == '__main__':
    main()
