"""Image transformations on HWC float images with RGB channel order."""
from math import ceil, sqrt
import numpy as np
from PIL import Image
import cv2
from data.augment import make_augment

import torch
import torchvision.transforms as transforms


def resize(img, size, interpolation=Image.BILINEAR, max_size=None):
    """Resize the image according to a given size.

    :param img: A image in PIL format.
    :param size: The target resolution.
    :param interpolation: Interpolation method.
    :param max_size: The maximum resolution.
    """
    if not Image.isImageType(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    if not isinstance(size, int):
        raise TypeError("size should be Int. Got {}".format(type(img)))
    else:
        w, h = img.size

        short, long = (w, h) if w < h else (h, w)
        if short == size:
            return img

        new_short, new_long = size, int(size * long / short)

        if max_size is not None:
            if max_size <= size:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if w < h else (new_long, new_short)
        return img.resize((new_w, new_h), resample=interpolation)


def scale_and_center_crop(img, scale_size, crop_size, interpolation=Image.BILINEAR):
    """Performs scaling and center cropping (used for testing)."""
    if not Image.isImageType(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    w, h = img.size

    if w < h and w != scale_size:
        w, h = scale_size, int(h / w * scale_size)
    elif h <= w and h != scale_size:
        w, h = int(w / h * scale_size), scale_size

    img = img.resize((w, h), resample=interpolation)
    x = ceil((w - crop_size) / 2)
    y = ceil((h - crop_size) / 2)
    return img.crop((x, y, x + crop_size, y + crop_size))


def padding_resize(img, size, interpolation=Image.BILINEAR):
    """Resize the image by resizing the long side to the given size."""
    if not Image.isImageType(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    bg = Image.new(img.mode, (size, size), (0, 0, 0))
    w, h = img.size

    if w < h:
        w, h = int(w / h * size), size
        img = img.resize((w, h), resample=interpolation)
        bg.paste(img, ((h - w) // 2, 0))
    elif h < w:
        w, h = size, int(h / w * size)
        img = img.resize((w, h), resample=interpolation)
        bg.paste(img, (0, (w - h) // 2))
    else:
        # w = h
        img = img.resize((size, size), resample=interpolation)
        bg.paste(img, (0, 0))

    return bg


def random_resized_crop(img, crop_size, area_frac=0.08, max_iter=10, interpolation=Image.BILINEAR):
    """Performs Inception-style cropping (used for training)."""
    if not Image.isImageType(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    w, h = img.size
    area = w * h

    for _ in range(max_iter):
        target_area = np.random.uniform(area_frac, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w_crop = round(sqrt(target_area * aspect_ratio))
        h_crop = round(sqrt(target_area / aspect_ratio))

        if np.random.uniform() < 0.5:
            w_crop, h_crop = h_crop, w_crop

        if h_crop <= h and w_crop <= w:
            y = 0 if h_crop == h else np.random.randint(0, h - h_crop)
            x = 0 if w_crop == w else np.random.randint(0, w - w_crop)
            img = img.crop((x, y, x + w_crop, y + h_crop))
            return img.resize((crop_size, crop_size), resample=interpolation)

    return scale_and_center_crop(img, crop_size, crop_size)


def horizontal_flip(img, prob=0.5):
    """Performs horizontal flip (used for training). Applicable for PIL images and tensors."""
    if Image.isImageType(img):
        return img.transpose(Image.FLIP_LEFT_RIGHT) if np.random.uniform() < prob else img
    elif torch.is_tensor(img):
        tensor_flip = transforms.RandomHorizontalFlip(prob)
        return tensor_flip(img)
    else:
        raise TypeError("img should be PIL Image or torch.Tensor. Got {}".format(type(img)))


def augment(img, augment_str):
    """Augments image (used for training)."""
    if not Image.isImageType(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    if augment_str:
        img = make_augment(augment_str)(img)
    return img


def lighting(img, alpha_std, eig_val, eig_vec):
    """Performs AlexNet-style PCA jitter (used for training)."""
    if Image.isImageType(img):
        img = np.asarray(img).astype(np.float32) / 255

    alpha = np.random.normal(0, alpha_std, size=(1, 3))
    alpha = np.repeat(alpha, 3, axis=0)
    eig_val = np.repeat(np.array(eig_val), 3, axis=0)
    rgb = np.sum(np.array(eig_vec) * alpha * eig_val, axis=1)

    for i in range(3):
        img[:, :, i] = img[:, :, i] + rgb[i]

    return img


def color_norm(img, mean, std):
    """Performs per-channel normalization (used for training and testing)."""
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
    return img


def crop_postprocess(cropped_tensor):
    return horizontal_flip(cropped_tensor)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


def get_random_crop_anchor(img, input_h, input_w, flip):
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
    w_border = get_border(56, img.shape[1])
    h_border = get_border(56, img.shape[0])
    c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
    c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

    trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    return inp, trans_input


def box_affine(box, trans_input, input_size, output_size, flip):
    width, height = input_size
    output_w, output_h = output_size
    if flip:
        box[[0, 2]] = width - box[[2, 0]] - 1

    box[:2] = affine_transform(box[:2], trans_input)
    box[2:] = affine_transform(box[2:], trans_input)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, output_w - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, output_h - 1)

    return box


# def crop_class_imgs(root, class_id, class_cams):
#     imgs, img_names = read_class_imgs(root, class_id, scale=True, crop=True)
#     bboxes = get_bboxes(class_cams)
#     cropped_imgs = crop_imgs(imgs, bboxes)
#
#     return cropped_imgs, img_names


# def composite_all_imgs(root, cropped_root, alpha, cam_model_name):
#     saveroot = '/home/hao/dataFolder/DATA/datasets/composite/' + cam_model_name + '/val/'
#
#     norm_class_names = os.listdir(os.path.join(root, 'val'))
#     cropped_class_names = os.listdir(os.path.join(cropped_root, 'val'))
#     norm_class_names.sort()
#     cropped_class_names.sort()
#     assert norm_class_names == cropped_class_names, "Mismatch of the normal dataset and the cropped dataset."
#
#     for class_id in range(len(norm_class_names)):
#         norm_imgs, img_names = read_class_imgs(root, class_id, scale=True, crop=True)
#         cropped_imgs, cropped_names = read_class_imgs(cropped_root, class_id, scale=True, crop=False)
#         assert img_names == cropped_names, "Mismatch of image names in the normal dataset and the cropped dataset."
#         comp_imgs = composite_imgs(cropped_imgs, norm_imgs, alpha)
#
#         class_dir = os.path.join(saveroot, norm_class_names[class_id])
#         save_imgs(comp_imgs, img_names, class_dir)
#
#         sys.stdout.write('Compositing: %d/%d' % (class_id + 1, len(norm_class_names)))
#         sys.stdout.flush()
#         sys.stdout.write('\r')


# def crop_all_imgs(root, cams):
#     saveroot = '/home/hao/dataFolder/DATA/datasets/crop/' + cam_model_name + '/val/'
#     class_names = os.listdir(os.path.join(root, 'val'))
#     assert len(class_names) == 1000, "Incomplete dataset. Got {} classes.".format(len(class_names))
#     class_names.sort()
#
#     for class_id in range(1000):
#         class_dir = os.path.join(saveroot, class_names[class_id])
#         if not os.path.isdir(class_dir):
#             os.makedirs(class_dir)
#
#         cropped_class_imgs, img_names = crop_class_imgs(root, class_id, cams[class_id])
#         save_imgs(cropped_class_imgs, img_names, class_dir)
#
#         sys.stdout.write('Cropping: %d/%d' % (class_id + 1, 1000))
#         sys.stdout.flush()
#         sys.stdout.write('\r')
