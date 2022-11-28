import torch

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from data import transforms


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def show_imgs(imgs, save=False):
    nrow, ncol = int(np.sqrt(len(imgs))), int(np.sqrt(len(imgs)))
    fig, axs = plt.subplots(nrow, ncol, figsize=(28, 28))
    for row_id in range(nrow):
        for col_id in range(ncol):
            img_id = int(ncol * row_id + col_id)
            axs[row_id][col_id].imshow(imgs[img_id])

    if save:
        save_dir = './figures/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'show_imgs.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')


def read_class_imgs(root, class_id, scale_size=0, crop_size=0):
    root = os.path.join(root, 'val')
    class_names = os.listdir(root)
    class_names.sort()
    class_dir = os.path.join(root, class_names[class_id])
    img_names = os.listdir(class_dir)
    img_names.sort()
    img_paths = [os.path.join(class_dir, img_path) for img_path in img_names]
    imgs = [Image.open(img) for img in img_paths]
    imgs = [img.convert('RGB') for img in imgs]

    if scale_size != 0:
        if crop_size != 0:
            imgs = [transforms.scale_and_center_crop(img, scale_size, crop_size) for img in imgs]
        else:
            imgs = [transforms.scale_and_center_crop(img, crop_size, crop_size) for img in imgs]

    return imgs, img_names


def show_class_imgs(root, class_id, nimgs=16):
    imgs, _ = read_class_imgs(root, class_id)
    show_imgs(imgs[:nimgs])


def pil_preprocess(imgs, mean, std):
    inputs = []
    for img in imgs:
        img = np.asarray(img).astype(np.float32) / 255
        img = transforms.color_norm(img, mean, std)
        img = np.ascontiguousarray(img.transpose([2, 0, 1]))
        inputs.append(img)
    inputs = torch.tensor(inputs)

    return inputs


def show_box_on_imgs(imgs, bboxes):
    assert len(imgs) == len(
        bboxes), "The number of images and boxes should be equal. Got {} and {}, respectively".format(len(imgs),
                                                                                                      len(bboxes))

    nrow, ncol = int(np.sqrt(len(imgs))), int(np.sqrt(len(imgs)))
    fig, axs = plt.subplots(nrow, ncol, figsize=(28, 28))
    for idx, (img, bbox) in enumerate(zip(imgs, bboxes)):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        row_id = idx // ncol
        col_id = idx % ncol
        ax = axs[row_id][col_id]
        ax.imshow(img)
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def composite_imgs(fg_imgs, bg_imgs, alpha):
    assert len(fg_imgs) == len(
        bg_imgs), "Mismatch of the number of foreground images and background images. Got {} and {}, respectively.".format(
        len(fg_imgs), len(bg_imgs))
    comp_imgs = []

    for idx, (fg, bg) in enumerate(zip(fg_imgs, bg_imgs)):
        comp = Image.blend(fg, bg, alpha)
        comp_imgs.append(comp)

    return comp_imgs


def save_imgs(imgs, img_names, savedir):
    assert len(imgs) == len(img_names), "Mismatch of the number of images and image names. " \
                                         "Got {} and {}, respectively.".format(len(imgs), len(img_names))
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    for idx, (img, name) in enumerate(zip(imgs, img_names)):
        savepath = os.path.join(savedir, name)
        img.save(savepath, quality=100, subsampling=0)


def get_interpolation(name):
    interpolations = {
        'bilinear': Image.BILINEAR
    }
    err_message = f"Interpolation {name} not supported."
    assert name in interpolations.keys()

    return interpolations[name]


# def show_class_cams(root, class_id, acts_blobs, grad_blobs, nimgs=16):
#     acts = np.asarray(acts_blobs[class_id])
#     grads = np.asarray(grad_blobs[class_id])
#     weights = np.mean(grads, axis=(2, 3), keepdims=True)
#
#     imgs, _ = read_class_imgs(root, class_id)
#
#     grascale_cams = get_cams(acts, weights, train_size)
#     agat_imgs = []
#
#     for idx, (rgb_img, mask) in enumerate(zip(imgs, grascale_cams)):
#         rgb_img = np.float32(rgb_img) / 255
#         agat_imgs.append(show_cam_on_image(rgb_img, mask, use_rgb=True))
#
#     show_imgs(agat_imgs[:nimgs])
