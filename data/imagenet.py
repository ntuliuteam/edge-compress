"""Preprocessing for ImageNet"""
import numpy
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

import os
import re
import numpy as np
from PIL import Image
import cv2
import math

import data.transforms as transforms
from core.utils import make_divisible
from configs.config import cfg
from core.image import get_interpolation
from data.guassian_hp import get_gaussian_hp

# Per-channel mean and standard deviation values on ImageNet (in RGB order)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Constants for lighting normalization on ImageNet (in RGB order)
_EIG_VALS = [[0.2175, 0.0188, 0.0045]]
_EIG_VECS = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]
_PCA_STD = 0.1
# Make the resolution divisible by 4 for higher execution efficiency.
ROUND = 4


class ImageNet(Dataset):
    """ImageNet (224x224)"""

    def __init__(self, root, split, cropped=False):
        assert os.path.exists(root), "Data path '{}' not found.".format(root)
        splits = ['train', 'val']
        assert split in splits, "Split '{}' not supported for ImageNet.".format(split)

        self._root = root
        self._split = split
        self._augment = cfg.TRAIN.AUGMENT
        self._res_mult = cfg.SCALING.RES_MULT_A
        self._cropped = cropped
        self._flip_ratio = 0.5
        self._train_crop = 'random'
        self._interpolation = get_interpolation(cfg.SCALING.INTERPOLATE)
        self.train_size = make_divisible(224 * self._res_mult, ROUND)
        self.test_size = make_divisible(256 * self._res_mult, ROUND)
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the image database."""
        # Compile the split data path
        split_path = os.path.join(self._root, self._split)
        # Images are stored per class in subdirs (format: n<number>)
        split_folders = os.listdir(split_path)
        self._class_ids = sorted(f for f in split_folders if re.match(r"^n[0-9]+$", str(f)))
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image database
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)

            im_names = os.listdir(im_dir)
            im_names.sort()
            for im_name in im_names:
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({'im_path': im_path, 'class': cont_id})

        # print("|| Number of {} images: {}".format(self._split, len(self._imdb)))
        # print("|| Number of {} classes: {}".format(self._split, len(self._class_ids)))

    def _preprocess_img(self, img):
        """Preprocess the image for network input.

        :param img: A image in PIL format.
        """
        if self._split == 'train':
            # For training use random_sized_crop, horizontal_flip, augment, lighting
            if self._train_crop == 'random':
                img = transforms.random_resized_crop(img, self.train_size, interpolation=self._interpolation)
            else:
                img = transforms.scale_and_center_crop(img, self.train_size, self.train_size)
            img = transforms.horizontal_flip(img, prob=self._flip_ratio)
            img = transforms.augment(img, self._augment)
            img = transforms.lighting(img, _PCA_STD, _EIG_VALS, _EIG_VECS)
        else:
            # For testing use scale and crop
            if self._cropped:
                # Only scale for cropped images
                img = transforms.scale_and_center_crop(img, self.train_size, self.train_size)
            else:
                # For normal images
                img = transforms.scale_and_center_crop(img, self.test_size, self.train_size)

        # transforms.lighting returns a np.array, while scale_and_center_crop returns a PIL Image
        if Image.isImageType(img):
            img = np.asarray(img).astype(np.float32) / 255
        # For training and testing use color normalization
        img = transforms.color_norm(img, _MEAN, _STD)
        # Convert HWC/RGB/float to CHW/RGB/float format
        img = img.transpose([2, 0, 1])

        return img

    def __getitem__(self, index):
        # Load PIL image
        img = Image.open(self._imdb[index]['im_path'])
        img = img.convert('RGB')
        # Preprocess the image for training and testing
        img = self._preprocess_img(img)
        # Retrieve the label
        label = self._imdb[index]['class']

        return img, label

    def __len__(self):
        return len(self._imdb)


class CroppedImageNet(ImageNet):
    def __init__(self, root, split):
        super(CroppedImageNet, self).__init__(root, split, cropped=True)

    def _preprocess_img(self, img):
        """Preprocess the image for network input.

        :param img: A image in PIL format.
        """
        if self._split == 'train':
            err_message = "Cropped training dataset not implemented yet."
            raise NotImplementedError(err_message)
        else:
            """For cropped images, we first transform PIL images into tensors and use 
            torchvision.transforms.functional.resize to resize them for higher efficiency 
            and fair comparison.
            """
            img = np.asarray(img).astype(np.float32) / 255
            # For training and testing use color normalization
            img = transforms.color_norm(img, _MEAN, _STD)
            # Convert HWC/RGB/float to CHW/RGB/float format
            img = img.transpose([2, 0, 1])
            img = torch.tensor(img)
            img = TF.resize(img, size=self.test_size)

        return img


class ImageNetForCrop(ImageNet):
    def __init__(self, root, split):
        super(ImageNetForCrop, self).__init__(root, split, cropped=True)
        self._train_crop = 'center'
        self._flip_ratio = 0.0


class ImageNetBox(ImageNetForCrop):
    def __init__(self, root, split):
        self.box_root = './tmp/box_annotation'
        self.box_name = cfg.CROP.TYPE + '_' + cfg.CROP.SHAPE + '_' + split + '.npy'
        super(ImageNetBox, self).__init__(root, split)

    def normalize_box(self, box):
        return [float(e / self.train_size) for e in box]

    def _construct_imdb(self):
        """Constructs the image database."""
        # Compile the split data path
        split_path = os.path.join(self._root, self._split)
        boxes_path = os.path.join(self.box_root, cfg.CROP.SHAPE + '_' + self._split + '.npy')
        # Images are stored per class in subdirs (format: n<number>)
        split_folders = os.listdir(split_path)
        self._class_ids = sorted(f for f in split_folders if re.match(r"^n[0-9]+$", str(f)))
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Load bounding boxes as the label
        self.boxes = np.load(boxes_path, allow_pickle=True)

        # Construct the image database
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)

            im_names = os.listdir(im_dir)
            im_names.sort()

            for img_id, im_name in enumerate(im_names):
                im_path = os.path.join(im_dir, im_name)
                box = self.boxes[cont_id][img_id]
                box = self.normalize_box(box)
                self._imdb.append({'im_path': im_path, 'class': cont_id, 'box': box})

    def __getitem__(self, index):
        # Load PIL image
        img = Image.open(self._imdb[index]['im_path'])
        img = img.convert('RGB')
        # Preprocess the image for training and testing
        img = self._preprocess_img(img)
        # Retrieve the label
        box = np.asarray(self._imdb[index]['box'], dtype=np.float32)

        return img, box


class ImageNetFull(ImageNetForCrop):
    def __init__(self, root, split, box_root):
        self.box_root = box_root
        self.box_fname = cfg.CROP.TYPE + '_' + cfg.CROP.SHAPE + '_' + split + '.npy'
        super(ImageNetFull, self).__init__(root, split)

    def _construct_imdb(self):
        """Constructs the image database."""
        # Compile the split data path
        split_path = os.path.join(self._root, self._split)
        boxes_path = os.path.join(self.box_root, self.box_fname)
        # Images are stored per class in subdirs (format: n<number>)
        split_folders = os.listdir(split_path)
        self._class_ids = sorted(f for f in split_folders if re.match(r"^n[0-9]+$", str(f)))
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Load bounding boxes as the label
        self.boxes = np.load(boxes_path, allow_pickle=True)

        # Construct the image database
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)

            im_names = os.listdir(im_dir)
            im_names.sort()

            for img_id, im_name in enumerate(im_names):
                im_path = os.path.join(im_dir, im_name)
                box = self.boxes[cont_id][img_id]
                # box = self.normalize_box(box)
                self._imdb.append({'im_path': im_path, 'class': cont_id, 'box': box})

    def __getitem__(self, index):
        # Load PIL image
        img = Image.open(self._imdb[index]['im_path'])
        img = img.convert('RGB')
        # Preprocess the image for training and testing
        img = self._preprocess_img(img)
        # Retrieve the label
        box = np.asarray(self._imdb[index]['box'], dtype=np.int)
        label = self._imdb[index]['class']

        return img, label, box


class ImageNetTune(ImageNetForCrop):
    def __init__(self, root, split):
        self.tune_root = cfg.OUT_DIR
        self.tune_fname = 'tune_ids_' + cfg.CROP.SHAPE + '_' + split + '.npy'
        self.box_root = cfg.OUT_DIR
        self.box_fname = 'cropnet_anynet_light_box' + '_' + cfg.CROP.SHAPE + '_' + split + '.npy'
        super(ImageNetTune, self).__init__(root, split)

    def _construct_imdb(self):
        """Construct the image database. Use the tune id as the label"""
        # Compile the split data path
        split_path = os.path.join(self._root, self._split)
        boxes_path = os.path.join(self.box_root, self.box_fname)
        tune_path = os.path.join(self.tune_root, self.tune_fname)
        # Images are stored per class in subdirs (format: n<number>)
        split_folders = os.listdir(split_path)
        self._class_ids = sorted(f for f in split_folders if re.match(r"^n[0-9]+$", str(f)))
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Load bounding boxes as the label
        self.boxes = np.load(boxes_path, allow_pickle=True)
        self.tunes = np.load(tune_path, allow_pickle=True)

        # Construct the image database
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)

            im_names = os.listdir(im_dir)
            im_names.sort()

            for img_id, im_name in enumerate(im_names):
                im_path = os.path.join(im_dir, im_name)
                box = self.boxes[cont_id][img_id]
                tune = self.tunes[cont_id][img_id]
                # box = self.normalize_box(box)
                self._imdb.append({'im_path': im_path, 'class': cont_id, 'box': box, 'tune': tune})

    def __getitem__(self, index):
        # Load PIL image
        img = Image.open(self._imdb[index]['im_path'])
        img = img.convert('RGB')
        # Preprocess the image for training and testing
        img = self._preprocess_img(img)
        # Retrieve the label
        box = np.asarray(self._imdb[index]['box'], dtype=np.float32)
        label = self._imdb[index]['class']
        tune_id = self._imdb[index]['tune']

        return img, box, tune_id


class ImageNetHP(ImageNetBox):
    def __init__(self, root, split):
        super(ImageNetHP, self).__init__(root, split)
        self.flip_ratio = 0.5
        self.flipped = False

    def map_box(self, img, box):
        w, h = img.size

        if h < w:
            scale = h / self.test_size
            h_pad = (self.test_size - self.train_size) / 2
            w_pad = (w / scale - self.train_size) / 2
            box[1], box[3] = int((box[1] + h_pad) * scale), int((box[3] + h_pad) * scale)
            box[0], box[2] = int((box[0] + w_pad) * scale), int((box[2] + w_pad) * scale)
        else:
            scale = w / self.test_size
            w_pad = (self.test_size - self.train_size) / 2
            h_pad = (h / scale - self.train_size) / 2
            box[0], box[2] = int((box[0] + w_pad) * scale), int((box[2] + w_pad) * scale)
            box[1], box[3] = int((box[1] + h_pad) * scale), int((box[3] + h_pad) * scale)

        return box

    @staticmethod
    def _get_border(border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _preprocess_img(self, img):
        if self._split == 'train':
            img, crop_info = transforms.random_resized_crop(img, self.test_size, self.train_size, max_iter=1)
            # Random horizontal flip
            if np.random.random() < self.flip_ratio:
                self.flipped = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                self.flipped = False
            # Color augmentation
            img = transforms.lighting(img, _PCA_STD, _EIG_VALS, _EIG_VECS)
        else:
            img = transforms.scale_and_center_crop(img, self.test_size, self.train_size)
            crop_info = None

        if Image.isImageType(img):
            img = np.array(img).astype(np.float32) / 255.
        # Color normalization
        img = transforms.color_norm(img, _MEAN, _STD)
        # Convert HWC/RGB/float to CHW/RGB/float format
        img = img.transpose([2, 0, 1])

        return img, crop_info

    def _box_affine(self, box, trans_inp):
        xmin, ymin, width, height = trans_inp
        x_ratio = self.train_size / width
        y_ratio = self.train_size / height

        box[0] = max(0, box[0] - xmin) * x_ratio
        box[1] = max(0, box[1] - ymin) * y_ratio
        box[2] = min(width, box[2] - xmin) * x_ratio - 1
        box[3] = min(height, box[3] - ymin) * y_ratio - 1

        return box

    def __getitem__(self, index):
        # Load the image
        img = Image.open(self._imdb[index]['im_path'])
        img = img.convert('RGB')
        img_size = img.size
        # Preprocess the image for training and testing
        inp, trans_inp = self._preprocess_img(img)

        # Preprocess the boundng box
        class_id = self._imdb[index]['class']
        box = np.array(self._imdb[index]['box'])
        if trans_inp is not None:
            # Map the box to the original image
            box = self.map_box(img, box)
            box = self._box_affine(box, trans_inp)

        if self.flipped:
            box[0], box[2] = self.train_size - box[2], self.train_size - box[0]
        # Generate gaussian heatmap
        class_hp = get_gaussian_hp(box)
        hp_size = class_hp.shape[-1]
        hp = np.zeros((cfg.MODEL.NUM_CLASSES, hp_size, hp_size))
        hp[class_id] = class_hp

        return inp, np.ascontiguousarray(hp)


if __name__ == '__main__':
    dataset = ImageNetHP(cfg.DATASET.PATH + '/imagenet', 'val')
    print(dataset.__getitem__(0))

